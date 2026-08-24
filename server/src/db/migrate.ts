import Database from "better-sqlite3";
import { readFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { log } from "../log.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

let db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (db) return db;
  const path = process.env.DB_PATH ?? join(process.cwd(), "llamatoaster.db");
  // better-sqlite3 will happily create the .db file itself but not any
  // missing parent directory (e.g. DB_PATH=.../data/llamatoaster.db with no
  // data/ yet) -- it throws "Cannot open database because the directory
  // does not exist" instead, which took down every DB-touching route on the
  // VPS after a redeploy that didn't preserve data/. Same mkdirSync-before-
  // write pattern the worker already uses for its own model_dir.
  mkdirSync(dirname(path), { recursive: true });
  db = new Database(path);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  migrate(db);
  return db;
}

function migrate(database: Database.Database): void {
  const sql = readFileSync(join(__dirname, "schema.sql"), "utf8");
  database.exec(sql);
  applyColumnMigrations(database);
  migrateHfGgufIndexPk(database);
  createResultsItemUniqueIndex(database);
  createWorkerEnrolmentIndexes(database);
  createMultiTenancyIndexes(database);
  createHfGgufIndexSha256Index(database);
  createHfGgufIndexDeletedAtIndex(database);
}

interface ColumnSpec {
  table: string;
  column: string;
  ddlType: string;
}

// CREATE TABLE IF NOT EXISTS (above) only helps a brand-new DB file -- it's a
// no-op against a table that already exists, so any column added to
// schema.sql after a DB has been created elsewhere (in particular the live
// VPS DB, which this repo has no automated deploy/migration path to -- see
// project memory) needs an explicit ALTER TABLE here too. Each entry is
// idempotent: checked via PRAGMA table_info before altering, so re-running
// this on every boot (including against a DB that already has the column) is
// safe.
const COLUMN_MIGRATIONS: ColumnSpec[] = [
  { table: "local_model_cache", column: "state", ddlType: "TEXT NOT NULL DEFAULT 'detected'" },
  { table: "run_items", column: "ram_avg_mib", ddlType: "INTEGER" },
  { table: "run_items", column: "vram_avg_mib", ddlType: "INTEGER" },
  { table: "run_items", column: "ram_free_before_mib", ddlType: "INTEGER" },
  { table: "run_items", column: "vram_free_before_mib", ddlType: "INTEGER" },
  { table: "run_items", column: "live_tps", ddlType: "REAL" },
  { table: "results", column: "idx", ddlType: "INTEGER" },
  { table: "results", column: "ram_avg_mib", ddlType: "INTEGER" },
  { table: "results", column: "vram_avg_mib", ddlType: "INTEGER" },
  { table: "results", column: "ram_free_before_mib", ddlType: "INTEGER" },
  { table: "results", column: "vram_free_before_mib", ddlType: "INTEGER" },
  { table: "run_items", column: "mtp", ddlType: "TEXT DEFAULT 'off'" },
  { table: "results", column: "mtp", ddlType: "TEXT DEFAULT 'off'" },
  // See shared/types.ts's ResultRow/IngestResultInput for what these mean --
  // only ever populated by the llama-server/MTP path (worker/src/serverBench.ts).
  // suspect_samples is a JSON-encoded number[] (SQLite has no array type),
  // NULL when there were no suspect readings for that row.
  { table: "results", column: "sample_count", ddlType: "INTEGER" },
  { table: "results", column: "suspect_count", ddlType: "INTEGER" },
  { table: "results", column: "suspect_samples", ddlType: "TEXT" },
  // Every individual repeat's raw reading (JSON number[], same storage
  // approach as suspect_samples) -- see shared/types.ts's ResultRow doc
  // comment for how this differs from suspect_samples.
  { table: "results", column: "repeat_samples", ddlType: "TEXT" },
  // llama-server /metrics speculative-decoding counters (tg row of an MTP
  // item only) -- see shared/types.ts's ResultRow doc comment.
  { table: "results", column: "spec_drafted", ddlType: "INTEGER" },
  { table: "results", column: "spec_accepted", ddlType: "INTEGER" },
  // Cross-backend RAM/VRAM/offload metrics -- see shared/types.ts's
  // ResultRow doc comments and worker/src/vram.ts's per-backend
  // readGpuMemory for what populates these. NULL on every row inserted
  // before this migration, which correctly reads as "unavailable" (this
  // data genuinely didn't exist yet, not lost).
  { table: "runs", column: "backend_device_name", ddlType: "TEXT" },
  { table: "results", column: "system_memory_total_mib", ddlType: "INTEGER" },
  { table: "results", column: "gpu_memory_total_mib", ddlType: "INTEGER" },
  { table: "results", column: "gpu_memory_total_accuracy", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_total_source", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_free_start_accuracy", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_free_start_source", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_model_avg_accuracy", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_model_avg_source", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_model_peak_accuracy", ddlType: "TEXT" },
  { table: "results", column: "gpu_memory_model_peak_source", ddlType: "TEXT" },
  { table: "results", column: "gpu_layers_loaded", ddlType: "INTEGER" },
  { table: "results", column: "total_model_layers", ddlType: "INTEGER" },
  // The MTP/draft companion model's own actual offload -- see
  // shared/types.ts's ResultRow doc comment and worker/src/index.ts's
  // parseOffloadLayers.
  { table: "results", column: "gpu_layers_loaded_draft", ddlType: "INTEGER" },
  { table: "results", column: "total_model_layers_draft", ddlType: "INTEGER" },
  // Estimated actually-VRAM-resident base-model layer count, written by a
  // current worker on every row where it was computable (llama.cpp's own
  // post-allocation buffer split, or the VRAM-discrepancy heuristic when
  // that report was missing) -- see shared/types.ts's ResultRow doc comment
  // and shared/vramEstimate.ts's estimateResidentGpuLayers(FromBufferSizes).
  // NULL on every row inserted before this migration and on any row with
  // nothing to estimate (no offload claim, or no buffer report/baseline).
  { table: "results", column: "gpu_layers_resident_est", ddlType: "INTEGER" },
  // The MTP/draft companion model's own actually-resident estimate -- see
  // shared/types.ts's ResultRow.gpu_layers_resident_est_draft. NULL on every
  // row inserted before this migration and on any non-MTP row (the draft
  // model's buffer lines only exist on an MTP item with a separate
  // --model-draft file).
  { table: "results", column: "gpu_layers_resident_est_draft", ddlType: "INTEGER" },
  // -ngld for the MTP/draft companion model -- see shared/sweep.ts's
  // SweepItem.n_gpu_layers_draft. DEFAULT 0 (not NULL) so a row/item
  // predating this column reads the same as "not applicable", matching
  // every other value on that row -- there's no way to recover what would
  // have been requested, and 0 is what worker/src/serverBench.ts's buildArgs
  // already treats as "don't offload the draft model".
  { table: "run_items", column: "n_gpu_layers_draft", ddlType: "INTEGER DEFAULT 0" },
  { table: "results", column: "n_gpu_layers_draft", ddlType: "INTEGER DEFAULT 0" },
  // Multi-user Stage 1 (MULTIUSER_PLAN.md §1.2) -- which `workers` row
  // dispatched this run. Nullable: legacy rows predating the workers table,
  // and any run whose worker was later deleted (ON DELETE SET NULL), simply
  // have no worker_id -- runs.worker_name stays the point-in-time snapshot
  // the export reads directly, so those rows remain fully readable.
  { table: "runs", column: "worker_id", ddlType: "TEXT REFERENCES workers(id) ON DELETE SET NULL" },
  // Multi-user Stage 3 (MULTIUSER_PLAN.md §3.3) -- device-flow machine
  // enrolment. NULL until approved (user_id) / not currently mid-enrolment
  // (the other four) -- a Stage 1 worker created via the shared secret has
  // all five NULL and is indistinguishable from any other not-yet-enrolled
  // machine, which is exactly what lets it adopt Stage 3 enrolment later
  // with no separate migration path (§3.4's own point).
  { table: "workers", column: "user_id", ddlType: "TEXT REFERENCES users(id) ON DELETE CASCADE" },
  // sha256(device_code) -- worker-held, never shown to a human, looked up by
  // hash exactly like a session token (§2.2). NOT the same value as
  // user_code below, which is the short human-facing code.
  { table: "workers", column: "enrolment_code_hash", ddlType: "TEXT" },
  { table: "workers", column: "user_code", ddlType: "TEXT" },
  { table: "workers", column: "enrolment_expires_at", ddlType: "INTEGER" },
  { table: "workers", column: "approved_at", ddlType: "INTEGER" },
  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.1) -- nullable: a run/result/
  // item predating auth (or created while AUTH_ENABLED is off) simply has no
  // owner, same "no migration forces a value" posture as runs.worker_id
  // above. §4.2's legacy migration backfills these to the first superadmin's
  // id once one exists; until then NULL correctly reads as single-tenant.
  { table: "runs", column: "user_id", ddlType: "TEXT REFERENCES users(id) ON DELETE CASCADE" },
  { table: "results", column: "user_id", ddlType: "TEXT REFERENCES users(id) ON DELETE CASCADE" },
  { table: "run_items", column: "user_id", ddlType: "TEXT REFERENCES users(id) ON DELETE CASCADE" },
  // No ON DELETE clause (unlike the three above) -- matches the plan's own
  // DDL exactly (§4.1): the model catalog is global, so a deleted user's
  // created_by attribution on a model nobody else has touched should block
  // that user's deletion rather than silently orphaning/cascading a shared
  // catalog row. No user-deletion feature exists yet, so this is currently
  // untested in practice, just faithful to the plan.
  { table: "models", column: "created_by", ddlType: "TEXT REFERENCES users(id)" },
  // --n-cpu-moe -- see shared/sweep.ts's SweepItem.n_cpu_moe. Same DEFAULT 0
  // reasoning as n_gpu_layers_draft above.
  { table: "run_items", column: "n_cpu_moe", ddlType: "INTEGER DEFAULT 0" },
  { table: "results", column: "n_cpu_moe", ddlType: "INTEGER DEFAULT 0" },
  // Cached worker.vram_json -- see shared/types.ts's Worker.vram doc comment
  // (pull-queue model: this is populated from the worker's own heartbeat,
  // not a live server->worker proxy read).
  { table: "workers", column: "vram_json", ddlType: "TEXT" },
  // Cancel Download (distinct from Pause) -- set alongside cancel_requested
  // when the user wants a download's partial .part file deleted, not kept
  // for resume (server/src/routes/workers.ts's cancel route,
  // worker/src/index.ts's discard handling). Meaningless for any other job
  // type; only ever read/set for 'download_model' rows.
  { table: "worker_jobs", column: "discard_requested", ddlType: "INTEGER DEFAULT 0" },
  // Soft-delete marker for the HF gguf index -- see server/src/hf-index.ts's
  // module doc comment. Nullable: NULL means live, non-null means confirmed
  // gone from Hugging Face as of that timestamp.
  { table: "hf_gguf_index", column: "deleted_at", ddlType: "INTEGER" },
];

function applyColumnMigrations(database: Database.Database): void {
  for (const { table, column, ddlType } of COLUMN_MIGRATIONS) {
    const columns = database.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[];
    if (columns.some((c) => c.name === column)) continue;
    database.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${ddlType}`);
  }
  backfillResultIdx(database);
}

// results.idx is new (see COLUMN_MIGRATIONS above) and ALTER TABLE ADD
// COLUMN doesn't backfill existing rows, so any results row inserted before
// this migration has idx = NULL -- which would break the client's
// item<->result join for every already-completed run. Recover it from the
// one ordering invariant that already held before idx existed: the worker
// processes run_items strictly in idx order, one at a time (see
// worker/src/index.ts's sweep loop), so a run's results rows -- ordered by
// created_at -- line up 1:1 with that run's 'done' run_items ordered by idx.
// Only touches rows still missing idx, so this is a no-op on every boot
// after the first.
function backfillResultIdx(database: Database.Database): void {
  const pending = database
    .prepare(`SELECT id, run_id FROM results WHERE idx IS NULL ORDER BY run_id, created_at ASC`)
    .all() as { id: string; run_id: string }[];
  if (pending.length === 0) return;

  const byRun = new Map<string, string[]>();
  for (const row of pending) {
    const list = byRun.get(row.run_id) ?? [];
    list.push(row.id);
    byRun.set(row.run_id, list);
  }

  const doneIdxStmt = database.prepare(
    `SELECT idx FROM run_items WHERE run_id = ? AND status = 'done' ORDER BY idx ASC`
  );
  const updateStmt = database.prepare(`UPDATE results SET idx = ? WHERE id = ?`);
  const tx = database.transaction(() => {
    for (const [runId, resultIds] of byRun) {
      const doneItems = doneIdxStmt.all(runId) as { idx: number }[];
      resultIds.forEach((resultId, i) => {
        const idx = doneItems[i]?.idx;
        if (idx != null) updateStmt.run(idx, resultId);
      });
    }
  });
  tx();
}

// Multi-user Stage 1 (MULTIUSER_PLAN.md §1.2): guards against a retried
// terminal item report (safeItemTerminal, worker/src/index.ts) inserting a
// SECOND set of results rows for the same (run_id, idx, test_type) -- which
// stops being theoretical once a worker talks to the server over the public
// internet instead of a trusted tailnet.
//
// Must run AFTER applyColumnMigrations/backfillResultIdx above: results.idx
// only exists as a column once that migration has applied, so creating this
// index any earlier (e.g. as static DDL in schema.sql, which execs before
// this function runs) would fail with "no such column: idx" on any DB whose
// results table predates that column.
//
// Self-healing rather than a manual pre-deploy step: any pre-existing
// duplicate rows are deduped (keep the earliest created_at, matching
// backfillResultIdx's own tie-breaking convention) before the index is
// created, so this is idempotent and safe to run on every boot -- the same
// posture as every other migration in this file. Every deletion is logged so
// it is never silent. The operational pre-migration backup
// (scripts/pre-multiuser-backup.sh) still gives an operator a chance to
// inspect duplicates BEFORE this ever runs against the live DB.
function createResultsItemUniqueIndex(database: Database.Database): void {
  const existing = database
    .prepare(`SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_results_item'`)
    .get();
  if (existing) return;

  const dupeGroups = database
    .prepare(
      `SELECT run_id, idx, test_type, COUNT(*) c FROM results
       WHERE idx IS NOT NULL
       GROUP BY run_id, idx, test_type HAVING c > 1`
    )
    .all() as { run_id: string; idx: number; test_type: string; c: number }[];

  if (dupeGroups.length > 0) {
    const keepEarliestExceptStmt = database.prepare(
      `DELETE FROM results WHERE run_id = ? AND idx = ? AND test_type = ? AND id NOT IN (
         SELECT id FROM results WHERE run_id = ? AND idx = ? AND test_type = ?
         ORDER BY created_at ASC LIMIT 1
       )`
    );
    const tx = database.transaction(() => {
      for (const g of dupeGroups) {
        const result = keepEarliestExceptStmt.run(g.run_id, g.idx, g.test_type, g.run_id, g.idx, g.test_type);
        log.warn(
          `[migrate] idx_results_item: deduped ${result.changes} duplicate result row(s) for ` +
            `run_id=${g.run_id} idx=${g.idx} test_type=${g.test_type} (kept earliest by created_at)`
        );
      }
    });
    tx();
  }

  database.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_results_item ON results(run_id, idx, test_type)`);
}

// Multi-user Stage 3 (MULTIUSER_PLAN.md §3.3): must run AFTER
// applyColumnMigrations above -- these all reference columns that only exist
// once that migration has applied, same "no such column" ordering issue
// createResultsItemUniqueIndex's own comment already explains for
// results.idx. The partial unique index is the actual security property
// from §3.2: no two PENDING enrolments can ever share a user_code, so a
// correct guess of one is never ambiguous about which machine it approves.
function createWorkerEnrolmentIndexes(database: Database.Database): void {
  database.exec(`CREATE INDEX IF NOT EXISTS idx_workers_enrolment ON workers(enrolment_code_hash)`);
  database.exec(`CREATE INDEX IF NOT EXISTS idx_workers_user ON workers(user_id)`);
  database.exec(
    `CREATE UNIQUE INDEX IF NOT EXISTS idx_workers_user_code_pending ON workers(user_code) WHERE approved_at IS NULL`
  );
}

// Migrate legacy hf_gguf_index primary key from sha256-only to
// composite (sha256, repo_id, filename). Old tables have
// `pk=1` only on sha256 column; new schema has pk on three columns.
// If legacy detected, recreate table preserving data (deduplicate via
// REPLACE semantics -- last_seen wins, same as upsert).
function migrateHfGgufIndexPk(database: Database.Database): void {
  const tableInfo = database.prepare(`PRAGMA table_info(hf_gguf_index)`).all() as { name: string; pk: number }[];
  if (tableInfo.length === 0) return;
  const pkCols = tableInfo.filter((c) => c.pk > 0).map((c) => c.name).sort();
  const needsMigration = pkCols.length === 1 && pkCols[0] === "sha256";
  if (!needsMigration) return;
  // Legacy table -- recreate with composite PK
  database.exec(`
    CREATE TABLE hf_gguf_index_new (
      sha256 TEXT NOT NULL,
      repo_id TEXT NOT NULL,
      filename TEXT NOT NULL,
      revision TEXT,
      file_size INTEGER,
      last_seen INTEGER NOT NULL,
      PRIMARY KEY (sha256, repo_id, filename)
    );
    INSERT OR REPLACE INTO hf_gguf_index_new (sha256, repo_id, filename, revision, file_size, last_seen)
      SELECT sha256, repo_id, filename, revision, file_size, last_seen FROM hf_gguf_index;
    DROP TABLE hf_gguf_index;
    ALTER TABLE hf_gguf_index_new RENAME TO hf_gguf_index;
  `);
}

function createHfGgufIndexSha256Index(database: Database.Database): void {
  database.exec(`CREATE INDEX IF NOT EXISTS idx_hf_gguf_index_sha256 ON hf_gguf_index(sha256)`);
}

// Same "must run after applyColumnMigrations" ordering rule as the function
// above -- deleted_at only exists as a column once that migration has
// applied (COLUMN_MIGRATIONS), so an index on it in schema.sql's static
// CREATE TABLE block would fail on any DB that already had hf_gguf_index
// before this column was added.
function createHfGgufIndexDeletedAtIndex(database: Database.Database): void {
  database.exec(`CREATE INDEX IF NOT EXISTS idx_hf_gguf_index_deleted_at ON hf_gguf_index(deleted_at)`);
}

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.1) -- same "must run after
// applyColumnMigrations" ordering rule as the two index functions above:
// user_id/created_by only exist as columns once that migration has applied.
// idx_runs_worker was always missing (an oversight from Stage 1 -- runs.worker_id
// itself was added via COLUMN_MIGRATIONS, but no index ever followed it),
// added here rather than left for another stage to rediscover.
function createMultiTenancyIndexes(database: Database.Database): void {
  database.exec(`CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_id)`);
  database.exec(`CREATE INDEX IF NOT EXISTS idx_runs_worker ON runs(worker_id)`);
  database.exec(`CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id)`);
  database.exec(`CREATE INDEX IF NOT EXISTS idx_run_items_user ON run_items(user_id)`);
  database.exec(`CREATE INDEX IF NOT EXISTS idx_models_created_by ON models(created_by)`);
}
