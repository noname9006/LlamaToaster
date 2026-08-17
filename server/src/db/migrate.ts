import Database from "better-sqlite3";
import { readFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

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
  // -ngld for the MTP/draft companion model -- see shared/sweep.ts's
  // SweepItem.n_gpu_layers_draft. DEFAULT 0 (not NULL) so a row/item
  // predating this column reads the same as "not applicable", matching
  // every other value on that row -- there's no way to recover what would
  // have been requested, and 0 is what worker/src/serverBench.ts's buildArgs
  // already treats as "don't offload the draft model".
  { table: "run_items", column: "n_gpu_layers_draft", ddlType: "INTEGER DEFAULT 0" },
  { table: "results", column: "n_gpu_layers_draft", ddlType: "INTEGER DEFAULT 0" },
  // --n-cpu-moe -- see shared/sweep.ts's SweepItem.n_cpu_moe. Same DEFAULT 0
  // reasoning as n_gpu_layers_draft above.
  { table: "run_items", column: "n_cpu_moe", ddlType: "INTEGER DEFAULT 0" },
  { table: "results", column: "n_cpu_moe", ddlType: "INTEGER DEFAULT 0" },
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
