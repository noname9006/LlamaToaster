// BENCHMARKING_PLAN_V8.md §0.11 -- every ADD COLUMN lands in the ordered
// column-migration list applied at boot; every new table and its indexes land
// in the static DDL; and any new FK column defaults to NULL (SQLite's
// requirement with foreign keys on).
//
// The upgrade drill these tests encode: a database that predates all of v8
// must come out the other side with every column, every table and every index
// present, every historical row still readable, and running the migration
// twice must change nothing.

import { afterAll, describe, expect, it, vi } from "vitest";
import Database from "better-sqlite3";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function freshMigrateModule(dbPath: string) {
  process.env.DB_PATH = dbPath;
  vi.resetModules();
  return (await import("./migrate.js")) as typeof import("./migrate.js");
}

const tmpDirs: string[] = [];
function makeTmpDbPath(): string {
  const dir = mkdtempSync(join(tmpdir(), "llamatoaster-v8-migrate-test-"));
  tmpDirs.push(dir);
  return join(dir, "test.db");
}

afterAll(() => {
  for (const dir of tmpDirs) {
    try {
      if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
    } catch {
      /* file still open on Windows -- fine, it's in the OS temp dir */
    }
  }
});

function columnsOf(db: Database.Database, table: string): string[] {
  return (db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[]).map((c) => c.name);
}

function indexesOf(db: Database.Database, table: string): string[] {
  return (db.prepare(`PRAGMA index_list(${table})`).all() as { name: string }[]).map((i) => i.name);
}

// A pre-v8 database: the tables as they existed before any of this plan.
function seedLegacyDatabase(dbPath: string): void {
  const db = new Database(dbPath);
  db.exec(`
    CREATE TABLE models (
      id TEXT PRIMARY KEY, filename TEXT, size_bytes INTEGER, source TEXT,
      hf_repo TEXT, hf_file TEXT, metadata TEXT, created_at INTEGER NOT NULL
    );
    CREATE TABLE runs (
      id TEXT PRIMARY KEY, worker_name TEXT, llama_cpp_build TEXT, llama_cpp_backend TEXT,
      model_id TEXT, config TEXT, status TEXT, error TEXT,
      started_at INTEGER NOT NULL, completed_at INTEGER
    );
    CREATE TABLE results (
      id TEXT PRIMARY KEY, run_id TEXT, idx INTEGER, model_id TEXT, test_type TEXT,
      n_prompt INTEGER, n_gen INTEGER, n_threads INTEGER, n_gpu_layers INTEGER,
      batch_size INTEGER, ubatch_size INTEGER, cache_type_k TEXT, cache_type_v TEXT,
      flash_attn TEXT, mtp TEXT, avg_tps REAL, stddev_tps REAL,
      ram_peak_mib INTEGER, vram_peak_mib INTEGER, created_at INTEGER NOT NULL
    );
    CREATE TABLE run_items (
      id TEXT PRIMARY KEY, run_id TEXT, idx INTEGER, n_prompt INTEGER, n_gen INTEGER,
      n_threads INTEGER, n_gpu_layers INTEGER, batch_size INTEGER, ubatch_size INTEGER,
      cache_type_k TEXT, cache_type_v TEXT, flash_attn TEXT, mtp TEXT, status TEXT
    );
    CREATE TABLE workers (
      id TEXT PRIMARY KEY, machine_id TEXT UNIQUE, display_name TEXT,
      created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
    );
    INSERT INTO models (id, filename, size_bytes, source, metadata, created_at)
      VALUES ('legacy-model', 'old.gguf', 1, 'local', '{}', 1);
    INSERT INTO runs (id, worker_name, llama_cpp_build, llama_cpp_backend, model_id, config, status, started_at)
      VALUES ('legacy-run', 'old-box', 'b1', 'cpu', 'legacy-model', '{}', 'done', 1);
    INSERT INTO results (id, run_id, idx, model_id, test_type, n_prompt, n_gen, n_threads,
                         n_gpu_layers, batch_size, ubatch_size, cache_type_k, cache_type_v,
                         flash_attn, mtp, avg_tps, stddev_tps, ram_peak_mib, vram_peak_mib, created_at)
      VALUES ('legacy-result', 'legacy-run', 0, 'legacy-model', 'tg', 512, 128, 4, 0,
              512, 512, 'f16', 'f16', 'on', 'off', 41.0, 0.3, 2000, NULL, 1);
    INSERT INTO run_items (id, run_id, idx, n_prompt, n_gen, n_threads, n_gpu_layers,
                           batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn, mtp, status)
      VALUES ('legacy-item', 'legacy-run', 0, 512, 128, 4, 0, 512, 512, 'f16', 'f16', 'on', 'off', 'done');
  `);
  db.close();
}

describe("v8 schema evolution (§0.11)", () => {
  it("upgrades a pre-v8 database with every new column, and keeps its historical rows", async () => {
    const dbPath = makeTmpDbPath();
    seedLegacyDatabase(dbPath);
    const { getDb } = await freshMigrateModule(dbPath);
    const db = getDb();

    // §0.5 run identity, §0.1 methodology stamp, §0.2 depth, §0.8 twin join,
    // §0.10 caveat flags, N5 concurrency, M6 thermal, N6 ISA, N7 import.
    expect(columnsOf(db, "runs")).toEqual(expect.arrayContaining(["root_run_id", "kind", "comparison_id"]));
    expect(columnsOf(db, "results")).toEqual(
      expect.arrayContaining([
        "method_version",
        "n_depth",
        "config_hash",
        "prompt_offset",
        "spec_type",
        "spec_n_max",
        "spec_n_min",
        "speedup",
        "speedup_status",
        "caveat_flags",
        "concurrency",
        "gpu_temp_c_max",
        "gpu_clock_mhz_min",
        "gpu_clock_samples",
        "cpu_isa",
        "worker_id",
        "llama_cpp_build",
        "engine",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_n",
        "e2e_ms_mean",
        "imported_bundle_id",
        "import_opt_in",
      ])
    );
    expect(columnsOf(db, "run_items")).toEqual(expect.arrayContaining(["n_depth", "concurrency"]));
    expect(columnsOf(db, "workers")).toEqual(
      expect.arrayContaining(["capabilities_json", "sensors_json", "cpu_isa"])
    );

    // Historical rows survive untouched, and every new column reads NULL (or
    // its stated default) on them.
    const legacy = db.prepare(`SELECT * FROM results WHERE id = 'legacy-result'`).get() as Record<string, unknown>;
    expect(legacy.avg_tps).toBe(41);
    expect(legacy.method_version).toBeNull();
    expect(legacy.config_hash).toBeNull();
    expect(legacy.caveat_flags).toBeNull();
    expect(legacy.imported_bundle_id).toBeNull();
    // The two columns with an explicit DEFAULT fill in rather than reading NULL.
    expect(legacy.n_depth).toBe(0);
    expect((db.prepare(`SELECT * FROM run_items WHERE id = 'legacy-item'`).get() as { concurrency: number }).concurrency).toBe(1);
  });

  it("creates the new tables with their FK-complete definitions and UNIQUE keys", async () => {
    const dbPath = makeTmpDbPath();
    seedLegacyDatabase(dbPath);
    const { getDb } = await freshMigrateModule(dbPath);
    const db = getDb();

    for (const table of ["model_machine_limits", "quality_results"]) {
      const exists = db
        .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`)
        .get(table);
      expect(exists, `${table} must exist`).toBeTruthy();
      // Every new FK has an explicit ON DELETE.
      const ddl = (db.prepare(`SELECT sql FROM sqlite_master WHERE name = ?`).get(table) as { sql: string }).sql;
      const references = ddl.match(/REFERENCES/g) ?? [];
      const onDeletes = ddl.match(/ON DELETE/g) ?? [];
      expect(onDeletes.length).toBe(references.length);
    }

    // N2's re-probe key and N4's retry key.
    expect(indexesOf(db, "model_machine_limits").length).toBeGreaterThan(0);
    expect(indexesOf(db, "quality_results").length).toBeGreaterThan(0);
    expect(() => {
      db.prepare(
        `INSERT INTO model_machine_limits (id, worker_id, model_id, llama_cpp_build, kv_type, placement_hash, verified_ctx_tokens, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).run("l1", "w1", "legacy-model", "b1", "f16/f16", "ph", 1000, 1);
    }).toThrow(); // worker_id FK: no such worker row.
  });

  it("creates the v8 indexes over columns that only exist after the ALTERs ran", async () => {
    const dbPath = makeTmpDbPath();
    seedLegacyDatabase(dbPath);
    const { getDb } = await freshMigrateModule(dbPath);
    const db = getDb();
    const names = (db.prepare(`SELECT name FROM sqlite_master WHERE type = 'index'`).all() as { name: string }[]).map(
      (r) => r.name
    );
    for (const index of [
      "idx_runs_root",
      "idx_runs_kind",
      "idx_runs_comparison",
      "idx_results_config_hash",
      "idx_results_curve",
    ]) {
      expect(names, `${index} must exist`).toContain(index);
    }
  });

  it("is idempotent: migrating an already-migrated database changes nothing", async () => {
    const dbPath = makeTmpDbPath();
    seedLegacyDatabase(dbPath);
    const first = await freshMigrateModule(dbPath);
    const before = {
      results: columnsOf(first.getDb(), "results"),
      runs: columnsOf(first.getDb(), "runs"),
      indexes: (first.getDb().prepare(`SELECT name FROM sqlite_master WHERE type='index'`).all() as { name: string }[])
        .map((r) => r.name)
        .sort(),
    };

    // Second boot against the same file.
    const second = await freshMigrateModule(dbPath);
    const after = {
      results: columnsOf(second.getDb(), "results"),
      runs: columnsOf(second.getDb(), "runs"),
      indexes: (second.getDb().prepare(`SELECT name FROM sqlite_master WHERE type='index'`).all() as { name: string }[])
        .map((r) => r.name)
        .sort(),
    };
    expect(after).toEqual(before);
    // And the historical row is still exactly where it was.
    expect(
      (second.getDb().prepare(`SELECT COUNT(*) AS n FROM results`).get() as { n: number }).n
    ).toBe(1);
  });

  it("migrates a brand-new database to the same shape as an upgraded one", async () => {
    const upgradedPath = makeTmpDbPath();
    seedLegacyDatabase(upgradedPath);
    const upgraded = await freshMigrateModule(upgradedPath);
    const upgradedResults = columnsOf(upgraded.getDb(), "results").sort();

    const freshPath = makeTmpDbPath();
    const fresh = await freshMigrateModule(freshPath);
    const freshResults = columnsOf(fresh.getDb(), "results").sort();

    // A column added by BOTH schema.sql and COLUMN_MIGRATIONS must not end up
    // in one path and not the other -- that divergence is what makes an
    // upgraded install behave differently from a fresh one.
    expect(freshResults).toEqual(upgradedResults);
  });

  // 2026-08-29 incident: local_model_cache existed (created by an earlier
  // schema.sql, before the `state` column was added to it) but predated
  // `state`. schema.sql's own idx_local_model_cache_state index -- created
  // unconditionally right after the table's CREATE TABLE IF NOT EXISTS no-op
  // -- threw "no such column: state" partway through database.exec(sql),
  // which aborted BEFORE applyColumnMigrations (the thing that would add
  // `state`) ever ran, so every other column migration silently got skipped
  // too. This must self-heal in one boot, not require a hand-run ALTER.
  it("self-heals a table that predates a column its own schema.sql index references (2026-08-29)", async () => {
    const dbPath = makeTmpDbPath();
    seedLegacyDatabase(dbPath);
    const seed = new Database(dbPath);
    // Every real local_model_cache column (schema.sql) except `state` --
    // matching the actual incident, where sha256/hf_model_id already existed
    // and only `state` (added later) was missing.
    seed.exec(`
      CREATE TABLE local_model_cache (
        path TEXT PRIMARY KEY,
        size INTEGER NOT NULL,
        mtime INTEGER NOT NULL,
        sha256 TEXT,
        hf_model_id TEXT,
        last_verified INTEGER NOT NULL
      );
      INSERT INTO local_model_cache (path, size, mtime, sha256, hf_model_id, last_verified)
        VALUES ('/models/old.gguf', 1, 1, 'abc123', NULL, 1);
    `);
    seed.close();

    const { getDb } = await freshMigrateModule(dbPath);
    // Must not throw -- the whole point of the fix.
    const db = getDb();

    expect(columnsOf(db, "local_model_cache")).toContain("state");
    expect(indexesOf(db, "local_model_cache")).toContain("idx_local_model_cache_state");
    // The historical row survives with the new column's stated default, and
    // every OTHER column migration (blocked by the same aborted first pass
    // in the old code) also actually applied -- not just this one column.
    const row = db.prepare(`SELECT * FROM local_model_cache WHERE path = '/models/old.gguf'`).get() as Record<
      string,
      unknown
    >;
    expect(row.state).toBe("detected");
    expect(columnsOf(db, "results")).toContain("gpu_clock_samples");
  });

  // Same 2026-08-29 incident, second half: migrateHfGgufIndexPk rebuilds
  // hf_gguf_index from a hardcoded column list to fix its legacy single-
  // column PK. That list predated `deleted_at` (added later via
  // COLUMN_MIGRATIONS, which runs BEFORE this function), so the rebuild
  // silently dropped a column that had just been backfilled onto the source
  // table moments earlier in the same boot -- then createHfGgufIndexDeletedAtIndex
  // crashed on the now-missing column right after.
  it("preserves a column added after migrateHfGgufIndexPk's rebuild SQL was written, when recreating a legacy-PK table", async () => {
    const dbPath = makeTmpDbPath();
    seedLegacyDatabase(dbPath);
    const seed = new Database(dbPath);
    // The legacy single-sha256-PK shape migrateHfGgufIndexPk detects --
    // predates `deleted_at` entirely, matching the real incident.
    seed.exec(`
      CREATE TABLE hf_gguf_index (
        sha256 TEXT NOT NULL PRIMARY KEY,
        repo_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        revision TEXT,
        file_size INTEGER,
        last_seen INTEGER NOT NULL
      );
      INSERT INTO hf_gguf_index (sha256, repo_id, filename, revision, file_size, last_seen)
        VALUES ('sha-legacy', 'org/repo', 'model.gguf', 'main', 1000, 1);
    `);
    seed.close();

    const { getDb } = await freshMigrateModule(dbPath);
    const db = getDb();

    // PK migrated to composite, AND deleted_at survived the rebuild.
    const pkCols = (db.prepare(`PRAGMA table_info(hf_gguf_index)`).all() as { name: string; pk: number }[])
      .filter((c) => c.pk > 0)
      .map((c) => c.name)
      .sort();
    expect(pkCols).toEqual(["filename", "repo_id", "sha256"]);
    expect(columnsOf(db, "hf_gguf_index")).toContain("deleted_at");
    expect(indexesOf(db, "hf_gguf_index")).toContain("idx_hf_gguf_index_deleted_at");

    const row = db.prepare(`SELECT * FROM hf_gguf_index WHERE sha256 = 'sha-legacy'`).get() as Record<
      string,
      unknown
    >;
    expect(row.repo_id).toBe("org/repo");
    expect(row.deleted_at).toBeNull();
  });
});
