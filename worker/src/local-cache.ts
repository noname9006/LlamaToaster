import Database from "better-sqlite3";
import { resolve } from "node:path";
import { existsSync, mkdirSync } from "node:fs";
import { log } from "./log.js";
import type { ModelDirFile, LocalModelState } from "../../shared/types.js";

export interface LocalCacheEntry {
  path: string;
  size: number;
  mtime: number;
  sha256?: string;
  hf_model_id?: string;
  // When this hf_model_id match was last confirmed against the server.
  // Absent for entries that predate re-verification support.
  hf_checked_at?: number;
  // Set when the server last reported this match's HF source as removed
  // (hf-index.ts's deleted_at). Cleared back to undefined once a
  // re-verification finds the match live again.
  hf_deleted_at?: number;
  // Per-file GGUF header metadata, read once by model-scanner.ts's
  // backfillGgufMetadata (and stamped at download time) -- the same fields
  // a download reports to the server's catalog, so a file dropped into
  // model_dir by hand (never downloaded through the app) can still get its
  // real quant/param_count/layer-count reported on the heartbeat instead of
  // sitting at "?" / "—". See worker/src/gguf.ts's readGgufInfo for what
  // each field means; all absent for a file whose header couldn't be parsed.
  n_layer?: number;
  mtp_layers?: number;
  quant?: string;
  param_count?: number;
  // Trained context + KV geometry (see gguf.ts's GgufInfo) -- persisted
  // alongside the fields above so heartbeats report the full set.
  trained_ctx?: number;
  n_head_kv?: number;
  head_dim_k?: number;
  head_dim_v?: number;
  n_embd?: number;
  n_head?: number;
  sliding_window?: number;
  // When this entry's GGUF header was last (successfully or unsuccessfully)
  // read -- gates backfillGgufMetadata's re-read window the same way
  // hf_checked_at gates HF re-verification, so a header that came back null
  // (unreadable/corrupt file) isn't re-read on every single reconciliation.
  gguf_checked_at?: number;
  last_verified: number;
  state: LocalModelState;
}

const LOCAL_CACHE_DB_NAME = "local-model-cache.sqlite";

// Public methods stay `async`/Promise-returning even though better-sqlite3
// itself is synchronous -- callers (model-scanner.ts, worker/src/index.ts)
// already await these, and keeping the same contract means this file is the
// only one that needed to change when the driver did. Was `sqlite`/`sqlite3`
// (async node-sqlite3 wrapper) until that package's unreliable prebuild
// coverage started crashing fresh `npm install`s with no C++ toolchain on
// PATH ("Could not locate the bindings file") -- better-sqlite3 is already
// this repo's server-side driver and has broad, actively-maintained prebuild
// coverage, so reusing it here removes the failure mode entirely instead of
// requiring every worker host to have Visual Studio Build Tools installed.
export class LocalModelCache {
  private db: Database.Database | null = null;
  private cacheDir: string;
  private dbPath: string;

  constructor(modelDir: string) {
    this.cacheDir = resolve(modelDir, ".cache");
    this.dbPath = resolve(this.cacheDir, LOCAL_CACHE_DB_NAME);
  }

  async init(): Promise<void> {
    try {
      if (!existsSync(this.cacheDir)) {
        mkdirSync(this.cacheDir, { recursive: true });
      }
    } catch (err) {
      // Read-only or permission error - fallback to in-memory DB to keep worker alive
      // Cache will be ephemeral but hashing will still work (just rehash each startup)
      log.warn(`Failed to create cache dir ${this.cacheDir}: ${err instanceof Error ? err.message : String(err)} - using in-memory cache`);
      this.db = new Database(":memory:");
      this.db.pragma("foreign_keys = ON");
      this.createTables();
      log.info(`Local model cache initialized in-memory (ephemeral)`);
      return;
    }

    // Try to open DB, with recovery for corrupted file
    try {
      this.db = new Database(this.dbPath);
    } catch (err) {
      log.warn(`Failed to open cache DB at ${this.dbPath}: ${err instanceof Error ? err.message : String(err)} - attempting recovery`);
      try {
        // Try to delete corrupted file and recreate
        const { unlinkSync } = await import("node:fs");
        try { unlinkSync(this.dbPath); } catch {}
        try { unlinkSync(`${this.dbPath}-wal`); } catch {}
        try { unlinkSync(`${this.dbPath}-shm`); } catch {}
        this.db = new Database(this.dbPath);
        log.info(`Cache DB recovered by deleting corrupted file`);
      } catch (recoveryErr) {
        log.error(`Cache recovery failed: ${recoveryErr instanceof Error ? recoveryErr.message : String(recoveryErr)} - using in-memory`);
        this.db = new Database(":memory:");
      }
    }

    try {
      // Enable WAL mode for better concurrency (may fail on in-memory or read-only)
      this.db.pragma("journal_mode = WAL");
    } catch {}
    try {
      this.db.pragma("synchronous = NORMAL");
    } catch {}
    try {
      this.db.pragma("foreign_keys = ON");
    } catch {}

    try {
      this.createTables();
    } catch (err) {
      // Table creation failed due to corruption or version mismatch - try to recreate from scratch
      log.warn(`Failed to create tables: ${err instanceof Error ? err.message : String(err)} - resetting cache`);
      try {
        this.db.exec(`DROP TABLE IF EXISTS local_model_cache`);
        this.createTables();
      } catch (e2) {
        log.error(`Cache reset failed: ${e2 instanceof Error ? e2.message : String(e2)} - using in-memory`);
        this.db.close();
        this.db = new Database(":memory:");
        this.createTables();
      }
    }
    log.info(`Local model cache initialized at ${this.dbPath}`);
  }

  private createTables(): void {
    if (!this.db) throw new Error("Database not initialized");

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS local_model_cache (
        path TEXT PRIMARY KEY,
        size INTEGER NOT NULL,
        mtime INTEGER NOT NULL,
        sha256 TEXT,
        hf_model_id TEXT,
        hf_checked_at INTEGER,
        hf_deleted_at INTEGER,
        last_verified INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'detected'
      );

      CREATE INDEX IF NOT EXISTS idx_local_model_cache_sha256 ON local_model_cache(sha256);
      CREATE INDEX IF NOT EXISTS idx_local_model_cache_hf ON local_model_cache(hf_model_id);
      CREATE INDEX IF NOT EXISTS idx_local_model_cache_state ON local_model_cache(state);
    `);

    // Migration for DBs created before these columns existed (CREATE TABLE IF
    // NOT EXISTS is a no-op on an already-existing table).
    const cols = this.db.prepare(`PRAGMA table_info(local_model_cache)`).all() as { name: string }[];
    if (!cols.some((c) => c.name === "state")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN state TEXT NOT NULL DEFAULT 'detected'`);
      this.db.exec(`CREATE INDEX IF NOT EXISTS idx_local_model_cache_state ON local_model_cache(state)`);
    }
    if (!cols.some((c) => c.name === "hf_checked_at")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN hf_checked_at INTEGER`);
    }
    if (!cols.some((c) => c.name === "hf_deleted_at")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN hf_deleted_at INTEGER`);
    }
    // GGUF header metadata (see LocalCacheEntry's doc comments) -- added
    // together so a single PRAGMA check covers all five.
    if (!cols.some((c) => c.name === "n_layer")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN n_layer INTEGER`);
    }
    if (!cols.some((c) => c.name === "mtp_layers")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN mtp_layers INTEGER`);
    }
    if (!cols.some((c) => c.name === "quant")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN quant TEXT`);
    }
    if (!cols.some((c) => c.name === "param_count")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN param_count INTEGER`);
    }
    if (!cols.some((c) => c.name === "gguf_checked_at")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN gguf_checked_at INTEGER`);
    }
    // Trained context + KV geometry -- added together so a single PRAGMA
    // check covers all seven (see LocalCacheEntry's doc comments).
    if (!cols.some((c) => c.name === "trained_ctx")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN trained_ctx INTEGER`);
    }
    if (!cols.some((c) => c.name === "n_head_kv")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN n_head_kv INTEGER`);
    }
    if (!cols.some((c) => c.name === "head_dim_k")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN head_dim_k INTEGER`);
    }
    if (!cols.some((c) => c.name === "head_dim_v")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN head_dim_v INTEGER`);
    }
    if (!cols.some((c) => c.name === "n_embd")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN n_embd INTEGER`);
    }
    if (!cols.some((c) => c.name === "n_head")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN n_head INTEGER`);
    }
    if (!cols.some((c) => c.name === "sliding_window")) {
      this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN sliding_window INTEGER`);
    }
    // First migration that adds the trained-context + KV-geometry columns
    // (this whole branch runs exactly once, the first time `trained_ctx`
    // appears). Rows written by an OLDER GGUF reader have those columns NULL
    // but a fresh `gguf_checked_at`, so the scanner's 24h staleness gate
    // (worker/src/model-scanner.ts's isGgufCheckStale) would suppress
    // re-reading them for up to a day. Clear the gate once so the next
    // reconciliation (backfillGgufMetadata) re-reads every header with the
    // richer reader -- a one-time cost per file, bounded by model count.
    // Subsequent startups hit the staleness gate as usual.
    this.db.exec(`UPDATE local_model_cache SET gguf_checked_at = NULL`);
  }

  // Every column, shared by all three SELECT sites (get/getAll/getBySha256)
  // and the mapRow read below -- one source of truth so a future column
  // addition can't silently drift between the reader and the writer.
  private static readonly SELECT_COLS =
    "path, size, mtime, sha256, hf_model_id, hf_checked_at, hf_deleted_at, " +
    "n_layer, mtp_layers, quant, param_count, gguf_checked_at, last_verified, state, " +
    "trained_ctx, n_head_kv, head_dim_k, head_dim_v, n_embd, n_head, sliding_window";

  async get(path: string): Promise<LocalCacheEntry | null> {
    if (!this.db) throw new Error("Database not initialized");

    const row = this.db
      .prepare(`SELECT ${LocalModelCache.SELECT_COLS} FROM local_model_cache WHERE path = ?`)
      .get(path);

    return row ? this.mapRow(row as Parameters<LocalModelCache["mapRow"]>[0]) : null;
  }

  async getAll(): Promise<LocalCacheEntry[]> {
    if (!this.db) throw new Error("Database not initialized");

    const rows = this.db
      .prepare(`SELECT ${LocalModelCache.SELECT_COLS} FROM local_model_cache`)
      .all();

    return (rows as Parameters<LocalModelCache["mapRow"]>[0][]).map((r) => this.mapRow(r));
  }

  async getBySha256(sha256: string): Promise<LocalCacheEntry | null> {
    if (!this.db) throw new Error("Database not initialized");

    const row = this.db
      .prepare(`SELECT ${LocalModelCache.SELECT_COLS} FROM local_model_cache WHERE sha256 = ?`)
      .get(sha256);

    return row ? this.mapRow(row as Parameters<LocalModelCache["mapRow"]>[0]) : null;
  }

  async upsert(entry: LocalCacheEntry): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    this.db
      .prepare(
        `INSERT INTO local_model_cache (path, size, mtime, sha256, hf_model_id, hf_checked_at, hf_deleted_at, n_layer, mtp_layers, quant, param_count, gguf_checked_at, last_verified, state, trained_ctx, n_head_kv, head_dim_k, head_dim_v, n_embd, n_head, sliding_window)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(path) DO UPDATE SET
           size = excluded.size,
           mtime = excluded.mtime,
           sha256 = excluded.sha256,
           hf_model_id = excluded.hf_model_id,
           hf_checked_at = excluded.hf_checked_at,
           hf_deleted_at = excluded.hf_deleted_at,
           n_layer = excluded.n_layer,
           mtp_layers = excluded.mtp_layers,
           quant = excluded.quant,
           param_count = excluded.param_count,
           gguf_checked_at = excluded.gguf_checked_at,
           last_verified = excluded.last_verified,
           state = excluded.state,
           trained_ctx = excluded.trained_ctx,
           n_head_kv = excluded.n_head_kv,
           head_dim_k = excluded.head_dim_k,
           head_dim_v = excluded.head_dim_v,
           n_embd = excluded.n_embd,
           n_head = excluded.n_head,
           sliding_window = excluded.sliding_window`
      )
      .run(
        entry.path,
        entry.size,
        entry.mtime,
        entry.sha256 ?? null,
        entry.hf_model_id ?? null,
        entry.hf_checked_at ?? null,
        entry.hf_deleted_at ?? null,
        entry.n_layer ?? null,
        entry.mtp_layers ?? null,
        entry.quant ?? null,
        entry.param_count ?? null,
        entry.gguf_checked_at ?? null,
        entry.last_verified,
        entry.state,
        entry.trained_ctx ?? null,
        entry.n_head_kv ?? null,
        entry.head_dim_k ?? null,
        entry.head_dim_v ?? null,
        entry.n_embd ?? null,
        entry.n_head ?? null,
        entry.sliding_window ?? null
      );
  }

  async updateState(path: string, state: LocalModelState): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    this.db
      .prepare(`UPDATE local_model_cache SET state = ?, last_verified = ? WHERE path = ?`)
      .run(state, Date.now(), path);
  }

  async updateHash(path: string, sha256: string): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    this.db
      .prepare(`UPDATE local_model_cache SET sha256 = ?, state = ?, last_verified = ? WHERE path = ?`)
      .run(sha256, "hashing", Date.now(), path);
  }

  // deletedAt: the server's hf_gguf_index.deleted_at for this match (null if
  // live). Always stamps hf_checked_at with now -- this call IS a fresh
  // verification, whether it's the first match or a periodic re-check of an
  // already-matched entry (see model-scanner.ts's resolveHfMetadata).
  async updateHfMatch(path: string, hf_model_id: string, deletedAt: number | null = null): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    this.db
      .prepare(
        `UPDATE local_model_cache SET hf_model_id = ?, hf_checked_at = ?, hf_deleted_at = ?, state = ?, last_verified = ? WHERE path = ?`
      )
      .run(hf_model_id, Date.now(), deletedAt, "verified", Date.now(), path);
  }

  async delete(path: string): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    this.db.prepare(`DELETE FROM local_model_cache WHERE path = ?`).run(path);
  }

  async deleteMissing(existingPaths: string[]): Promise<number> {
    if (!this.db) throw new Error("Database not initialized");

    if (existingPaths.length === 0) {
      // Delete all entries if no files exist
      const result = this.db.prepare(`DELETE FROM local_model_cache`).run();
      return result.changes ?? 0;
    }

    const placeholders = existingPaths.map(() => "?").join(",");
    const result = this.db
      .prepare(`DELETE FROM local_model_cache WHERE path NOT IN (${placeholders})`)
      .run(...existingPaths);
    return result.changes ?? 0;
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  private mapRow(row: {
    path: string;
    size: number;
    mtime: number;
    sha256: string | null;
    hf_model_id: string | null;
    hf_checked_at: number | null;
    hf_deleted_at: number | null;
    n_layer: number | null;
    mtp_layers: number | null;
    quant: string | null;
    param_count: number | null;
    gguf_checked_at: number | null;
    last_verified: number;
    state: string;
    trained_ctx: number | null;
    n_head_kv: number | null;
    head_dim_k: number | null;
    head_dim_v: number | null;
    n_embd: number | null;
    n_head: number | null;
    sliding_window: number | null;
  }): LocalCacheEntry {
    return {
      path: row.path,
      size: row.size,
      mtime: row.mtime,
      sha256: row.sha256 ?? undefined,
      hf_model_id: row.hf_model_id ?? undefined,
      hf_checked_at: row.hf_checked_at ?? undefined,
      hf_deleted_at: row.hf_deleted_at ?? undefined,
      n_layer: row.n_layer ?? undefined,
      mtp_layers: row.mtp_layers ?? undefined,
      quant: row.quant ?? undefined,
      param_count: row.param_count ?? undefined,
      gguf_checked_at: row.gguf_checked_at ?? undefined,
      last_verified: row.last_verified,
      state: row.state as LocalModelState,
      trained_ctx: row.trained_ctx ?? undefined,
      n_head_kv: row.n_head_kv ?? undefined,
      head_dim_k: row.head_dim_k ?? undefined,
      head_dim_v: row.head_dim_v ?? undefined,
      n_embd: row.n_embd ?? undefined,
      n_head: row.n_head ?? undefined,
      sliding_window: row.sliding_window ?? undefined,
    };
  }
}

// Factory function to create and initialize the cache
export async function createLocalModelCache(modelDir: string): Promise<LocalModelCache> {
  const cache = new LocalModelCache(modelDir);
  await cache.init();
  return cache;
}
