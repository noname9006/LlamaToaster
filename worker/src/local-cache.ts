import { open, type Database } from "sqlite";
import sqlite3 from "sqlite3";
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
  last_verified: number;
  state: LocalModelState;
}

const LOCAL_CACHE_DB_NAME = "local-model-cache.sqlite";

export class LocalModelCache {
  private db: Database | null = null;
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
      this.db = await open({
        filename: ":memory:",
        driver: sqlite3.Database,
      });
      await this.db.exec("PRAGMA foreign_keys = ON;");
      await this.createTables();
      log.info(`Local model cache initialized in-memory (ephemeral)`);
      return;
    }

    // Try to open DB, with recovery for corrupted file
    try {
      this.db = await open({
        filename: this.dbPath,
        driver: sqlite3.Database,
      });
    } catch (err) {
      log.warn(`Failed to open cache DB at ${this.dbPath}: ${err instanceof Error ? err.message : String(err)} - attempting recovery`);
      try {
        // Try to delete corrupted file and recreate
        const { unlinkSync } = await import("node:fs");
        try { unlinkSync(this.dbPath); } catch {}
        try { unlinkSync(`${this.dbPath}-wal`); } catch {}
        try { unlinkSync(`${this.dbPath}-shm`); } catch {}
        this.db = await open({
          filename: this.dbPath,
          driver: sqlite3.Database,
        });
        log.info(`Cache DB recovered by deleting corrupted file`);
      } catch (recoveryErr) {
        log.error(`Cache recovery failed: ${recoveryErr instanceof Error ? recoveryErr.message : String(recoveryErr)} - using in-memory`);
        this.db = await open({
          filename: ":memory:",
          driver: sqlite3.Database,
        });
      }
    }

    try {
      // Enable WAL mode for better concurrency (may fail on in-memory or read-only)
      await this.db.exec("PRAGMA journal_mode = WAL;");
    } catch {}
    try {
      await this.db.exec("PRAGMA synchronous = NORMAL;");
    } catch {}
    try {
      await this.db.exec("PRAGMA foreign_keys = ON;");
    } catch {}

    try {
      await this.createTables();
    } catch (err) {
      // Table creation failed due to corruption or version mismatch - try to recreate from scratch
      log.warn(`Failed to create tables: ${err instanceof Error ? err.message : String(err)} - resetting cache`);
      try {
        await this.db.exec(`DROP TABLE IF EXISTS local_model_cache`);
        await this.createTables();
      } catch (e2) {
        log.error(`Cache reset failed: ${e2 instanceof Error ? e2.message : String(e2)} - using in-memory`);
        await this.db.close().catch(() => {});
        this.db = await open({ filename: ":memory:", driver: sqlite3.Database });
        await this.createTables();
      }
    }
    log.info(`Local model cache initialized at ${this.dbPath}`);
  }

  private async createTables(): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    await this.db.exec(`
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
    const cols = await this.db.all(`PRAGMA table_info(local_model_cache)`) as { name: string }[];
    if (!cols.some((c) => c.name === "state")) {
      await this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN state TEXT NOT NULL DEFAULT 'detected'`);
      await this.db.exec(`CREATE INDEX IF NOT EXISTS idx_local_model_cache_state ON local_model_cache(state)`);
    }
    if (!cols.some((c) => c.name === "hf_checked_at")) {
      await this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN hf_checked_at INTEGER`);
    }
    if (!cols.some((c) => c.name === "hf_deleted_at")) {
      await this.db.exec(`ALTER TABLE local_model_cache ADD COLUMN hf_deleted_at INTEGER`);
    }
  }

  async get(path: string): Promise<LocalCacheEntry | null> {
    if (!this.db) throw new Error("Database not initialized");

    const row = await this.db.get(
      `SELECT path, size, mtime, sha256, hf_model_id, hf_checked_at, hf_deleted_at, last_verified, state
       FROM local_model_cache WHERE path = ?`,
      path
    );

    return row ? this.mapRow(row) : null;
  }

  async getAll(): Promise<LocalCacheEntry[]> {
    if (!this.db) throw new Error("Database not initialized");

    const rows = await this.db.all(
      `SELECT path, size, mtime, sha256, hf_model_id, hf_checked_at, hf_deleted_at, last_verified, state
       FROM local_model_cache`
    );

    return rows.map(this.mapRow);
  }

  async getBySha256(sha256: string): Promise<LocalCacheEntry | null> {
    if (!this.db) throw new Error("Database not initialized");

    const row = await this.db.get(
      `SELECT path, size, mtime, sha256, hf_model_id, hf_checked_at, hf_deleted_at, last_verified, state
       FROM local_model_cache WHERE sha256 = ?`,
      sha256
    );

    return row ? this.mapRow(row) : null;
  }

  async upsert(entry: LocalCacheEntry): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    await this.db.run(
      `INSERT INTO local_model_cache (path, size, mtime, sha256, hf_model_id, hf_checked_at, hf_deleted_at, last_verified, state)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(path) DO UPDATE SET
         size = excluded.size,
         mtime = excluded.mtime,
         sha256 = excluded.sha256,
         hf_model_id = excluded.hf_model_id,
         hf_checked_at = excluded.hf_checked_at,
         hf_deleted_at = excluded.hf_deleted_at,
         last_verified = excluded.last_verified,
         state = excluded.state`,
      entry.path,
      entry.size,
      entry.mtime,
      entry.sha256 ?? null,
      entry.hf_model_id ?? null,
      entry.hf_checked_at ?? null,
      entry.hf_deleted_at ?? null,
      entry.last_verified,
      entry.state
    );
  }

  async updateState(path: string, state: LocalModelState): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    await this.db.run(
      `UPDATE local_model_cache SET state = ?, last_verified = ? WHERE path = ?`,
      state,
      Date.now(),
      path
    );
  }

  async updateHash(path: string, sha256: string): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    await this.db.run(
      `UPDATE local_model_cache SET sha256 = ?, state = ?, last_verified = ? WHERE path = ?`,
      sha256,
      "hashing",
      Date.now(),
      path
    );
  }

  // deletedAt: the server's hf_gguf_index.deleted_at for this match (null if
  // live). Always stamps hf_checked_at with now -- this call IS a fresh
  // verification, whether it's the first match or a periodic re-check of an
  // already-matched entry (see model-scanner.ts's resolveHfMetadata).
  async updateHfMatch(path: string, hf_model_id: string, deletedAt: number | null = null): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    await this.db.run(
      `UPDATE local_model_cache SET hf_model_id = ?, hf_checked_at = ?, hf_deleted_at = ?, state = ?, last_verified = ? WHERE path = ?`,
      hf_model_id,
      Date.now(),
      deletedAt,
      "verified",
      Date.now(),
      path
    );
  }

  async delete(path: string): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");

    await this.db.run(`DELETE FROM local_model_cache WHERE path = ?`, path);
  }

  async deleteMissing(existingPaths: string[]): Promise<number> {
    if (!this.db) throw new Error("Database not initialized");

    if (existingPaths.length === 0) {
      // Delete all entries if no files exist
      const result = await this.db.run(`DELETE FROM local_model_cache`);
      return result.changes ?? 0;
    }

    const placeholders = existingPaths.map(() => "?").join(",");
    const result = await this.db.run(
      `DELETE FROM local_model_cache WHERE path NOT IN (${placeholders})`,
      ...existingPaths
    );
    return result.changes ?? 0;
  }

  async close(): Promise<void> {
    if (this.db) {
      await this.db.close();
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
    last_verified: number;
    state: string;
  }): LocalCacheEntry {
    return {
      path: row.path,
      size: row.size,
      mtime: row.mtime,
      sha256: row.sha256 ?? undefined,
      hf_model_id: row.hf_model_id ?? undefined,
      hf_checked_at: row.hf_checked_at ?? undefined,
      hf_deleted_at: row.hf_deleted_at ?? undefined,
      last_verified: row.last_verified,
      state: row.state as LocalModelState,
    };
  }
}

// Factory function to create and initialize the cache
export async function createLocalModelCache(modelDir: string): Promise<LocalModelCache> {
  const cache = new LocalModelCache(modelDir);
  await cache.init();
  return cache;
}