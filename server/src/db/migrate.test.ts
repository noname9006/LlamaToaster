import { afterAll, describe, expect, it, vi } from "vitest";
import Database from "better-sqlite3";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// getDb() is a module-level singleton keyed on process.env.DB_PATH at first
// call -- each test in this file gets its own throwaway DB file (set up
// BEFORE the re-import below) so migrate() runs against exactly the
// pre-existing on-disk shape each test constructs, not a fresh CREATE TABLE.
// vi.resetModules() clears vitest's module registry so the following
// import("./migrate.js") is a genuinely fresh module instance (a new `db`
// singleton) instead of the previous test's already-initialized one.
async function freshMigrateModule(dbPath: string) {
  process.env.DB_PATH = dbPath;
  vi.resetModules();
  return (await import("./migrate.js")) as typeof import("./migrate.js");
}

const tmpDirs: string[] = [];
function makeTmpDbPath(): string {
  const dir = mkdtempSync(join(tmpdir(), "llamatoaster-migrate-test-"));
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

describe("migrate", () => {
  it("adds deleted_at + its index to an hf_gguf_index table that predates the column", async () => {
    // Simulates the real VPS DB this was caught against: an hf_gguf_index
    // table created before deleted_at existed. schema.sql's CREATE TABLE IF
    // NOT EXISTS is a no-op against this, so the deleted_at column can only
    // come from COLUMN_MIGRATIONS' ALTER TABLE -- and the index on it must
    // run after that, not alongside the static CREATE TABLE block.
    const dbPath = makeTmpDbPath();
    const preexisting = new Database(dbPath);
    preexisting.exec(`
      CREATE TABLE hf_gguf_index (
        sha256 TEXT NOT NULL,
        repo_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        revision TEXT,
        file_size INTEGER,
        last_seen INTEGER NOT NULL,
        PRIMARY KEY (sha256, repo_id, filename)
      );
    `);
    preexisting.close();

    const { getDb } = await freshMigrateModule(dbPath);
    // Previously threw "SqliteError: no such column: deleted_at" here.
    expect(() => getDb()).not.toThrow();

    const db = getDb();
    const cols = db.prepare(`PRAGMA table_info(hf_gguf_index)`).all() as { name: string }[];
    expect(cols.some((c) => c.name === "deleted_at")).toBe(true);

    const indexes = db.prepare(`PRAGMA index_list(hf_gguf_index)`).all() as { name: string }[];
    expect(indexes.some((i) => i.name === "idx_hf_gguf_index_deleted_at")).toBe(true);
  });

  it("initializes a brand-new DB (no pre-existing file) without error", async () => {
    const dbPath = makeTmpDbPath();
    const { getDb } = await freshMigrateModule(dbPath);
    expect(() => getDb()).not.toThrow();

    const cols = getDb().prepare(`PRAGMA table_info(hf_gguf_index)`).all() as { name: string }[];
    expect(cols.some((c) => c.name === "deleted_at")).toBe(true);
  });
});
