import { afterAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";

// An install whose admin once saved the old default budget of 24 must come up
// at 40 after the context tests v2 migration -- once, and never over a value
// that is not exactly 24.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-probe-budget-test-"));
const dbPath = join(tmpDir, "test.db");
process.env.DB_PATH = dbPath;

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

describe("raising a stored probe_max_loads of 24", () => {
  it("rewrites the old default to 40 once, and leaves it alone after an admin changes it back", async () => {
    const seed = new Database(dbPath);
    seed.exec(`CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)`);
    seed.prepare(`INSERT INTO meta (key, value) VALUES ('probe_max_loads', '24')`).run();
    seed.close();

    const { getDb } = await import("./migrate.js");
    const { repo } = await import("./repo.js");
    const db = getDb();
    expect(repo.appSettingsRepo.getProbeMaxLoads()).toBe(40);

    // An admin who deliberately wants 24 back keeps it: the migration is marked done.
    repo.appSettingsRepo.setProbeMaxLoads(24);
    const marker = db.prepare(`SELECT value FROM meta WHERE key = 'probe_max_loads_v2_raised'`).get();
    expect(marker).toBeTruthy();
    expect(repo.appSettingsRepo.getProbeMaxLoads()).toBe(24);
  });
});
