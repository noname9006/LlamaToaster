import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./repo.js";
import type { getDb as GetDbType } from "./migrate.js";
import type { TestConfig } from "../../../shared/types.js";

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.2): repo.claimLegacyHistory's own
// isolated test file -- getDb() is a module-level singleton keyed on
// DB_PATH at first call, so this needs its own throwaway DB.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-legacy-history-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;
let getDb: typeof GetDbType;

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
  ({ getDb } = await import("./migrate.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

// Simulates a row that predates AUTH_ENABLED -- inserted via raw SQL rather
// than repo.createTest/createTestItems/etc, which (as of this same Stage 4
// change) always stamp a real userId when one is passed. A real legacy row
// has NULL because no such column existed, or no user was logged in, when it
// was written.
function insertLegacyRun(db: ReturnType<typeof getDb>, id: string): void {
  db.prepare(`INSERT INTO runs (id, status, started_at) VALUES (?, 'done', ?)`).run(id, Date.now());
}
function insertLegacyResult(db: ReturnType<typeof getDb>, id: string, runId: string): void {
  db.prepare(`INSERT INTO results (id, run_id, created_at) VALUES (?, ?, ?)`).run(id, runId, Date.now());
}
function insertLegacyRunItem(db: ReturnType<typeof getDb>, id: string, runId: string): void {
  db.prepare(`INSERT INTO run_items (id, run_id, idx, status) VALUES (?, ?, 0, 'done')`).run(id, runId);
}

describe("repo.claimLegacyHistory", () => {
  it("claims every NULL-user_id row, never touches an already-owned one, and is one-shot", () => {
    const db = getDb();
    insertLegacyRun(db, "legacy-run-1");
    insertLegacyResult(db, "legacy-result-1", "legacy-run-1");
    insertLegacyRunItem(db, "legacy-item-1", "legacy-run-1");
    const legacyWorker = repo.workerRepo.getOrCreateByMachineId("legacy-machine-1", "legacy-box");

    // A run created normally, already owned by someone real -- must survive
    // the claim untouched. Inserted BEFORE the first claim below so this
    // actually exercises the WHERE user_id IS NULL guard, not just "the
    // meta flag already blocked everything."
    const originalOwner = repo.userRepo.upsertByIdentity("github", {
      providerUserId: "already-owned-owner",
      login: "real-owner",
      avatarUrl: null,
    });
    repo.createTest(originalOwner.id, {
      id: "owned-run-1",
      worker_name: "box",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: "m1",
      config: { model_id: "m1" } as TestConfig,
      status: "done",
      started_at: Date.now(),
    });

    const superadmin = repo.userRepo.upsertByIdentity("github", {
      providerUserId: "legacy-superadmin",
      login: "operator",
      avatarUrl: null,
    });

    const firstClaim = repo.claimLegacyHistory(superadmin.id);
    expect(firstClaim).toBe(true);

    const runRow = db.prepare(`SELECT user_id FROM runs WHERE id = ?`).get("legacy-run-1") as { user_id: string | null };
    expect(runRow.user_id).toBe(superadmin.id);
    const resultRow = db.prepare(`SELECT user_id FROM results WHERE id = ?`).get("legacy-result-1") as {
      user_id: string | null;
    };
    expect(resultRow.user_id).toBe(superadmin.id);
    const itemRow = db.prepare(`SELECT user_id FROM run_items WHERE id = ?`).get("legacy-item-1") as {
      user_id: string | null;
    };
    expect(itemRow.user_id).toBe(superadmin.id);
    const workerEnrolment = repo.workerRepo.getEnrolmentById(legacyWorker.id);
    expect(workerEnrolment?.userId).toBe(superadmin.id);

    // The already-owned run must be completely untouched by the claim.
    const ownedRow = db.prepare(`SELECT user_id FROM runs WHERE id = ?`).get("owned-run-1") as {
      user_id: string | null;
    };
    expect(ownedRow.user_id).toBe(originalOwner.id);

    // A second call (a second login, or the same superadmin logging in
    // again) must be a pure no-op -- the meta flag is what makes this
    // one-shot, not "every row is already claimed so there's nothing left to
    // do." A NEW legacy-looking row created after the first claim must NOT
    // be swept up by a later login; it stays whatever it genuinely is.
    const otherUser = repo.userRepo.upsertByIdentity("github", {
      providerUserId: "legacy-other-user",
      login: "someone-else",
      avatarUrl: null,
    });
    insertLegacyRun(db, "legacy-run-2");
    const secondClaim = repo.claimLegacyHistory(otherUser.id);
    expect(secondClaim).toBe(false);

    const stillFirstOwner = db.prepare(`SELECT user_id FROM runs WHERE id = ?`).get("legacy-run-1") as {
      user_id: string | null;
    };
    expect(stillFirstOwner.user_id).toBe(superadmin.id); // untouched by the second call

    const newRunUntouched = db.prepare(`SELECT user_id FROM runs WHERE id = ?`).get("legacy-run-2") as {
      user_id: string | null;
    };
    expect(newRunUntouched.user_id).toBeNull(); // the one-shot already fired; this row is simply unclaimed
  });
});
