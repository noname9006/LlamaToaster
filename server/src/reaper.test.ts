import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./db/repo.js";
import type { getDb as GetDbType } from "./db/migrate.js";

// §1.7's own maintenance-sweep spec named three sweeps beyond the original
// Stage 1 job-lease reaper (already covered in routes/runs.test.ts):
// expired-session pruning, terminal-job pruning, and expired-enrolment
// pruning. None of the three existed at all until this file's own subject
// (reaper.ts's runMaintenanceSweep) was added -- sessions/worker_jobs/
// abandoned-enrolment rows would otherwise grow forever in production. Own
// isolated DB, same reasoning as every other db-layer test file here.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-reaper-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;
let getDb: typeof GetDbType;
let runMaintenanceSweep: typeof import("./reaper.js")["runMaintenanceSweep"];

const silentLog = { warn: () => {} };

beforeAll(async () => {
  ({ repo } = await import("./db/repo.js"));
  ({ getDb } = await import("./db/migrate.js"));
  ({ runMaintenanceSweep } = await import("./reaper.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

const DAY_MS = 24 * 3600 * 1000;

describe("sessionRepo.pruneExpired", () => {
  it("deletes only sessions past their expires_at, leaves valid ones alone", () => {
    const db = getDb();
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "prune-sess-1", login: "prune-sess", avatarUrl: null });
    db.prepare(
      `INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)`
    ).run("sess-expired", user.id, "hash-expired", Date.now() - DAY_MS, Date.now() - 2 * DAY_MS);
    db.prepare(
      `INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)`
    ).run("sess-valid", user.id, "hash-valid", Date.now() + DAY_MS, Date.now());

    const deleted = repo.sessionRepo.pruneExpired();
    expect(deleted).toBe(1);

    const remaining = db.prepare(`SELECT id FROM sessions WHERE user_id = ?`).all(user.id) as { id: string }[];
    expect(remaining.map((r) => r.id)).toEqual(["sess-valid"]);
  });
});

describe("queueRepo.pruneCompletedOlderThan", () => {
  it("deletes old terminal jobs (completed/failed/cancelled), never a recent one or a still-pending one", () => {
    const db = getDb();
    const worker = repo.workerRepo.getOrCreateByMachineId("prune-jobs-machine", "prune-jobs-machine");
    const insertJob = (id: string, status: string, completedAt: number | null) =>
      db
        .prepare(
          `INSERT INTO worker_jobs (id, worker_id, job_type, payload_json, status, completed_at, created_at)
           VALUES (?, ?, 'benchmark', '{}', ?, ?, ?)`
        )
        .run(id, worker.id, status, completedAt, Date.now());

    insertJob("job-old-completed", "completed", Date.now() - 10 * DAY_MS);
    insertJob("job-old-failed", "failed", Date.now() - 10 * DAY_MS);
    insertJob("job-old-cancelled", "cancelled", Date.now() - 10 * DAY_MS);
    insertJob("job-recent-completed", "completed", Date.now() - 1 * DAY_MS);
    insertJob("job-pending", "pending", null);

    const deleted = repo.queueRepo.pruneCompletedOlderThan(7);
    expect(deleted).toBe(3);

    const remaining = db.prepare(`SELECT id FROM worker_jobs WHERE worker_id = ?`).all(worker.id) as { id: string }[];
    expect(remaining.map((r) => r.id).sort()).toEqual(["job-pending", "job-recent-completed"]);
  });
});

describe("workerRepo.pruneExpiredEnrolments", () => {
  it("clears code fields on an expired, never-approved enrolment; leaves everything else untouched", () => {
    const db = getDb();

    // Expired, never approved -- the one row this should actually touch.
    db.prepare(
      `INSERT INTO workers (id, machine_id, display_name, user_code, enrolment_code_hash, enrolment_expires_at, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    ).run("w-expired-pending", "m-expired-pending", "m-expired-pending", "CODE-EXPD", "hash1", Date.now() - DAY_MS, Date.now(), Date.now());

    // Still within its window -- must not be touched even though it's unapproved.
    db.prepare(
      `INSERT INTO workers (id, machine_id, display_name, user_code, enrolment_code_hash, enrolment_expires_at, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    ).run("w-still-pending", "m-still-pending", "m-still-pending", "CODE-LIVE", "hash2", Date.now() + DAY_MS, Date.now(), Date.now());

    // Already approved (a real, owned, historied machine) -- even with a
    // stale enrolment_expires_at left over from its original approval code,
    // approved_at NOT NULL must protect it unconditionally.
    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "prune-enrol-owner", login: "owner", avatarUrl: null });
    db.prepare(
      `INSERT INTO workers (id, machine_id, display_name, user_id, approved_at, user_code, enrolment_code_hash, enrolment_expires_at, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run("w-approved", "m-approved", "m-approved", owner.id, Date.now(), "CODE-OLD", "hash3", Date.now() - DAY_MS, Date.now(), Date.now());

    // Stage 1 legacy worker -- never went through device-flow at all
    // (enrolment_expires_at IS NULL), must never be touched.
    repo.workerRepo.getOrCreateByMachineId("m-legacy", "m-legacy");

    const changed = repo.workerRepo.pruneExpiredEnrolments();
    expect(changed).toBe(1);

    const expiredRow = db.prepare(`SELECT user_code, enrolment_code_hash, enrolment_expires_at FROM workers WHERE id = ?`).get("w-expired-pending") as {
      user_code: string | null;
      enrolment_code_hash: string | null;
      enrolment_expires_at: number | null;
    };
    expect(expiredRow).toEqual({ user_code: null, enrolment_code_hash: null, enrolment_expires_at: null });

    const stillPending = db.prepare(`SELECT user_code FROM workers WHERE id = ?`).get("w-still-pending") as { user_code: string | null };
    expect(stillPending.user_code).toBe("CODE-LIVE");

    const approvedRow = db.prepare(`SELECT user_code, enrolment_expires_at FROM workers WHERE id = ?`).get("w-approved") as {
      user_code: string | null;
      enrolment_expires_at: number | null;
    };
    expect(approvedRow.user_code).toBe("CODE-OLD"); // untouched -- approved_at protects it
    expect(approvedRow.enrolment_expires_at).not.toBeNull();

    const legacyRow = db.prepare(`SELECT enrolment_expires_at FROM workers WHERE machine_id = 'm-legacy'`).get() as {
      enrolment_expires_at: number | null;
    };
    expect(legacyRow.enrolment_expires_at).toBeNull(); // was already null, still null
  });
});

describe("repo.pruneOldGpuClockSamples (M6)", () => {
  it("nulls only the sample series on rows past 30 days, and only that column", () => {
    const db = getDb();
    const insertResult = (id: string, createdAt: number) =>
      db
        .prepare(
          `INSERT INTO results
             (id, gpu_temp_c_max, gpu_clock_mhz_min, gpu_clock_samples, caveat_flags, created_at)
           VALUES (?, ?, ?, ?, ?, ?)`
        )
        .run(id, 89, 1200, JSON.stringify([1400, 1350, 1300, 1200]), JSON.stringify(["thermally_throttled"]), createdAt);

    insertResult("result-old-thermal", Date.now() - 31 * DAY_MS);
    insertResult("result-recent-thermal", Date.now() - 1 * DAY_MS);

    const changed = repo.pruneOldGpuClockSamples(30);
    expect(changed).toBe(1);

    const old = db
      .prepare(`SELECT gpu_temp_c_max, gpu_clock_mhz_min, gpu_clock_samples, caveat_flags FROM results WHERE id = ?`)
      .get("result-old-thermal") as {
      gpu_temp_c_max: number | null;
      gpu_clock_mhz_min: number | null;
      gpu_clock_samples: string | null;
      caveat_flags: string | null;
    };
    // The bulky sample series is gone -- everything a later reader (scoring,
    // N6 policy) actually consumes persists indefinitely.
    expect(old.gpu_clock_samples).toBeNull();
    expect(old.gpu_temp_c_max).toBe(89);
    expect(old.gpu_clock_mhz_min).toBe(1200);
    expect(JSON.parse(old.caveat_flags!)).toEqual(["thermally_throttled"]);

    const recent = db.prepare(`SELECT gpu_clock_samples FROM results WHERE id = ?`).get("result-recent-thermal") as {
      gpu_clock_samples: string | null;
    };
    expect(recent.gpu_clock_samples).not.toBeNull();
  });
});

describe("runMaintenanceSweep", () => {
  it("runs the job-lease reaper and all four prune sweeps together without throwing", () => {
    const db = getDb();
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "sweep-integration", login: "sweep", avatarUrl: null });
    db.prepare(`INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)`).run(
      "sess-sweep-expired",
      user.id,
      "hash-sweep",
      Date.now() - DAY_MS,
      Date.now() - 2 * DAY_MS
    );
    db.prepare(
      `INSERT INTO results (id, gpu_clock_samples, created_at) VALUES (?, ?, ?)`
    ).run("result-sweep-old", JSON.stringify([1400, 1200]), Date.now() - 31 * DAY_MS);

    expect(() => runMaintenanceSweep(silentLog)).not.toThrow();

    const row = db.prepare(`SELECT id FROM sessions WHERE id = ?`).get("sess-sweep-expired");
    expect(row).toBeUndefined();
    const result = db.prepare(`SELECT gpu_clock_samples FROM results WHERE id = ?`).get("result-sweep-old") as {
      gpu_clock_samples: string | null;
    };
    expect(result.gpu_clock_samples).toBeNull();
  });
});
