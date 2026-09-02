import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./repo.js";

// getDb() (migrate.ts) is a module-level singleton keyed on process.env.DB_PATH
// at first call -- set it before importing repo.js so this test file gets its
// own throwaway DB, isolated from any other test file or the real dev DB.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-queue-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
});

afterAll(() => {
  // better-sqlite3 keeps the DB (and its WAL/SHM sidecars) open for the
  // process lifetime -- migrate.ts exposes no close(), and on Windows an
  // open file handle makes rmSync fail with EBUSY. Best-effort only; this is
  // an OS temp dir, not repo state, so leaving it behind is harmless.
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

function makeWorker(machineId: string): string {
  return repo.workerRepo.getOrCreateByMachineId(machineId, machineId).id;
}

function minimalState(machineId: string) {
  return {
    machine_id: machineId,
    capabilities: [],
    hostname: machineId,
    backend: "cpu",
    hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
    installed_builds: [],
    model_files: [],
    status: "idle" as const,
  };
}

describe("queueRepo.claimNextJob", () => {
  it("hands a single pending job to exactly one of many concurrent callers", async () => {
    const workerId = makeWorker("machine-concurrent-1");
    const jobId = repo.queueRepo.enqueueJob(workerId, { type: "benchmark", payload: { run_id: "r1" } });

    // Genuinely concurrent from the event loop's perspective (each call is
    // deferred to a microtask), even though better-sqlite3's synchronous
    // transaction inside claimNextJob is what actually guarantees only one
    // wins -- this is the same shape as N worker/queue long-poll handlers
    // racing to claim off the same worker.
    const results = await Promise.all(
      Array.from({ length: 8 }, () => Promise.resolve().then(() => repo.queueRepo.claimNextJob(workerId)))
    );

    const claimed = results.filter((r) => r !== undefined);
    expect(claimed).toHaveLength(1);
    expect(claimed[0]!.job_id).toBe(jobId);
  });

  it("never claims a job flagged cancel_requested while still pending", () => {
    const workerId = makeWorker("machine-concurrent-2");
    const jobId = repo.queueRepo.enqueueJob(workerId, { type: "benchmark", payload: { run_id: "r2" } });
    repo.queueRepo.requestCancel(jobId);

    expect(repo.queueRepo.claimNextJob(workerId)).toBeUndefined();
  });

  it("claims jobs for one worker in FIFO (created_at) order", () => {
    const workerId = makeWorker("machine-fifo");
    const first = repo.queueRepo.enqueueJob(workerId, { type: "install_build", payload: { tag: "b1" } });
    const second = repo.queueRepo.enqueueJob(workerId, { type: "install_build", payload: { tag: "b2" } });

    expect(repo.queueRepo.claimNextJob(workerId)?.job_id).toBe(first);
    expect(repo.queueRepo.claimNextJob(workerId)?.job_id).toBe(second);
    expect(repo.queueRepo.claimNextJob(workerId)).toBeUndefined();
  });

  it("preserves insertion order even when two jobs share the same created_at millisecond", () => {
    // A trigger enqueues install_build then benchmark back-to-back in the
    // same request -- Date.now() can return the same value for both. Without
    // a tie-breaker, SQLite's ORDER BY created_at ASC alone doesn't guarantee
    // which comes out first, and install_build MUST be claimed before
    // benchmark (MULTIUSER_PLAN.md §1.12).
    const workerId = makeWorker("machine-fifo-tie");
    const originalNow = Date.now;
    Date.now = () => 1_700_000_000_000; // freeze time for both inserts
    let install: string, benchmark: string;
    try {
      install = repo.queueRepo.enqueueJob(workerId, { type: "install_build", payload: { tag: "b1" } });
      benchmark = repo.queueRepo.enqueueJob(workerId, { type: "benchmark", payload: { run_id: "r-tie" } });
    } finally {
      Date.now = originalNow;
    }

    expect(repo.queueRepo.claimNextJob(workerId)?.job_id).toBe(install);
    expect(repo.queueRepo.claimNextJob(workerId)?.job_id).toBe(benchmark);
  });
});

describe("lease expiry", () => {
  it("returnToPending clears the lease and increments attempts (not claimNextJob itself)", () => {
    const workerId = makeWorker("machine-lease-1");
    const jobId = repo.queueRepo.enqueueJob(workerId, { type: "benchmark", payload: { run_id: "r3" } });

    const claimed = repo.queueRepo.claimNextJob(workerId);
    expect(claimed?.job_id).toBe(jobId);

    // listExpiredLeases scans across every worker (it's a global reaper
    // scan, MULTIUSER_PLAN.md §1.7), so other tests in this file may have
    // left their own expired-looking claims behind -- find this job
    // specifically rather than assuming it's the only row.
    const expiredBeforeReturn = repo.queueRepo
      .listExpiredLeases(Date.now() + 10 * 60_000)
      .find((j) => j.id === jobId);
    expect(expiredBeforeReturn).toBeDefined();
    expect(expiredBeforeReturn?.attempts).toBe(0); // claiming alone must not bump attempts

    repo.queueRepo.returnToPending(jobId);

    // Back in the pending pool, claimable again.
    const reclaimed = repo.queueRepo.claimNextJob(workerId);
    expect(reclaimed?.job_id).toBe(jobId);

    const expiredAfterReturn = repo.queueRepo
      .listExpiredLeases(Date.now() + 10 * 60_000)
      .find((j) => j.id === jobId);
    expect(expiredAfterReturn?.attempts).toBe(1); // one lease-expiry cycle recorded
  });

  it("listExpiredLeases only returns claimed jobs past their lease, not pending ones", () => {
    const workerId = makeWorker("machine-lease-2");
    const jobId = repo.queueRepo.enqueueJob(workerId, { type: "benchmark", payload: { run_id: "r4" } });
    // Still pending -- must never show up as an "expired lease" candidate,
    // even though other tests in this file have real expired claims by now.
    const expired = repo.queueRepo.listExpiredLeases(Date.now() + 10 * 60_000);
    expect(expired.find((j) => j.id === jobId)).toBeUndefined();
  });
});

describe("workerRepo.getOrCreateByMachineId", () => {
  it("is idempotent -- the same machine_id always resolves to the same worker row", () => {
    const a = repo.workerRepo.getOrCreateByMachineId("machine-idempotent", "box-1");
    const b = repo.workerRepo.getOrCreateByMachineId("machine-idempotent", "box-1-renamed-hint");
    expect(b.id).toBe(a.id);
    // Re-announcing doesn't rewrite display_name -- that's user-editable,
    // not re-derived from the hostname on every heartbeat.
    expect(b.displayName).toBe("box-1");
  });
});

describe("Worker.activeTestId / activeJobProgress (client-facing progress plumbing)", () => {
  it("is undefined for a worker with no active job", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-no-job", "box");
    const fetched = repo.workerRepo.getWorker(worker.id)!;
    expect(fetched.activeTestId).toBeUndefined();
    expect(fetched.activeJobProgress).toBeUndefined();
  });

  it("surfaces run_id (via enqueueJob's runId param, the worker_jobs column -- not payload_json) and live progress once claimed+reported", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-active-benchmark", "box");
    const jobId = repo.queueRepo.enqueueJob(worker.id, {
      type: "benchmark",
      payload: { run_id: "r1" },
      runId: "run-r1",
    });
    const claimed = repo.queueRepo.claimNextJob(worker.id);
    expect(claimed?.job_id).toBe(jobId);

    repo.queueRepo.extendLeaseAndGetFlags(jobId, worker.id, {
      job_id: jobId,
      phase: "benchmarking",
      item_idx: 2,
      items_total: 8,
    });

    const fetched = repo.workerRepo.getWorker(worker.id)!;
    expect(fetched.activeJobId).toBe(jobId);
    expect(fetched.activeTestId).toBe("run-r1");
    expect(fetched.activeJobProgress).toMatchObject({ phase: "benchmarking", item_idx: 2, items_total: 8 });
  });

  it("leaves activeTestId undefined for a non-benchmark job (e.g. download_model), but still surfaces progress", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-active-download", "box");
    const jobId = repo.queueRepo.enqueueJob(worker.id, {
      type: "download_model",
      payload: { hf_repo: "a/b", hf_file: "c.gguf" },
    });
    repo.queueRepo.claimNextJob(worker.id);
    repo.queueRepo.extendLeaseAndGetFlags(jobId, worker.id, {
      job_id: jobId,
      phase: "downloading",
      bytes: 500_000_000,
      total_bytes: 5_000_000_000,
    });

    const fetched = repo.workerRepo.getWorker(worker.id)!;
    expect(fetched.activeTestId).toBeUndefined();
    expect(fetched.activeJobProgress).toMatchObject({ phase: "downloading", bytes: 500_000_000, total_bytes: 5_000_000_000 });
  });

  it("listWorkers also carries the same fields (used by GET /api/workers)", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-active-listcheck", "box");
    const jobId = repo.queueRepo.enqueueJob(worker.id, {
      type: "benchmark",
      payload: { run_id: "r3" },
      runId: "run-r3",
    });
    repo.queueRepo.claimNextJob(worker.id);
    repo.queueRepo.extendLeaseAndGetFlags(jobId, worker.id, { job_id: jobId, phase: "loading" });

    const all = repo.workerRepo.listWorkers();
    const found = all.find((w) => w.id === worker.id);
    expect(found?.activeTestId).toBe("run-r3");
    expect(found?.activeJobProgress?.phase).toBe("loading");
  });

  it("clears workers.active_job_id (and status flips back to idle) once a job completes", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-active-complete", "box");
    // In the real flow, claimNextJob is only ever called from the
    // /api/worker/queue route handler, which always calls recordHeartbeat
    // first -- do the same here so last_heartbeat_at is set and status
    // derivation (offline-if-never-heartbeated takes priority over
    // active_job_id) reflects a realistic just-claimed worker.
    repo.workerRepo.recordHeartbeat(worker.id, minimalState("machine-active-complete"), null);
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "install_build", payload: { tag: "b1" } });
    repo.queueRepo.claimNextJob(worker.id);
    expect(repo.workerRepo.getWorker(worker.id)?.status).toBe("busy");

    repo.queueRepo.markJobCompleted(jobId);
    const after = repo.workerRepo.getWorker(worker.id)!;
    expect(after.activeJobId).toBeNull();
    expect(after.status).toBe("idle");
  });

  it("clears workers.active_job_id on markJobFailed too", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-active-fail", "box");
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "install_build", payload: { tag: "b1" } });
    repo.queueRepo.claimNextJob(worker.id);

    repo.queueRepo.markJobFailed(jobId, "boom");
    expect(repo.workerRepo.getWorker(worker.id)?.activeJobId).toBeNull();
  });

  it("clears workers.active_job_id on returnToPending (lease reaped) too", () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("machine-active-requeue", "box");
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "install_build", payload: { tag: "b1" } });
    repo.queueRepo.claimNextJob(worker.id);
    expect(repo.workerRepo.getWorker(worker.id)?.activeJobId).toBe(jobId);

    repo.queueRepo.returnToPending(jobId);
    expect(repo.workerRepo.getWorker(worker.id)?.activeJobId).toBeNull();
  });
});
