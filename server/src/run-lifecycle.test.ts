import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./db/repo.js";
import type { getDb as GetDbType } from "./db/migrate.js";
import type { TestItemTerminalInput, TestItemTickInput } from "../../shared/types.js";

// Audit 2026-10-18 A2: an item tick is fire-and-forget beside the item's
// terminal report, so it can arrive after it. A finished item has to stay
// finished, a finished run has to keep its outcome, and a run whose jobs have
// all ended without every item reporting must still be finalized.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-run-lifecycle-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;
let getDb: typeof GetDbType;
let runMaintenanceSweep: typeof import("./reaper.js")["runMaintenanceSweep"];
let ORPHANED_RUN_GRACE_MS: number;
let expandSweep: typeof import("../../shared/sweep.js")["expandSweep"];

const silentLog = { warn: () => {} };
let workerId: string;

beforeAll(async () => {
  ({ repo } = await import("./db/repo.js"));
  ({ getDb } = await import("./db/migrate.js"));
  ({ runMaintenanceSweep, ORPHANED_RUN_GRACE_MS } = await import("./reaper.js"));
  ({ expandSweep } = await import("../../shared/sweep.js"));
  repo.registerModel({ id: "m".repeat(64), filename: "m.gguf", size_bytes: 1, source: "local" });
  workerId = repo.workerRepo.getOrCreateByMachineId("lc-machine", "lc-box").id;
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

const sweep = {
  n_prompt: [512, 1024],
  n_gen: [0],
  threads: [8],
  n_gpu_layers: [99],
  batch_size: [2048],
  ubatch_size: [512],
  cache_type_k: ["f16"],
  cache_type_v: ["f16"],
  flash_attn: ["on"],
  mtp: ["off"],
  n_gpu_layers_draft: [0],
  n_cpu_moe: [0],
  repeats: 3,
};

let seq = 0;
function makeRun(status: "running" | "scheduled" = "running", startedAgo = 0): string {
  const id = `run-lc-${++seq}`;
  repo.createTest(undefined, {
    id,
    root_run_id: id,
    kind: null,
    comparison_id: null,
    worker_name: "w",
    worker_id: null,
    llama_cpp_build: "b1",
    llama_cpp_backend: "vulkan",
    model_id: "m".repeat(64),
    config: { model_id: "m".repeat(64), sweep } as never,
    status,
    started_at: Date.now() - startedAgo,
  } as never);
  repo.createTestItems(undefined, id, expandSweep(sweep));
  return id;
}

function addJob(runId: string, status: string, endedAgo: number): void {
  const t = Date.now() - endedAgo;
  getDb()
    .prepare(
      `INSERT INTO worker_jobs (id, worker_id, job_type, payload_json, status, run_id, created_at, claimed_at, completed_at, lease_expires_at)
       VALUES (?, ?, 'benchmark', '{}', ?, ?, ?, ?, ?, ?)`
    )
    .run(
      `job-${runId}-${status}`,
      workerId,
      status,
      runId,
      t,
      t,
      status === "pending" || status === "claimed" ? null : t,
      status === "claimed" ? Date.now() + 60_000 : null
    );
}

const failed: TestItemTerminalInput = { status: "failed", error: "x" } as TestItemTerminalInput;
const tick: TestItemTickInput = { status: "processing", detail: "late tick" } as TestItemTickInput;
const items = (runId: string) => repo.getTestItems(runId).map((i) => i.status);

describe("item ticks never reopen a finished item", () => {
  it("a tick arriving after the item's terminal report is ignored, so the run still finalizes", () => {
    const runId = makeRun();
    repo.recordTestItemTerminal(runId, 0, failed);
    repo.updateTestItemTick(runId, 0, tick);
    expect(items(runId)[0]).toBe("failed");
    const run = repo.recordTestItemTerminal(runId, 1, failed);
    expect(run?.status).toBe("failed");
  });

  it("a tick on an unfinished item still applies", () => {
    const runId = makeRun();
    repo.updateTestItemTick(runId, 0, tick);
    expect(items(runId)[0]).toBe("processing");
  });
});

describe("a late terminal report on a finished run", () => {
  it("records the item but keeps the run's status and completed_at", () => {
    const runId = makeRun();
    const stopped = repo.reconcileStaleTest(undefined, runId, "stopped by user")!;
    expect(stopped.status).toBe("cancelled");
    const { completed_at } = repo.getTest(undefined, runId)!;
    const after = repo.recordTestItemTerminal(runId, 0, failed)!;
    expect(items(runId)[0]).toBe("failed");
    expect(after.status).toBe("cancelled");
    expect(after.completed_at).toBe(completed_at);
  });
});

describe("finalizeOrphanedRuns", () => {
  it("finalizes a running run whose jobs all ended past the grace, marking unreported items cancelled", () => {
    const runId = makeRun();
    repo.recordTestItemTerminal(runId, 0, failed);
    addJob(runId, "completed", ORPHANED_RUN_GRACE_MS + 60_000);
    runMaintenanceSweep(silentLog);
    const run = repo.getTest(undefined, runId)!;
    expect(run.status).not.toBe("running");
    expect(items(runId)).toEqual(["failed", "cancelled"]);
  });

  it("leaves it alone inside the grace window", () => {
    const runId = makeRun();
    addJob(runId, "completed", 60_000);
    runMaintenanceSweep(silentLog);
    expect(repo.getTest(undefined, runId)!.status).toBe("running");
  });

  it("never touches a run with a pending or claimed job (e.g. between install_build and benchmark)", () => {
    const pending = makeRun("scheduled");
    addJob(pending, "completed", ORPHANED_RUN_GRACE_MS + 60_000);
    addJob(pending, "pending", ORPHANED_RUN_GRACE_MS + 60_000);
    const claimed = makeRun();
    addJob(claimed, "claimed", ORPHANED_RUN_GRACE_MS + 60_000);
    runMaintenanceSweep(silentLog);
    expect(repo.getTest(undefined, pending)!.status).toBe("scheduled");
    expect(repo.getTest(undefined, claimed)!.status).toBe("running");
  });

  it("finalizes an old run whose terminal jobs were already pruned", () => {
    const old = makeRun("running", ORPHANED_RUN_GRACE_MS + 60_000);
    const fresh = makeRun("running");
    runMaintenanceSweep(silentLog);
    expect(repo.getTest(undefined, old)!.status).not.toBe("running");
    expect(repo.getTest(undefined, fresh)!.status).toBe("running");
  });
});
