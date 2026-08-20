import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-queue-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "route-test-secret";

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let queueEvents: typeof import("../queue-events.js")["queueEvents"];

function hardwareState(machineId: string, status: "idle" | "busy") {
  return {
    machine_id: machineId,
    hostname: machineId,
    backend: "cpu",
    hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
    installed_builds: [],
    model_files: [],
    status,
  };
}

async function postJson(path: string, body: unknown, token = "route-test-secret") {
  return fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ queueEvents } = await import("../queue-events.js"));
  const { queueRoutes } = await import("./queue.js");

  app = Fastify({ logger: false });
  await app.register(queueRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless, OS temp dir */
  }
});

describe("POST /api/worker/queue", () => {
  it("rejects a wrong bearer token with 401", async () => {
    const res = await postJson("/api/worker/queue", hardwareState("q-auth", "idle"), "wrong-token");
    expect(res.status).toBe(401);
  });

  it("returns 204 immediately for a worker reporting busy, never hanging on the long poll", async () => {
    const started = Date.now();
    const res = await postJson("/api/worker/queue", hardwareState("q-busy", "busy"));
    expect(res.status).toBe(204);
    expect(Date.now() - started).toBeLessThan(2000); // must not fall into the 25s long-poll wait
  });

  it("delivers an already-pending job immediately (the synchronous claim before the wait)", async () => {
    // First poll (idle, nothing queued yet) would normally hang -- avoid it
    // by creating the worker row directly, matching what a prior heartbeat
    // would have done, then enqueueing before ever calling /queue.
    const worker = repo.workerRepo.getOrCreateByMachineId("q-immediate", "q-immediate");
    const jobId = repo.queueRepo.enqueueJob(worker.id, {
      type: "install_build",
      payload: { tag: "b1", asset_name: "a.zip", download_url: "http://x" },
    });

    const started = Date.now();
    const res = await postJson("/api/worker/queue", hardwareState("q-immediate", "idle"));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { job_id: string; type: string };
    expect(body.job_id).toBe(jobId);
    expect(body.type).toBe("install_build");
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it("wakes the long-poll via queueEvents as soon as a job is enqueued, without waiting for the backstop", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("q-wake", "q-wake");

    const pollPromise = postJson("/api/worker/queue", hardwareState("q-wake", "idle"));

    // Give the long-poll a moment to actually start listening (register its
    // queueEvents handler) before enqueueing -- otherwise this would just be
    // testing the synchronous pre-wait claim, same as the previous test.
    await new Promise((r) => setTimeout(r, 100));
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "benchmark", payload: { run_id: "r-wake" } });
    queueEvents.emit(worker.id);

    const started = Date.now();
    const res = await pollPromise;
    expect(res.status).toBe(200);
    const body = (await res.json()) as { job_id: string };
    expect(body.job_id).toBe(jobId);
    // Must resolve via the event, not the 5s backstop poll.
    expect(Date.now() - started).toBeLessThan(2000);
  });
});

describe("POST /api/worker/heartbeat", () => {
  it("returns worker_id, empty control, and a lease_until in the future for an idle worker", async () => {
    const res = await postJson("/api/worker/heartbeat", hardwareState("hb-1", "idle"));
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      worker_id: string;
      control: { cancel_job_ids: string[]; discard_job_ids: string[]; pause: boolean };
      lease_until: number;
    };
    expect(body.worker_id).toBeTruthy();
    expect(body.control).toEqual({ cancel_job_ids: [], discard_job_ids: [], pause: false });
    expect(body.lease_until).toBeGreaterThan(Date.now());
  });

  it("reflects workers.pause_requested in control.pause", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("hb-pause", "hb-pause");
    repo.workerRepo.setPauseRequested(worker.id, true);

    const res = await postJson("/api/worker/heartbeat", hardwareState("hb-pause", "idle"));
    const body = (await res.json()) as { control: { pause: boolean } };
    expect(body.control.pause).toBe(true);
  });

  it("tells the worker to cancel a job whose lease the server no longer recognizes", async () => {
    const res = await postJson("/api/worker/heartbeat", {
      ...hardwareState("hb-unknown-job", "busy"),
      active_job: { job_id: "does-not-exist", phase: "benchmarking" },
    });
    const body = (await res.json()) as { control: { cancel_job_ids: string[] } };
    expect(body.control.cancel_job_ids).toEqual(["does-not-exist"]);
  });

  it("extends the lease and surfaces cancel_requested for a real claimed job", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("hb-real-job", "hb-real-job");
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "benchmark", payload: { run_id: "r-hb" } });
    const claimed = repo.queueRepo.claimNextJob(worker.id);
    expect(claimed?.job_id).toBe(jobId);

    const notCancelled = await postJson("/api/worker/heartbeat", {
      ...hardwareState("hb-real-job", "busy"),
      active_job: { job_id: jobId, phase: "benchmarking", item_idx: 1, items_total: 4 },
    });
    expect(((await notCancelled.json()) as { control: { cancel_job_ids: string[] } }).control.cancel_job_ids).toEqual(
      []
    );

    repo.queueRepo.requestCancel(jobId);
    const cancelled = await postJson("/api/worker/heartbeat", {
      ...hardwareState("hb-real-job", "busy"),
      active_job: { job_id: jobId, phase: "benchmarking" },
    });
    expect(((await cancelled.json()) as { control: { cancel_job_ids: string[] } }).control.cancel_job_ids).toEqual([
      jobId,
    ]);
  });
});

describe("POST /api/worker/jobs/:jobId/complete", () => {
  it("marks a claimed job completed on success", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("complete-1", "complete-1");
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "install_build", payload: { tag: "b1" } });
    repo.queueRepo.claimNextJob(worker.id);

    const res = await postJson(`/api/worker/jobs/${jobId}/complete`, { machine_id: "complete-1", ok: true });
    expect(res.status).toBe(200);

    // A completed job must never be handed out again -- this is the
    // zombie-requeue bug this endpoint exists to prevent.
    const expired = repo.queueRepo.listExpiredLeases(Date.now() + 10 * 60_000).find((j) => j.id === jobId);
    expect(expired).toBeUndefined();
  });

  it("rejects a job that belongs to a different worker", async () => {
    const owner = repo.workerRepo.getOrCreateByMachineId("complete-owner", "complete-owner");
    const intruder = repo.workerRepo.getOrCreateByMachineId("complete-intruder", "complete-intruder");
    void intruder;
    const jobId = repo.queueRepo.enqueueJob(owner.id, { type: "install_build", payload: { tag: "b1" } });
    repo.queueRepo.claimNextJob(owner.id);

    const res = await fetch(`${baseUrl}/api/worker/jobs/${jobId}/complete`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer route-test-secret" },
      // machine_id in the body picks up (or creates) "complete-intruder", but
      // the job belongs to "complete-owner" -- authenticateWorker resolves
      // identity from THIS request's own machine_id, not from the job.
      body: JSON.stringify({ ...hardwareState("complete-intruder", "idle"), ok: true }),
    });
    expect(res.status).toBe(404);
  });

  it("failing a benchmark job reconciles its run and cancels any still-pending sibling job", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("complete-fail", "complete-fail");
    const runId = "run-complete-fail";
    // A sibling install_build job for the same run, enqueued first so it's
    // claimed first (FIFO) -- the benchmark job stays pending behind it.
    repo.queueRepo.enqueueJob(worker.id, { type: "install_build", payload: { tag: "b2" }, runId });
    const benchmarkJobId = repo.queueRepo.enqueueJob(worker.id, { type: "benchmark", payload: { run_id: runId }, runId });

    const installClaim = repo.queueRepo.claimNextJob(worker.id); // claims install_build, not benchmark
    expect(installClaim?.type).toBe("install_build");

    const res = await postJson(`/api/worker/jobs/${installClaim!.job_id}/complete`, {
      machine_id: "complete-fail",
      ok: false,
      error: "install failed",
    });
    expect(res.status).toBe(200);

    const failedJob = repo.queueRepo.getJob(installClaim!.job_id);
    expect(failedJob?.status).toBe("failed");
    // The still-pending benchmark sibling must be cancelled, not left to
    // fire later against a run that's already been reconciled.
    const sibling = repo.queueRepo.getJob(benchmarkJobId);
    expect(sibling?.status).toBe("cancelled");
  });
});
