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

  it("registers a hash-verified model file into the catalog on heartbeat (manual-drop flow)", async () => {
    const sha = "c".repeat(64);
    const files = [
      {
        path: "dropped.gguf",
        size_bytes: 456,
        sha256: sha,
        state: "verified",
        hf_match: { repo_id: "org/repo", filename: "dropped.gguf", revision: "main", deleted: false },
      },
      // Not registered: no hash match yet.
      { path: "unidentified.gguf", size_bytes: 1 },
      // Not registered: the HF match was soft-deleted (removed from HF).
      {
        path: "gone-from-hf.gguf",
        size_bytes: 2,
        sha256: "d".repeat(64),
        state: "verified",
        hf_match: { repo_id: "org/gone", filename: "gone.gguf", revision: "main", deleted: true },
      },
    ];

    await postJson("/api/worker/heartbeat", { ...hardwareState("hb-hash-register", "idle"), model_files: files });
    expect(repo.getModel(sha)).toBeTruthy();
    expect(repo.getModel("d".repeat(64))).toBeUndefined();

    // Idempotent across beats -- a second identical heartbeat must not throw
    // or duplicate (models.id is the primary key).
    await postJson("/api/worker/heartbeat", { ...hardwareState("hb-hash-register", "idle"), model_files: files });
    expect(repo.listModels().filter((m) => m.id === sha)).toHaveLength(1);
  });

  it("self-heals a catalog row whose hf_repo/hf_file were mis-split by the old buildHfMatch bug, without wiping its existing metadata", async () => {
    const sha = "e".repeat(64);
    // Simulate a row written before the model-scanner.ts buildHfMatch fix:
    // repo_id truncated to just the namespace, the real repo name dumped
    // into hf_file -- and it already carries a backfilled param_count that
    // must survive the correction.
    repo.registerModel({
      id: sha,
      filename: "repo-name-GGUF/model.Q4_K_M.gguf",
      size_bytes: 789,
      source: "huggingface",
      hf_repo: "org",
      hf_file: "repo-name-GGUF/model.Q4_K_M.gguf",
      metadata: { param_count: 8_000_000_000 },
    });

    await postJson("/api/worker/heartbeat", {
      ...hardwareState("hb-hash-selfheal", "idle"),
      model_files: [
        {
          path: "model.Q4_K_M.gguf",
          size_bytes: 789,
          sha256: sha,
          state: "verified",
          hf_match: { repo_id: "org/repo-name-GGUF", filename: "model.Q4_K_M.gguf", revision: "main", deleted: false },
        },
      ],
    });

    const healed = repo.getModel(sha);
    expect(healed?.hf_repo).toBe("org/repo-name-GGUF");
    expect(healed?.hf_file).toBe("model.Q4_K_M.gguf");
    expect(healed?.metadata.param_count).toBe(8_000_000_000);
  });

  it("adopts per-file GGUF metadata (quant/param_count/n_layer) into a hash-verified model's catalog row", async () => {
    const sha = "f".repeat(64);
    // Simulate the user's real-world row: a hand-dropped file whose
    // param_count was backfilled from HF's repo-level gguf.total -- wrong for
    // a multi-model repo (HF reports the 0.8B file's count for the 9B file).
    repo.registerModel({
      id: sha,
      filename: "Qwen3.5-9B/Qwen3.5-9B-PRISM-DQ.gguf",
      size_bytes: 4_651_885_536,
      source: "huggingface",
      hf_repo: "Ex0bit/Qwen3.5-PRISM-Dynamic-Quant-GGUF",
      hf_file: "Qwen3.5-9B/Qwen3.5-9B-PRISM-DQ.gguf",
      metadata: { param_count: 752_393_024 },
    });

    await postJson("/api/worker/heartbeat", {
      ...hardwareState("hb-hash-gguf-adopt", "idle"),
      model_files: [
        {
          path: "Qwen3.5-9B-PRISM-DQ.gguf", // flat, not in the HF subfolder
          size_bytes: 4_651_885_536,
          sha256: sha,
          state: "verified",
          n_layer: 32,
          param_count: 8_953_803_264,
          quant: "Q3_K_M",
          hf_match: {
            repo_id: "Ex0bit/Qwen3.5-PRISM-Dynamic-Quant-GGUF",
            filename: "Qwen3.5-9B/Qwen3.5-9B-PRISM-DQ.gguf",
            revision: "main",
            deleted: false,
          },
        },
      ],
    });

    const adopted = repo.getModel(sha);
    expect(adopted?.metadata.param_count).toBe(8_953_803_264);
    expect(adopted?.metadata.quant).toBe("Q3_K_M");
    expect(adopted?.metadata.n_layer).toBe(32);

    // Idempotent: a second identical heartbeat must not keep rewriting (no
    // throw, and values stay the same).
    await postJson("/api/worker/heartbeat", {
      ...hardwareState("hb-hash-gguf-adopt", "idle"),
      model_files: [
        {
          path: "Qwen3.5-9B-PRISM-DQ.gguf",
          size_bytes: 4_651_885_536,
          sha256: sha,
          state: "verified",
          n_layer: 32,
          param_count: 8_953_803_264,
          quant: "Q3_K_M",
          hf_match: {
            repo_id: "Ex0bit/Qwen3.5-PRISM-Dynamic-Quant-GGUF",
            filename: "Qwen3.5-9B/Qwen3.5-9B-PRISM-DQ.gguf",
            revision: "main",
            deleted: false,
          },
        },
      ],
    });
    expect(repo.getModel(sha)?.metadata.param_count).toBe(8_953_803_264);
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

  it("failing a benchmark job fails its run (not 'cancelled') and cancels any still-pending sibling job", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("complete-fail", "complete-fail");
    const runId = "run-complete-fail";
    repo.createTest(undefined, {
      id: runId,
      kind: null,
      root_run_id: runId,
      worker_id: worker.id,
      worker_name: "complete-fail",
      llama_cpp_build: "b2",
      llama_cpp_backend: "cpu",
      model_id: "m1",
      config: { model_id: "m1", sweep: {} } as never,
      status: "scheduled",
      started_at: Date.now(),
    } as never);
    repo.createTestItems(undefined, runId, [
      { idx: 0, n_prompt: 1, n_gen: 1, n_depth: 0, concurrency: 1, threads: 1, n_gpu_layers: 0, batch_size: 1, ubatch_size: 1, cache_type_k: "f16", cache_type_v: "f16", flash_attn: "on", mtp: "off", n_gpu_layers_draft: 0, n_cpu_moe: 0 },
    ] as never);
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

    // The RUN itself reads "failed" (nothing ever completed) with the
    // worker's real reported reason, never "cancelled" -- that label is
    // reserved for a genuine user stop or a run this server lost track of,
    // neither of which happened here (see repo.ts's reportJobFailure).
    const run = repo.getTest(undefined, runId);
    expect(run?.status).toBe("failed");
    expect(run?.error).toBe("install failed");
  });
});
