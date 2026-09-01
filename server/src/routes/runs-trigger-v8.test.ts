import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-v8-trigger-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "v8-trigger-secret";
process.env.LOG_DIR = join(tmpDir, "run-logs");

vi.mock("../github-releases.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../github-releases.js")>();
  return {
    ...actual,
    getReleases: vi.fn(async () => [
      {
        tag: "b9999",
        published_at: "2026-01-01T00:00:00Z",
        assets: [{ name: "llama-b9999-bin-ubuntu-x64.zip", download_url: "http://example.invalid/x.zip", size_bytes: 123 }],
      },
    ]),
  };
});

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

const baseSweep = {
  n_prompt: [512],
  n_gen: [128],
  threads: [8],
  n_gpu_layers: [0],
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

async function postJson(path: string, body: unknown) {
  return fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function heartbeat(
  machineId: string,
  opts: { capabilities?: string[]; model_files?: { path: string; size_bytes: number; sha256?: string }[] } = {}
) {
  return fetch(`${baseUrl}/api/worker/heartbeat`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer v8-trigger-secret" },
    body: JSON.stringify({
      machine_id: machineId,
      capabilities: opts.capabilities ?? ["benchmark"],
      hostname: machineId,
      backend: "cpu",
      hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
      installed_builds: [{ tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip", installed_at: 1, active: true }],
      model_files: opts.model_files ?? [],
      status: "idle",
    }),
  });
}

// Tests that exercise the active-roots quota need a clean slate -- earlier
// tests' scheduled runs would otherwise count against the same (global,
// single-tenant) quota.
function drainActiveRuns() {
  for (const run of repo.listRuns(undefined)) {
    if (run.status === "running" || run.status === "scheduled") {
      repo.reconcileStaleRun(undefined, run.id, "test cleanup");
    }
  }
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { runsRoutes } = await import("./runs.js");
  const { queueRoutes } = await import("./queue.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(runsRoutes);
  await app.register(queueRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  repo.registerModel({
    id: "v8-model",
    filename: "model.gguf",
    size_bytes: 1000,
    source: "local",
    metadata: {},
  });
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

describe("§0.2 depth rule + §0.5 budgets at trigger", () => {
  it("rejects n_depth > 0 on server-engine sweeps with the fix-naming copy", async () => {
    await heartbeat("v8-depth");
    const worker = repo.workerRepo.getByMachineId("v8-depth")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      sweep: { ...baseSweep, mtp: ["on"], n_depth: [4096] },
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("llama-server has no KV-prefill flag");
  });

  it("hard-rejects sweeps expanding past MAX_SWEEP_ITEMS with a 400", async () => {
    await heartbeat("v8-budget");
    const worker = repo.workerRepo.getByMachineId("v8-budget")!;
    const bigAxis = Array.from({ length: 64 }, (_, i) => i);
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      // 64 x 64 x 8 = 32768 items
      sweep: { ...baseSweep, n_prompt: bigAxis, n_gen: bigAxis, ubatch_size: [256, 512, 1024, 2048, 128, 64, 32, 16] },
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("hard cap is 20000");
  });

  it("rejects axis arrays longer than MAX_AXIS_VALUES and repeats outside 1..25", async () => {
    await heartbeat("v8-axis");
    const worker = repo.workerRepo.getByMachineId("v8-axis")!;
    const tooMany = Array.from({ length: 65 }, (_, i) => i);
    const res1 = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      sweep: { ...baseSweep, threads: tooMany },
    });
    expect(res1.status).toBe(400);
    const res2 = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      sweep: { ...baseSweep, repeats: 26 },
    });
    expect(res2.status).toBe(400);
    const res3 = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      sweep: { ...baseSweep, cache_type_k: ["q6_K"] },
    });
    expect(res3.status).toBe(400);
  });
});

describe("§0.5 duplicate-trigger guard", () => {
  it("refuses a second same-kind run on the same (user, model, worker) and names the root", async () => {
    await heartbeat("v8-dup");
    const worker = repo.workerRepo.getByMachineId("v8-dup")!;
    const first = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
    });
    expect(first.status).toBe(201);
    const firstRun = ((await first.json()) as { run: { id: string; root_run_id: string; kind: string } }).run;
    expect(firstRun.root_run_id).toBe(firstRun.id);

    const second = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
    });
    expect(second.status).toBe(409);
    const body = (await second.json()) as { error: string };
    expect(body.error).toContain(`a sweep run for this model`);
    expect(body.error).toContain(firstRun.id);
    expect(body.error).toMatch(/Stop it first or wait for it to finish/);

    // A different kind is not blocked by the standalone/sweep guard.
    drainActiveRuns();
    await heartbeat("v8-dup", { capabilities: ["benchmark", "curve-v1"] });
    const otherKind = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "runtime",
      sweep: baseSweep,
      curve_point: { effective_ctx: 2048 },
    });
    expect(otherKind.status).toBe(201);
  });

  it("NULL-kind (standalone) triggers only conflict with NULL-kind runs", async () => {
    drainActiveRuns();
    await heartbeat("v8-nullkind");
    const worker = repo.workerRepo.getByMachineId("v8-nullkind")!;
    const standalone = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    expect(standalone.status).toBe(201);
    const dupStandalone = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    expect(dupStandalone.status).toBe(409);
    const body = (await dupStandalone.json()) as { error: string };
    expect(body.error).toContain("standalone");
  });
});

describe("M2 goals block", () => {
  it("echoes the goals block on the created run verbatim", async () => {
    await heartbeat("v8-goals-a");
    const worker = repo.workerRepo.getByMachineId("v8-goals-a")!;
    const goals = { goal: "max_context", target_ctx: 32768, speed_floor_frac: 0.5, workload: "chat", kv_preset: "extended" };
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "sweep",
      goals,
      sweep: { ...baseSweep, cache_type_v: ["q8_0"] },
    });
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { config: { goals?: unknown } } }).run;
    expect(run.config.goals).toEqual(goals);
  });

  it("omits the goals key entirely when skipped -- byte-identical legacy payload", async () => {
    await heartbeat("v8-goals-b");
    const worker = repo.workerRepo.getByMachineId("v8-goals-b")!;
    const payload = { model_id: "v8-model", worker_id: worker.id, sweep: baseSweep };
    const res = await postJson("/api/runs/trigger", payload);
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { config: Record<string, unknown> } }).run;
    expect("goals" in run.config).toBe(false);
  });
});

describe("§0.7/N2 capability gates", () => {
  it("refuses probes on workers without probe-v1 with the update copy", async () => {
    await heartbeat("v8-cap-old");
    const worker = repo.workerRepo.getByMachineId("v8-cap-old")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 32768, placement: { ngl: 16, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
    });
    expect(res.status).toBe(409);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("older worker build");
  });

  it("accepts probes on workers advertising probe-v1 and enqueues a run_probe job", async () => {
    await heartbeat("v8-cap-new", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-new")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 32768, placement: { ngl: 16, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
    });
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { id: string; kind: string } }).run;
    expect(run.kind).toBe("probe");
  });

  it("validates probe bounds -- out-of-bounds yields 400, never a silent clamp", async () => {
    await heartbeat("v8-cap-bounds", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-bounds")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 100, placement: { ngl: 16, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
    });
    expect(res.status).toBe(400);
    const res2 = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 4096, placement: { ngl: 99999, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
    });
    expect(res2.status).toBe(400);
    const res3 = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 4096, placement: { ngl: 0, slots: 1 }, kv_pair: ["bogus", "f16"] },
      sweep: baseSweep,
    });
    expect(res3.status).toBe(400);
  });

  it("stores a probe's mode and granularity on the run, and passes them to the worker job", async () => {
    await heartbeat("v8-cap-mode", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-mode")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: {
        candidate_ctx: 32768,
        placement: { ngl: 16, slots: 1 },
        kv_pair: ["f16", "f16"],
        mode: "max_gpu",
        granularity: "fine",
      },
      sweep: baseSweep,
    });
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { id: string; config: { probe: Record<string, unknown> } } }).run;
    expect(run.config.probe.mode).toBe("max_gpu");
    expect(run.config.probe.granularity).toBe("fine");

    const job = repo.queueRepo.claimNextJob(worker.id);
    expect(job?.type).toBe("run_probe");
    expect((job?.payload as { mode?: string; granularity?: string }).mode).toBe("max_gpu");
    expect((job?.payload as { mode?: string; granularity?: string }).granularity).toBe("fine");
  });

  // A probe (and a quality measurement, same job-type family) rides its own
  // job type -- 'run_probe', not 'benchmark' -- so claimNextJob's flip-to-
  // 'running' check has to name it explicitly or the run sits at 'scheduled'
  // for its entire (often minutes-long) execution: the Runs page mislabels a
  // live probe, and RunDetail hides its elapsed-time/Stop controls the whole
  // time it's actually running.
  it("flips a probe run to 'running' the moment the worker claims its run_probe job", async () => {
    await heartbeat("v8-cap-running", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-running")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 32768, placement: { ngl: 16, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
    });
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { id: string; status: string } }).run;
    expect(run.status).toBe("scheduled");

    const job = repo.queueRepo.claimNextJob(worker.id);
    expect(job?.type).toBe("run_probe");
    expect(repo.getRun(undefined, run.id)?.status).toBe("running");
  });

  it("rejects an unknown probe mode or granularity rather than silently searching differently", async () => {
    await heartbeat("v8-cap-mode-bad", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-mode-bad")!;
    const badMode = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 4096, placement: { ngl: 0, slots: 1 }, kv_pair: ["f16", "f16"], mode: "go-fast" },
      sweep: baseSweep,
    });
    expect(badMode.status).toBe(400);

    const badGranularity = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 4096, placement: { ngl: 0, slots: 1 }, kv_pair: ["f16", "f16"], granularity: "coarse" },
      sweep: baseSweep,
    });
    expect(badGranularity.status).toBe(400);
  });

  it("refuses N4 quality runs on workers without quality-v1, with the update copy", async () => {
    drainActiveRuns();
    await heartbeat("v8-cap-quality-old");
    const worker = repo.workerRepo.getByMachineId("v8-cap-quality-old")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "quality",
      quality: { ctx_tokens: 32768, kv_pair: ["f16", "f16"], dataset_hash: `sha256:${"a".repeat(64)}` },
      sweep: baseSweep,
    });
    expect(res.status).toBe(409);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("older worker build");
  });

  it("accepts N4 quality runs on workers advertising quality-v1 and enqueues a measure_quality job", async () => {
    drainActiveRuns();
    await heartbeat("v8-cap-quality-new", { capabilities: ["benchmark", "quality-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-quality-new")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "quality",
      quality: { ctx_tokens: 32768, kv_pair: ["f16", "f16"], dataset_hash: `sha256:${"a".repeat(64)}` },
      sweep: baseSweep,
    });
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { id: string; kind: string } }).run;
    expect(run.kind).toBe("quality");
    const jobs = repo.queueRepo.getNonTerminalJobsForRun(run.id);
    expect(jobs.map((j) => j.jobType)).toContain("measure_quality");
  });

  it("validates quality bounds -- out-of-bounds yields 400, never a silent clamp", async () => {
    drainActiveRuns();
    await heartbeat("v8-cap-quality-bounds", { capabilities: ["benchmark", "quality-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-quality-bounds")!;
    const badCtx = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "quality",
      quality: { ctx_tokens: 100, kv_pair: ["f16", "f16"], dataset_hash: `sha256:${"a".repeat(64)}` },
      sweep: baseSweep,
    });
    expect(badCtx.status).toBe(400);
    const badKv = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "quality",
      quality: { ctx_tokens: 4096, kv_pair: ["bogus", "f16"], dataset_hash: `sha256:${"a".repeat(64)}` },
      sweep: baseSweep,
    });
    expect(badKv.status).toBe(400);
    const badHash = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "quality",
      quality: { ctx_tokens: 4096, kv_pair: ["f16", "f16"], dataset_hash: "not-a-hash" },
      sweep: baseSweep,
    });
    expect(badHash.status).toBe(400);
  });

  it("refuses N1 curve points on workers without curve-v1", async () => {
    await heartbeat("v8-cap-curve-old");
    const worker = repo.workerRepo.getByMachineId("v8-cap-curve-old")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "runtime",
      curve_point: { effective_ctx: 4096 },
      sweep: baseSweep,
    });
    expect(res.status).toBe(409);
  });

  it("refuses curve points beyond trained ctx -- nothing beyond it can appear as measured", async () => {
    repo.registerModel({
      id: "v8-trained",
      filename: "trained.gguf",
      size_bytes: 1000,
      source: "local",
      metadata: { trained_ctx: 32768 },
    });
    await heartbeat("v8-cap-curve-new", { capabilities: ["benchmark", "curve-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-curve-new")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-trained",
      worker_id: worker.id,
      kind: "runtime",
      curve_point: { effective_ctx: 65536 },
      sweep: baseSweep,
    });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toContain("trained context");
  });

  // N1's "Measure missing points" prices every uncovered ladder cell and
  // must enqueue them as ONE run -- a run per context would 409 against
  // itself past the first via the §0.5 duplicate-trigger guard.
  it("accepts a list of curve-point contexts and creates one run_item per context, all in one run", async () => {
    drainActiveRuns();
    await heartbeat("v8-cap-curve-multi", { capabilities: ["benchmark", "curve-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-curve-multi")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "runtime",
      curve_point: { effective_ctx: [2048, 4096, 8192] },
      sweep: baseSweep,
    });
    expect(res.status).toBe(201);
    const run = ((await res.json()) as { run: { id: string } }).run;
    const items = repo.getRunItems(run.id);
    expect(items.map((i) => i.idx).sort((a, b) => a - b)).toEqual([0, 1, 2]);
    expect(items.map((i) => i.n_prompt).sort((a, b) => a - b)).toEqual([2048, 4096, 8192]);
  });

  it("rejects a curve-point list if any entry exceeds trained ctx", async () => {
    drainActiveRuns();
    await heartbeat("v8-cap-curve-multi-bad", { capabilities: ["benchmark", "curve-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-cap-curve-multi-bad")!;
    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-trained",
      worker_id: worker.id,
      kind: "runtime",
      curve_point: { effective_ctx: [2048, 65536] },
      sweep: baseSweep,
    });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toContain("trained context");
  });
});

// N2 batching -- lets several probe search modes fired from one "Run test"
// click collapse into one Runs-list row instead of each 409ing against the
// duplicate-trigger guard / becoming its own root.
describe("N2 probe batching", () => {
  async function triggerProbe(workerId: string, extra: Record<string, unknown> = {}) {
    return postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: workerId,
      kind: "probe",
      probe: { candidate_ctx: 4096, placement: { ngl: 0, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
      ...extra,
    });
  }

  it("a same-batch sibling bypasses the duplicate-trigger guard and shares the first run's root", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-a", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-a")!;

    const first = await triggerProbe(worker.id);
    expect(first.status).toBe(201);
    const firstRun = ((await first.json()) as { run: { id: string; root_run_id: string } }).run;
    expect(firstRun.root_run_id).toBe(firstRun.id);
    // Still non-terminal -- claimNextJob is never called here, so the sibling
    // trigger below has to bypass findBlockingRun on its own merits, not
    // because the first run happened to finish first.
    expect(repo.getRun(undefined, firstRun.id)?.status).toBe("scheduled");

    const second = await triggerProbe(worker.id, { probe_batch_root_id: firstRun.id });
    expect(second.status).toBe(201);
    const secondRun = ((await second.json()) as { run: { id: string; root_run_id: string } }).run;
    expect(secondRun.root_run_id).toBe(firstRun.id);
    expect(secondRun.id).not.toBe(firstRun.id);
  });

  it("without probe_batch_root_id, a second probe on the same worker still 409s", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-b", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-b")!;

    const first = await triggerProbe(worker.id);
    expect(first.status).toBe(201);
    const second = await triggerProbe(worker.id);
    expect(second.status).toBe(409);
  });

  it("rejects a probe_batch_root_id that isn't a probe run", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-c", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-c")!;
    const sweepRun = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
    });
    expect(sweepRun.status).toBe(201);
    const sweepRunId = ((await sweepRun.json()) as { run: { id: string } }).run.id;

    const res = await triggerProbe(worker.id, { probe_batch_root_id: sweepRunId });
    expect(res.status).toBe(400);
  });

  it("rejects probe_batch_root_id on a non-probe kind", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-d", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-d")!;
    const first = await triggerProbe(worker.id);
    const firstRunId = ((await first.json()) as { run: { id: string } }).run.id;

    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-model",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      probe_batch_root_id: firstRunId,
    });
    expect(res.status).toBe(400);
  });

  it("a 5-member probe batch never hits the chain depth cap, unlike parent_run_id chaining", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-depth", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-depth")!;

    const first = await triggerProbe(worker.id);
    expect(first.status).toBe(201);
    const rootId = ((await first.json()) as { run: { id: string } }).run.id;
    for (let i = 0; i < 4; i++) {
      const res = await triggerProbe(worker.id, { probe_batch_root_id: rootId });
      expect(res.status).toBe(201);
      const run = ((await res.json()) as { run: { id: string; config: { chain_depth?: number } } }).run;
      expect(run.config.chain_depth).toBeUndefined();
    }
    const members = repo.listRunsUnderRoot(undefined, rootId);
    expect(members).toHaveLength(5);
  });

  it("GET /api/runs/:id/batch-members lists every sibling under the root", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-members", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-members")!;
    const first = await triggerProbe(worker.id);
    const firstRunId = ((await first.json()) as { run: { id: string } }).run.id;
    const second = await triggerProbe(worker.id, { probe_batch_root_id: firstRunId });
    const secondRunId = ((await second.json()) as { run: { id: string } }).run.id;

    const res = await fetch(`${baseUrl}/api/runs/${firstRunId}/batch-members`);
    expect(res.status).toBe(200);
    const { members } = (await res.json()) as { members: { id: string }[] };
    expect(members.map((m) => m.id).sort()).toEqual([firstRunId, secondRunId].sort());

    // Reachable from any member's own id too, not just the root's.
    const fromSibling = await fetch(`${baseUrl}/api/runs/${secondRunId}/batch-members`);
    const { members: fromSiblingMembers } = (await fromSibling.json()) as { members: { id: string }[] };
    expect(fromSiblingMembers).toHaveLength(2);
  });

  // Item 6's actual ask: not just "another trigger while the first sits
  // unclaimed", but "while a scenario is genuinely RUNNING on the worker".
  it("a sibling can be added while the first batch member is actually claimed/running, not just scheduled", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-running", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-running")!;

    const first = await triggerProbe(worker.id);
    const firstRunId = ((await first.json()) as { run: { id: string } }).run.id;
    // Simulate the worker actually starting it -- claimNextJob flips a
    // run_probe job's run to 'running' in the same transaction (repo.ts).
    const claimed = repo.queueRepo.claimNextJob(worker.id);
    expect(claimed?.type).toBe("run_probe");
    expect(repo.getRun(undefined, firstRunId)?.status).toBe("running");

    const second = await triggerProbe(worker.id, { probe_batch_root_id: firstRunId });
    expect(second.status).toBe(201);
    const secondRun = ((await second.json()) as { run: { id: string; root_run_id: string; status: string } }).run;
    expect(secondRun.root_run_id).toBe(firstRunId);
    // Queued behind the running one -- not claimed by this same call, so it
    // sits exactly where the FIFO leaves any fresh job: scheduled/pending.
    expect(secondRun.status).toBe("scheduled");
  });

  it("stopping any member cascades to every non-terminal sibling under the same root", async () => {
    drainActiveRuns();
    await heartbeat("v8-batch-stop", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-batch-stop")!;
    const first = await triggerProbe(worker.id);
    const firstRunId = ((await first.json()) as { run: { id: string } }).run.id;
    const second = await triggerProbe(worker.id, { probe_batch_root_id: firstRunId });
    const secondRunId = ((await second.json()) as { run: { id: string } }).run.id;
    const third = await triggerProbe(worker.id, { probe_batch_root_id: firstRunId });
    const thirdRunId = ((await third.json()) as { run: { id: string } }).run.id;

    // Neither job has been claimed yet -- both are still pending, so stop
    // reconciles them immediately rather than waiting on a heartbeat.
    const stopRes = await postJson(`/api/runs/${firstRunId}/stop`, {});
    expect(stopRes.status).toBe(200);

    expect(repo.getRun(undefined, firstRunId)?.status).toBe("cancelled");
    expect(repo.getRun(undefined, secondRunId)?.status).toBe("cancelled");
    expect(repo.getRun(undefined, thirdRunId)?.status).toBe("cancelled");
  });
});

describe("N3 comparisons require the model already registered on the target worker", () => {
  it("refuses a comparison member whose model the worker never reported having, when the worker CAN report hashes", async () => {
    drainActiveRuns();
    repo.registerModel({ id: "v8-cmp-a", filename: "cmp-a.gguf", size_bytes: 1000, source: "local", metadata: {} });
    repo.registerModel({ id: "v8-cmp-b", filename: "cmp-b.gguf", size_bytes: 1000, source: "local", metadata: {} });
    // Reports at least one hash -- a hashing-capable worker -- but not for
    // either comparison model, so the membership check applies and finds no
    // match.
    await heartbeat("v8-cmp-worker-missing", {
      model_files: [{ path: "unrelated.gguf", size_bytes: 500, sha256: "f".repeat(64) }],
    });
    const worker = repo.workerRepo.getByMachineId("v8-cmp-worker-missing")!;

    const first = await postJson("/api/runs/trigger", {
      model_id: "v8-cmp-a",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      comparison_id: "cmp-missing-1",
    });
    expect(first.status).toBe(400);
    const body = (await first.json()) as { error: string };
    expect(body.error).toContain("local cache");
    expect(body.error).toContain("cmp-a.gguf");
  });

  it("accepts comparison members the worker actually reports having by sha256 (models.id)", async () => {
    drainActiveRuns();
    // models.id IS the file's sha256 in production -- a real 64-char hex
    // digest, not an arbitrary id, since that identity is exactly what this
    // check matches on.
    const hashC = "c".repeat(64);
    const hashD = "d".repeat(64);
    repo.registerModel({ id: hashC, filename: "cmp-c.gguf", size_bytes: 1000, source: "local", metadata: {} });
    repo.registerModel({ id: hashD, filename: "cmp-d.gguf", size_bytes: 1000, source: "local", metadata: {} });
    await heartbeat("v8-cmp-worker-present", {
      model_files: [
        { path: "cmp-c.gguf", size_bytes: 1000, sha256: hashC },
        { path: "cmp-d.gguf", size_bytes: 1000, sha256: hashD },
      ],
    });
    const worker = repo.workerRepo.getByMachineId("v8-cmp-worker-present")!;

    const first = await postJson("/api/runs/trigger", {
      model_id: hashC,
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      comparison_id: "cmp-present-1",
    });
    expect(first.status).toBe(201);
    const second = await postJson("/api/runs/trigger", {
      model_id: hashD,
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      comparison_id: "cmp-present-1",
    });
    expect(second.status).toBe(201);
  });

  it("does not enforce the check against a worker that has never reported any file hash (version-skew: degrade, don't block)", async () => {
    drainActiveRuns();
    repo.registerModel({ id: "v8-cmp-e", filename: "cmp-e.gguf", size_bytes: 1000, source: "local", metadata: {} });
    // No sha256 on any reported file (or no files at all) -- an older,
    // pre-hashing worker build.
    await heartbeat("v8-cmp-worker-legacy", { model_files: [{ path: "cmp-e.gguf", size_bytes: 1000 }] });
    const worker = repo.workerRepo.getByMachineId("v8-cmp-worker-legacy")!;

    const res = await postJson("/api/runs/trigger", {
      model_id: "v8-cmp-e",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      comparison_id: "cmp-legacy-1",
    });
    expect(res.status).toBe(201);
  });
});

describe("§0.5 chain quotas", () => {
  it("enforces the <= 3 active roots quota per user (probes exempt)", async () => {
    drainActiveRuns();
    await heartbeat("v8-quota", { capabilities: ["benchmark", "probe-v1"] });
    const worker = repo.workerRepo.getByMachineId("v8-quota")!;
    void repo;
    const ids: string[] = [];
    for (let i = 0; i < 3; i++) {
      const res = await postJson("/api/runs/trigger", {
        model_id: `v8-model-${i}`,
        worker_id: worker.id,
        sweep: baseSweep,
      });
      if (res.status !== 201) {
        // register missing models then retry once
        repo.registerModel({ id: `v8-model-${i}`, filename: `m${i}.gguf`, size_bytes: 1, source: "local", metadata: {} });
        const retry = await postJson("/api/runs/trigger", {
          model_id: `v8-model-${i}`,
          worker_id: worker.id,
          sweep: baseSweep,
        });
        expect(retry.status).toBe(201);
        ids.push(((await retry.json()) as { run: { id: string } }).run.id);
      } else {
        repo.registerModel({ id: `v8-model-${i}`, filename: `m${i}.gguf`, size_bytes: 1, source: "local", metadata: {} });
        ids.push(((await res.json()) as { run: { id: string } }).run.id);
      }
    }
    // A fourth non-probe run must be refused...
    repo.registerModel({ id: "v8-model-x", filename: "mx.gguf", size_bytes: 1, source: "local", metadata: {} });
    const fourth = await postJson("/api/runs/trigger", {
      model_id: "v8-model-x",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    expect(fourth.status).toBe(409);
    // ...but a probe is exempt.
    const probe = await postJson("/api/runs/trigger", {
      model_id: "v8-model-x",
      worker_id: worker.id,
      kind: "probe",
      probe: { candidate_ctx: 4096, placement: { ngl: 0, slots: 1 }, kv_pair: ["f16", "f16"] },
      sweep: baseSweep,
    });
    expect(probe.status).toBe(201);
  });

  it("cancels running roots past the 48h wall clock", async () => {
    await heartbeat("v8-wallclock");
    const worker = repo.workerRepo.getByMachineId("v8-wallclock")!;
    repo.registerModel({ id: "v8-old-model", filename: "old.gguf", size_bytes: 1, source: "local", metadata: {} });
    repo.createRun(undefined, {
      id: "v8-old-root",
      kind: "sweep",
      worker_id: worker.id,
      worker_name: "w",
      llama_cpp_build: "b9999",
      llama_cpp_backend: "cpu",
      model_id: "v8-old-model",
      config: { model_id: "v8-old-model", sweep: baseSweep } as never,
      status: "running",
      started_at: Date.now() - 49 * 3600 * 1000,
    });
    repo.createRunItems(
      undefined,
      "v8-old-root",
      [{ idx: 0, n_prompt: 1, n_gen: 1, n_depth: 0, concurrency: 1, threads: 1, n_gpu_layers: 0, batch_size: 1, ubatch_size: 1, cache_type_k: "f16", cache_type_v: "f16", flash_attn: "on", mtp: "off", n_gpu_layers_draft: 0, n_cpu_moe: 0 }]
    );
    const cancelled = repo.cancelExpiredRoots();
    expect(cancelled).toContain("v8-old-root");
    expect(repo.getRun(undefined, "v8-old-root")!.status).not.toBe("running");
  });
});
