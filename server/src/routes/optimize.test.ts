import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-optimize-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "optimize-secret";
process.env.LOG_DIR = join(tmpDir, "run-logs");

vi.mock("../github-releases.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../github-releases.js")>();
  return {
    ...actual,
    getReleases: vi.fn(async () => [
      {
        tag: "b12000",
        published_at: "2026-02-01T00:00:00Z",
        assets: [{ name: "llama-b12000-bin-ubuntu-x64.zip", download_url: "http://example.invalid/y.zip", size_bytes: 40000000 }],
      },
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
let getDb: typeof import("../db/migrate.js")["getDb"];

const sweep = {
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
  repeats: 1,
};
const ALL_CAPS = ["benchmark", "fit-map-v1", "speed-run-v1", "thread-sweep-v1"];

async function post(path: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}
const workerAuth = { authorization: "Bearer optimize-secret" };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const json = async (res: Response): Promise<any> => res.json();

const DEFAULT_BUILDS: Record<string, unknown>[] = [
  { tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip", installed_at: 1, active: true, fit_params_path: "/x/llama-fit-params" },
];

async function heartbeat(machineId: string, capabilities = ALL_CAPS, builds = DEFAULT_BUILDS) {
  const res = await post(
    "/api/worker/heartbeat",
    {
      machine_id: machineId,
      capabilities,
      hostname: machineId,
      backend: "cpu",
      hardware: {
        platform: "linux",
        arch: "x64",
        cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 },
        gpu: [],
        mem_total_bytes: 32 * 2 ** 30,
        cpu_topology: {
          source: "linux",
          cores: [{ id: 0, logical: [0, 1], l3: 0, die: null, effClass: 0 }],
          logicalCount: 2,
          masksSupported: true,
          bogus: "dropped",
        },
      },
      installed_builds: builds,
      model_files: [],
      status: "idle",
    },
    workerAuth
  );
  expect(res.status).toBe(200);
  return repo.workerRepo.getByMachineId(machineId)!;
}

function jobsFor(runId: string) {
  return getDb().prepare(`SELECT job_type, payload_json FROM worker_jobs WHERE run_id = ?`).all(runId) as {
    job_type: string;
    payload_json: string;
  }[];
}

function markRunning(runId: string) {
  getDb().prepare(`UPDATE runs SET status = 'running' WHERE id = ?`).run(runId);
}

const point = (over: Record<string, unknown> = {}) => ({
  ctx: 32768, fa: "on", ctk: "f16", ctv: "f16", margin_mib: 1024, verdict: "partial", reason: null, ngl: 16, layers_gpu: 16,
  layers_total: 65, overflow_layers: 0, ot: null, dev_used_mib: 6156, dev_free_mib: 1072, host_used_mib: 15184, total_mib: 21340,
  kv_mib: 2197, compute_mib: 633, need_all_gpu_mib: 20530, splits: 2, fa_disabled: false, n_expert: 0, n_expert_used: 0,
  inferred: false, raw_args: "-c 32768 -ngl 16", error: null, ...over,
});

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ getDb } = await import("../db/migrate.js"));
  const { testsRoutes } = await import("./tests.js");
  const { queueRoutes } = await import("./queue.js");
  const { optimizeRoutes } = await import("./optimize.js");
  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(testsRoutes);
  await app.register(queueRoutes);
  await app.register(optimizeRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;
  repo.registerModel({ id: "opt-model", filename: "model.gguf", size_bytes: 1000, source: "local", metadata: { trained_ctx: 262144 } });
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

const fitMap = { margin_mib: 1024, ctx_stops: [4096, 32768, 262144], machine: "dgpu", kv_detect: true, kv_fit: true, target_ctx: 32768 };

describe("heartbeat carries the new hardware fields", () => {
  it("keeps cpu_topology (sanitized) and fit_params_path", async () => {
    const w = await heartbeat("opt-hw");
    expect(w.hardware?.cpu_topology?.cores[0].logical).toEqual([0, 1]);
    expect((w.hardware?.cpu_topology as unknown as Record<string, unknown>).bogus).toBeUndefined();
    expect(w.installedBuilds?.[0].fit_params_path).toBe("/x/llama-fit-params");
  });
});

describe("fit-map runs", () => {
  it("rejects a fit block without kind fit, a bad spec, and an old worker", async () => {
    const w = await heartbeat("opt-a");
    expect((await post("/api/tests/trigger", { model_id: "opt-model", worker_id: w.id, sweep, fit_map: fitMap })).status).toBe(400);
    const bad = await post("/api/tests/trigger", { model_id: "opt-model", worker_id: w.id, sweep, kind: "fit", fit_map: { ...fitMap, margin_mib: 700 } });
    expect(bad.status).toBe(400);
    expect((await json(bad)).error).toMatch(/margin/);
    const old = await heartbeat("opt-old", ["benchmark"]);
    const res = await post("/api/tests/trigger", { model_id: "opt-model", worker_id: old.id, sweep, kind: "fit", fit_map: fitMap });
    expect(res.status).toBe(409);
  });

  it("creates one item and a fit_map job with the RAM budget; points and KV rows round-trip", async () => {
    const w = await heartbeat("opt-b");
    const res = await post("/api/tests/trigger", { model_id: "opt-model", worker_id: w.id, sweep, kind: "fit", fit_map: fitMap, flow_id: "flow-abc123" });
    expect(res.status).toBe(201);
    const { run } = await json(res);
    const jobs = jobsFor(run.id).filter((j) => j.job_type === "fit_map");
    expect(jobs).toHaveLength(1);
    const payload = JSON.parse(jobs[0].payload_json);
    expect(payload.ram_budget_mib).toBe(32 * 1024 - 1024);
    expect(payload.spec.ctx_stops).toEqual([4096, 32768, 262144]);
    expect(repo.getTestWithResults(undefined, run.id)!.items).toHaveLength(1);

    // Until a worker claims it the run is closed to posts.
    const early = await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-b", points: [point()] }, workerAuth);
    expect(early.status).toBe(409);
    markRunning(run.id);

    // Another machine can't post into this run.
    const other = await heartbeat("opt-c");
    const forbidden = await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-c", points: [point()] }, workerAuth);
    expect(forbidden.status).toBe(403);
    void other;

    const ok = await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-b", points: [point(), point({ ctx: 4096, verdict: "full", layers_gpu: 65 })] }, workerAuth);
    expect(ok.status).toBe(200);
    // Re-posting the same cell replaces it.
    await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-b", points: [point({ layers_gpu: 17, ngl: 17 })] }, workerAuth);
    const invalid = await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-b", points: [point({ ctk: "f32" })] }, workerAuth);
    expect(invalid.status).toBe(400);

    const kv = await post(
      `/api/tests/${run.id}/kv-support`,
      { machine_id: "opt-b", rows: [{ ctk: "q8_0", ctv: "f16", fa: "off", status: "cpu_fallback", splits: 98, baseline_splits: 2, detail: "x" }] },
      workerAuth
    );
    expect(kv.status).toBe(200);

    const got = await json(await fetch(`${baseUrl}/api/tests/${run.id}/fit-points`));
    expect(got.points).toHaveLength(2);
    expect(got.points.find((p: { ctx: number }) => p.ctx === 32768).layers_gpu).toBe(17);
    expect(got.kv).toEqual([{ ctk: "q8_0", ctv: "f16", fa: "off", status: "cpu_fallback", splits: 98, baseline_splits: 2, detail: "x" }]);
    expect(got.spec.margin_mib).toBe(1024);

    const flow = await json(await fetch(`${baseUrl}/api/flows/flow-abc123`));
    expect(flow.tests.map((t: { id: string }) => t.id)).toEqual([run.id]);

    // Known KV rows ride along on the next fit job for this machine/build/model.
    repo.reconcileStaleTest(undefined, run.id, "done");
    const again = await json(await post("/api/tests/trigger", { model_id: "opt-model", worker_id: w.id, sweep, kind: "fit", fit_map: fitMap }));
    const p2 = JSON.parse(jobsFor(again.run.id).find((j) => j.job_type === "fit_map")!.payload_json);
    expect(p2.kv_known).toHaveLength(1);
    repo.reconcileStaleTest(undefined, again.run.id, "done");

    // A finished run takes no more points or KV rows.
    const late = await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-b", points: [point({ ctx: 8192 })] }, workerAuth);
    expect(late.status).toBe(409);
    const lateKv = await post(`/api/tests/${run.id}/kv-support`, { machine_id: "opt-b", rows: [] }, workerAuth);
    expect(lateKv.status).toBe(409);
  });

  it("caps the points one run can hold; re-posting stored cells never counts", async () => {
    const { MAX_POINTS_PER_RUN } = await import("./optimize.js");
    const w = await heartbeat("opt-cap");
    const { run } = await json(await post("/api/tests/trigger", { model_id: "opt-model", worker_id: w.id, sweep, kind: "fit", fit_map: fitMap }));
    markRunning(run.id);
    const cells = (from: number, n: number) => Array.from({ length: n }, (_, i) => point({ ctx: 4096 + from + i }));
    for (let at = 0; at < MAX_POINTS_PER_RUN; at += 400) {
      const n = Math.min(400, MAX_POINTS_PER_RUN - at);
      expect((await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-cap", points: cells(at, n) }, workerAuth)).status).toBe(200);
    }
    // Full: a retry of stored cells still lands, a new cell is refused and rolls its whole post back.
    expect((await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-cap", points: cells(0, 5) }, workerAuth)).status).toBe(200);
    const over = await post(`/api/tests/${run.id}/fit-points`, { machine_id: "opt-cap", points: [point({ ctx: 4096 + MAX_POINTS_PER_RUN, verdict: "full" }), ...cells(0, 1)] }, workerAuth);
    expect(over.status).toBe(400);
    const n = (getDb().prepare(`SELECT COUNT(*) AS n FROM fit_points WHERE run_id = ?`).get(run.id) as { n: number }).n;
    expect(n).toBe(MAX_POINTS_PER_RUN);
    repo.reconcileStaleTest(undefined, run.id, "done");
  });
});

describe("speed runs and thread sweeps", () => {
  const candidate = { label: "Default settings", source: "baseline", ngl: 999, ot: null, ctk: "f16", ctv: "f16", fa: "auto", threads: null };
  const speedRun = { target_ctx: 32768, prompt_tokens: 4096, n_gen: 512, repeats: 3, candidates: [candidate] };

  it("validates the workload against the target context", async () => {
    const w = await heartbeat("opt-d");
    const small = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime",
      speed_run: { ...speedRun, target_ctx: 4096 },
    });
    expect(small.status).toBe(400);
    expect((await json(small)).error).toMatch(/3328-token prompt/);
    const tooMany = await post("/api/tests/trigger", { model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime", speed_run: { ...speedRun, repeats: 11 } });
    expect(tooMany.status).toBe(400);
    const badOt = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime",
      speed_run: { ...speedRun, candidates: [{ ...candidate, ot: "x; rm -rf /" }] },
    });
    expect(badOt.status).toBe(400);
  });

  it("one item per candidate, dispatched as a speed_run benchmark job", async () => {
    const w = await heartbeat("opt-e");
    const fitCand = {
      label: "Most layers", source: "fit", ngl: 41, fa: "on", ctk: "q8_0", ctv: "q8_0",
      ot: "blk\\.8\\.ffn_down.*=CPU,blk\\.9\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU",
      threads: { t: 6, tb: 6, mask: "0x555", maskBatch: "0x555", strict: true },
    };
    const res = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime", flow_id: "flow-speed1",
      speed_run: { ...speedRun, candidates: [candidate, fitCand] },
    });
    const body = await json(res);
    expect(res.status, JSON.stringify(body)).toBe(201);
    const { run } = body;
    const items = repo.getTestWithResults(undefined, run.id)!.items;
    expect(items.map((i) => [i.n_gpu_layers, i.cache_type_k, i.n_prompt, i.n_gen])).toEqual([
      [999, "f16", 4096, 512],
      [41, "q8_0", 4096, 512],
    ]);
    const job = jobsFor(run.id).find((j) => j.job_type === "benchmark")!;
    const payload = JSON.parse(job.payload_json);
    expect(payload.mode).toBe("speed_run");
    expect(payload.speed_run.candidates[1].ot).toContain("ffn_down");
    repo.reconcileStaleTest(undefined, run.id, "done");
  });

  it("batch sizes reach the items, and an older worker is refused rather than mislabelled", async () => {
    const old = await heartbeat("opt-batch-old");
    const batched = { ...speedRun, candidates: [{ ...candidate, b: 2048, ub: 1024 }] };
    const refused = await post("/api/tests/trigger", { model_id: "opt-model", worker_id: old.id, sweep, kind: "runtime", speed_run: batched });
    expect(refused.status).toBe(409);
    expect((await json(refused)).error).toMatch(/batch sizes/);

    const w = await heartbeat("opt-batch", [...ALL_CAPS, "speed-batch-v1"]);
    const bad = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime",
      speed_run: { ...speedRun, candidates: [{ ...candidate, b: 512, ub: 1024 }] },
    });
    expect(bad.status).toBe(400);
    const res = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime", flow_id: "flow-batch1", flow_step: "batch",
      speed_run: { ...speedRun, candidates: [candidate, { ...candidate, label: "-b 2048 -ub 1024", b: 2048, ub: 1024 }] },
    });
    const body = await json(res);
    expect(res.status, JSON.stringify(body)).toBe(201);
    const items = repo.getTestWithResults(undefined, body.run.id)!.items;
    expect(items.map((i) => [i.batch_size, i.ubatch_size])).toEqual([[2048, 512], [2048, 1024]]);
    repo.reconcileStaleTest(undefined, body.run.id, "done");
  });

  it("thread sweep items carry role-specific workloads", async () => {
    const w = await heartbeat("opt-f");
    const res = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime",
      thread_sweep: {
        placement: { ngl: 16, ot: null, ctk: "f16", ctv: "f16", fa: "on" },
        repeats: 3,
        items: [
          { role: "t", kind: "default", label: "all", threads: 6, mask: "0x555", strict: true },
          { role: "tb", kind: "one_core", label: "1 core", threads: 1, mask: "0x1", strict: true },
        ],
      },
    });
    expect(res.status).toBe(201);
    const { run } = await json(res);
    const items = repo.getTestWithResults(undefined, run.id)!.items;
    expect(items.map((i) => [i.n_threads, i.n_prompt, i.n_gen])).toEqual([[6, 0, 128], [1, 512, 0]]);
    expect(JSON.parse(jobsFor(run.id).find((j) => j.job_type === "benchmark")!.payload_json).mode).toBe("thread_sweep");
    repo.reconcileStaleTest(undefined, run.id, "done");
  });

  it("a runtime run carries one block at most", async () => {
    const w = await heartbeat("opt-g");
    const res = await post("/api/tests/trigger", {
      model_id: "opt-model", worker_id: w.id, sweep, kind: "runtime", speed_run: speedRun,
      thread_sweep: { placement: { ngl: 1, ot: null, ctk: "f16", ctv: "f16", fa: "on" }, repeats: 1, items: [] },
    });
    expect(res.status).toBe(400);
  });
});

describe("fit checks (docs/plans/HEADER_FIT_PLAN.md)", () => {
  const CHECK_CAPS = [...ALL_CAPS, "fit-check-v1"];
  const pendingChecks = (workerId: string) =>
    getDb().prepare(`SELECT payload_json FROM worker_jobs WHERE worker_id = ? AND job_type = 'fit_check' AND status = 'pending'`).all(workerId) as {
      payload_json: string;
    }[];

  it("refuses an older worker and a bad body", async () => {
    const old = await heartbeat("fc-old");
    expect((await post(`/api/workers/${old.id}/fit-check`, { model_ids: ["opt-model"], ctx: 32768 })).status).toBe(409);
    const w = await heartbeat("fc-bad", CHECK_CAPS);
    expect((await post(`/api/workers/${w.id}/fit-check`, { model_ids: [], ctx: 32768 })).status).toBe(400);
    expect((await post(`/api/workers/${w.id}/fit-check`, { model_ids: ["opt-model"], ctx: 32768, margin_mib: 700 })).status).toBe(400);
    expect((await post(`/api/workers/${w.id}/fit-check`, { model_ids: ["nope"], ctx: 32768 })).status).toBe(400);
  });

  it("queues one job with the newest build, refuses a second while it is pending, and stores answers", async () => {
    const w = await heartbeat("fc-a", CHECK_CAPS);
    const res = await post(`/api/workers/${w.id}/fit-check`, { model_ids: ["opt-model"], ctx: 32768 });
    expect(res.status).toBe(202);
    expect((await json(res)).queued).toBe(1);
    const jobs = pendingChecks(w.id);
    expect(jobs).toHaveLength(1);
    const payload = JSON.parse(jobs[0].payload_json);
    expect(payload).toMatchObject({ ctx: 32768, margin_mib: 1024, llama_cpp_build: "b9999", llama_cpp_backend: "cpu" });
    expect(payload.models.map((m: { id: string }) => m.id)).toEqual(["opt-model"]);
    expect(payload.ram_budget_mib).toBe(32 * 1024 - 1024);

    const again = await json(await post(`/api/workers/${w.id}/fit-check`, { model_ids: ["opt-model"], ctx: 32768 }));
    expect(again).toMatchObject({ queued: 0, pending: true });
    expect(pendingChecks(w.id)).toHaveLength(1);

    const result = { model_id: "opt-model", ctx: 32768, margin_mib: 1024, llama_cpp_build: "b9999", source: "header", vram_free_mib: 7378, point: point(), error: null, fetched_bytes: 11_000_000 };
    const stored = await post("/api/worker/fit-checks", { machine_id: "fc-a", results: [result] }, workerAuth);
    expect(stored.status).toBe(200);
    const list = await json(await fetch(`${baseUrl}/api/workers/${w.id}/fit-checks?ctx=32768`));
    expect(list.results).toHaveLength(1);
    expect(list.results[0]).toMatchObject({ model_id: "opt-model", source: "header", vram_free_mib: 7378, fetched_bytes: 11_000_000 });
    expect(list.results[0].point.layers_gpu).toBe(16);
  });

  it("does not ask again for a fresh answer from the same build", async () => {
    const w = await heartbeat("fc-b", CHECK_CAPS);
    const result = { model_id: "opt-model", ctx: 8192, margin_mib: 1024, llama_cpp_build: "b9999", source: "local", vram_free_mib: null, point: point({ ctx: 8192 }), error: null };
    expect((await post("/api/worker/fit-checks", { machine_id: "fc-b", results: [result] }, workerAuth)).status).toBe(200);
    const res = await json(await post(`/api/workers/${w.id}/fit-check`, { model_ids: ["opt-model"], ctx: 8192 }));
    expect(res.queued).toBe(0);
    expect(pendingChecks(w.id)).toHaveLength(0);
  });

  it("asks again after a failed check, and rejects a result with neither point nor error", async () => {
    const w = await heartbeat("fc-c", CHECK_CAPS);
    const failed = { model_id: "opt-model", ctx: 16384, margin_mib: 1024, llama_cpp_build: "b9999", source: "header", vram_free_mib: null, point: null, error: "unknown model architecture: 'glm5-next'" };
    expect((await post("/api/worker/fit-checks", { machine_id: "fc-c", results: [failed] }, workerAuth)).status).toBe(200);
    expect((await json(await post(`/api/workers/${w.id}/fit-check`, { model_ids: ["opt-model"], ctx: 16384 }))).queued).toBe(1);
    const empty = { ...failed, error: null };
    expect((await post("/api/worker/fit-checks", { machine_id: "fc-c", results: [empty] }, workerAuth)).status).toBe(400);
  });
});

describe("Will it fit? (docs/plans/MODEL_FIT_CHECK_PLAN.md)", () => {
  const CAPS = [...ALL_CAPS, "fit-check-v2"];
  const target = (file: string) => ({ hf_repo: "unsloth/gemma-GGUF", hf_file: file, size_bytes: 1000 });
  const keyOf = (file: string) => `hf:unsloth/gemma-GGUF/${file}`;
  const memory = { devices: [{ name: "Vulkan0", description: "GPU", total_mib: 8176, free_mib: 7378 }], ram_available_mib: 29000, ram_total_mib: 49000, unified: false };
  const report = (file: string, over: Record<string, unknown> = {}) => ({
    target_key: keyOf(file),
    ctx: 1024,
    margin_mib: 1024,
    llama_cpp_build: "b9999",
    backend: "cpu",
    source: "header",
    memory,
    point: point({ ctx: 1024 }),
    error: null,
    fetched_bytes: 16_000_000,
    ...over,
  });
  const jobsOf = (workerId: string, type = "fit_check") =>
    getDb().prepare(`SELECT payload_json, status FROM worker_jobs WHERE worker_id = ? AND job_type = ? ORDER BY created_at, rowid`).all(workerId, type) as {
      payload_json: string;
      status: string;
    }[];
  const finishJobs = (workerId: string) => getDb().prepare(`UPDATE worker_jobs SET status = 'completed' WHERE worker_id = ?`).run(workerId);

  it("refuses an older worker and a bad target, and never touches the catalog", async () => {
    const old = await heartbeat("mf-old");
    expect((await post(`/api/workers/${old.id}/model-fit`, { targets: [target("a.gguf")] })).status).toBe(409);
    const w = await heartbeat("mf-bad", CAPS);
    const bad = async (t: unknown) => (await post(`/api/workers/${w.id}/model-fit`, { targets: [t] })).status;
    expect(await bad({ hf_repo: "nope", hf_file: "a.gguf" })).toBe(400);
    expect(await bad({ hf_repo: "u/r", hf_file: "../../etc/passwd.gguf" })).toBe(400);
    expect(await bad({ hf_repo: "u/r", hf_file: "/abs.gguf" })).toBe(400);
    expect(await bad({ hf_repo: "u/r", hf_file: "a\\b.gguf" })).toBe(400);
    expect(await bad({ hf_repo: "u/r", hf_file: "notes.txt" })).toBe(400);
    expect(await bad({ model_id: "nope" })).toBe(400);
    expect((await post(`/api/workers/${w.id}/model-fit`, { targets: [] })).status).toBe(400);

    const before = (getDb().prepare(`SELECT COUNT(*) AS n FROM models`).get() as { n: number }).n;
    expect((await post(`/api/workers/${w.id}/model-fit`, { targets: [target("uncatalogued.gguf")] })).status).toBe(202);
    expect((getDb().prepare(`SELECT COUNT(*) AS n FROM models`).get() as { n: number }).n).toBe(before);
  });

  it("the plan names the build a machine would use", async () => {
    const w = await heartbeat("mf-plan", CAPS);
    const plan = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit/plan`));
    expect(plan).toMatchObject({ online: true, capable: true, ctx: 1024, margin_mib: 1024, build: { tag: "b9999", installed: true, backend: "cpu" } });
  });

  it("a second click adds its own job instead of being lost, and repeats are skipped", async () => {
    const w = await heartbeat("mf-two", CAPS);
    const first = await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")] }));
    expect(first).toMatchObject({ queued: 1, queued_keys: [keyOf("a.gguf")] });
    const second = await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf"), target("b.gguf")] }));
    expect(second).toMatchObject({ queued: 1, queued_keys: [keyOf("b.gguf")] });
    const jobs = jobsOf(w.id);
    expect(jobs).toHaveLength(2);
    expect(JSON.parse(jobs[1].payload_json)).toMatchObject({ ctx: 1024, target_keys: [keyOf("b.gguf")], llama_cpp_build: "b9999" });
    const list = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit`));
    expect(list.pending_keys.sort()).toEqual([keyOf("a.gguf"), keyOf("b.gguf")]);
  });

  it("stores answers with a level, clears pending, skips fresh ones unless forced", async () => {
    const w = await heartbeat("mf-store", CAPS);
    await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")] });
    await new Promise((r) => setTimeout(r, 5));
    expect((await post("/api/worker/model-fit", { machine_id: "mf-store", results: [report("a.gguf")] }, workerAuth)).status).toBe(200);
    const list = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit`));
    expect(list.pending_keys).toEqual([]);
    expect(list.results[0]).toMatchObject({ target_key: keyOf("a.gguf"), level: "partial", fetched_bytes: 16_000_000, memory });
    finishJobs(w.id);

    expect(await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")] }))).toMatchObject({ queued: 0 });
    expect(await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")], force: true }))).toMatchObject({ queued: 1 });
  });

  it("keeps the trained-context answer (bread grades) and rejects a malformed one", async () => {
    const w = await heartbeat("mf-ceiling", CAPS);
    const ceiling = { trained_ctx: 262144, fits_at_trained: false, max_full_ctx: 99072 };
    expect((await post("/api/worker/model-fit", { machine_id: "mf-ceiling", results: [report("a.gguf", { ceiling })] }, workerAuth)).status).toBe(200);
    const list = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit`));
    expect(list.results[0]).toMatchObject({ target_key: keyOf("a.gguf"), ceiling });
    // An older worker's answer has none.
    expect((await post("/api/worker/model-fit", { machine_id: "mf-ceiling", results: [report("b.gguf")] }, workerAuth)).status).toBe(200);
    const both = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit`));
    expect(both.results.find((r: { target_key: string }) => r.target_key === keyOf("b.gguf")).ceiling).toBeUndefined();
    for (const bad of [{ ...ceiling, trained_ctx: -1 }, { ...ceiling, fits_at_trained: "yes" }, { ...ceiling, max_full_ctx: 0.5 }]) {
      expect((await post("/api/worker/model-fit", { machine_id: "mf-ceiling", results: [report("c.gguf", { ceiling: bad })] }, workerAuth)).status).toBe(400);
    }
  });

  it("a failed check is kept with its error, has no level, and is asked again", async () => {
    const w = await heartbeat("mf-err", CAPS);
    const failed = report("a.gguf", { point: null, error: "401 (gated or private repo)", source: null });
    expect((await post("/api/worker/model-fit", { machine_id: "mf-err", results: [failed] }, workerAuth)).status).toBe(200);
    const list = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit`));
    expect(list.results[0]).toMatchObject({ level: null, error: "401 (gated or private repo)" });
    expect(await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")] }))).toMatchObject({ queued: 1 });
    const empty = report("b.gguf", { point: null, error: null });
    expect((await post("/api/worker/model-fit", { machine_id: "mf-err", results: [empty] }, workerAuth)).status).toBe(400);
    const badKey = report("c.gguf", { target_key: "../evil" });
    expect((await post("/api/worker/model-fit", { machine_id: "mf-err", results: [badKey] }, workerAuth)).status).toBe(400);
  });

  it("with no usable build: refuses without consent, then installs, activates and checks in order", async () => {
    const w = await heartbeat("mf-none", CAPS, []);
    const plan = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit/plan`));
    expect(plan.build).toMatchObject({ tag: "b12000", installed: false, backend: "cpu", size_bytes: 40_000_000 });

    expect((await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")] })).status).toBe(409);
    expect(jobsOf(w.id)).toHaveLength(0);

    const res = await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")], auto_install: true }));
    expect(res).toMatchObject({ queued: 1, installing: "b12000" });
    const order = getDb().prepare(`SELECT job_type FROM worker_jobs WHERE worker_id = ? ORDER BY rowid`).all(w.id) as { job_type: string }[];
    expect(order.map((j) => j.job_type)).toEqual(["install_build", "activate_build", "fit_check"]);
    expect(JSON.parse(jobsOf(w.id)[0].payload_json).llama_cpp_build).toBe("b12000");

    await post(`/api/workers/${w.id}/model-fit`, { targets: [target("b.gguf")], auto_install: true });
    expect(jobsOf(w.id, "install_build")).toHaveLength(1);
    expect((await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit`))).installing).toBe("b12000");
  });

  it("a build without fit-params is passed over, but an active one is never replaced", async () => {
    const w = await heartbeat("mf-old-build", CAPS, [
      { tag: "b5000", asset_name: "llama-b5000-bin-ubuntu-x64.zip", installed_at: 1, active: true },
    ]);
    await post(`/api/workers/${w.id}/model-fit`, { targets: [target("a.gguf")], auto_install: true });
    const order = getDb().prepare(`SELECT job_type FROM worker_jobs WHERE worker_id = ? ORDER BY rowid`).all(w.id) as { job_type: string }[];
    expect(order.map((j) => j.job_type)).toEqual(["install_build", "fit_check"]);
  });

  it("prefers the active build, whatever its backend", async () => {
    const w = await heartbeat("mf-active", CAPS, [
      { tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip", installed_at: 1, active: false, fit_params_path: "/x/fit" },
      { tag: "b9000", asset_name: "llama-b9000-bin-ubuntu-vulkan-x64.zip", installed_at: 2, active: true, fit_params_path: "/y/fit" },
    ]);
    const plan = await json(await fetch(`${baseUrl}/api/workers/${w.id}/model-fit/plan`));
    expect(plan.build).toMatchObject({ tag: "b9000", backend: "vulkan", installed: true });
  });

  it("a catalogued model is sent as its own row, under the same key as its Hugging Face file", async () => {
    const w = await heartbeat("mf-model", CAPS);
    expect(await json(await post(`/api/workers/${w.id}/model-fit`, { targets: [{ model_id: "opt-model" }] }))).toMatchObject({ queued: 1 });
    const payload = JSON.parse(jobsOf(w.id)[0].payload_json);
    expect(payload.models[0].id).toBe("opt-model");
    expect(payload.target_keys).toHaveLength(1);
    expect(payload.target_keys[0]).toMatch(/^(hf|model):/);
  });

  it("caps pending jobs per machine", async () => {
    const w = await heartbeat("mf-cap", CAPS);
    for (let i = 0; i < 8; i++) {
      expect((await post(`/api/workers/${w.id}/model-fit`, { targets: [target(`f${i}.gguf`)] })).status).toBe(202);
    }
    expect((await post(`/api/workers/${w.id}/model-fit`, { targets: [target("over.gguf")] })).status).toBe(409);
  });
});
