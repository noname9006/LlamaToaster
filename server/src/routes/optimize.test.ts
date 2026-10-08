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

async function heartbeat(machineId: string, capabilities = ALL_CAPS) {
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
      installed_builds: [
        { tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip", installed_at: 1, active: true, fit_params_path: "/x/llama-fit-params" },
      ],
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
