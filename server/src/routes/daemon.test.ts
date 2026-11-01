import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-daemon-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "route-test-secret";

let sweepLiveState: typeof import("./daemon.js")["sweepLiveState"];
let liveStateSizes: typeof import("./daemon.js")["liveStateSizes"];
let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let daemonRepo: typeof import("../db/daemonRepo.js")["daemonRepo"];
let getDb: typeof import("../db/migrate.js")["getDb"];

const MACHINE = "serve-machine";

function workerState(capabilities = ["daemon-v1"]) {
  return {
    machine_id: MACHINE,
    hostname: MACHINE,
    backend: "cpu",
    capabilities,
    hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
    installed_builds: [],
    model_files: [],
    status: "idle",
  };
}

async function send(method: string, path: string, body?: unknown, token?: string) {
  return fetch(`${baseUrl}${path}`, {
    method,
    headers: { ...(body !== undefined ? { "content-type": "application/json" } : {}), ...(token ? { authorization: `Bearer ${token}` } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

function telemetry(extra: Record<string, unknown> = {}) {
  return send("POST", "/api/worker/daemon/telemetry", { machine_id: MACHINE, hostname: MACHINE, config_version: 0, ...extra }, "route-test-secret");
}

const status = (over: Record<string, unknown> = {}) => ({
  phase: "ready",
  message: null,
  model_id: "model-1",
  model_name: "a.gguf",
  alias: "a",
  build_tag: "b1",
  load_hash: "hash-1",
  settings: { ctx_size: 4096 },
  actual: { gpu_layers: 10, total_layers: 10, n_ctx: 4096, n_slots: 4 },
  loaded_at: 1,
  in_flight: 0,
  requests_total: 0,
  generation_defaults: { temperature: 0.8 },
  gateway: { listening: true, host: "127.0.0.1", port: 8642, error: null },
  interfaces: [],
  active_build_tag: "b1",
  supported_settings: ["ctx_size"],
  restarts: 0,
  usage_dropped: 0,
  autostart_model_id: null,
  memory: null,
  ...over,
});

const usageRow = (over: Record<string, unknown> = {}) => ({
  id: `row-${Math.random()}`,
  at: Date.now(),
  model_id: "model-1",
  load_hash: "hash-1",
  endpoint: "chat",
  stream: false,
  status: 200,
  cancelled: false,
  processed_tokens: 500,
  cached_tokens: 100,
  generated_tokens: 50,
  prompt_ms: 500,
  predicted_ms: 500,
  total_ms: 1010,
  concurrency: 1,
  watched: false,
  ...over,
});

let workerId: string;

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ daemonRepo } = await import("../db/daemonRepo.js"));
  ({ getDb } = await import("../db/migrate.js"));
  const { queueRoutes } = await import("./queue.js");
  const { daemonRoutes } = await import("./daemon.js");
  ({ sweepLiveState, liveStateSizes } = await import("./daemon.js"));
  const { appErrorHandler } = await import("../error-handler.js");

  app = Fastify({ logger: false });
  app.setErrorHandler(appErrorHandler);
  await app.register(queueRoutes);
  await app.register(daemonRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const addr = app.server.address();
  baseUrl = `http://127.0.0.1:${typeof addr === "object" && addr ? addr.port : 0}`;

  // A first heartbeat creates the machine, same as a real worker.
  const res = await send("POST", "/api/worker/heartbeat", workerState(), "route-test-secret");
  workerId = ((await res.json()) as { worker_id: string }).worker_id;

  repo.registerModel({ id: "model-1", source: "huggingface", hf_repo: "org/repo", hf_file: "a.gguf", filename: "a.gguf", size_bytes: 1, metadata: {} });
});

afterAll(async () => {
  await app.close();
  try {
    rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    // The database file is still open on Windows; same as the other route tests.
  }
});

describe("telemetry", () => {
  it("opens no port on a machine that never used the page", async () => {
    const body = (await (await telemetry({ status: status({ phase: "idle", model_id: null, load_hash: null, settings: null, actual: null }) })).json()) as { config?: unknown };
    expect(body.config).toBeUndefined();
  });

  it("needs a worker's token", async () => {
    const res = await send("POST", "/api/worker/daemon/telemetry", { machine_id: MACHINE }, "wrong");
    expect(res.status).toBe(401);
  });

  it("says watched only while someone has the page open", async () => {
    expect(((await (await telemetry()).json()) as { watched: boolean }).watched).toBe(false);
    await send("GET", `/api/workers/${workerId}/daemon`);
    expect(((await (await telemetry()).json()) as { watched: boolean }).watched).toBe(true);
  });

  it("drops a malformed status instead of storing it", async () => {
    await telemetry({ status: status({ phase: "hacked" }) });
    const view = (await (await send("GET", `/api/workers/${workerId}/daemon`)).json()) as { status: { phase: string } | null };
    expect(view.status?.phase).not.toBe("hacked");
  });

  it("stores usage idempotently and acknowledges everything, including bad rows", async () => {
    const row = usageRow({ id: "dup" });
    const first = (await (await telemetry({ usage: [row, { id: "bad", at: "soon" }] })).json()) as { acked: string[] };
    expect(first.acked.sort()).toEqual(["bad", "dup"]);
    await telemetry({ usage: [row] });
    const count = getDb().prepare(`SELECT COUNT(*) AS n FROM daemon_usage WHERE id = 'dup'`).get() as { n: number };
    expect(count.n).toBe(1);
    expect((getDb().prepare(`SELECT COUNT(*) AS n FROM daemon_usage WHERE id = 'bad'`).get() as { n: number }).n).toBe(0);
  });

  it("has no column that could hold text from a request", () => {
    const columns = (getDb().prepare(`PRAGMA table_info(daemon_usage)`).all() as { name: string; type: string }[]);
    const text = columns.filter((c) => c.type === "TEXT").map((c) => c.name).sort();
    expect(text).toEqual(["endpoint", "id", "load_hash", "model_id", "worker_id"]);
  });
});

describe("load and unload", () => {
  it("refuses bad settings without queueing anything", async () => {
    const before = daemonRepo.latestJob(workerId);
    for (const body of [
      { model_id: "model-1", settings: { tools: "all" } },
      { model_id: "model-1", settings: { ctx_size: -5 } },
      { model_id: "model-1", alias: "--tools" },
      { model_id: "model-1", mmproj_filename: "../../etc/passwd" },
      { model_id: "model-1", draft_model_id: "nope" },
      { model_id: "model-1", sampling: { values: { temperature: 99 } } },
    ]) {
      expect((await send("POST", `/api/workers/${workerId}/daemon/load`, body)).status).toBe(400);
    }
    expect((await send("POST", `/api/workers/${workerId}/daemon/load`, { model_id: "missing" })).status).toBe(404);
    expect(daemonRepo.latestJob(workerId)).toEqual(before);
  });

  it("queues a load with a default identifier, a config, and remembers the settings", async () => {
    const res = await send("POST", `/api/workers/${workerId}/daemon/load`, { model_id: "model-1", settings: { ctx_size: 8192 }, sampling: { values: { temperature: 0.4 } } });
    expect(res.status).toBe(202);
    const claimed = await send("POST", "/api/worker/queue", workerState(), "route-test-secret");
    const job = (await claimed.json()) as { type: string; job_id: string; payload: { spec: { alias: string; settings: unknown }; config: { gateway: { host: string; port: number } } } };
    expect(job.type).toBe("daemon_load");
    expect(job.payload.spec.alias).toBe("a");
    expect(job.payload.spec.settings).toEqual({ ctx_size: 8192 });
    expect(job.payload.config.gateway).toMatchObject({ host: "127.0.0.1", port: 8642 });
    const saved = (await (await send("GET", `/api/workers/${workerId}/daemon/models/model-1/settings`)).json()) as { settings: { settings: unknown; sampling: unknown } };
    expect(saved.settings.settings).toEqual({ ctx_size: 8192 });
    expect(saved.settings.sampling).toEqual({ values: { temperature: 0.4 } });

    const view = (await (await send("GET", `/api/workers/${workerId}/daemon`)).json()) as { pending_job: { type: string; status: string } };
    expect(view.pending_job).toMatchObject({ type: "daemon_load", status: "claimed" });

    // A second click supersedes: the running load is told to stop.
    await send("POST", `/api/workers/${workerId}/daemon/unload`);
    const hb = (await (await send("POST", "/api/worker/heartbeat", { ...workerState(), active_job: { job_id: job.job_id, phase: "loading" } }, "route-test-secret")).json()) as { control: { cancel_job_ids: string[] } };
    expect(hb.control.cancel_job_ids).toContain(job.job_id);
    await send("POST", `/api/worker/jobs/${job.job_id}/complete`, { ok: false, error: "load cancelled", machine_id: MACHINE }, "route-test-secret");
  });

  it("is last-wins across pending jobs", async () => {
    await send("POST", `/api/workers/${workerId}/daemon/load`, { model_id: "model-1" });
    await send("POST", `/api/workers/${workerId}/daemon/load`, { model_id: "model-1" });
    await send("POST", `/api/workers/${workerId}/daemon/unload`);
    const pending = getDb().prepare(`SELECT job_type FROM worker_jobs WHERE worker_id = ? AND status = 'pending' AND job_type LIKE 'daemon_%'`).all(workerId) as { job_type: string }[];
    expect(pending.map((p) => p.job_type)).toEqual(["daemon_unload"]);
    const claimed = (await (await send("POST", "/api/worker/queue", workerState(), "route-test-secret")).json()) as { job_id: string };
    await send("POST", `/api/worker/jobs/${claimed.job_id}/complete`, { ok: true, machine_id: MACHINE }, "route-test-secret");
  });

  it("delivers config once the worker is behind, with hashes and live sampling", async () => {
    const created = (await (await send("POST", `/api/workers/${workerId}/daemon/keys`, { name: "agent" })).json()) as { key: string };
    await send("PUT", `/api/workers/${workerId}/daemon/sampling`, { model_id: "model-1", sampling: { values: { top_k: 11 } } });
    const body = (await (await telemetry({ status: status() })).json()) as { config: { version: number; key_hashes: string[]; sampling: { values: unknown }; sampling_model_id: string } };
    expect(body.config.key_hashes).toContain(createHash("sha256").update(created.key).digest("hex"));
    expect(JSON.stringify(body.config)).not.toContain(created.key);
    expect(body.config.sampling).toEqual({ values: { top_k: 11 } });
    expect(body.config.sampling_model_id).toBe("model-1");
    const again = (await (await telemetry({ status: status(), config_version: body.config.version, config_model_id: "model-1" })).json()) as { config?: unknown };
    expect(again.config).toBeUndefined();
  });

  it("resends config when the worker's is for another model, at the same version", async () => {
    const first = (await (await telemetry({ status: status() })).json()) as { config: { version: number } };
    const wrongModel = (await (await telemetry({ status: status(), config_version: first.config.version, config_model_id: "another-model" })).json()) as { config?: { sampling_model_id: string } };
    expect(wrongModel.config?.sampling_model_id).toBe("model-1");
  });
});

describe("audit 2026-11-01", () => {
  it("accepts the stopping phase and the fields a reload compares", async () => {
    await telemetry({ status: status({ phase: "stopping", mmproj_filename: "p.gguf", draft_model_id: "d", memory: { at: 1, ram: { total: 1, app: 1, other: 0, free: 0 }, gpu_count: 2, breakdown: {}, spill: {} } }) });
    const view = (await (await send("GET", `/api/workers/${workerId}/daemon`)).json()) as { status: { phase: string; mmproj_filename: string; draft_model_id: string; memory: { gpu_count: number } } };
    expect(view.status).toMatchObject({ phase: "stopping", mmproj_filename: "p.gguf", draft_model_id: "d" });
    expect(view.status.memory.gpu_count).toBe(2);
    await telemetry({ status: status() });
  });

  it("writes a load only when it changes, not on every report", async () => {
    const spy = vi.spyOn(daemonRepo, "recordLoad");
    const report = (actual: Record<string, number>) => telemetry({ status: status({ load_hash: "steady", actual }) });
    await report({ n_ctx: 1 });
    await report({ n_ctx: 1 });
    await report({ n_ctx: 1 });
    expect(spy).toHaveBeenCalledTimes(1);
    await report({ n_ctx: 2 });
    expect(spy).toHaveBeenCalledTimes(2);
    // Cleared history: the row is gone, so the next report has to write it again.
    getDb().prepare(`DELETE FROM daemon_loads WHERE worker_id = ? AND load_hash = 'steady'`).run(workerId);
    await report({ n_ctx: 2 });
    expect(spy).toHaveBeenCalledTimes(3);
    spy.mockRestore();
  });

  it("forgets live state nothing has refreshed in an hour", async () => {
    await telemetry({ status: status() });
    await send("GET", `/api/workers/${workerId}/daemon`);
    expect(liveStateSizes().live).toBeGreaterThan(0);
    expect(liveStateSizes().watched).toBeGreaterThan(0);
    sweepLiveState(Date.now());
    expect(liveStateSizes().live).toBeGreaterThan(0);
    sweepLiveState(Date.now() + 2 * 60 * 60 * 1000);
    expect(liveStateSizes()).toEqual({ live: 0, watched: 0, loads: 0 });
  });
});

describe("gateway settings and keys", () => {
  it("shows a key once, and never after", async () => {
    const created = (await (await send("POST", `/api/workers/${workerId}/daemon/keys`, { name: "shown once" })).json()) as { id: string; key: string; prefix: string };
    expect(created.key).toMatch(/^lt-/);
    const view = await (await send("GET", `/api/workers/${workerId}/daemon`)).text();
    expect(view).not.toContain(created.key);
    expect(view).toContain(created.prefix);
  });

  it("refuses to open up with no active key, and closes when the last one goes", async () => {
    const keys = daemonRepo.listKeys(workerId).filter((k) => !k.revoked_at);
    for (const k of keys) await send("DELETE", `/api/workers/${workerId}/daemon/keys/${k.id}`);
    const open = { host: "0.0.0.0", port: 8642, cors_origins: [], autostart: false };
    expect((await send("PUT", `/api/workers/${workerId}/daemon/gateway`, open)).status).toBe(400);

    const created = (await (await send("POST", `/api/workers/${workerId}/daemon/keys`, {})).json()) as { id: string };
    expect((await send("PUT", `/api/workers/${workerId}/daemon/gateway`, open)).status).toBe(200);
    const revoked = (await (await send("DELETE", `/api/workers/${workerId}/daemon/keys/${created.id}`)).json()) as { gateway: { host: string } };
    expect(revoked.gateway.host).toBe("127.0.0.1");
    expect(daemonRepo.activeKeyHashes(workerId)).toEqual([]);
  });

  it("404s an unknown key", async () => {
    expect((await send("DELETE", `/api/workers/${workerId}/daemon/keys/nope`)).status).toBe(404);
  });
});

describe("usage", () => {
  it("returns a profile, a curve and points, per load", async () => {
    await telemetry({
      status: status({ load_hash: "hash-2", settings: { ctx_size: 2048 } }),
      usage: [
        ...Array.from({ length: 25 }, () => usageRow({ load_hash: "hash-2" })),
        usageRow({ load_hash: "hash-2", concurrency: 3 }),
      ],
    });
    const res = await send("GET", `/api/workers/${workerId}/daemon/usage?model_id=model-1&load_hash=hash-2`);
    const body = (await res.json()) as { loads: { load_hash: string; requests: number }[]; profile: { requests: number; workload: unknown }; curve: unknown[]; points: unknown[]; concurrent_excluded: number };
    expect(body.profile.requests).toBe(26);
    expect(body.profile.workload).not.toBeNull();
    expect(body.points).toHaveLength(25);
    expect(body.concurrent_excluded).toBe(1);
    expect(body.curve.length).toBeGreaterThan(0);
    expect(body.loads.find((l) => l.load_hash === "hash-2")?.requests).toBe(26);

    const all = (await (await send("GET", `/api/workers/${workerId}/daemon/usage?model_id=model-1&load_hash=hash-2&include_concurrent=1`)).json()) as { points: unknown[]; concurrent_excluded: number };
    expect(all.points).toHaveLength(26);
    expect(all.concurrent_excluded).toBe(0);
  });

  it("can be cleared", async () => {
    await telemetry({ usage: [usageRow()] });
    const res = (await (await send("DELETE", `/api/workers/${workerId}/daemon/usage?model_id=model-1`)).json()) as { removed: number };
    expect(res.removed).toBeGreaterThan(0);
    const body = (await (await send("GET", `/api/workers/${workerId}/daemon/usage?model_id=model-1`)).json()) as { profile: { requests: number } };
    expect(body.profile.requests).toBe(0);
  });

  it("prunes by age and cascades with the machine", async () => {
    await telemetry({ usage: [usageRow({ id: "old", at: Date.now() - 100 * 86_400_000 }), usageRow({ id: "new" })] });
    expect(daemonRepo.pruneUsage()).toBeGreaterThanOrEqual(1);
    const ids = (getDb().prepare(`SELECT id FROM daemon_usage WHERE worker_id = ?`).all(workerId) as { id: string }[]).map((r) => r.id);
    expect(ids).toContain("new");
    expect(ids).not.toContain("old");
    repo.workerRepo.deleteWorker(workerId);
    for (const table of ["daemon_usage", "daemon_config", "daemon_api_keys", "daemon_model_settings", "daemon_loads"]) {
      expect((getDb().prepare(`SELECT COUNT(*) AS n FROM ${table} WHERE worker_id = ?`).get(workerId) as { n: number }).n).toBe(0);
    }
    expect(existsSync(tmpDir)).toBe(true);
  });
});

// Audit 2026-11-01 E6: the load hash comes from the worker, and each new one used to
// add a row that nothing charged, capped or pruned.
describe("load history stays bounded", () => {
  const loadRows = (workerId: string) => getDb().prepare(`SELECT load_hash FROM daemon_loads WHERE worker_id = ?`).all(workerId) as { load_hash: string }[];
  const workerId = () => repo.workerRepo.getByMachineId(MACHINE)!.id;

  it("a new load waits out an exhausted budget, and the report still succeeds", async () => {
    const prev = process.env.INGEST_BYTES_PER_HOUR;
    process.env.INGEST_BYTES_PER_HOUR = "1";
    try {
      expect((await telemetry({ status: status({ load_hash: "e6-over-budget" }) })).status).toBe(200);
      expect(loadRows(workerId()).some((r) => r.load_hash === "e6-over-budget")).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.INGEST_BYTES_PER_HOUR;
      else process.env.INGEST_BYTES_PER_HOUR = prev;
    }
    expect((await telemetry({ status: status({ load_hash: "e6-over-budget" }) })).status).toBe(200);
    expect(loadRows(workerId()).some((r) => r.load_hash === "e6-over-budget")).toBe(true);
  });

  it("a load already on record keeps updating over budget", async () => {
    await telemetry({ status: status({ load_hash: "e6-known" }) });
    const prev = process.env.INGEST_BYTES_PER_HOUR;
    process.env.INGEST_BYTES_PER_HOUR = "1";
    try {
      await telemetry({ status: status({ load_hash: "e6-known", actual: { gpu_layers: 7, total_layers: 10, n_ctx: 4096, n_slots: 4 } }) });
    } finally {
      if (prev === undefined) delete process.env.INGEST_BYTES_PER_HOUR;
      else process.env.INGEST_BYTES_PER_HOUR = prev;
    }
    const row = getDb().prepare(`SELECT actual_json FROM daemon_loads WHERE worker_id = ? AND load_hash = 'e6-known'`).get(workerId()) as { actual_json: string };
    expect(JSON.parse(row.actual_json).gpu_layers).toBe(7);
  });

  it("keeps the newest loads per machine", async () => {
    const { MAX_LOADS_PER_WORKER } = await import("../db/daemonRepo.js");
    const id = repo.workerRepo.getOrCreateByMachineId("e6-cap", "e6-cap").id;
    const load = (hash: string) => ({ load_hash: hash, model_id: "m", build_tag: "b", settings: { ctx_size: 1 }, actual: null });
    for (let i = 0; i < MAX_LOADS_PER_WORKER + 20; i++) daemonRepo.recordLoad(id, load(`hash-${i}`));
    const hashes = loadRows(id).map((r) => r.load_hash);
    expect(hashes).toHaveLength(MAX_LOADS_PER_WORKER);
    expect(hashes).toContain(`hash-${MAX_LOADS_PER_WORKER + 19}`);
    expect(hashes).not.toContain("hash-0");
  });

  it("prunes old loads nothing refers to, and only those", () => {
    const id = repo.workerRepo.getOrCreateByMachineId("e6-prune", "e6-prune").id;
    const load = (hash: string) => ({ load_hash: hash, model_id: "m", build_tag: "b", settings: { ctx_size: 1 }, actual: null });
    for (const hash of ["old-unused", "old-used", "new-unused"]) daemonRepo.recordLoad(id, load(hash));
    const longAgo = Date.now() - 3650 * 86_400_000;
    getDb().prepare(`UPDATE daemon_loads SET first_seen = ? WHERE worker_id = ? AND load_hash LIKE 'old-%'`).run(longAgo, id);
    getDb()
      .prepare(
        `INSERT INTO daemon_usage (id, worker_id, model_id, load_hash, endpoint, stream, status, cancelled, total_ms, concurrency, watched, at)
         VALUES ('e6-usage', ?, 'm', 'old-used', 'chat', 0, 200, 0, 1, 1, 0, ?)`
      )
      .run(id, Date.now());
    daemonRepo.pruneUsage();
    expect(loadRows(id).map((r) => r.load_hash).sort()).toEqual(["new-unused", "old-used"]);
  });
});
