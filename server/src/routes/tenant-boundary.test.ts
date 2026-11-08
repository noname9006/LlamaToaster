// Audit 2026-11-01: what one tenant writes must not reach, or be usable by,
// another -- exports, community data, shared catalog rows, quotas -- and one
// tenant can't fill the disk. Every case here but the two that pin behaviour
// which must survive (an unknown cache type, a real community row) failed
// against the code before the audit's fixes.
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";
import fastifyRateLimit from "@fastify/rate-limit";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-tenant-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.LOG_DIR = join(tmpDir, "logs");
process.env.AUTH_ENABLED = "true";

let app: FastifyInstance;
let repo: (typeof import("../db/repo.js"))["repo"];
let getDb: (typeof import("../db/migrate.js"))["getDb"];
let closeDb: (typeof import("../db/migrate.js"))["closeDb"];
let formatResultsExport: (typeof import("./results.js"))["formatResultsExport"];
let loadExportRows: (typeof import("./results.js"))["loadExportRows"];
let stampConfigHash: (typeof import("../../../shared/exchange.js"))["stampConfigHash"];
let BUNDLE_FORMAT: string;
let BUNDLE_FORMAT_VERSION: number;

const sweep = {
  n_prompt: [512],
  n_gen: [128],
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
  repeats: 5,
};

const resultRow = (over: Record<string, unknown> = {}) => ({
  test_type: "tg",
  n_prompt: 512,
  n_gen: 128,
  n_depth: 0,
  n_threads: 8,
  n_gpu_layers: 99,
  batch_size: 2048,
  ubatch_size: 512,
  cache_type_k: "f16",
  cache_type_v: "f16",
  flash_attn: "on",
  mtp: "off",
  n_gpu_layers_draft: 0,
  n_cpu_moe: 0,
  avg_tps: 10,
  stddev_tps: 0,
  ram_peak_mib: 1,
  vram_peak_mib: 1,
  ram_avg_mib: 1,
  vram_avg_mib: 1,
  ram_free_before_mib: 1,
  vram_free_before_mib: 1,
  system_memory_total_mb: 1,
  gpu_memory_total_mb: 1,
  gpu_memory_total_accuracy: "exact",
  gpu_memory_total_source: null,
  gpu_memory_free_start_accuracy: "exact",
  gpu_memory_free_start_source: null,
  gpu_memory_model_avg_accuracy: "exact",
  gpu_memory_model_avg_source: null,
  gpu_memory_model_peak_accuracy: "exact",
  gpu_memory_model_peak_source: null,
  gpu_layers_loaded: 1,
  total_model_layers: 1,
  sample_count: 5,
  suspect_count: 0,
  method_version: 1,
  ...over,
});

interface Tenant {
  id: string;
  h: Record<string, string>;
  worker: { id: string; h: Record<string, string> };
}

let seq = 0;
function tenant(name: string): Tenant {
  const label = `${name}-${++seq}`;
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: label, login: label, avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label });
  repo.workerRepo.createPending({
    machineId: label,
    hostname: label,
    platform: "linux",
    arch: "x64",
    deviceCode: `dc-${label}`,
    userCode: `UC-${label}`,
    expiresAt: Date.now() + 1e6,
    hardware: undefined,
    publicKey: undefined,
  });
  const workerId = repo.workerRepo.getByMachineId(label)!.id;
  repo.workerRepo.approve(workerId, user.id);
  const workerSession = repo.sessionRepo.create(user.id, { isWorker: true, workerId, label });
  repo.workerRepo.recordHeartbeat(
    workerId,
    {
      machine_id: label,
      capabilities: ["probe-v1", "probe-frontier-v1", "quality-v1", "curve-v1", "fit-check-v1", "fit-check-v2"],
      hostname: label,
      backend: "vulkan",
      hardware: {
        platform: "linux",
        arch: "x64",
        cpu: { manufacturer: "x", brand: "x", flags: [], cores: 8 },
        gpu: [{ vendor: "AMD", model: `GPU ${label}`, vram_mb: 8192 }],
      },
      installed_builds: [{ tag: "b1", asset_name: "llama-b1-bin-ubuntu-vulkan-x64.tar.gz", installed_at: 1, active: true }],
      model_files: [],
      status: "idle",
    } as never,
    null
  );
  return {
    id: user.id,
    h: { authorization: `Bearer ${token}` },
    worker: { id: workerId, h: { authorization: `Bearer ${workerSession.token}` } },
  };
}

function modelFor(owner: Tenant | null, name: string): string {
  const id = `model-${name}-${++seq}`;
  repo.registerModel({ id, filename: `${name}.gguf`, size_bytes: 1, source: "local", metadata: {}, created_by: owner?.id });
  return id;
}

const trigger = (t: Tenant, modelId: string, extra: Record<string, unknown> = {}) =>
  app.inject({ method: "POST", url: "/api/tests/trigger", headers: t.h, payload: { model_id: modelId, worker_id: t.worker.id, sweep, ...extra } });

function importBundle(run: Record<string, unknown>, hardware: Record<string, unknown> = {}, rowOver: Record<string, unknown> = {}) {
  const row = stampConfigHash({
    test_type: "tg",
    n_prompt: 1,
    n_gen: 1,
    n_depth: 0,
    n_threads: 1,
    n_gpu_layers: 0,
    batch_size: 1,
    ubatch_size: 1,
    cache_type_k: "f16",
    cache_type_v: "f16",
    flash_attn: "on",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
    engine: null,
    concurrency: null,
    method_version: 1,
    avg_tps: 99999,
    stddev_tps: 0,
    sample_count: 5,
    suspect_count: 0,
    vram_peak_mib: 1,
    ram_peak_mib: 1,
    ttft_ms_p50: null,
    ttft_ms_p95: null,
    ttft_n: null,
    e2e_ms_mean: null,
    gpu_temp_c_max: null,
    gpu_clock_mhz_min: null,
    created_at: 1,
    ...rowOver,
  } as never);
  return {
    bundle: {
      format: BUNDLE_FORMAT,
      format_version: BUNDLE_FORMAT_VERSION,
      run: { id: "x", kind: "sweep", model_filename: "plain.gguf", model_quant: null, model_sha256: null, llama_cpp_build: "b1", config: {}, ...run },
      hardware_class: { gpu_name: "Some GPU", backend: "vulkan", vram_class_mib: 8192, cpu_isa: null, ...hardware },
      rows: [row],
      quality_rows: [],
      method_versions: [1],
      methods: [],
    },
  };
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ getDb, closeDb } = await import("../db/migrate.js"));
  ({ formatResultsExport, loadExportRows } = await import("./results.js"));
  const exchange = await import("../../../shared/exchange.js");
  stampConfigHash = exchange.stampConfigHash;
  BUNDLE_FORMAT = exchange.BUNDLE_FORMAT;
  BUNDLE_FORMAT_VERSION = exchange.BUNDLE_FORMAT_VERSION;

  const { testsRoutes } = await import("./tests.js");
  const { profilesRoutes } = await import("./profiles.js");
  const { measurementRoutes } = await import("./measurements.js");
  const { optimizeRoutes } = await import("./optimize.js");
  const { curveRoutes } = await import("./curves.js");
  const { comparisonRoutes } = await import("./comparisons.js");
  const { exchangeRoutes } = await import("./exchange.js");
  const { modelsRoutes } = await import("./models.js");
  const { sessionRoutes } = await import("./sessions.js");
  const { authMiddleware } = await import("../auth-middleware.js");
  const { appErrorHandler } = await import("../error-handler.js");

  app = Fastify();
  await app.register(fastifyCookie);
  await app.register(fastifyRateLimit, { global: false });
  app.addHook("preHandler", authMiddleware);
  for (const plugin of [testsRoutes, profilesRoutes, measurementRoutes, optimizeRoutes, curveRoutes, comparisonRoutes, exchangeRoutes, modelsRoutes, sessionRoutes]) {
    await app.register(plugin);
  }
  app.setErrorHandler(appErrorHandler);
  await app.ready();
});

afterAll(async () => {
  await app.close();
  closeDb();
});

describe("exports don't carry another tenant's formulas (B3)", () => {
  it("refuses formula-shaped result labels at ingestion", async () => {
    const t = tenant("csv-ingest");
    const run = (await trigger(t, modelFor(t, "a"))).json().run;
    for (const bad of ['=HYPERLINK("http://evil.example","open")', "+1+1", "@SUM(1)", "-1+1"]) {
      const res = await app.inject({
        method: "POST",
        url: `/api/tests/${run.id}/items/0`,
        headers: t.worker.h,
        payload: { status: "done", results: [resultRow({ cache_type_k: bad })] },
      });
      expect(res.statusCode).toBe(400);
    }
    expect(repo.getResultsForTest(run.id)).toHaveLength(0);
  });

  it("keeps a cache type this server has never heard of", async () => {
    const t = tenant("csv-newer");
    const run = (await trigger(t, modelFor(t, "a"))).json().run;
    const res = await app.inject({
      method: "POST",
      url: `/api/tests/${run.id}/items/0`,
      headers: t.worker.h,
      payload: { status: "done", results: [resultRow({ cache_type_k: "q3_future" })] },
    });
    expect(res.statusCode).toBe(200);
  });

  it("neutralizes every string cell for rows already stored, and leaves numbers alone", () => {
    const t = tenant("csv-old");
    const modelId = modelFor(t, "a");
    const runId = `old-run-${++seq}`;
    const db = getDb();
    db.prepare(
      `INSERT INTO runs (id, root_run_id, user_id, worker_name, llama_cpp_backend, backend_device_name, model_id, config, status, started_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, '{}', 'done', 1)`
    ).run(runId, runId, t.id, "=worker()", "=cmd|'/c calc'!A1", " =device()", modelId);
    db.prepare(
      `INSERT INTO results (id, run_id, user_id, idx, model_id, test_type, n_prompt, n_gen, n_threads, n_gpu_layers, batch_size, ubatch_size,
         cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft, n_cpu_moe, avg_tps, stddev_tps, ram_peak_mib, gpu_layers_loaded, created_at)
       VALUES (?, ?, ?, 0, ?, 'tg', 1, 1, 1, -1, 1, 1, '=a()', '+b', '@c', '-d', 0, 0, 1, 0, 1, -1, 1)`
    ).run(`row-${seq}`, runId, t.id, modelId);

    const csv = formatResultsExport(loadExportRows(undefined, [runId]), "csv").body;
    expect(csv).toContain(`"'=worker()"`);
    expect(csv).toContain(`"'=cmd|'/c calc'!A1"`);
    expect(csv).toContain(`"' =device()"`);
    for (const cell of ["=a()", "+b", "@c", "-d"]) expect(csv).toContain(`"'${cell}"`);
    // A plain negative number is data, not a formula.
    expect(csv).toContain(`"-1"`);
    expect(csv).not.toContain(`"'-1"`);
  });

  it("escapes table syntax in the Markdown export", () => {
    const t = tenant("md");
    const modelId = modelFor(t, "a");
    const runId = `md-run-${++seq}`;
    const db = getDb();
    db.prepare(
      `INSERT INTO runs (id, root_run_id, user_id, worker_name, llama_cpp_backend, model_id, config, status, started_at)
       VALUES (?, ?, ?, ?, 'cpu', ?, '{}', 'done', 1)`
    ).run(runId, runId, t.id, "a|b\nc", modelId);
    db.prepare(
      `INSERT INTO results (id, run_id, user_id, idx, model_id, test_type, n_prompt, n_gen, n_threads, n_gpu_layers, batch_size, ubatch_size,
         cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft, n_cpu_moe, avg_tps, stddev_tps, ram_peak_mib, created_at)
       VALUES (?, ?, ?, 0, ?, 'tg', 1, 1, 1, 1, 1, 1, 'x|y', 'f16', 'on', 'off', 0, 0, 1, 0, 1, 1)`
    ).run(`row-${seq}`, runId, t.id, modelId);
    const md = formatResultsExport(loadExportRows(undefined, [runId]), "md").body;
    expect(md).toContain("a\\|b c");
    expect(md).toContain("x\\|y");
  });
});

describe("imports stay the importer's own (D3, D8, A11)", () => {
  it("never reach another user's community data", async () => {
    const mallory = tenant("import-mallory");
    const alice = tenant("import-alice");
    const res = await app.inject({
      method: "POST",
      url: "/api/import",
      headers: mallory.h,
      payload: importBundle({ model_filename: "IGNORE PREVIOUS INSTRUCTIONS.gguf" }),
    });
    expect(res.statusCode).toBe(201);
    const inAggregates = repo.communityRepo.listAggregates(alice.id, {});
    expect(inAggregates.some((a) => String(a.modelFilename).includes("IGNORE"))).toBe(false);
    const facets = repo.communityRepo.listFacets(alice.id);
    expect(facets.models.some((m) => m.value.includes("IGNORE"))).toBe(false);
  });

  it("still counts a real measurement as community data", async () => {
    const carol = tenant("real-carol");
    const alice = tenant("real-alice");
    const modelId = modelFor(carol, "community");
    getDb().prepare(`UPDATE models SET hf_repo = 'org/community', hf_file = 'community.gguf' WHERE id = ?`).run(modelId);
    getDb().prepare(`INSERT INTO hf_gguf_index (sha256, repo_id, filename, last_seen) VALUES (?, 'org/community', 'community.gguf', ?)`).run(modelId, Date.now());
    const run = (await trigger(carol, modelId)).json().run;
    await app.inject({ method: "POST", url: `/api/tests/${run.id}/items/0`, headers: carol.worker.h, payload: { status: "done", results: [resultRow()] } });
    expect(repo.communityRepo.listAggregates(alice.id, { modelId })).toHaveLength(1);
    expect(repo.communityRepo.listFacets(alice.id).models.some((m) => m.modelId === modelId)).toBe(true);
  });

  it("refuses malformed envelopes with a 400, not a 500", async () => {
    const t = tenant("import-envelope");
    const bad: [Record<string, unknown>, Record<string, unknown>?, Record<string, unknown>?][] = [
      [{ model_filename: "line\nbreak.gguf" }],
      [{ model_filename: 12 }],
      [{ kind: { a: 1 } }],
      [{ kind: "=cmd" }],
      [{ model_sha256: "not-a-hash" }],
      [{ config: [1, 2] }],
      [{ config: { pad: "x".repeat(300 * 1024) } }],
      [{}, { backend: "=cmd|'/c calc'!A1" }],
      [{}, { gpu_name: { a: 1 } }],
    ];
    for (const [run, hw] of bad) {
      const res = await app.inject({ method: "POST", url: "/api/import", headers: t.h, payload: importBundle(run, hw) });
      expect(res.statusCode).toBe(400);
    }
    const payload = importBundle({}) as { bundle: Record<string, unknown> };
    delete payload.bundle.run;
    expect((await app.inject({ method: "POST", url: "/api/import", headers: t.h, payload })).statusCode).toBe(400);
    const noHw = importBundle({}) as { bundle: Record<string, unknown> };
    delete noHw.bundle.hardware_class;
    expect((await app.inject({ method: "POST", url: "/api/import", headers: t.h, payload: noHw })).statusCode).toBe(400);
  });

  it("drops a row whose labels aren't labels, keeping the rest of the bundle's rules", async () => {
    const t = tenant("import-row");
    const res = await app.inject({ method: "POST", url: "/api/import", headers: t.h, payload: importBundle({}, {}, { cache_type_k: "=2+2" }) });
    expect(res.statusCode).toBe(400);
  });

  it("naming a hash doesn't attach to a row the importer can't see", async () => {
    const alice = tenant("import-link-alice");
    const mallory = tenant("import-link-mallory");
    const secret = "f".repeat(63) + "1";
    repo.registerModel({ id: secret, filename: "secret-private.gguf", size_bytes: 1, source: "local", metadata: {}, created_by: alice.id });
    const res = await app.inject({ method: "POST", url: "/api/import", headers: mallory.h, payload: importBundle({ model_sha256: secret }) });
    expect(res.statusCode).toBe(201);
    expect(repo.listModelsVisibleTo(mallory.id).some((m) => m.id === secret)).toBe(false);
    // A row they can see still links.
    const mine = "e".repeat(63) + "1";
    repo.registerModel({ id: mine, filename: "mine.gguf", size_bytes: 1, source: "local", metadata: {}, created_by: mallory.id });
    const linked = await app.inject({ method: "POST", url: "/api/import", headers: mallory.h, payload: importBundle({ model_sha256: mine }) });
    expect(repo.getResultsForTest(linked.json().run_id)[0].model_id).toBe(mine);
  });
});

describe("quality rows and comparisons don't cross tenants (N8, A5, N15)", () => {
  const dataset = `sha256:${"0".repeat(64)}`;
  const quality = { ctx_tokens: 2048, kv_pair: ["f16", "f16"], dataset_hash: dataset };
  const report = { ppl: 6.1, dataset_hash: dataset, ctx_tokens: 2048, cache_type_k: "f16", cache_type_v: "f16", method_version: 1 };

  it("shows each owner only their own quality rows", async () => {
    const alice = tenant("q-alice");
    const mallory = tenant("q-mallory");
    const modelId = modelFor(null, "shared-quality");
    repo.addModelRegistration(alice.id, modelId);
    repo.addModelRegistration(mallory.id, modelId);
    const aRun = (await trigger(alice, modelId, { kind: "quality", quality })).json().run;
    expect((await app.inject({ method: "POST", url: `/api/tests/${aRun.id}/quality-result`, headers: alice.worker.h, payload: report })).statusCode).toBe(200);
    expect(repo.qualityRepo.listForModel(modelId, alice.id)).toHaveLength(1);
    expect(repo.qualityRepo.listForModel(modelId, mallory.id)).toHaveLength(0);
    expect(repo.qualityRepo.listForModel(modelId, undefined)).toHaveLength(1);

    const mRun = (await trigger(mallory, modelId)).json().run;
    const profiles = await app.inject({ method: "GET", url: `/api/tests/${mRun.id}/profiles`, headers: mallory.h });
    expect(profiles.statusCode).toBe(200);
    expect(profiles.json().quality_results).toEqual([]);
  });

  it("refuses a quality report that doesn't match what the run asked for", async () => {
    const t = tenant("q-label");
    const run = (await trigger(t, modelFor(t, "q"), { kind: "quality", quality })).json().run;
    for (const over of [{ cache_type_k: "q8_0" }, { ctx_tokens: 4096 }, { dataset_hash: `sha256:${"1".repeat(64)}` }]) {
      const res = await app.inject({ method: "POST", url: `/api/tests/${run.id}/quality-result`, headers: t.worker.h, payload: { ...report, ...over } });
      expect(res.statusCode).toBe(400);
    }
  });

  it("another tenant's comparison id doesn't lift the active-run limit", () => {
    const alice = tenant("cmp-alice");
    const mallory = tenant("cmp-mallory");
    const runId = `cmp-run-${++seq}`;
    getDb()
      .prepare(`INSERT INTO runs (id, root_run_id, user_id, comparison_id, model_id, config, status, started_at) VALUES (?, ?, ?, 'cmp-shared', 'm', '{}', 'running', 1)`)
      .run(runId, runId, alice.id);
    expect(repo.hasActiveComparisonMember("cmp-shared", alice.id)).toBe(true);
    expect(repo.hasActiveComparisonMember("cmp-shared", mallory.id)).toBe(false);
    expect(repo.hasActiveComparisonMember("cmp-shared", undefined)).toBe(true);
  });
});

describe("a shared catalog row carries no adder's claims (D4)", () => {
  const body = { source: "huggingface", hf_repo: "bartowski/Popular-GGUF", hf_file: "Popular-Q4.gguf" };

  it("what the first adder typed isn't what the second is served", async () => {
    const mallory = tenant("squat-mallory");
    const alice = tenant("squat-alice");
    const poisoned = { ...body, filename: "attacker.gguf", size_bytes: 1, metadata: { n_layer: 2, trained_ctx: 512, mtp_layers: 1 } };
    const first = await app.inject({ method: "POST", url: "/api/models", headers: mallory.h, payload: poisoned });
    expect(first.statusCode).toBe(201);
    // The adder keeps seeing what they typed, on their own list.
    expect(first.json().model.metadata.n_layer).toBe(2);
    const modelId = first.json().model.id as string;
    expect(repo.listModelsVisibleTo(mallory.id).find((m) => m.id === modelId)?.metadata.n_layer).toBe(2);

    const second = await app.inject({ method: "POST", url: "/api/models", headers: alice.h, payload: body });
    expect(second.statusCode).toBe(200);
    const seen = second.json().model;
    expect(seen.filename).toBe("Popular-Q4.gguf");
    expect(seen.metadata).toEqual({});
    expect(repo.listModelsVisibleTo(alice.id).find((m) => m.id === modelId)?.metadata).toEqual({});
    expect(repo.getModelForWorker(modelId, alice.worker.id)?.metadata).toEqual({});
  });

  it("refuses repo and file names that aren't one", async () => {
    const t = tenant("squat-shape");
    for (const over of [{ hf_repo: "../../spaces/x" }, { hf_repo: "a/b?x=1" }, { hf_file: "../evil.gguf" }, { hf_file: "x.exe" }, { hf_file: "a\\b.gguf" }]) {
      const res = await app.inject({ method: "POST", url: "/api/models", headers: t.h, payload: { ...body, ...over } });
      expect(res.statusCode).toBe(400);
    }
  });

  it("caps client metadata", async () => {
    const t = tenant("squat-big");
    const res = await app.inject({ method: "POST", url: "/api/models", headers: t.h, payload: { ...body, metadata: { pad: "x".repeat(70_000) } } });
    expect(res.statusCode).toBe(400);
  });
});

describe("a model id only works for someone who can see it (A11)", () => {
  it("answers exactly like an unknown id on every route that takes one", async () => {
    const alice = tenant("vis-alice");
    const mallory = tenant("vis-mallory");
    const secret = modelFor(alice, "secret-private");

    const hidden = await trigger(mallory, secret);
    const unknown = await trigger(mallory, "no-such-model");
    expect(hidden.statusCode).toBe(unknown.statusCode);
    expect(hidden.json()).toEqual(unknown.json());

    const fit = (id: string) =>
      app.inject({ method: "POST", url: `/api/workers/${mallory.worker.id}/fit-check`, headers: mallory.h, payload: { ctx: 4096, model_ids: [id] } });
    expect((await fit(secret)).statusCode).toBe((await fit("no-such-model")).statusCode);

    const modelFit = (id: string) =>
      app.inject({ method: "POST", url: `/api/workers/${mallory.worker.id}/model-fit`, headers: mallory.h, payload: { targets: [{ model_id: id }] } });
    expect((await modelFit(secret)).statusCode).toBe(400);

    for (const path of [
      `/api/models/${secret}/curve`,
      `/api/models/${secret}/rates`,
      `/api/models/${secret}/verified-limits?worker=${mallory.worker.id}`,
    ]) {
      expect((await app.inject({ method: "GET", url: path, headers: mallory.h })).statusCode).toBe(404);
    }
    expect((await app.inject({ method: "DELETE", url: `/api/models/${secret}`, headers: mallory.h })).statusCode).toBe(404);
    expect(repo.getModel(secret)).toBeDefined();
  });

  it("lets the owner use their own, by every path a model becomes theirs", async () => {
    const t = tenant("vis-own");
    expect(repo.isModelVisibleTo(t.id, modelFor(t, "created"))).toBe(true);
    const registered = modelFor(null, "registered");
    expect(repo.isModelVisibleTo(t.id, registered)).toBe(false);
    repo.addModelRegistration(t.id, registered);
    expect(repo.isModelVisibleTo(t.id, registered)).toBe(true);

    const sha = "d".repeat(64);
    repo.registerModel({ id: sha, filename: "x.gguf", size_bytes: 1, source: "local", metadata: {} });
    expect(repo.isModelVisibleTo(t.id, sha)).toBe(false);
    getDb().prepare(`UPDATE workers SET model_files_json = ? WHERE id = ?`).run(JSON.stringify([{ path: "x.gguf", size_bytes: 1, sha256: sha }]), t.worker.id);
    expect(repo.isModelVisibleTo(t.id, sha)).toBe(true);
    // The list and the single check are the same rule.
    expect(repo.listModelsVisibleTo(t.id).some((m) => m.id === sha)).toBe(true);
    repo.recordHeaderReport(sha, t.id, { n_layer: 7 });
    expect(repo.getModelVisibleTo(t.id, sha)?.metadata.n_layer).toBe(7);
    expect(repo.getModelVisibleTo(tenant("vis-stranger").id, sha)).toBeUndefined();
    expect(repo.isModelVisibleTo(undefined, "anything")).toBe(true);
  });
});

describe("one account can't fill the disk (D1)", () => {
  it("charges reports against an hourly budget and refuses past it", async () => {
    const t = tenant("budget");
    const run = (await trigger(t, modelFor(t, "b"))).json().run;
    const prev = process.env.INGEST_BYTES_PER_HOUR;
    process.env.INGEST_BYTES_PER_HOUR = "1000";
    try {
      const big = await app.inject({
        method: "POST",
        url: `/api/tests/${run.id}/items/0`,
        headers: t.worker.h,
        payload: { status: "done", results: [resultRow()] },
      });
      expect(big.statusCode).toBe(429);
      expect(repo.getResultsForTest(run.id)).toHaveLength(0);
      const log = await app.inject({
        method: "POST",
        url: `/api/tests/${run.id}/log`,
        headers: { ...t.worker.h, "content-type": "application/octet-stream" },
        payload: Buffer.alloc(2000),
      });
      expect(log.statusCode).toBe(429);
      expect(existsSync(join(process.env.LOG_DIR!, `run-${run.id}.log.gz`))).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.INGEST_BYTES_PER_HOUR;
      else process.env.INGEST_BYTES_PER_HOUR = prev;
    }
  });

  it("a trigger's item rows count too, and nothing is created past it", async () => {
    const t = tenant("budget-trigger");
    const modelId = modelFor(t, "b");
    const prev = process.env.INGEST_BYTES_PER_HOUR;
    process.env.INGEST_BYTES_PER_HOUR = "100";
    try {
      const res = await trigger(t, modelId);
      expect(res.statusCode).toBe(429);
      expect(repo.listTests(t.id)).toHaveLength(0);
    } finally {
      if (prev === undefined) delete process.env.INGEST_BYTES_PER_HOUR;
      else process.env.INGEST_BYTES_PER_HOUR = prev;
    }
  });

  it("one owner's spend doesn't touch another's", () => {
    expect(repo.takeIngestBudget("budget-a", 900, 1000)).toBe(true);
    expect(repo.takeIngestBudget("budget-a", 200, 1000)).toBe(false);
    expect(repo.takeIngestBudget("budget-a", 100, 1000)).toBe(true);
    expect(repo.takeIngestBudget("budget-b", 1000, 1000)).toBe(true);
  });

  it("stores bounded free text however long the report's is", async () => {
    const t = tenant("budget-text");
    const run = (await trigger(t, modelFor(t, "b"))).json().run;
    const res = await app.inject({
      method: "POST",
      url: `/api/tests/${run.id}/items/0`,
      headers: t.worker.h,
      payload: { status: "failed", error: "E".repeat(900_000) },
    });
    expect(res.statusCode).toBe(200);
    expect(repo.getTestItems(run.id)[0].error?.length).toBeLessThanOrEqual(8_000);
  });
});

describe("run logs leave with their account (M3)", () => {
  it("deleting an account removes its run logs", async () => {
    const t = tenant("logs");
    const run = (await trigger(t, modelFor(t, "l"))).json().run;
    const posted = await app.inject({
      method: "POST",
      url: `/api/tests/${run.id}/log`,
      headers: { ...t.worker.h, "content-type": "application/octet-stream" },
      payload: Buffer.from("gz"),
    });
    expect(posted.statusCode).toBe(200);
    const path = join(process.env.LOG_DIR!, `run-${run.id}.log.gz`);
    expect(existsSync(path)).toBe(true);

    const del = await app.inject({ method: "DELETE", url: "/api/auth/account", headers: t.h, payload: { confirm: true } });
    expect(del.statusCode).toBe(200);
    expect(existsSync(path)).toBe(false);
  });

  it("prunes by age, honours the setting, and ignores other files", async () => {
    const { pruneRunLogs } = await import("../run-logs.js");
    const dir = process.env.LOG_DIR!;
    mkdirSync(dir, { recursive: true });
    const old = join(dir, "run-old.log.gz");
    const fresh = join(dir, "run-fresh.log.gz");
    const other = join(dir, "notes.txt");
    for (const p of [old, fresh, other]) writeFileSync(p, "x");
    const longAgo = new Date(Date.now() - 200 * 24 * 3600 * 1000);
    utimesSync(old, longAgo, longAgo);
    utimesSync(other, longAgo, longAgo);

    process.env.RUN_LOG_RETENTION_DAYS = "0";
    expect(pruneRunLogs()).toBe(0);
    expect(existsSync(old)).toBe(true);

    delete process.env.RUN_LOG_RETENTION_DAYS;
    expect(pruneRunLogs()).toBe(1);
    expect(existsSync(old)).toBe(false);
    expect(existsSync(fresh)).toBe(true);
    expect(existsSync(other)).toBe(true);
  });
});

describe("the trigger route (L6, L7, D9)", () => {
  it("answers a malformed probe or quality block with a 400", async () => {
    const t = tenant("trigger-shape");
    const modelId = modelFor(t, "s");
    const cases = [
      { kind: "probe", probe: { candidate_ctx: 2048, mode: "frontier" } },
      { kind: "probe", probe: { candidate_ctx: 2048, mode: "frontier", placement: { ngl: 1 } } },
      { kind: "quality", quality: { ctx_tokens: 2048 } },
    ];
    for (const extra of cases) expect((await trigger(t, modelId, extra)).statusCode).toBe(400);
    expect((await trigger(t, modelId, { main_gpu_backend: "=cmd" })).statusCode).toBe(400);
  });

  it("two simultaneous triggers can't both pass the duplicate check", async () => {
    const t = tenant("trigger-race");
    const modelId = modelFor(t, "r");
    // No build installed: the handler waits on GitHub, the widest window there is.
    getDb().prepare(`UPDATE workers SET installed_builds_json = '[]' WHERE id = ?`).run(t.worker.id);
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      await new Promise((r) => setTimeout(r, 50));
      return new Response(
        JSON.stringify([
          {
            tag_name: "b9",
            published_at: "2026-01-01",
            draft: false,
            prerelease: true,
            assets: [{ name: "llama-b9-bin-ubuntu-vulkan-x64.tar.gz", browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b9/a.tar.gz", size: 1 }],
          },
        ]),
        { status: 200 }
      );
    }) as typeof fetch;
    try {
      const [a, b] = await Promise.all([trigger(t, modelId), trigger(t, modelId)]);
      expect([a.statusCode, b.statusCode].sort()).toEqual([201, 409]);
      expect(repo.listTests(t.id)).toHaveLength(1);
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});
