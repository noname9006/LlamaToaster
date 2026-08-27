import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-measurements-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "shared-secret";

let app: FastifyInstance;
let baseUrl: string;
let repo: (typeof import("../db/repo.js"))["repo"];
let workerToken: string;
let otherWorkerToken: string;
let workerId: string;
let otherWorkerId: string;

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

function post(path: string, body: unknown, token?: string) {
  return fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });
}

function makeRun(
  id: string,
  opts: { kind?: string; worker: string; config?: Record<string, unknown> } = { worker: "" }
): void {
  repo.createRun(undefined, {
    id,
    kind: (opts.kind as never) ?? null,
    root_run_id: id,
    worker_id: opts.worker,
    worker_name: "w",
    llama_cpp_build: "b1",
    llama_cpp_backend: "cpu",
    model_id: "m1",
    config: { model_id: "m1", sweep, ...(opts.config ?? {}) } as never,
    status: "running",
    started_at: Date.now(),
  } as never);
  repo.createRunItems(undefined, id, [
    {
      idx: 0,
      n_prompt: 512,
      n_gen: 128,
      n_depth: 0,
      concurrency: 1,
      threads: 8,
      n_gpu_layers: 0,
      batch_size: 2048,
      ubatch_size: 512,
      cache_type_k: "f16",
      cache_type_v: "f16",
      flash_attn: "on",
      mtp: "off",
      n_gpu_layers_draft: 0,
      n_cpu_moe: 0,
    },
  ] as never);
}

const probeSpec = {
  candidate_ctx: 44_000,
  placement: { ngl: 16, n_cpu_moe: 32, slots: 1 },
  kv_pair: ["q8_0", "q8_0"],
};

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { measurementRoutes } = await import("./measurements.js");
  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(measurementRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  repo.registerModel({ id: "m1", filename: "m.gguf", size_bytes: 1, source: "local", metadata: {} });

  const w1 = repo.workerRepo.getOrCreateByMachineId("probe-machine", "probe-machine");
  const w2 = repo.workerRepo.getOrCreateByMachineId("other-machine", "other-machine");
  workerId = w1.id;
  otherWorkerId = w2.id;
  // An enrolled worker session -- the per-worker bearer credential the
  // machine gets when it is approved.
  const owner = repo.userRepo.upsertByIdentity("github", {
    providerUserId: "measurements-owner",
    login: "owner",
    avatarUrl: null,
  });
  workerToken = repo.sessionRepo.create(owner.id, { isWorker: true, workerId: w1.id }).token;
  otherWorkerToken = repo.sessionRepo.create(owner.id, { isWorker: true, workerId: w2.id }).token;
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* Windows temp-dir handle */
  }
});

describe("POST /api/runs/:id/probe-result (N2)", () => {
  it("requires an enrolled worker session and refuses the shared deployment secret", async () => {
    makeRun("probe-auth", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const noToken = await post("/api/runs/probe-auth/probe-result", { status: "verified", verified_ctx_tokens: 1000 });
    expect(noToken.status).toBe(401);

    const sharedSecret = await post(
      "/api/runs/probe-auth/probe-result",
      { status: "verified", verified_ctx_tokens: 1000 },
      "shared-secret"
    );
    expect(sharedSecret.status).toBe(401);
    expect(((await sharedSecret.json()) as { error: string }).error).toContain("enrolled worker session");
  });

  it("binds authorization by path, not payload: another machine cannot poison this ceiling", async () => {
    makeRun("probe-path", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-path/probe-result",
      { status: "verified", verified_ctx_tokens: 999_999, worker_id: workerId },
      otherWorkerToken
    );
    expect(res.status).toBe(403);
    expect(repo.limitsRepo.listForModelAndWorker("m1", otherWorkerId)).toHaveLength(0);
  });

  it("writes a verified ceiling with its observed margin, and a re-probe replaces it", async () => {
    makeRun("probe-ok", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-ok/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 41_000,
        margin_observed_frac: 0.06,
        attempts: [{ candidate_ctx: 44_000, ok: false, oom: true, spill: false }, { candidate_ctx: 41_000, ok: true, oom: false, spill: false }],
      },
      workerToken
    );
    expect(res.status).toBe(200);
    const rows = repo.limitsRepo.listForModelAndWorker("m1", workerId);
    expect(rows).toHaveLength(1);
    expect(rows[0].verified_ctx_tokens).toBe(41_000);
    expect(rows[0].margin_observed_frac).toBeCloseTo(0.06, 6);
    expect(rows[0].kv_type).toBe("q8_0/q8_0");

    // Re-probe the same (machine, build, KV pair, placement): replaced, not
    // duplicated.
    makeRun("probe-ok-2", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    await post(
      "/api/runs/probe-ok-2/probe-result",
      { status: "verified", verified_ctx_tokens: 39_000, margin_observed_frac: 0.1 },
      workerToken
    );
    const after = repo.limitsRepo.listForModelAndWorker("m1", workerId);
    expect(after).toHaveLength(1);
    expect(after[0].verified_ctx_tokens).toBe(39_000);
  });

  it("a different KV type yields a fresh row, never a reused verdict", async () => {
    makeRun("probe-f16", {
      kind: "probe",
      worker: workerId,
      config: { probe: { ...probeSpec, kv_pair: ["f16", "f16"] } },
    });
    await post(
      "/api/runs/probe-f16/probe-result",
      { status: "verified", verified_ctx_tokens: 20_000, margin_observed_frac: 0.02 },
      workerToken
    );
    const rows = repo.limitsRepo.listForModelAndWorker("m1", workerId);
    expect(rows.map((r) => r.kv_type).sort()).toEqual(["f16/f16", "q8_0/q8_0"]);
  });

  it("rejects out-of-bounds contexts with a 400 rather than a silent clamp", async () => {
    makeRun("probe-bounds", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const tooSmall = await post(
      "/api/runs/probe-bounds/probe-result",
      { status: "verified", verified_ctx_tokens: 4 },
      workerToken
    );
    expect(tooSmall.status).toBe(400);
    const tooBig = await post(
      "/api/runs/probe-bounds/probe-result",
      { status: "verified", verified_ctx_tokens: 99_999_999 },
      workerToken
    );
    expect(tooBig.status).toBe(400);
  });

  it("refuses more than three loads -- the hard ceiling is part of the contract", async () => {
    makeRun("probe-loads", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-loads/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 10_000,
        attempts: [1, 2, 3, 4].map((n) => ({ candidate_ctx: n * 1000, ok: true, oom: false, spill: false })),
      },
      workerToken
    );
    expect(res.status).toBe(400);
  });

  it("records a failed probe without writing any verified ceiling", async () => {
    makeRun("probe-fail", { kind: "probe", worker: workerId, config: { probe: { ...probeSpec, kv_pair: ["q4_0", "q4_0"] } } });
    const res = await post(
      "/api/runs/probe-fail/probe-result",
      { status: "failed_oom", verified_ctx_tokens: null, error: "OOM at 44000 and at 33000" },
      workerToken
    );
    expect(res.status).toBe(200);
    expect(repo.limitsRepo.listForModelAndWorker("m1", workerId).some((r) => r.kv_type === "q4_0/q4_0")).toBe(false);
    expect(repo.getRunItems("probe-fail")[0].status).toBe("failed_oom");
  });
});

describe("POST /api/runs/:id/quality-result (N4)", () => {
  const dataset = `sha256:${"a".repeat(64)}`;

  it("rejects a malformed dataset hash before anything is stored", async () => {
    makeRun("q-bad-hash", { kind: "quality", worker: workerId });
    const res = await post(
      "/api/runs/q-bad-hash/quality-result",
      { ppl: 6.1, dataset_hash: "not-a-hash", ctx_tokens: 32_768, cache_type_k: "f16", cache_type_v: "f16" },
      workerToken
    );
    expect(res.status).toBe(400);
    expect(repo.qualityRepo.listForModel("m1")).toHaveLength(0);
  });

  it("rejects non-finite or non-positive readings -- never stored as data", async () => {
    makeRun("q-bad-ppl", { kind: "quality", worker: workerId });
    for (const ppl of [0, -1, Number.NaN]) {
      const res = await post(
        "/api/runs/q-bad-ppl/quality-result",
        { ppl, dataset_hash: dataset, ctx_tokens: 32_768, cache_type_k: "f16", cache_type_v: "f16" },
        workerToken
      );
      expect(res.status).toBe(400);
    }
    expect(repo.qualityRepo.listForModel("m1")).toHaveLength(0);
  });

  it("records ppl with its corpus hash, and leaves KLD null unless the measurement reported one", async () => {
    makeRun("q-baseline", { kind: "quality", worker: workerId });
    const baseline = await post(
      "/api/runs/q-baseline/quality-result",
      { ppl: 6.0, dataset_hash: dataset, ctx_tokens: 32_768, cache_type_k: "f16", cache_type_v: "f16" },
      workerToken
    );
    expect(baseline.status).toBe(200);
    expect(((await baseline.json()) as { quality: { kld_vs_baseline: number | null } }).quality.kld_vs_baseline).toBeNull();

    makeRun("q-q8", { kind: "quality", worker: workerId });
    const quantized = await post(
      "/api/runs/q-q8/quality-result",
      { ppl: 6.024, dataset_hash: dataset, ctx_tokens: 32_768, cache_type_k: "q8_0", cache_type_v: "q8_0" },
      workerToken
    );
    const body = (await quantized.json()) as { quality: { kld_vs_baseline: number | null; dataset_hash: string } };
    // A log-perplexity ratio is NOT a KL divergence, so nothing is derived
    // into this column: with no measured KLD the row lands ppl-only, which is
    // the plan's own stated fallback.
    expect(body.quality.kld_vs_baseline).toBeNull();
    // A perplexity number without its corpus hash is meaningless.
    expect(body.quality.dataset_hash).toBe(dataset);
  });

  it("forces KLD null even if reported, when no f16 baseline row exists yet at that ctx", async () => {
    makeRun("q-no-baseline", { kind: "quality", worker: workerId });
    const res = await post(
      "/api/runs/q-no-baseline/quality-result",
      {
        ppl: 6.024,
        kld_vs_baseline: 0.004,
        dataset_hash: dataset,
        ctx_tokens: 4096, // no f16/f16 row has ever been measured at this ctx
        cache_type_k: "q8_0",
        cache_type_v: "q8_0",
      },
      workerToken
    );
    expect(res.status).toBe(200);
    // "No baseline row => ppl only" is an enforced invariant, not merely a
    // description of what the worker happens to report today.
    expect(((await res.json()) as { quality: { kld_vs_baseline: number | null } }).quality.kld_vs_baseline).toBeNull();
  });

  it("stores a KLD the measurement DID report, against the same baseline pairing", async () => {
    makeRun("q-kld-baseline", { kind: "quality", worker: workerId });
    const baseline = await post(
      "/api/runs/q-kld-baseline/quality-result",
      { ppl: 6.0, dataset_hash: dataset, ctx_tokens: 16_384, cache_type_k: "f16", cache_type_v: "f16" },
      workerToken
    );
    expect(baseline.status).toBe(200);

    makeRun("q-measured-kld", { kind: "quality", worker: workerId });
    const res = await post(
      "/api/runs/q-measured-kld/quality-result",
      {
        ppl: 6.024,
        kld_vs_baseline: 0.004,
        dataset_hash: dataset,
        ctx_tokens: 16_384,
        cache_type_k: "q8_0",
        cache_type_v: "q8_0",
      },
      workerToken
    );
    expect(res.status).toBe(200);
    expect(((await res.json()) as { quality: { kld_vs_baseline: number } }).quality.kld_vs_baseline).toBeCloseTo(
      0.004,
      6
    );
  });

  it("a retried job replaces through the UNIQUE key instead of duplicating", async () => {
    makeRun("q-retry", { kind: "quality", worker: workerId });
    const payload = {
      ppl: 7.5,
      dataset_hash: dataset,
      ctx_tokens: 8192,
      cache_type_k: "q4_0",
      cache_type_v: "q4_0",
    };
    await post("/api/runs/q-retry/quality-result", payload, workerToken);
    await post("/api/runs/q-retry/quality-result", { ...payload, ppl: 7.4 }, workerToken);
    const rows = repo.qualityRepo.listForRoot("q-retry");
    expect(rows).toHaveLength(1);
    expect(rows[0].ppl).toBeCloseTo(7.4, 6);
  });

  it("orphans dependent KLD rows loudly when the baseline goes away", async () => {
    const before = repo.qualityRepo
      .listForModel("m1")
      .filter((r) => r.cache_type_k === "q8_0" && r.kld_vs_baseline != null);
    expect(before.length).toBeGreaterThan(0);
    const changed = repo.qualityRepo.orphanDependents("m1", "b1", 16_384, dataset);
    expect(changed).toBeGreaterThan(0);
    // Deleting a baseline marks dependents orphaned rather than silently
    // re-baselining them onto some other row.
    const after = repo.qualityRepo.listForModel("m1").filter((r) => r.ctx_tokens === 16_384);
    expect(after.every((r) => r.kld_vs_baseline == null)).toBe(true);
  });

  it("requires an enrolled worker session here too", async () => {
    makeRun("q-auth", { kind: "quality", worker: workerId });
    const res = await post(
      "/api/runs/q-auth/quality-result",
      { ppl: 6.1, dataset_hash: dataset, ctx_tokens: 32_768, cache_type_k: "f16", cache_type_v: "f16" },
      "shared-secret"
    );
    expect(res.status).toBe(401);
  });
});
