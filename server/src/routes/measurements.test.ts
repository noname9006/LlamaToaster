import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import { placementHash } from "../../../shared/configHash.js";

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
// Read from the route module in beforeAll rather than imported statically:
// a top-level import would pull in repo.js and pin the getDb singleton
// before DB_PATH above takes effect.
let MAX_PROBE_ATTEMPTS: number;

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
  const measurements = await import("./measurements.js");
  const { measurementRoutes } = measurements;
  MAX_PROBE_ATTEMPTS = measurements.MAX_PROBE_ATTEMPTS;
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

  it("keys the ceiling on the placement it was VERIFIED at, not the one requested", async () => {
    // The ladder moves ngl in every searching mode, so a probe requested at
    // ngl=16 routinely proves its ceiling somewhere else entirely. Filing the
    // row under the request would attribute a loaded context to a placement
    // that may never have loaded at all.
    makeRun("probe-ngl", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-ngl/probe-result",
      { status: "verified", verified_ctx_tokens: 8192, verified_ngl: 31 },
      workerToken
    );
    expect(res.status).toBe(200);
    const row = repo.limitsRepo.listForModelAndWorker("m1", workerId).find((l) => l.verified_ctx_tokens === 8192)!;
    expect(row.verified_ngl).toBe(31);
    expect(row.placement_hash).toBe(placementHash({ ngl: 31, n_cpu_moe: 32, slots: 1 }));
    expect(row.placement_hash).not.toBe(placementHash({ ngl: 16, n_cpu_moe: 32, slots: 1 }));
    // A different placement means a genuinely new row -- drop it again so the
    // row-counting tests in this block keep seeing only what they wrote.
    repo.limitsRepo.deleteById(row.id);
  });

  it("falls back to the requested placement for a worker that reports no verified_ngl", async () => {
    makeRun("probe-no-ngl", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-no-ngl/probe-result",
      { status: "verified", verified_ctx_tokens: 7000 },
      workerToken
    );
    expect(res.status).toBe(200);
    const row = repo.limitsRepo.listForModelAndWorker("m1", workerId).find((l) => l.verified_ctx_tokens === 7000)!;
    expect(row.verified_ngl).toBeNull();
    expect(row.placement_hash).toBe(placementHash({ ngl: 16, n_cpu_moe: 32, slots: 1 }));
    // Same placement hash as the requested-placement row above, so this
    // REPLACES it rather than adding one -- exactly the pre-verified_ngl
    // behaviour, and what the row-counting test below still expects.
  });

  it("rejects an out-of-range verified_ngl rather than hashing a nonsense placement", async () => {
    makeRun("probe-bad-ngl", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-bad-ngl/probe-result",
      { status: "verified", verified_ctx_tokens: 6000, verified_ngl: -3 },
      workerToken
    );
    expect(res.status).toBe(400);
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

  it("refuses more loads than the ladder's ceiling -- the cap is part of the contract", async () => {
    makeRun("probe-loads", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const atCap = await post(
      "/api/runs/probe-loads/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 10_000,
        attempts: Array.from({ length: MAX_PROBE_ATTEMPTS }, (_, i) => ({
          candidate_ctx: (i + 1) * 1000,
          ok: true,
          oom: false,
          spill: false,
        })),
      },
      workerToken
    );
    expect(atCap.status).toBe(200);

    makeRun("probe-loads-2", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const overCap = await post(
      "/api/runs/probe-loads-2/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 10_000,
        attempts: Array.from({ length: MAX_PROBE_ATTEMPTS + 1 }, (_, i) => ({
          candidate_ctx: (i + 1) * 1000,
          ok: true,
          oom: false,
          spill: false,
        })),
      },
      workerToken
    );
    expect(overCap.status).toBe(400);
  });

  it("enforces the LIVE admin-configurable probeMaxLoads setting, not the MAX_PROBE_ATTEMPTS constant", async () => {
    // Default is 24 (matches MAX_PROBE_ATTEMPTS) on a fresh DB -- lower it to
    // prove the request-time check reads appSettingsRepo, not the constant.
    expect(repo.appSettingsRepo.getProbeMaxLoads()).toBe(MAX_PROBE_ATTEMPTS);
    repo.appSettingsRepo.setProbeMaxLoads(3);
    try {
      const attempt = (ctx: number) => ({ candidate_ctx: ctx, ok: true, oom: false, spill: false });

      makeRun("probe-loads-live-ok", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
      const atLoweredCap = await post(
        "/api/runs/probe-loads-live-ok/probe-result",
        { status: "verified", verified_ctx_tokens: 10_000, attempts: [attempt(1000), attempt(2000), attempt(3000)] },
        workerToken
      );
      expect(atLoweredCap.status).toBe(200);

      makeRun("probe-loads-live-over", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
      const overLoweredCap = await post(
        "/api/runs/probe-loads-live-over/probe-result",
        { status: "verified", verified_ctx_tokens: 10_000, attempts: [attempt(1000), attempt(2000), attempt(3000), attempt(4000)] },
        workerToken
      );
      expect(overLoweredCap.status).toBe(400);
      const body = (await overLoweredCap.json()) as { error: string };
      expect(body.error).toContain("at most 3 loads");
    } finally {
      // Restore the default for later tests in this file/process.
      repo.appSettingsRepo.setProbeMaxLoads(MAX_PROBE_ATTEMPTS);
    }
  });

  it("persists every ladder rung, not just the winning one", async () => {
    makeRun("probe-rungs", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-rungs/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 16_384,
        attempts: [
          {
            candidate_ctx: 32_768,
            ngl: 27,
            ok: false,
            oom: true,
            spill: false,
            vram_needed_mib: 9119,
            vram_free_mib: 8000,
            ram_needed_mib: 6743,
            ram_free_mib: 32_000,
          },
          {
            candidate_ctx: 16_384,
            ngl: 27,
            ok: true,
            oom: false,
            spill: false,
            vram_needed_mib: 5000,
            vram_free_mib: 8000,
            vram_peak_mib: 5422,
            ram_needed_mib: 4000,
            ram_free_mib: 32_000,
            gen_tps: 41.5,
          },
        ],
      },
      workerToken
    );
    expect(res.status).toBe(200);

    const rungs = repo.probeAttemptsRepo.listForRun("probe-rungs");
    expect(rungs).toHaveLength(2);
    expect(rungs.map((r) => r.seq)).toEqual([0, 1]);
    expect(rungs[0].candidate_ctx).toBe(32_768);
    expect(rungs[0].ok).toBe(0);
    expect(rungs[0].oom).toBe(1);
    // The predicted-vs-real pair that previously only reached a log line.
    expect(rungs[0].vram_needed_mib).toBe(9119);
    expect(rungs[0].vram_free_mib).toBe(8000);
    expect(rungs[0].vram_peak_mib).toBeNull();
    expect(rungs[1].ok).toBe(1);
    expect(rungs[1].ngl).toBe(27);
    expect(rungs[1].vram_peak_mib).toBe(5422);
    expect(rungs[1].gen_tps).toBeCloseTo(41.5, 6);
  });

  it("keeps the rungs of a failed probe, which writes no verified ceiling", async () => {
    makeRun("probe-rungs-fail", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-rungs-fail/probe-result",
      {
        status: "failed_oom",
        verified_ctx_tokens: null,
        attempts: [{ candidate_ctx: 8192, ngl: 20, ok: false, oom: true, spill: false }],
        error: "OOM at 8192",
      },
      workerToken
    );
    expect(res.status).toBe(200);
    expect(repo.probeAttemptsRepo.listForRun("probe-rungs-fail")).toHaveLength(1);
  });

  it("rejects a malformed attempt reading rather than storing it as data", async () => {
    makeRun("probe-bad-attempt", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    const res = await post(
      "/api/runs/probe-bad-attempt/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 10_000,
        attempts: [{ candidate_ctx: 10_000, ok: true, oom: false, spill: false, vram_peak_mib: -5 }],
      },
      workerToken
    );
    expect(res.status).toBe(400);
    expect(repo.probeAttemptsRepo.listForRun("probe-bad-attempt")).toHaveLength(0);
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

// Read side. Ownership binds to the MACHINE the run was dispatched to (the
// same assertOwnsWorker gate GET /api/models/:id/verified-limits uses), not
// to the run's own user_id -- these rows describe that machine's memory.
describe("GET /api/runs/:id/probe-attempts", () => {
  function authed(token: string): Record<string, string> {
    return { authorization: `Bearer ${token}` };
  }

  async function sessionFor(login: string): Promise<{ userId: string; token: string }> {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: login });
    return { userId: user.id, token };
  }

  async function claimWorker(id: string, userId: string): Promise<void> {
    const { getDb } = await import("../db/migrate.js");
    getDb().prepare(`UPDATE workers SET user_id = ? WHERE id = ?`).run(userId, id);
  }

  it("returns the ladder in seq order for the machine's owner", async () => {
    makeRun("probe-read", { kind: "probe", worker: workerId, config: { probe: probeSpec } });
    await post(
      "/api/runs/probe-read/probe-result",
      {
        status: "verified",
        verified_ctx_tokens: 8192,
        attempts: [
          { candidate_ctx: 16_384, ngl: 27, ok: false, oom: true, spill: false },
          { candidate_ctx: 8192, ngl: 27, ok: true, oom: false, spill: false, gen_tps: 30 },
        ],
      },
      workerToken
    );

    const { userId, token } = await sessionFor("probe-read-owner");
    await claimWorker(workerId, userId);
    const res = await fetch(`${baseUrl}/api/runs/probe-read/probe-attempts`, { headers: authed(token) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { attempts: { seq: number; candidate_ctx: number }[] };
    expect(body.attempts.map((a) => a.candidate_ctx)).toEqual([16_384, 8192]);
  });

  it("refuses a different authenticated user", async () => {
    const { token: intruderToken } = await sessionFor("probe-read-intruder");
    const res = await fetch(`${baseUrl}/api/runs/probe-read/probe-attempts`, { headers: authed(intruderToken) });
    expect(res.status).toBe(403);
  });

  it("404s an unknown run rather than leaking an empty list", async () => {
    const res = await fetch(`${baseUrl}/api/runs/no-such-run/probe-attempts`);
    expect(res.status).toBe(404);
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
