import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-profiles-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: (typeof import("../db/repo.js"))["repo"];
let getDb: (typeof import("../db/migrate.js"))["getDb"];

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

function item(idx: number, over: Record<string, unknown> = {}) {
  return {
    idx,
    n_prompt: 512,
    n_gen: 128,
    n_depth: 0,
    concurrency: 1,
    threads: 8,
    n_gpu_layers: 99,
    batch_size: 2048,
    ubatch_size: 512,
    cache_type_k: "f16",
    cache_type_v: "f16",
    flash_attn: "on",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
    ...over,
  };
}

function result(testType: "pp" | "tg", tps: number, over: Record<string, unknown> = {}) {
  return {
    test_type: testType,
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
    avg_tps: tps,
    stddev_tps: 0,
    ram_peak_mib: 2000,
    vram_peak_mib: 4000,
    ram_avg_mib: 1800,
    vram_avg_mib: 3800,
    ram_free_before_mib: 16000,
    vram_free_before_mib: 8000,
    system_memory_total_mb: 32768,
    gpu_memory_total_mb: 8192,
    gpu_memory_total_accuracy: "exact" as const,
    gpu_memory_total_source: null,
    gpu_memory_free_start_accuracy: "exact" as const,
    gpu_memory_free_start_source: null,
    gpu_memory_model_avg_accuracy: "exact" as const,
    gpu_memory_model_avg_source: null,
    gpu_memory_model_peak_accuracy: "exact" as const,
    gpu_memory_model_peak_source: null,
    gpu_layers_loaded: 49,
    total_model_layers: 49,
    sample_count: 5,
    suspect_count: 0,
    method_version: 1,
    ...over,
  };
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ getDb } = await import("../db/migrate.js"));
  const { profilesRoutes } = await import("./profiles.js");
  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(profilesRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  repo.registerModel({
    id: "m1",
    filename: "model.gguf",
    size_bytes: 17_000_000_000,
    source: "local",
    metadata: {
      trained_ctx: 32_768,
      n_layer: 48,
      n_head_kv: 4,
      head_dim_k: 128,
      head_dim_v: 128,
    },
  });
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* Windows temp-dir handle */
  }
});

function makeScoredRun(
  runId: string,
  configs: { ngl: number; pp: number; tg: number; vramPeak?: number }[],
  extra: Partial<{ goals: unknown; kind: string }> = {}
): void {
  repo.createRun(undefined, {
    id: runId,
    kind: (extra.kind as never) ?? "sweep",
    worker_id: null as never,
    worker_name: "w",
    llama_cpp_build: "b1",
    llama_cpp_backend: "cpu",
    model_id: "m1",
    config: { model_id: "m1", sweep, ...(extra.goals ? { goals: extra.goals } : {}) } as never,
    status: "running",
    started_at: Date.now(),
  });
  repo.createRunItems(
    undefined,
    runId,
    configs.map((c, i) => item(i, { n_gpu_layers: c.ngl })) as never
  );
  configs.forEach((c, i) => {
    repo.recordRunItemTerminal(runId, i, {
      status: "done",
      results: [
        result("pp", c.pp, { n_gpu_layers: c.ngl, vram_peak_mib: c.vramPeak ?? 4000 }),
        result("tg", c.tg, { n_gpu_layers: c.ngl, vram_peak_mib: c.vramPeak ?? 4000 }),
      ] as never,
    });
  });
}

describe("GET /api/runs/:id/profiles (M3)", () => {
  it("scores stored results and names hidden cards instead of silently dropping them", async () => {
    makeScoredRun("run-basic", [
      { ngl: 99, pp: 1400, tg: 41, vramPeak: 7000 },
      { ngl: 50, pp: 1200, tg: 39, vramPeak: 4200 },
      { ngl: 8, pp: 900, tg: 30, vramPeak: 2100 },
    ]);
    const res = await fetch(`${baseUrl}/api/runs/run-basic/profiles`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      scoring: {
        profiles: { id: string }[];
        hidden: { id: string; reason: string }[];
        tallies: Record<string, number>;
        scoredCount: number;
      };
      goals: { goal: string };
      method_versions_present: number[];
    };
    expect(body.scoring.scoredCount).toBe(3);
    expect(body.scoring.profiles.map((p) => p.id).sort()).toEqual(["balanced", "low_memory", "max_speed"]);
    expect(body.scoring.hidden.map((h) => h.id)).toContain("max_context");
    expect(body.goals.goal).toBe("balanced");
    expect(body.method_versions_present).toEqual([1]);
  });

  it("re-scores on a query override without touching stored data (M2's instant re-score)", async () => {
    const before = await (await fetch(`${baseUrl}/api/runs/run-basic/profiles`)).json();
    const res = await fetch(
      `${baseUrl}/api/runs/run-basic/profiles?goal=max_context&target_ctx=8192&workload=chat&speed_floor_frac=0.5`
    );
    const body = (await res.json()) as {
      scoring: { profiles: { id: string; basis: string }[]; weights: { wPP: number; wTG: number } };
      goals: { goal: string; target_ctx: number };
      goals_overridden: boolean;
    };
    expect(body.goals_overridden).toBe(true);
    expect(body.goals.goal).toBe("max_context");
    expect(body.scoring.weights).toEqual({ wPP: 0.25, wTG: 0.75 });
    expect(body.scoring.profiles.map((p) => p.id)).toContain("max_context");
    expect(body.scoring.profiles.find((p) => p.id === "max_context")!.basis).toContain(
      "not measured at your target"
    );
    // The stored run is untouched: the default read still answers as before.
    const after = await (await fetch(`${baseUrl}/api/runs/run-basic/profiles`)).json();
    expect(after).toEqual(before);
  });

  it("clamps a target past the model's trained context, and says it clamped", async () => {
    const res = await fetch(`${baseUrl}/api/runs/run-basic/profiles?goal=max_context&target_ctx=131072`);
    const body = (await res.json()) as { goals: { target_ctx: number }; target_ctx_clamped: boolean };
    expect(body.goals.target_ctx).toBe(32_768);
    expect(body.target_ctx_clamped).toBe(true);
  });

  it("uses the run's stored goals when the caller states none (M2: intent stored as configuration)", async () => {
    makeScoredRun(
      "run-goals",
      [
        { ngl: 99, pp: 1400, tg: 41, vramPeak: 7000 },
        { ngl: 50, pp: 1200, tg: 39, vramPeak: 4200 },
      ],
      { goals: { goal: "max_context", target_ctx: 16_384, workload: "chat", speed_floor_frac: 0.5, kv_preset: "extended" } }
    );
    const body = (await (await fetch(`${baseUrl}/api/runs/run-goals/profiles`)).json()) as {
      goals: { goal: string; target_ctx: number };
      goals_overridden: boolean;
      scoring: { profiles: { id: string }[] };
    };
    expect(body.goals_overridden).toBe(false);
    expect(body.goals.goal).toBe("max_context");
    expect(body.goals.target_ctx).toBe(16_384);
    expect(body.scoring.profiles.map((p) => p.id)).toContain("max_context");
  });

  it("excludes N3 comparison members from profile scoring", async () => {
    repo.createRun(undefined, {
      id: "run-cmp",
      kind: "sweep",
      comparison_id: "cmp-1",
      worker_id: null as never,
      worker_name: "w",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: "m1",
      config: { model_id: "m1", sweep } as never,
      status: "running",
      started_at: Date.now(),
    });
    repo.createRunItems(undefined, "run-cmp", [item(0)] as never);
    repo.recordRunItemTerminal("run-cmp", 0, {
      status: "done",
      results: [result("pp", 9999), result("tg", 999)] as never,
    });
    const body = (await (await fetch(`${baseUrl}/api/runs/run-cmp/profiles`)).json()) as {
      scoring: { scoredCount: number; candidateCount: number };
    };
    // Its own rows are excluded from its own scoring universe, so nothing is
    // scored -- a trimmed comparison grid never competes with a full grid.
    expect(body.scoring.candidateCount).toBe(0);
    expect(body.scoring.scoredCount).toBe(0);
  });

  // M1's stated fallback: "weightsMib === null -> interpolate from any prior
  // run's peak for this model+machine when one exists". Both runs share the
  // worked-example geometry (48 layers, n_head_kv 4, head dim 128/128) so the
  // interpolated estimate can be checked against the plan's own numbers.
  it("interpolates weightsMib from a prior run's peak at the same placement, when this chain's own reading is unusable", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("profiles-interp-worker", "profiles-interp-worker");

    // Baseline run: a real VRAM reading at ngl=99/f16/f16. vram_peak_mib is
    // picked so residentWeightsMibFromPeak backs out to the worked example's
    // 2969 MiB (context 640 tokens -> 60 MiB of KV at 96 KiB/token, +256
    // scratch): 2969 + 60 + 256 = 3285.
    repo.createRun(undefined, {
      id: "run-interp-baseline",
      kind: "sweep",
      worker_id: worker.id,
      worker_name: "w",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cuda",
      model_id: "m1",
      config: { model_id: "m1", sweep } as never,
      status: "running",
      started_at: Date.now(),
    });
    repo.createRunItems(undefined, "run-interp-baseline", [item(0)] as never);
    repo.recordRunItemTerminal("run-interp-baseline", 0, {
      status: "done",
      results: [
        result("pp", 1000, { vram_peak_mib: 3285, gpu_memory_total_mb: 8192 }),
        result("tg", 40, { vram_peak_mib: 3285, gpu_memory_total_mb: 8192 }),
      ] as never,
    });

    // Target run: the SAME placement, SAME worker+model, but its own VRAM
    // reading is unusable (e.g. the GPU read failed) -- null, not just
    // absent.
    repo.createRun(undefined, {
      id: "run-interp-target",
      kind: "sweep",
      worker_id: worker.id,
      worker_name: "w",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cuda",
      model_id: "m1",
      config: { model_id: "m1", sweep, goals: { goal: "max_context", target_ctx: 8192, workload: "even" } } as never,
      status: "running",
      started_at: Date.now(),
    });
    repo.createRunItems(undefined, "run-interp-target", [item(0)] as never);
    repo.recordRunItemTerminal("run-interp-target", 0, {
      status: "done",
      results: [
        result("pp", 1000, { vram_peak_mib: null, gpu_memory_total_mb: null }),
        result("tg", 40, { vram_peak_mib: null, gpu_memory_total_mb: null }),
      ] as never,
    });

    const body = (await (await fetch(`${baseUrl}/api/runs/run-interp-target/profiles`)).json()) as {
      scoring: { profiles: { id: string; config: { maxCtx: { tokens: number; confidence: string } | null } }[] };
    };
    const maxContext = body.scoring.profiles.find((p) => p.id === "max_context");
    expect(maxContext).toBeDefined();
    expect(maxContext!.config.maxCtx?.confidence).toBe("good");
    // ~44.2k, same as the worked example -- this chain's own row contributed
    // nothing; every bit of the number came from the baseline run.
    expect(maxContext!.config.maxCtx!.tokens).toBeGreaterThan(44_000);
    expect(maxContext!.config.maxCtx!.tokens).toBeLessThan(44_400);
  });

  it("offers a conservative floor candidate (full trained_ctx) when neither this chain nor a prior run has usable VRAM data", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("profiles-no-data-worker", "profiles-no-data-worker");
    repo.createRun(undefined, {
      id: "run-no-vram-data",
      kind: "sweep",
      worker_id: worker.id,
      worker_name: "w",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: "m1",
      config: { model_id: "m1", sweep, goals: { goal: "max_context", target_ctx: 8192, workload: "even" } } as never,
      status: "running",
      started_at: Date.now(),
    });
    repo.createRunItems(undefined, "run-no-vram-data", [item(0)] as never);
    repo.recordRunItemTerminal("run-no-vram-data", 0, {
      status: "done",
      results: [
        result("pp", 1000, { vram_peak_mib: null, gpu_memory_total_mb: null }),
        result("tg", 40, { vram_peak_mib: null, gpu_memory_total_mb: null }),
      ] as never,
    });

    const body = (await (await fetch(`${baseUrl}/api/runs/run-no-vram-data/profiles`)).json()) as {
      scoring: {
        profiles: {
          id: string;
          config: { maxCtx: { confidence: string; conservativeFloorTokens?: number | null } | null };
        }[];
      };
    };
    const maxContext = body.scoring.profiles.find((p) => p.id === "max_context");
    expect(maxContext).toBeDefined();
    expect(maxContext!.config.maxCtx?.confidence).toBe("unknown");
    // m1's own trained_ctx is 32768 -- named explicitly, never fabricated.
    expect(maxContext!.config.maxCtx?.conservativeFloorTokens).toBe(32_768);
  });

  it("surfaces N4 quality measurements for this model, matched per card by KV pair", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("profiles-quality-worker", "profiles-quality-worker");
    repo.qualityRepo.upsert({
      root_run_id: "run-basic",
      model_id: "m1",
      worker_id: worker.id,
      llama_cpp_build: "b1",
      ctx_tokens: 8192,
      cache_type_k: "f16",
      cache_type_v: "f16",
      ppl: 6.1234,
      kld_vs_baseline: null,
      dataset_hash: `sha256:${"a".repeat(64)}`,
    });

    const body = (await (await fetch(`${baseUrl}/api/runs/run-basic/profiles`)).json()) as {
      quality_results: { model_id: string; cache_type_k: string; ppl: number | null }[];
    };
    expect(body.quality_results.length).toBeGreaterThan(0);
    const row = body.quality_results.find((r) => r.cache_type_k === "f16");
    expect(row).toBeDefined();
    expect(row!.ppl).toBeCloseTo(6.1234, 4);
    expect(row!.model_id).toBe("m1");
  });

  it("404s for an unknown run", async () => {
    const res = await fetch(`${baseUrl}/api/runs/nope/profiles`);
    expect(res.status).toBe(404);
  });
});

// New Step-2 fit-check route: verified limits readable WITHOUT an existing
// sweep run. Security-critical -- must authorize exactly like every other
// worker-scoped GET route (assertOwnsWorker), same 403 as
// GET /api/workers/:id/vram, not a reuse of a run's own ownership.
describe("GET /api/models/:id/verified-limits", () => {
  function authed(token: string): Record<string, string> {
    return { authorization: `Bearer ${token}` };
  }

  async function sessionFor(login: string): Promise<{ userId: string; token: string }> {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: login });
    return { userId: user.id, token };
  }

  function claimWorker(workerId: string, userId: string): void {
    getDb().prepare(`UPDATE workers SET user_id = ? WHERE id = ?`).run(userId, workerId);
  }

  it("400s when the worker query parameter is missing", async () => {
    const res = await fetch(`${baseUrl}/api/models/m1/verified-limits`);
    expect(res.status).toBe(400);
  });

  it("404s for an unknown model", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("limits-model-404", "limits-model-404");
    const res = await fetch(`${baseUrl}/api/models/nope/verified-limits?worker=${worker.id}`);
    expect(res.status).toBe(404);
  });

  it("404s for an unknown worker", async () => {
    const res = await fetch(`${baseUrl}/api/models/m1/verified-limits?worker=nope`);
    expect(res.status).toBe(404);
  });

  it("a different authenticated user cannot read another user's claimed worker's verified limits", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("limits-intruder", "limits-intruder");
    const { userId } = await sessionFor("limits-real-owner");
    claimWorker(worker.id, userId);
    const { token: intruderToken } = await sessionFor("limits-intruder-user");

    const res = await fetch(`${baseUrl}/api/models/m1/verified-limits?worker=${worker.id}`, {
      headers: authed(intruderToken),
    });
    expect(res.status).toBe(403);
  });

  it("the owner can read their own machine's verified limits", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("limits-owner-ok", "limits-owner-ok");
    const { userId, token } = await sessionFor("limits-owner");
    claimWorker(worker.id, userId);
    repo.limitsRepo.upsert({
      worker_id: worker.id,
      model_id: "m1",
      llama_cpp_build: "b1",
      cache_type_k: "f16",
      cache_type_v: "f16",
      placement_hash: "hash-1",
      verified_ctx_tokens: 8192,
    });

    const res = await fetch(`${baseUrl}/api/models/m1/verified-limits?worker=${worker.id}`, { headers: authed(token) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { model_id: string; worker_id: string; limits: { verified_ctx_tokens: number }[] };
    expect(body.model_id).toBe("m1");
    expect(body.worker_id).toBe(worker.id);
    expect(body.limits.some((l) => l.verified_ctx_tokens === 8192)).toBe(true);
  });

  it("an unauthenticated caller (single-tenant mode) can read any machine's limits regardless of ownership", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("limits-single-tenant", "limits-single-tenant");
    const { userId } = await sessionFor("limits-single-tenant-owner");
    claimWorker(worker.id, userId);

    const res = await fetch(`${baseUrl}/api/models/m1/verified-limits?worker=${worker.id}`);
    expect(res.status).toBe(200);
  });

  // Lets a stale ceiling be forgotten so ProfileCards offers "Verify with a
  // probe" again -- otherwise a card stuck reading "Verified" has no route
  // back to re-testing. Authorization mirrors the GET route above exactly.
  describe("DELETE /api/verified-limits/:id", () => {
    it("404s for an unknown id", async () => {
      const res = await fetch(`${baseUrl}/api/verified-limits/nope`, { method: "DELETE" });
      expect(res.status).toBe(404);
    });

    it("a different authenticated user cannot delete another user's claimed worker's verified limit", async () => {
      const worker = repo.workerRepo.getOrCreateByMachineId("limits-del-intruder", "limits-del-intruder");
      const { userId } = await sessionFor("limits-del-real-owner");
      claimWorker(worker.id, userId);
      const { token: intruderToken } = await sessionFor("limits-del-intruder-user");
      const limit = repo.limitsRepo.upsert({
        worker_id: worker.id,
        model_id: "m1",
        llama_cpp_build: "b1",
        cache_type_k: "f16",
        cache_type_v: "f16",
        placement_hash: "hash-del-1",
        verified_ctx_tokens: 8192,
      });

      const res = await fetch(`${baseUrl}/api/verified-limits/${limit.id}`, {
        method: "DELETE",
        headers: authed(intruderToken),
      });
      expect(res.status).toBe(403);
      expect(repo.limitsRepo.getById(limit.id)).toBeDefined();
    });

    it("the owner can delete their own machine's verified limit", async () => {
      const worker = repo.workerRepo.getOrCreateByMachineId("limits-del-owner-ok", "limits-del-owner-ok");
      const { userId, token } = await sessionFor("limits-del-owner");
      claimWorker(worker.id, userId);
      const limit = repo.limitsRepo.upsert({
        worker_id: worker.id,
        model_id: "m1",
        llama_cpp_build: "b1",
        cache_type_k: "f16",
        cache_type_v: "f16",
        placement_hash: "hash-del-2",
        verified_ctx_tokens: 4096,
      });

      const res = await fetch(`${baseUrl}/api/verified-limits/${limit.id}`, {
        method: "DELETE",
        headers: authed(token),
      });
      expect(res.status).toBe(200);
      expect(repo.limitsRepo.getById(limit.id)).toBeUndefined();
    });

    it("an unauthenticated caller (single-tenant mode) can delete any machine's verified limit regardless of ownership", async () => {
      const worker = repo.workerRepo.getOrCreateByMachineId("limits-del-single-tenant", "limits-del-single-tenant");
      const { userId } = await sessionFor("limits-del-single-tenant-owner");
      claimWorker(worker.id, userId);
      const limit = repo.limitsRepo.upsert({
        worker_id: worker.id,
        model_id: "m1",
        llama_cpp_build: "b1",
        cache_type_k: "f16",
        cache_type_v: "f16",
        placement_hash: "hash-del-3",
        verified_ctx_tokens: 2048,
      });

      const res = await fetch(`${baseUrl}/api/verified-limits/${limit.id}`, { method: "DELETE" });
      expect(res.status).toBe(200);
      expect(repo.limitsRepo.getById(limit.id)).toBeUndefined();
    });
  });
});
