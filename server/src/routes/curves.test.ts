import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { CURVE_METHOD_VERSION } from "../../../shared/types.js";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-curves-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: (typeof import("../db/repo.js"))["repo"];
let workerId: string;

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
    stddev_tps: 0.1,
    ram_peak_mib: 2000,
    vram_peak_mib: 4000,
    ram_avg_mib: 1900,
    vram_avg_mib: 3900,
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
    sample_count: 5,
    suspect_count: 0,
    method_version: CURVE_METHOD_VERSION,
    ...over,
  };
}

function makeRun(id: string, kind: string | null, items: ReturnType<typeof item>[]): void {
  repo.createTest(undefined, {
    id,
    kind: kind as never,
    root_run_id: id,
    worker_id: workerId,
    worker_name: "toaster-01",
    llama_cpp_build: "b10516",
    llama_cpp_backend: "vulkan",
    backend_device_name: "AMD Radeon RX 6600 XT",
    model_id: "m1",
    config: { model_id: "m1", sweep } as never,
    status: "running",
    started_at: Date.now(),
  } as never);
  repo.createTestItems(undefined, id, items as never);
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { curveRoutes } = await import("./curves.js");
  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(curveRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  workerId = repo.workerRepo.getOrCreateByMachineId("toaster-01", "toaster-01").id;
  repo.registerModel({
    id: "m1",
    filename: "model.gguf",
    size_bytes: 17_000_000_000,
    source: "local",
    metadata: { trained_ctx: 32_768, n_layer: 48, n_head_kv: 4, head_dim_k: 128, head_dim_v: 128 },
  });

  // Three measured curve points on the server engine, all METHOD_VERSION 2.
  makeRun("curve-run", "runtime", [0, 1, 2].map((idx) => item(idx, { n_prompt: [8192, 16384, 32768][idx] })));
  [8192, 16384, 32768].forEach((ctx, idx) => {
    repo.recordTestItemTerminal("curve-run", idx, {
      status: "done",
      results: [
        result("pp", 310, { n_prompt: ctx, engine: "server", ttft_ms_p50: (ctx / 310) * 1000, ttft_n: 1 }),
        result("tg", 39 - idx * 0.3, { n_prompt: ctx, engine: "server" }),
      ] as never,
    });
  });

  // An ordinary METHOD_VERSION 1 runtime row at the same context: it must not
  // land in the curve.
  makeRun("ordinary-run", null, [item(0, { n_prompt: 8192 })]);
  repo.recordTestItemTerminal("ordinary-run", 0, {
    status: "done",
    results: [result("tg", 99, { n_prompt: 8192, engine: "server", method_version: 1 })] as never,
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

describe("GET /api/models/:id/curve (N1)", () => {
  it("renders the measured points for one (model, machine, build, engine)", async () => {
    const res = await fetch(`${baseUrl}/api/models/m1/curve?worker=${workerId}&build=b10516&engine=server`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      points: { effectiveCtx: number; tg: number | null; ttftN: number | null }[];
      method_version: number;
    };
    expect(body.method_version).toBe(CURVE_METHOD_VERSION);
    expect(body.points.map((p) => p.effectiveCtx)).toEqual([8192, 16384, 32768]);
    // Cold TTFT points are visibly single-shot.
    expect(body.points.every((p) => p.ttftN === 1)).toBe(true);
  });

  it("keeps ordinary runtime rows out of the curve via the method-version stamp", async () => {
    const body = (await (
      await fetch(`${baseUrl}/api/models/m1/curve?worker=${workerId}&engine=server`)
    ).json()) as { points: { tg: number | null }[] };
    expect(body.points.some((p) => p.tg === 99)).toBe(false);
  });

  it("renders the ladder past trained context as unavailable-with-reason, never a silent gap", async () => {
    const body = (await (
      await fetch(`${baseUrl}/api/models/m1/curve?worker=${workerId}&engine=server`)
    ).json()) as { ladder: { effectiveCtx: number; available: boolean; unavailableReason: string | null }[] };
    const beyond = body.ladder.find((c) => c.effectiveCtx === 65_536)!;
    expect(beyond.available).toBe(false);
    expect(beyond.unavailableReason).toBeTruthy();
  });

  it("says whether any ladder cell is still uncovered, so the CTA can disable itself with a reason", async () => {
    const body = (await (
      await fetch(`${baseUrl}/api/models/m1/curve?worker=${workerId}&engine=server`)
    ).json()) as { all_points_covered: boolean };
    // 2k and 4k are still uncovered here.
    expect(body.all_points_covered).toBe(false);
  });

  it("rejects an unknown engine and 404s an unknown model", async () => {
    expect((await fetch(`${baseUrl}/api/models/m1/curve?engine=nonsense`)).status).toBe(400);
    expect((await fetch(`${baseUrl}/api/models/nope/curve`)).status).toBe(404);
  });
});

describe("GET /api/models/:id/rates (§0.6)", () => {
  it("prices from the server rate and names its source", async () => {
    const body = (await (await fetch(`${baseUrl}/api/models/m1/rates?worker=${workerId}`)).json()) as {
      pp: { tps: number; source: string; label: string };
      tg: { tps: number; source: string };
    };
    // These rows carry engine "server" with mtp off -- a {server, spec:"off"}
    // baseline, which is exactly the first tier of §0.6's order.
    expect(body.pp.source).toBe("server_measured");
    expect(body.pp.label).toBe("measured on this machine");
    expect(body.tg.tps).toBeGreaterThan(0);
  });

  it("returns unavailable rather than a number for an unmeasured machine", async () => {
    const body = (await (await fetch(`${baseUrl}/api/models/m1/rates?worker=nonexistent`)).json()) as {
      pp: { tps: number | null; source: string };
    };
    expect(body.pp.tps).toBeNull();
    expect(body.pp.source).toBe("unavailable");
  });
});

describe("GET /api/runs/:id/knee (N5)", () => {
  it("derives the knee from stored rows rather than reading a stored verdict", async () => {
    makeRun(
      "knee-run",
      "runtime",
      [1, 2, 4, 8].map((slots, idx) => item(idx, { concurrency: slots, n_prompt: 32_768 }))
    );
    [
      { slots: 1, tps: 39, ttft: 107_000 },
      { slots: 2, tps: 68, ttft: 118_000 },
      { slots: 4, tps: 101, ttft: 226_000 },
      { slots: 8, tps: 108, ttft: 415_000 },
    ].forEach((row, idx) => {
      repo.recordTestItemTerminal("knee-run", idx, {
        status: "done",
        results: [
          result("tg", row.tps, {
            n_prompt: 32_768,
            engine: "server",
            concurrency: row.slots,
            ttft_ms_p95: row.ttft,
            method_version: 1,
          }),
        ] as never,
      });
    });

    const body = (await (await fetch(`${baseUrl}/api/runs/knee-run/knee`)).json()) as {
      knee: number | null;
      thresholdMs: number;
      summary: string;
      samples: { slots: number; aggregateTps: number | null }[];
    };
    expect(body.knee).toBe(4);
    expect(body.thresholdMs).toBe(214_000);
    expect(body.summary).toContain("knee at slots = 4");
    expect(body.samples.map((s) => s.aggregateTps)).toEqual([39, 68, 101, 108]);
  });
});

describe("GET /api/runs/:id/sustained (N6)", () => {
  it("offers a priced re-run only past the one-third threshold, and never schedules it", async () => {
    makeRun("thermal-run", null, [0, 1, 2].map((idx) => item(idx)));
    // Two of three items throttled: past 1/3.
    [0, 1].forEach((idx) => {
      repo.recordTestItemTerminal("thermal-run", idx, {
        status: "done",
        results: [
          result("tg", 38, { method_version: 1, caveat_flags: ["thermally_throttled"], gpu_temp_c_max: 89 }),
        ] as never,
      });
    });
    repo.recordTestItemTerminal("thermal-run", 2, {
      status: "done",
      results: [result("tg", 41, { method_version: 1 })] as never,
    });

    const body = (await (await fetch(`${baseUrl}/api/runs/thermal-run/sustained`)).json()) as {
      flagged_items: number[];
      denominator: number;
      ratio: number;
      offer_rerun: boolean;
      rerun_estimate: string;
      steady_state: { available: boolean; reason: string | null };
    };
    expect(body.flagged_items).toEqual([0, 1]);
    expect(body.denominator).toBe(3);
    expect(body.ratio).toBeCloseTo(2 / 3, 6);
    expect(body.offer_rerun).toBe(true);
    expect(body.rerun_estimate).toBeTruthy();
    // repeats is 5 here, so discarding one still leaves n = 4.
    expect(body.steady_state.available).toBe(true);
  });

  it("does not offer a re-run when a single item out of many is flagged", async () => {
    makeRun("thermal-quiet", null, [0, 1, 2, 3].map((idx) => item(idx)));
    repo.recordTestItemTerminal("thermal-quiet", 0, {
      status: "done",
      results: [result("tg", 38, { method_version: 1, caveat_flags: ["thermally_throttled"] })] as never,
    });
    [1, 2, 3].forEach((idx) =>
      repo.recordTestItemTerminal("thermal-quiet", idx, {
        status: "done",
        results: [result("tg", 41, { method_version: 1 })] as never,
      })
    );
    const body = (await (await fetch(`${baseUrl}/api/runs/thermal-quiet/sustained`)).json()) as {
      ratio: number;
      offer_rerun: boolean;
    };
    expect(body.ratio).toBeCloseTo(0.25, 6);
    expect(body.offer_rerun).toBe(false);
  });
});
