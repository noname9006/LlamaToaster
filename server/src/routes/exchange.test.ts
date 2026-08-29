import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import type { Bundle } from "../../../shared/exchange.js";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-exchange-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: (typeof import("../db/repo.js"))["repo"];

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

function result(testType: "pp" | "tg", tps: number) {
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
    stddev_tps: 0.2,
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
    method_version: 1,
    gpu_temp_c_max: 78,
    gpu_clock_mhz_min: 2480,
  };
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { exchangeRoutes } = await import("./exchange.js");
  const { profilesRoutes } = await import("./profiles.js");
  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(exchangeRoutes);
  await app.register(profilesRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  repo.registerModel({
    id: "a".repeat(64),
    filename: "model-Q4_K_M.gguf",
    size_bytes: 17_000_000_000,
    source: "local",
    metadata: { quant: "Q4_K_M", trained_ctx: 32_768, n_layer: 48, n_head_kv: 4, head_dim_k: 128, head_dim_v: 128 },
  });

  repo.createRun(undefined, {
    id: "run-export",
    kind: "sweep",
    root_run_id: "run-export",
    worker_id: null as never,
    worker_name: "toaster-01",
    llama_cpp_build: "b10516",
    llama_cpp_backend: "vulkan",
    backend_device_name: "AMD Radeon RX 6600 XT",
    model_id: "a".repeat(64),
    config: {
      model_id: "a".repeat(64),
      sweep,
      goals: { goal: "max_context", target_ctx: 32_768, workload: "chat", speed_floor_frac: 0.5, kv_preset: "extended" },
    } as never,
    status: "running",
    started_at: Date.now(),
  } as never);
  repo.createRunItems(undefined, "run-export", [
    {
      idx: 0,
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
    },
  ] as never);
  repo.recordRunItemTerminal("run-export", 0, {
    status: "done",
    results: [result("pp", 1400), result("tg", 41)] as never,
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

async function exportBundle(): Promise<Bundle> {
  const res = await fetch(`${baseUrl}/api/runs/run-export/export`);
  expect(res.status).toBe(200);
  return (await res.json()) as Bundle;
}

describe("GET /api/runs/:id/export (N7)", () => {
  it("carries measurements, the goals block, the build tag and a methods section", async () => {
    const bundle = await exportBundle();
    expect(bundle.rows).toHaveLength(2);
    expect(bundle.run.llama_cpp_build).toBe("b10516");
    expect(bundle.run.config.goals).toMatchObject({ goal: "max_context", target_ctx: 32_768 });
    expect(bundle.methods.map((m) => m.method_version)).toEqual([1]);
    expect(bundle.methods[0].pipeline.length).toBeGreaterThan(0);
  });

  it("anonymizes the hardware to a class -- GPU name plus a VRAM bucket, no machine identity", async () => {
    const bundle = await exportBundle();
    expect(bundle.hardware_class).toMatchObject({
      gpu_name: "AMD Radeon RX 6600 XT",
      backend: "vulkan",
      vram_class_mib: 8192,
    });
    expect(JSON.stringify(bundle)).not.toContain("machine_id");
  });

  it("stamps every row with a hash recomputed from its canonical form", async () => {
    const bundle = await exportBundle();
    expect(bundle.rows.every((r) => /^[0-9a-f]{64}$/.test(r.config_hash))).toBe(true);
  });
});

describe("POST /api/import (N7)", () => {
  it("round-trips: the imported rows keep byte-equal hashes", async () => {
    const bundle = await exportBundle();
    const res = await fetch(`${baseUrl}/api/import`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bundle }),
    });
    expect(res.status).toBe(201);
    const body = (await res.json()) as { run_id: string; imported_rows: number; opted_into_scoring: boolean };
    expect(body.imported_rows).toBe(2);
    expect(body.opted_into_scoring).toBe(false);
    const rows = repo.getResultsForRun(body.run_id);
    expect(rows.map((r) => r.config_hash).sort()).toEqual(bundle.rows.map((r) => r.config_hash).sort());
    expect(rows.every((r) => r.imported_bundle_id != null)).toBe(true);
  });

  it("badges imported rows and keeps them OUT of local scoring unless opted in", async () => {
    const bundle = await exportBundle();
    const notOptedIn = await (
      await fetch(`${baseUrl}/api/import`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ bundle }),
      })
    ).json();
    const excluded = (await (
      await fetch(`${baseUrl}/api/runs/${(notOptedIn as { run_id: string }).run_id}/profiles`)
    ).json()) as { scoring: { candidateCount: number } };
    expect(excluded.scoring.candidateCount).toBe(0);

    const optedIn = await (
      await fetch(`${baseUrl}/api/import`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ bundle, opt_in_scoring: true }),
      })
    ).json();
    const included = (await (
      await fetch(`${baseUrl}/api/runs/${(optedIn as { run_id: string }).run_id}/profiles`)
    ).json()) as { scoring: { candidateCount: number } };
    expect(included.scoring.candidateCount).toBe(1);
  });

  it("tampering one field rejects exactly that row and imports the rest", async () => {
    const bundle = await exportBundle();
    bundle.rows[0].avg_tps = 999_999;
    bundle.rows[0].n_gpu_layers = 3; // breaks the hash for this row only
    const res = await fetch(`${baseUrl}/api/import`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bundle }),
    });
    expect(res.status).toBe(201);
    const body = (await res.json()) as {
      imported_rows: number;
      rejected_rows: { index: number; reason: string }[];
    };
    expect(body.imported_rows).toBe(1);
    expect(body.rejected_rows).toHaveLength(1);
    expect(body.rejected_rows[0].reason).toContain("altered after export");
  });

  it("refuses a bundle envelope it does not understand", async () => {
    const res = await fetch(`${baseUrl}/api/import`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bundle: { format: "not-ours" } }),
    });
    expect(res.status).toBe(400);
  });

  it("never overwrites a local model record from an import", async () => {
    const before = repo.getModel("a".repeat(64))!;
    const bundle = await exportBundle();
    bundle.run.model_filename = "attacker-renamed.gguf";
    bundle.run.model_quant = "Q2_K";
    await fetch(`${baseUrl}/api/import`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bundle }),
    });
    const after = repo.getModel("a".repeat(64))!;
    expect(after.filename).toBe(before.filename);
    expect(after.metadata.quant).toBe(before.metadata.quant);
  });
});
