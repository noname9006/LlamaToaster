import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-v8-ingest-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "v8-ingest-secret";

let repo: typeof import("../db/repo.js")["repo"];

const baseResult = {
  test_type: "pp" as const,
  n_prompt: 512,
  n_gen: 0,
  n_threads: 8,
  n_gpu_layers: 99,
  batch_size: 2048,
  ubatch_size: 512,
  cache_type_k: "f16",
  cache_type_v: "f16",
  flash_attn: "on",
  mtp: "off",
  avg_tps: 1000,
  stddev_tps: 5,
  ram_peak_mib: 100,
  vram_peak_mib: 1000,
  ram_avg_mib: 90,
  vram_avg_mib: 900,
  ram_free_before_mib: 8000,
  vram_free_before_mib: 7000,
  system_memory_total_mb: 32000,
  gpu_memory_total_mb: 8192,
  gpu_memory_total_accuracy: "exact" as const,
  gpu_memory_total_source: "driver_reported_memory" as const,
  gpu_memory_free_start_accuracy: "exact" as const,
  gpu_memory_free_start_source: "driver_reported_memory" as const,
  gpu_memory_model_avg_accuracy: "exact" as const,
  gpu_memory_model_avg_source: "driver_reported_memory" as const,
  gpu_memory_model_peak_accuracy: "exact" as const,
  gpu_memory_model_peak_source: "driver_reported_memory" as const,
  gpu_layers_loaded: null,
  total_model_layers: null,
};

function makeRun(id: string, kind: string | null = "runtime", rootId?: string) {
  repo.registerModel({ id: `${id}-model`, filename: `${id}.gguf`, size_bytes: 10, source: "local", metadata: {} });
  repo.createRun(undefined, {
    id,
    root_run_id: rootId ?? id,
    kind: kind as never,
    worker_name: "w",
    llama_cpp_build: "b1",
    llama_cpp_backend: "cpu",
    model_id: `${id}-model`,
    config: { model_id: `${id}-model`, sweep: {} } as never,
    status: "running",
    started_at: Date.now(),
  });
  repo.createRunItems(
    undefined,
    id,
    [0, 1].map((idx) => ({
      idx,
      n_prompt: 512,
      n_gen: 0,
      n_depth: 0,
      concurrency: 1,
      threads: 8,
      n_gpu_layers: 99,
      batch_size: 2048,
      ubatch_size: 512,
      cache_type_k: "f16",
      cache_type_v: "f16",
      flash_attn: "on",
      mtp: idx === 1 ? "on" : "off",
      n_gpu_layers_draft: 0,
      n_cpu_moe: 0,
    }))
  );
}

function terminal(runId: string, idx: number, overrides: Record<string, unknown> = {}) {
  // The llama-server path reports up to one pp row and one tg row per item;
  // the twin join pairs within each test_type separately.
  const mk = (testType: "pp" | "tg") => ({
    ...baseResult,
    test_type: testType,
    method_version: 1,
    prompt_offset: 0,
    ...(idx === 1 ? { mtp: "on", avg_tps: 1500 } : {}),
    ...(overrides as object),
  });
  return repo.recordRunItemTerminal(runId, idx, {
    status: "done",
    results: [mk("pp"), mk("tg")] as never,
  })!;
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
});

afterAll(async () => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

describe("§0.8 twin join and speedup statuses", () => {
  it("pairs a speculative row with its server/off baseline and writes speedup ok on both", () => {
    makeRun("twin-ok");
    terminal("twin-ok", 0); // baseline (runtime run -> engine server)
    terminal("twin-ok", 1, { avg_tps: 1500 });

    const results = repo.getResultsForRun("twin-ok");
    const spec = results.find((r) => r.mtp === "on")!;
    const baseline = results.find((r) => r.mtp !== "on")!;
    expect(spec.speedup_status).toBe("ok");
    expect(spec.speedup).toBeCloseTo(1.5, 6);
    expect(baseline.speedup_status).toBe("ok");
    expect(spec.config_hash).toBe(baseline.config_hash);
    expect(spec.config_hash).toMatch(/^[0-9a-f]{64}$/);
    // Acceptance rate is derived on read; counters stored verbatim.
    expect(spec.engine).toBe("server");
    expect(baseline.engine).toBe("server");
  });

  it("stamps METHOD_VERSION and prompt_offset verbatim from the worker report", () => {
    makeRun("twin-stamp");
    terminal("twin-stamp", 0, { method_version: 1, prompt_offset: 274 });
    const results = repo.getResultsForRun("twin-stamp");
    expect(results[0].method_version).toBe(1);
    expect(results[0].prompt_offset).toBe(274);
  });

  it("marks unverified when the pair's offsets mismatch and writes spec_pair_prompt_mismatch", () => {
    makeRun("twin-offset");
    terminal("twin-offset", 0, { method_version: 1, prompt_offset: 0 });
    terminal("twin-offset", 1, { avg_tps: 2000, method_version: 1, prompt_offset: 137 });
    const results = repo.getResultsForRun("twin-offset");
    const spec = results.find((r) => r.mtp === "on")!;
    const baseline = results.find((r) => r.mtp !== "on")!;
    expect(spec.speedup_status).toBe("unverified");
    expect(baseline.speedup_status).toBe("unverified");
    expect(spec.caveat_flags).toContain("spec_pair_prompt_mismatch");
  });

  it("marks unavailable when no baseline exists yet, then upgrades when the baseline lands later", () => {
    makeRun("twin-late");
    terminal("twin-late", 1, { avg_tps: 1200, method_version: 1, prompt_offset: 0 });
    let spec = repo.getResultsForRun("twin-late").find((r) => r.mtp === "on")!;
    expect(spec.speedup_status).toBe("unavailable");

    terminal("twin-late", 0, { method_version: 1, prompt_offset: 0 });
    const results = repo.getResultsForRun("twin-late");
    spec = results.find((r) => r.mtp === "on")!;
    expect(spec.speedup_status).toBe("ok");
    expect(results.find((r) => r.mtp !== "on")!.speedup_status).toBe("ok");
  });

  it("marks unverified for zero drafted/accepted spec counters and for suspect-only sides", () => {
    makeRun("twin-zero-spec");
    terminal("twin-zero-spec", 0, { method_version: 1, prompt_offset: 0 });
    terminal("twin-zero-spec", 1, { avg_tps: 900, spec_drafted: 0, spec_accepted: 0, method_version: 1, prompt_offset: 0 });
    expect(repo.getResultsForRun("twin-zero-spec").find((r) => r.mtp === "on")!.speedup_status).toBe("unverified");

    makeRun("twin-suspect");
    terminal("twin-suspect", 0, { method_version: 1, prompt_offset: 0, sample_count: 3, suspect_count: 3 });
    terminal("twin-suspect", 1, { avg_tps: 800, method_version: 1, prompt_offset: 0 });
    expect(repo.getResultsForRun("twin-suspect").find((r) => r.mtp === "on")!.speedup_status).toBe("unverified");
  });

  it("marks unverified when the pair's method versions differ", () => {
    makeRun("twin-version");
    terminal("twin-version", 0, { method_version: undefined });
    terminal("twin-version", 1, { avg_tps: 1600 });
    const results = repo.getResultsForRun("twin-version");
    expect(results.find((r) => r.mtp === "on")!.speedup_status).toBe("unverified");
  });

  it("never joins across different roots or depths", () => {
    makeRun("twin-root-a");
    makeRun("twin-root-b");
    // Two separate roots, identical configs: no twin either side.
    terminal("twin-root-a", 1, { avg_tps: 1300, method_version: 1, prompt_offset: 0 });
    terminal("twin-root-b", 0, { method_version: 1, prompt_offset: 0 });
    const a = repo.getResultsForRun("twin-root-a").find((r) => r.mtp === "on")!;
    expect(a.speedup_status).toBe("unavailable");

    // Same root but different n_depth: no join.
    repo.registerModel({ id: "depth-model", filename: "d.gguf", size_bytes: 10, source: "local", metadata: {} });
    repo.createRun(undefined, {
      id: "twin-depth",
      kind: "runtime",
      worker_name: "w",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: "depth-model",
      config: { model_id: "depth-model", sweep: {} } as never,
      status: "running",
      started_at: Date.now(),
    });
    repo.createRunItems(
      undefined,
      "twin-depth",
      [
        { idx: 0, n_prompt: 512, n_gen: 0, n_depth: 4096, concurrency: 1, threads: 8, n_gpu_layers: 99, batch_size: 2048, ubatch_size: 512, cache_type_k: "f16", cache_type_v: "f16", flash_attn: "on", mtp: "off", n_gpu_layers_draft: 0, n_cpu_moe: 0 },
        { idx: 1, n_prompt: 512, n_gen: 0, n_depth: 0, concurrency: 1, threads: 8, n_gpu_layers: 99, batch_size: 2048, ubatch_size: 512, cache_type_k: "f16", cache_type_v: "f16", flash_attn: "on", mtp: "on", n_gpu_layers_draft: 0, n_cpu_moe: 0 },
      ]
    );
    terminal("twin-depth", 0, { n_depth: 4096, method_version: 1, prompt_offset: 0 });
    terminal("twin-depth", 1, { n_depth: 0, avg_tps: 1400, method_version: 1, prompt_offset: 0 });
    const depthSpec = repo.getResultsForRun("twin-depth").find((r) => r.mtp === "on")!;
    expect(depthSpec.speedup_status).toBe("unavailable");
  });
});

describe("skipped items (§0.7)", () => {
  it("skipped items are terminal, do not fail the run, and surface in the error line", () => {
    repo.registerModel({ id: "skip-model", filename: "s.gguf", size_bytes: 10, source: "local", metadata: {} });
    repo.createRun(undefined, {
      id: "skip-run",
      worker_name: "w",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: "skip-model",
      config: { model_id: "skip-model", sweep: {} } as never,
      status: "running",
      started_at: Date.now(),
    });
    repo.createRunItems(
      undefined,
      "skip-run",
      [0, 1].map((idx) => ({
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
      }))
    );
    repo.recordRunItemTerminal("skip-run", 0, { status: "done" });
    const run = repo.recordRunItemTerminal("skip-run", 1, { status: "skipped", error: "flag not supported by this build" })!;
    expect(run.status).toBe("done");
    expect(run.error).toContain("1 of 2 tests skipped");
    const items = repo.getRunItems("skip-run");
    expect(items[1].status).toBe("skipped");
  });
});

describe("M6 thermal columns (storage)", () => {
  it("stores worker-derived clock/temp telemetry verbatim", () => {
    makeRun("thermal-store");
    terminal("thermal-store", 0, {
      gpu_temp_c_max: 89,
      gpu_clock_mhz_min: 1971,
      gpu_clock_samples: [2480, 2479, 2100, 1971],
      caveat_flags: ["thermally_throttled"],
    });
    const row = repo.getResultsForRun("thermal-store")[0];
    expect(row.gpu_temp_c_max).toBe(89);
    expect(row.gpu_clock_mhz_min).toBe(1971);
    expect(row.gpu_clock_samples).toEqual([2480, 2479, 2100, 1971]);
    expect(row.caveat_flags).toEqual(["thermally_throttled"]);
  });
});
