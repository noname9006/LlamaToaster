import { describe, expect, it } from "vitest";
import {
  BUNDLE_FORMAT,
  BUNDLE_FORMAT_VERSION,
  methodsFor,
  stampConfigHash,
  validateBundle,
  vramClass,
  type Bundle,
  type BundleRow,
} from "./exchange.js";
import {
  CURVE_METHOD_VERSION,
  LEGACY_CURVE_METHOD_VERSION,
  METHOD_VERSION,
  SERVER_METHOD_VERSION,
} from "./types.js";

function bundleRow(over: Partial<BundleRow> = {}): BundleRow {
  return stampConfigHash({
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
    engine: "bench",
    concurrency: 1,
    method_version: 1,
    avg_tps: 41,
    stddev_tps: 0.4,
    sample_count: 5,
    suspect_count: 0,
    vram_peak_mib: 4000,
    ram_peak_mib: 2000,
    ttft_ms_p50: null,
    ttft_ms_p95: null,
    ttft_n: null,
    e2e_ms_mean: null,
    gpu_temp_c_max: 78,
    gpu_clock_mhz_min: 2480,
    caveat_flags: [],
    created_at: 1_000,
    ...over,
  });
}

function bundle(rows: BundleRow[]): Bundle {
  return {
    format: BUNDLE_FORMAT,
    format_version: BUNDLE_FORMAT_VERSION,
    exported_at: 2_000,
    run: {
      id: "r1",
      root_run_id: "r1",
      kind: "sweep",
      model_filename: "model.gguf",
      model_quant: "Q4_K_M",
      llama_cpp_build: "b10516",
      config: { goals: { goal: "max_context", target_ctx: 32_768, workload: "chat" } },
    },
    hardware_class: { gpu_name: "AMD Radeon RX 6600 XT", backend: "vulkan", vram_class_mib: 8192, cpu_isa: "AVX AVX2" },
    rows,
    quality_rows: [],
    method_versions: [1],
    methods: [methodsFor(1)],
  } as Bundle;
}

describe("N7 export bundle", () => {
  it("recomputes config_hash from the row's canonical form, not from a stored column", () => {
    const row = bundleRow();
    expect(row.config_hash).toMatch(/^[0-9a-f]{64}$/);
    // A spec field is not part of §0.4's canonical form, so it cannot change
    // the hash even when carried alongside.
    const withSpec = stampConfigHash({ ...row, avg_tps: row.avg_tps } as BundleRow);
    expect(withSpec.config_hash).toBe(row.config_hash);
  });

  it("carries the goals block so the export stays reproducible without local defaults", () => {
    const exported = bundle([bundleRow()]);
    expect(exported.run.config.goals).toMatchObject({ goal: "max_context", target_ctx: 32_768 });
  });

  it("buckets VRAM into a class rather than exporting an exact fingerprint", () => {
    expect(vramClass(8192)).toBe(8192);
    expect(vramClass(7900)).toBe(8192);
    expect(vramClass(24_000)).toBe(24_576);
    expect(vramClass(null)).toBeNull();
    expect(vramClass(0)).toBeNull();
  });

  it("embeds a methods section keyed to the method version", () => {
    expect(methodsFor(METHOD_VERSION).pipeline.join(" ")).toContain("stddev");
    expect(methodsFor(CURVE_METHOD_VERSION).summary).toContain("cold timed prefill");
    expect(methodsFor(CURVE_METHOD_VERSION).pipeline.join(" ")).toContain("cache_evicted");
    // Server-measured rows say so, and say what the prompt was.
    expect(methodsFor(SERVER_METHOD_VERSION).pipeline.join(" ")).toContain("mixed-register");
    expect(methodsFor(SERVER_METHOD_VERSION).pipeline.join(" ")).toContain("grammar_constrained");
    // A stored row from before the filler rewrite keeps its OWN description --
    // relabelling it with today's pipeline would misdescribe shared data.
    expect(methodsFor(LEGACY_CURVE_METHOD_VERSION).method_version).toBe(LEGACY_CURVE_METHOD_VERSION);
    expect(methodsFor(LEGACY_CURVE_METHOD_VERSION).summary).toContain("before the filler prompt was rewritten");
    expect(methodsFor(LEGACY_CURVE_METHOD_VERSION).pipeline.join(" ")).toContain("not comparable");
    // llama-bench rows were never affected by the filler change; their section
    // says why nothing about the prompt is ours to control.
    expect(methodsFor(METHOD_VERSION).pipeline.join(" ")).toContain("builds its own prompt");
  });
});

describe("N7 import validation", () => {
  it("round-trips a clean bundle byte-for-byte on the hashes", () => {
    const exported = bundle([bundleRow(), bundleRow({ test_type: "pp", avg_tps: 1400 })]);
    const roundTripped = JSON.parse(JSON.stringify(exported)) as unknown;
    const validation = validateBundle(roundTripped);
    expect(validation.ok).toBe(true);
    expect(validation.rows.every((r) => r.ok)).toBe(true);
    expect(validation.acceptedRows.map((r) => r.config_hash)).toEqual(exported.rows.map((r) => r.config_hash));
  });

  it("tampering one field rejects exactly that row and imports the rest", () => {
    const exported = bundle([bundleRow(), bundleRow({ test_type: "pp", avg_tps: 1400 })]);
    const tampered = JSON.parse(JSON.stringify(exported)) as Bundle;
    tampered.rows[0].n_gpu_layers = 8; // hash no longer matches the canonical form
    const validation = validateBundle(tampered);
    expect(validation.rows[0].ok).toBe(false);
    expect(validation.rows[0].reason).toContain("altered after export");
    expect(validation.rows[1].ok).toBe(true);
    expect(validation.acceptedRows).toHaveLength(1);
  });

  it("surfaces mixed vintages instead of silently averaging them", () => {
    const validation = validateBundle(
      bundle([bundleRow({ method_version: 1 }), bundleRow({ method_version: 2, n_prompt: 8192 })])
    );
    expect(validation.mixedVintages).toBe(true);
    expect(validation.methodVersions.sort()).toEqual([1, 2]);
  });

  it("refuses an envelope it does not understand, without pretending to import rows", () => {
    expect(validateBundle({ format: "something-else" }).fatal).toContain("not a");
    expect(validateBundle({ format: BUNDLE_FORMAT, format_version: 99, rows: [] }).fatal).toContain("newer than");
    expect(validateBundle(null).fatal).toContain("must be an object");
  });

  it("rejects a structurally malformed row on its own terms", () => {
    const exported = bundle([bundleRow()]);
    const broken = JSON.parse(JSON.stringify(exported)) as Bundle;
    (broken.rows[0] as unknown as Record<string, unknown>).avg_tps = "fast";
    const validation = validateBundle(broken);
    expect(validation.rows[0].ok).toBe(false);
    expect(validation.rows[0].reason).toContain("avg_tps");
  });
});
