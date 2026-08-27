import { describe, expect, it } from "vitest";
import {
  checkComparisonFairness,
  gridSignature,
  paretoFrontier,
  MAX_COMPARISON_MEMBERS,
  type ComparisonFairnessFacts,
  type ComparisonMemberRow,
} from "./comparison.js";

const reference: ComparisonFairnessFacts = {
  worker_id: "toaster-01",
  llama_cpp_build: "b10516",
  llama_cpp_backend: "vulkan",
  backend_device_name: "AMD Radeon RX 6600 XT",
  repeats: 5,
  method_version: 1,
  grid_signature: "flash_attn=on;n_gpu_layers=16",
};

describe("N3 fairness rules (blocking)", () => {
  it("passes an identical member", () => {
    expect(checkComparisonFairness(reference, { ...reference })).toEqual([]);
  });

  it("fails a mid-group build swap loudly", () => {
    const violations = checkComparisonFairness(reference, { ...reference, llama_cpp_build: "b10600" });
    expect(violations).toHaveLength(1);
    expect(violations[0].field).toBe("llama_cpp_build");
    expect(violations[0].message).toContain("llama.cpp build differs");
  });

  it("fails a different machine, backend, GPU, repeats or methodology version", () => {
    const cases: [keyof ComparisonFairnessFacts, string | number][] = [
      ["worker_id", "toaster-02"],
      ["llama_cpp_backend", "cuda"],
      ["backend_device_name", "NVIDIA RTX 4090"],
      ["repeats", 3],
      ["method_version", 2],
      ["grid_signature", "flash_attn=off;n_gpu_layers=16"],
    ];
    for (const [field, value] of cases) {
      const violations = checkComparisonFairness(reference, { ...reference, [field]: value });
      expect(violations.map((v) => v.field)).toEqual([field]);
    }
  });

  it("treats a not-yet-known field as not-yet-drift, never as a violation", () => {
    // method_version is only known once rows land -- the per-member re-check
    // is what catches it then.
    expect(checkComparisonFairness({ ...reference, method_version: null }, reference)).toEqual([]);
    expect(checkComparisonFairness(reference, { ...reference, method_version: null })).toEqual([]);
  });

  it("reports every drifted field, not just the first", () => {
    const violations = checkComparisonFairness(reference, {
      ...reference,
      llama_cpp_build: "b1",
      repeats: 1,
    });
    expect(violations.map((v) => v.field).sort()).toEqual(["llama_cpp_build", "repeats"]);
  });

  it("caps a comparison at five models", () => {
    expect(MAX_COMPARISON_MEMBERS).toBe(5);
  });
});

describe("grid signature", () => {
  it("ignores key order and the model, so two identical grids match", () => {
    const a = gridSignature({ model_id: "m1", flash_attn: ["on"], n_gpu_layers: [16, 8] });
    const b = gridSignature({ n_gpu_layers: [8, 16], flash_attn: ["on"], model_id: "m2" });
    expect(a).toBe(b);
  });

  it("ignores repeats, which is checked as its own field", () => {
    expect(gridSignature({ flash_attn: ["on"], repeats: 5 })).toBe(gridSignature({ flash_attn: ["on"], repeats: 3 }));
  });

  it("distinguishes a genuinely different grid", () => {
    expect(gridSignature({ flash_attn: ["on"] })).not.toBe(gridSignature({ flash_attn: ["off"] }));
  });
});

describe("Pareto scatter across models", () => {
  const row = (over: Partial<ComparisonMemberRow>): ComparisonMemberRow => ({
    run_id: "r",
    model_id: "m",
    model_filename: "m.gguf",
    quant_label: null,
    file_sha256: null,
    status: "done",
    pp: null,
    tg: null,
    vram_peak_mib: null,
    ram_peak_mib: null,
    ppl: null,
    kld_vs_baseline: null,
    dataset_hash: null,
    verified_ctx_tokens: null,
    violations: [],
    ...over,
  });

  it("marks a config that is both slower and hungrier as dominated", () => {
    const points = paretoFrontier([
      row({ model_id: "q4", quant_label: "Q4_K_M", tg: 45, vram_peak_mib: 4000, ram_peak_mib: 0 }),
      row({ model_id: "q5", quant_label: "Q5_K_M", tg: 40, vram_peak_mib: 5000, ram_peak_mib: 0 }),
      row({ model_id: "q6", quant_label: "Q6_K", tg: 38, vram_peak_mib: 6000, ram_peak_mib: 0 }),
    ]);
    expect(points.find((p) => p.model_id === "q4")!.dominated).toBe(false);
    expect(points.find((p) => p.model_id === "q5")!.dominated).toBe(true);
    expect(points.find((p) => p.model_id === "q6")!.dominated).toBe(true);
  });

  it("keeps a genuine trade-off on the frontier", () => {
    const points = paretoFrontier([
      row({ model_id: "fast", tg: 45, vram_peak_mib: 6000, ram_peak_mib: 0 }),
      row({ model_id: "small", tg: 30, vram_peak_mib: 3000, ram_peak_mib: 0 }),
    ]);
    expect(points.every((p) => !p.dominated)).toBe(true);
  });

  it("skips members with nothing measured rather than plotting a zero", () => {
    expect(paretoFrontier([row({ model_id: "pending", tg: null })])).toHaveLength(0);
  });
});
