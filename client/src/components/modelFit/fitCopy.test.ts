import { describe, expect, it } from "vitest";
import type { ModelFitResult } from "../../types";
import { fitDetails, fitHeadline, headerDownloadMib } from "./fitCopy";

const result = (over: Partial<ModelFitResult> = {}, point: Partial<NonNullable<ModelFitResult["point"]>> = {}): ModelFitResult => ({
  target_key: "hf:u/r/m.gguf",
  ctx: 1024,
  margin_mib: 1024,
  llama_cpp_build: "b11840",
  backend: "vulkan",
  source: "header",
  memory: { devices: [{ name: "Vulkan0", description: "GPU", total_mib: 8176, free_mib: 7378 }], ram_available_mib: 29000, ram_total_mib: 49000, unified: false },
  point: {
    ctx: 1024, fa: "on", ctk: "f16", ctv: "f16", margin_mib: 1024, verdict: "partial", reason: null, ngl: 31, layers_gpu: 31, layers_total: 31,
    overflow_layers: 25, ot: null, dev_used_mib: 6203, dev_free_mib: 1174, host_used_mib: 11471, total_mib: 17674, kv_mib: 220, compute_mib: 560,
    need_all_gpu_mib: 16903, splits: 1, fa_disabled: false, n_expert: 128, n_expert_used: 8, inferred: false, raw_args: null, error: null, ...point,
  },
  error: null,
  fetched_bytes: 16_000_000,
  level: "partial",
  created_at: 1,
  ...over,
});

describe("fitHeadline", () => {
  it("names the memory that holds a model that fits", () => {
    expect(fitHeadline(result({ level: "fits" }))).toBe("Fits in VRAM");
    expect(fitHeadline(result({ level: "fits", memory: { ...result().memory, unified: true } }))).toBe("Fits in unified memory");
    expect(fitHeadline(result({ level: "fits", memory: { ...result().memory, devices: [] } }))).toBe("Fits in RAM");
  });

  it("says what is partial about a partial offload", () => {
    expect(fitHeadline(result())).toContain("experts of 25 layers in RAM");
    expect(fitHeadline(result({}, { layers_gpu: 12, overflow_layers: 0 }))).toContain("12 of 31 layers on the GPU");
    expect(fitHeadline(result({}, { layers_gpu: 0, overflow_layers: 0 }))).toContain("run on the CPU");
  });

  it("warns that a model without enough memory may run badly or not at all", () => {
    expect(fitHeadline(result({ level: "no_fit" }))).toBe("Not enough memory: will not run, or will run with degraded speed");
  });

  it("does not present a failed check as a verdict", () => {
    expect(fitHeadline(result({ level: null, point: null, error: "401" }))).toBe("The check could not finish");
  });
});

describe("fitDetails", () => {
  it("states need against available on both sides, and how it was sized", () => {
    const lines = fitDetails(result());
    expect(lines[0]).toBe("VRAM: needs 6.1 GB of 7.2 GB available, keeping 1.0 GB free");
    expect(lines[1]).toBe("RAM: needs 11.2 GB of 28.3 GB available");
    expect(lines).toContain("Context: 1,024 tokens");
    expect(lines).toContain("llama.cpp b11840 (vulkan)");
    expect(lines.at(-1)).toMatch(/header.*downloaded, then deleted/);
  });

  it("has no VRAM line without a GPU, and shows the context actually sized", () => {
    const lines = fitDetails(result({ memory: { ...result().memory, devices: [] } }, { ctx: 512 }));
    expect(lines.some((l) => l.startsWith("VRAM"))).toBe(false);
    expect(lines).toContain("Context: 512 tokens");
  });
});

describe("headerDownloadMib", () => {
  it("charges the tokenizer once for a split model", () => {
    expect(headerDownloadMib(1)).toBe(15);
    expect(headerDownloadMib(3)).toBe(17);
  });
});
