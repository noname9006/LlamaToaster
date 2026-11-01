import { describe, expect, it } from "vitest";
import { fitLevel, fitTargetKey, type FitMemory } from "./fitLevel.js";
import type { FitPoint } from "./fitParams.js";

const point = (over: Partial<FitPoint>): FitPoint => ({
  ctx: 1024,
  fa: "on",
  ctk: "f16",
  ctv: "f16",
  margin_mib: 1024,
  verdict: "full",
  reason: null,
  ngl: -1,
  layers_gpu: 31,
  layers_total: 31,
  overflow_layers: 0,
  ot: null,
  dev_used_mib: 4000,
  dev_free_mib: 3000,
  host_used_mib: 500,
  total_mib: 4500,
  kv_mib: 100,
  compute_mib: 100,
  need_all_gpu_mib: 4000,
  splits: 1,
  fa_disabled: false,
  n_expert: null,
  n_expert_used: null,
  inferred: false,
  raw_args: null,
  error: null,
  ...over,
});

const gpu: FitMemory = {
  devices: [{ name: "Vulkan0", description: "RX 6600 XT", total_mib: 8176, free_mib: 7378 }],
  ram_available_mib: 29000,
  ram_total_mib: 49000,
  unified: false,
};

describe("fitLevel", () => {
  it("full offload on a discrete GPU fits", () => {
    expect(fitLevel(point({}), gpu)).toBe("fits");
  });

  it("MoE experts in RAM are partial even with every layer on the GPU", () => {
    // Measured: gemma-4-26B UD-Q4_K_M on an RX 6600 XT -- 31/31 layers, 25 overflowing.
    const moe = point({ verdict: "partial", overflow_layers: 25, dev_used_mib: 6203, host_used_mib: 11471 });
    expect(fitLevel(moe, gpu)).toBe("partial");
  });

  it("some layers on the CPU is partial while RAM holds them", () => {
    expect(fitLevel(point({ verdict: "partial", layers_gpu: 12, host_used_mib: 9000 }), gpu)).toBe("partial");
  });

  it("does not fit when the RAM side exceeds what is available", () => {
    expect(fitLevel(point({ verdict: "partial", layers_gpu: 12, host_used_mib: 30000 }), gpu)).toBe("no_fit");
    expect(fitLevel(point({ host_used_mib: 30000 }), gpu)).toBe("no_fit");
  });

  it("a GPU too small for the margin can still run the model on the CPU", () => {
    const small = point({ verdict: "doesnt_fit", reason: "vram", layers_gpu: 0, dev_used_mib: 200, dev_free_mib: 300, host_used_mib: 9000 });
    expect(fitLevel(small, gpu)).toBe("partial");
    expect(fitLevel({ ...small, host_used_mib: 40000 }, gpu)).toBe("no_fit");
  });

  it("no layers on the GPU is still not enough when the device cannot hold the rest", () => {
    // Measured: gemma-4-26B BF16 with a game holding the card -- 0 layers, needs 2.4 GB, 0.5 GB there.
    const p = point({ verdict: "doesnt_fit", reason: "vram", layers_gpu: 0, dev_used_mib: 2458, dev_free_mib: -1945, host_used_mib: 9000 });
    expect(fitLevel(p, gpu)).toBe("no_fit");
  });

  it("layers placed with the device over budget do not fit", () => {
    expect(fitLevel(point({ verdict: "doesnt_fit", reason: "vram", layers_gpu: 5, dev_free_mib: -200 }), gpu)).toBe("no_fit");
    expect(fitLevel(point({ verdict: "doesnt_fit", reason: "vram", layers_gpu: 5, dev_free_mib: 300 }), gpu)).toBe("partial");
  });

  it("a machine with no GPU has one pool and no partial level", () => {
    const cpu: FitMemory = { ...gpu, devices: [] };
    const p = point({ verdict: "cpu_only", layers_gpu: 0, dev_used_mib: 0, host_used_mib: 9000 });
    expect(fitLevel(p, cpu)).toBe("fits");
    expect(fitLevel({ ...p, host_used_mib: 30000 }, cpu)).toBe("no_fit");
  });

  it("falls back to fit's own verdict with no RAM reading on a CPU machine", () => {
    const cpu: FitMemory = { ...gpu, devices: [], ram_available_mib: null };
    expect(fitLevel(point({ verdict: "cpu_only" }), cpu)).toBe("fits");
    expect(fitLevel(point({ verdict: "doesnt_fit", reason: "ram" }), cpu)).toBe("no_fit");
  });

  it("unified memory judges both sides against one pool, trusting the larger reading", () => {
    // macOS: free RAM under-reports, Metal's budget is what counts.
    const mac: FitMemory = {
      devices: [{ name: "Metal", description: "Apple M3", total_mib: 24000, free_mib: 21000 }],
      ram_available_mib: 3000,
      ram_total_mib: 32000,
      unified: true,
    };
    expect(fitLevel(point({ dev_used_mib: 15000, host_used_mib: 500 }), mac)).toBe("fits");
    expect(fitLevel(point({ verdict: "partial", dev_used_mib: 15000, host_used_mib: 4000 }), mac)).toBe("partial");
    expect(fitLevel(point({ verdict: "partial", dev_used_mib: 15000, host_used_mib: 9000 }), mac)).toBe("no_fit");
  });

  it("gives no level for an error", () => {
    expect(fitLevel(point({ verdict: "error" }), gpu)).toBeNull();
  });
});

describe("fitTargetKey", () => {
  it("shares one key for the same Hugging Face file", () => {
    expect(fitTargetKey({ id: "abc", hf_repo: "u/r", hf_file: "BF16/m-00001-of-00002.gguf" })).toBe("hf:u/r/BF16/m-00001-of-00002.gguf");
    expect(fitTargetKey({ hf_repo: "u/r", hf_file: "BF16/m-00001-of-00002.gguf" })).toBe("hf:u/r/BF16/m-00001-of-00002.gguf");
  });

  it("uses the id for a model with no Hugging Face source", () => {
    expect(fitTargetKey({ id: "abc" })).toBe("model:abc");
    expect(fitTargetKey({})).toBeNull();
  });
});
