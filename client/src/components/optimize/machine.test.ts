import { describe, expect, it } from "vitest";
import { modelVerdict, type FitAnswer } from "./machine";
import type { Model, Worker } from "../../types";

const MiB = 1048576;

// Numbers from llama-fit-params (Vulkan b9058) on a real Qwen3.5-9B-PRISM-DQ:
// 4009 MiB of weights on the GPU, ~416 MiB (the untied token embedding) in RAM.
const model = {
  id: "m",
  filename: "q.gguf",
  size_bytes: 4425 * MiB,
  source: "huggingface",
  metadata: {
    n_layer: 2,
    tensor_layer_bytes: { dense: [2000 * MiB, 2009 * MiB], moe: [0, 0], embed: 416 * MiB, output: 416 * MiB, other: 0 },
    kv_layout: { full_k: 8192, full_v: 8192, swa_k: 0, swa_v: 0, window: 0, vocab: 248320 },
  },
} as unknown as Model;

function worker(over: { vramMib: number; ramMib: number; backend?: string }): Worker {
  return {
    id: "w",
    status: "online",
    backend: over.backend ?? "vulkan",
    vram: { vram_free_before_mib: over.vramMib },
    hardware: {
      gpu: [{ name: "gpu", vram_mb: over.vramMib, backend: over.backend ?? "vulkan" }],
      mem_total_bytes: over.ramMib * MiB,
    },
  } as unknown as Worker;
}

describe("modelVerdict -- VRAM and RAM pools", () => {
  it("fits: reports both pools, with the embedding counted in RAM not VRAM", () => {
    const v = modelVerdict(model, worker({ vramMib: 8000, ramMib: 16000 }), 8192)!;
    expect(v.icon).toBe("ok");
    expect(v.text).toMatch(/^Fits · VRAM .* \/ .* · RAM .* \/ .*/);
  });

  // 3000 MiB free - fit's 1024 MiB margin: the output layer (416 MiB + the
  // ~500 MiB compute buffer) fits, a 2 GB block does not.
  it("over VRAM: says how many layers llama.cpp would put on the GPU", () => {
    const v = modelVerdict(model, worker({ vramMib: 3000, ramMib: 16000 }), 8192)!;
    expect(v.icon).toBe("spill");
    expect(v.text).toMatch(/^1 of 3 layers on the GPU at 8k · RAM .* \(estimate\)$/);
    expect(v.tone).toBe("text-warning");
  });

  it("over VRAM and the blocks left on the CPU do not fit RAM -> hard no", () => {
    const v = modelVerdict(model, worker({ vramMib: 3000, ramMib: 2500 }), 8192)!;
    expect(v.icon).toBe("oom");
    expect(v.text).toMatch(/^Too big for RAM/);
  });

  it("fits VRAM but the CPU-side tensors do not fit RAM", () => {
    const v = modelVerdict(model, worker({ vramMib: 8000, ramMib: 400 }), 8192)!;
    expect(v.icon).toBe("oom");
    expect(v.text).toMatch(/^Too big for RAM/);
  });

  it("a llama-fit-params answer replaces the estimate and says so", () => {
    const fit: FitAnswer = { layers_gpu: 3, layers_total: 3, overflow_layers: 0, dev_used_mib: 5000, host_used_mib: 500, source: "fit" };
    const v = modelVerdict(model, worker({ vramMib: 3000, ramMib: 16000 }), 8192, fit)!;
    expect(v.icon).toBe("ok");
    expect(v.text).toMatch(/^Fits · VRAM .* · llama\.cpp fit$/);
  });

  it("MoE: every layer on the GPU with some experts in RAM", () => {
    const fit: FitAnswer = { layers_gpu: 41, layers_total: 41, overflow_layers: 34, dev_used_mib: 6191, host_used_mib: 15801, source: "fit" };
    const v = modelVerdict(model, worker({ vramMib: 7378, ramMib: 32000 }), 32768, fit)!;
    expect(v.icon).toBe("spill");
    expect(v.text).toMatch(/^Fits with 34 layers' experts in RAM/);
  });

  it("a CPU-only machine needs the whole model in RAM", () => {
    const w = worker({ vramMib: 0, ramMib: 6500, backend: "cpu" });
    expect(modelVerdict(model, w, 8192)!.icon).toBe("chip");
    expect(modelVerdict(model, worker({ vramMib: 0, ramMib: 5000, backend: "cpu" }), 8192)!.icon).toBe("oom");
  });
});
