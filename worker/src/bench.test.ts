import { describe, expect, it } from "vitest";
import { parseModelBufferSizes } from "./bench.js";

describe("parseModelBufferSizes", () => {
  it("returns null when the line was never seen at all (older build, or failed pre-load)", () => {
    expect(parseModelBufferSizes("some unrelated stderr\nwith no buffer lines\n")).toBeNull();
    expect(parseModelBufferSizes("")).toBeNull();
  });

  it("sums a CPU_Mapped + CUDA0 pair from a genuine full-offload run", () => {
    const stderr = [
      "load_tensors: loading model tensors, type = q4_K",
      "load_tensors: offloading 32 repeating layers to GPU",
      "load_tensors: offloaded 33/33 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size =   137.42 MiB",
      "load_tensors:        CUDA0 model buffer size =  4368.51 MiB",
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result).toEqual({ gpuMib: 4368.51, cpuMib: 137.42 });
  });

  it("reproduces the diagnosed sysmem-fallback fingerprint: claimed GPU, everything actually landed on CPU_Mapped", () => {
    // The scheduler-level equivalent of run 2bf0ce32 item 1 (33/33 claimed,
    // ~0 actually resident) -- here the allocator itself never got a GPU
    // buffer at all, which is the case this ground-truth check exists to
    // catch even when the external VRAM sampler's own reading is unreliable.
    const stderr = [
      "load_tensors: offloaded 33/33 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size =  5880.00 MiB",
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result).toEqual({ gpuMib: 0, cpuMib: 5880 });
  });

  it("treats a bare CPU buffer name (non-mmap) the same as CPU_Mapped", () => {
    const stderr = "load_tensors:          CPU model buffer size =   256.00 MiB";
    expect(parseModelBufferSizes(stderr)).toEqual({ gpuMib: 0, cpuMib: 256 });
  });

  it("sums multiple GPU devices from a multi-GPU split", () => {
    const stderr = [
      "load_tensors:        CUDA0 model buffer size =  2000.00 MiB",
      "load_tensors:        CUDA1 model buffer size =  1500.50 MiB",
      "load_tensors:   CPU_Mapped model buffer size =    50.00 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)).toEqual({ gpuMib: 3500.5, cpuMib: 50 });
  });

  it("recognizes non-CUDA backend device names as GPU buffers (ROCm, Vulkan, Metal)", () => {
    expect(parseModelBufferSizes("load_tensors:        ROCm0 model buffer size =  1000.00 MiB")).toEqual({
      gpuMib: 1000,
      cpuMib: 0,
    });
    expect(parseModelBufferSizes("load_tensors:      Vulkan0 model buffer size =  1000.00 MiB")).toEqual({
      gpuMib: 1000,
      cpuMib: 0,
    });
    expect(parseModelBufferSizes("load_tensors:        Metal model buffer size =  1000.00 MiB")).toEqual({
      gpuMib: 1000,
      cpuMib: 0,
    });
  });

  it("a plain -ngl 0 run reports everything on CPU", () => {
    const stderr = "load_tensors:   CPU_Mapped model buffer size =  5880.00 MiB";
    expect(parseModelBufferSizes(stderr)).toEqual({ gpuMib: 0, cpuMib: 5880 });
  });

  it("aggregates base+draft MTP buffers together rather than separating them", () => {
    // Known, documented simplification -- see parseModelBufferSizes's own
    // comment. The draft's tiny buffer just adds a little noise, never masks
    // a real base-model shortfall.
    const stderr = [
      "load_tensors:        CUDA0 model buffer size =  4000.00 MiB", // base
      "load_tensors:        CUDA0 model buffer size =    50.00 MiB", // draft
    ].join("\n");
    expect(parseModelBufferSizes(stderr)).toEqual({ gpuMib: 4050, cpuMib: 0 });
  });
});
