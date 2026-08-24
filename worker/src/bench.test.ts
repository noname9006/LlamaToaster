import { describe, expect, it } from "vitest";
import { parseModelBufferSizes, extractCudaDiagnosticLines, MAX_CUDA_DIAGNOSTIC_LINES } from "./bench.js";

describe("parseModelBufferSizes", () => {
  it("returns null when the line was never seen at all (older build, or failed pre-load)", () => {
    expect(parseModelBufferSizes("some unrelated stderr\nwith no buffer lines\n")).toBeNull();
    expect(parseModelBufferSizes("")).toBeNull();
  });

  it("sums a CPU_Mapped + CUDA0 pair from a genuine full-offload run, attributed to main", () => {
    const stderr = [
      "load_tensors: loading model tensors, type = q4_K",
      "load_tensors: offloading 32 repeating layers to GPU",
      "load_tensors: offloaded 33/33 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size =   137.42 MiB",
      "load_tensors:        CUDA0 model buffer size =  4368.51 MiB",
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result).toEqual({ main: { gpuMib: 4368.51, cpuMib: 137.42 }, draft: null });
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
    expect(result).toEqual({ main: { gpuMib: 0, cpuMib: 5880 }, draft: null });
  });

  it("treats a bare CPU buffer name (non-mmap) the same as CPU_Mapped", () => {
    const stderr = "load_tensors:          CPU model buffer size =   256.00 MiB";
    expect(parseModelBufferSizes(stderr)).toEqual({ main: { gpuMib: 0, cpuMib: 256 }, draft: null });
  });

  it("sums multiple GPU devices from a multi-GPU split", () => {
    const stderr = [
      "load_tensors:        CUDA0 model buffer size =  2000.00 MiB",
      "load_tensors:        CUDA1 model buffer size =  1500.50 MiB",
      "load_tensors:   CPU_Mapped model buffer size =    50.00 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)).toEqual({ main: { gpuMib: 3500.5, cpuMib: 50 }, draft: null });
  });

  it("recognizes non-CUDA backend device names as GPU buffers (ROCm, Vulkan, Metal)", () => {
    expect(parseModelBufferSizes("load_tensors:        ROCm0 model buffer size =  1000.00 MiB")).toEqual({
      main: { gpuMib: 1000, cpuMib: 0 },
      draft: null,
    });
    expect(parseModelBufferSizes("load_tensors:      Vulkan0 model buffer size =  1000.00 MiB")).toEqual({
      main: { gpuMib: 1000, cpuMib: 0 },
      draft: null,
    });
    expect(parseModelBufferSizes("load_tensors:        Metal model buffer size =  1000.00 MiB")).toEqual({
      main: { gpuMib: 1000, cpuMib: 0 },
      draft: null,
    });
  });

  it("a plain -ngl 0 run reports everything on CPU", () => {
    const stderr = "load_tensors:   CPU_Mapped model buffer size =  5880.00 MiB";
    expect(parseModelBufferSizes(stderr)).toEqual({ main: { gpuMib: 0, cpuMib: 5880 }, draft: null });
  });

  it("splits base+draft MTP buffers apart by their preceding offload line", () => {
    // Ordering verified live against the b10605 llama-server binary: each
    // model prints its own "offloaded X/Y" line, then its own buffer lines.
    const stderr = [
      "load_tensors: offloaded 36/36 layers to GPU", // base
      "load_tensors:   CPU_Mapped model buffer size =  1756.00 MiB", // base
      "load_tensors:      Vulkan0 model buffer size =  1290.62 MiB", // base
      "load_tensors: offloaded 5/5 layers to GPU", // draft
      "load_tensors:   CPU_Mapped model buffer size =    68.00 MiB", // draft
      "load_tensors:      Vulkan0 model buffer size =    78.25 MiB", // draft
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result).toEqual({
      main: { gpuMib: 1290.62, cpuMib: 1756 },
      draft: { gpuMib: 78.25, cpuMib: 68 },
    });
  });

  it("tolerates llama-server's logger prefix on every line", () => {
    const stderr = [
      "0.02.638.073 I load_tensors: offloaded 36/36 layers to GPU",
      "0.02.638.078 I load_tensors:   CPU_Mapped model buffer size =  1756.00 MiB",
      "0.02.638.079 I load_tensors:      Vulkan0 model buffer size =  1290.62 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)).toEqual({
      main: { gpuMib: 1290.62, cpuMib: 1756 },
      draft: null,
    });
  });
});

describe("extractCudaDiagnosticLines", () => {
  it("pulls device enumeration, claim, buffer split and allocation failures from realistic stderr while skipping per-tensor noise", () => {
    const stderr = [
      "ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no",
      "ggml_cuda_init: found 1 CUDA devices:",
      "  Device 0: NVIDIA GeForce RTX 3090, compute capability (8, 6), VMM: yes",
      "load_tensors: tensor 'blk.0.attn_q.weight' loaded in 1.23 ms", // noise
      "load_tensors: offloaded 43/43 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size = 14341.00 MiB",
      "llm_layer_count: 42", // noise
      "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 14868.00 MiB on device 0: out of memory",
    ].join("\n");
    expect(extractCudaDiagnosticLines(stderr)).toEqual([
      "ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no",
      "ggml_cuda_init: found 1 CUDA devices:",
      "Device 0: NVIDIA GeForce RTX 3090, compute capability (8, 6), VMM: yes",
      "load_tensors: offloaded 43/43 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size = 14341.00 MiB",
      "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 14868.00 MiB on device 0: out of memory",
    ]);
  });

  it("matches llama-server's logger-prefixed diagnostic lines too", () => {
    const stderr = [
      "0.02.100 I ggml_cuda_init: found 1 CUDA devices:",
      "0.02.101 I load_tensors: offloaded 5/5 layers to GPU",
    ].join("\n");
    expect(extractCudaDiagnosticLines(stderr)).toEqual([
      "0.02.100 I ggml_cuda_init: found 1 CUDA devices:",
      "0.02.101 I load_tensors: offloaded 5/5 layers to GPU",
    ]);
  });

  it("returns nothing for a clean transcript with no matching lines", () => {
    expect(extractCudaDiagnosticLines("some unrelated line\nanother one\n")).toEqual([]);
    expect(extractCudaDiagnosticLines("")).toEqual([]);
  });

  it("caps output with an elision marker when over the limit", () => {
    const stderr = Array.from({ length: MAX_CUDA_DIAGNOSTIC_LINES + 4 }, (_, i) => `ggml_cuda_init: line ${i}`).join("\n");
    const result = extractCudaDiagnosticLines(stderr);
    expect(result).toHaveLength(MAX_CUDA_DIAGNOSTIC_LINES + 1);
    expect(result[MAX_CUDA_DIAGNOSTIC_LINES]).toBe(
      "…[4 more diagnostic lines elided -- see the raw JSON dump]"
    );
  });

  it("catches host-fallback wording and CUDA API names", () => {
    const stderr = [
      "load_tensors: falling back to CPU buffer for blk.0",
      "cudaMalloc failed: unspecified driver error",
      "cuMemCreate returned CUDA_ERROR_OUT_OF_MEMORY",
    ].join("\n");
    expect(extractCudaDiagnosticLines(stderr)).toEqual(stderr.split("\n"));
  });
});
