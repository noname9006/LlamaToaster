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
    expect(result).toEqual({
      main: { gpuMib: 4368.51, cpuMib: 137.42, gpu_layers_exact: null },
      draft: null,
    });
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
    expect(result).toEqual({
      main: { gpuMib: 0, cpuMib: 5880, gpu_layers_exact: null },
      draft: null,
    });
  });

  it("treats a bare CPU buffer name (non-mmap) the same as CPU_Mapped", () => {
    const stderr = "load_tensors:          CPU model buffer size =   256.00 MiB";
    expect(parseModelBufferSizes(stderr)).toEqual({
      main: { gpuMib: 0, cpuMib: 256, gpu_layers_exact: null },
      draft: null,
    });
  });

  it("sums multiple GPU devices from a multi-GPU split", () => {
    const stderr = [
      "load_tensors:        CUDA0 model buffer size =  2000.00 MiB",
      "load_tensors:        CUDA1 model buffer size =  1500.50 MiB",
      "load_tensors:   CPU_Mapped model buffer size =    50.00 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)).toEqual({
      main: { gpuMib: 3500.5, cpuMib: 50, gpu_layers_exact: null },
      draft: null,
    });
  });

  it("recognizes non-CUDA backend device names as GPU buffers (ROCm, Vulkan, Metal)", () => {
    expect(parseModelBufferSizes("load_tensors:        ROCm0 model buffer size =  1000.00 MiB")).toEqual({
      main: { gpuMib: 1000, cpuMib: 0, gpu_layers_exact: null },
      draft: null,
    });
    expect(parseModelBufferSizes("load_tensors:      Vulkan0 model buffer size =  1000.00 MiB")).toEqual({
      main: { gpuMib: 1000, cpuMib: 0, gpu_layers_exact: null },
      draft: null,
    });
    expect(parseModelBufferSizes("load_tensors:        Metal model buffer size =  1000.00 MiB")).toEqual({
      main: { gpuMib: 1000, cpuMib: 0, gpu_layers_exact: null },
      draft: null,
    });
  });

  it("a plain -ngl 0 run reports everything on CPU", () => {
    const stderr = "load_tensors:   CPU_Mapped model buffer size =  5880.00 MiB";
    expect(parseModelBufferSizes(stderr)).toEqual({
      main: { gpuMib: 0, cpuMib: 5880, gpu_layers_exact: null },
      draft: null,
    });
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
      main: { gpuMib: 1290.62, cpuMib: 1756, gpu_layers_exact: null },
      draft: { gpuMib: 78.25, cpuMib: 68, gpu_layers_exact: null },
    });
  });

  it("tolerates llama-server's logger prefix on every line", () => {
    const stderr = [
      "0.02.638.073 I load_tensors: offloaded 36/36 layers to GPU",
      "0.02.638.078 I load_tensors:   CPU_Mapped model buffer size =  1756.00 MiB",
      "0.02.638.079 I load_tensors:      Vulkan0 model buffer size =  1290.62 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)).toEqual({
      main: { gpuMib: 1290.62, cpuMib: 1756, gpu_layers_exact: null },
      draft: null,
    });
  });

  // --- exact per-layer assignment counting (llama.cpp DEBUG lines) ---

  function layerLines(n: number, deviceFor: (i: number) => string): string[] {
    return Array.from({ length: n }, (_, i) => `load_tensors: layer ${String(i).padStart(3)} assigned to device ${deviceFor(i)}, is_swa = 0`);
  }

  it("counts EXACT non-CPU layer assignments for a partial offload", () => {
    // b10612 prints one of these per layer at DEBUG level (which llama-bench's
    // -v enables): 30 CPU + 10 CUDA0 = exactly 10 resident, no ratios.
    const devices = (i: number) => (i < 20 ? "CPU" : "CUDA0");
    const stderr = [
      ...layerLines(30, devices),
      "load_tensors: offloaded 10/31 layers to GPU",
      "load_tensors:        CUDA0 model buffer size =  1500.00 MiB",
      "load_tensors:   CPU_Mapped model buffer size =  3000.00 MiB",
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result?.main).toEqual({ gpuMib: 1500, cpuMib: 3000, gpu_layers_exact: 10 });
  });

  it("resolves the MX150 driver-mismatch case to an exact 0: claimed 49/49 but every layer assigned to CPU", () => {
    // The real-world shape this feature was built for: cuInit failed (driver
    // too old for the CUDA runtime), so ZERO devices existed and every layer
    // silently stayed on CPU while the -ngl-derived claim still said 49/49.
    const stderr = [
      ...layerLines(49, () => "CPU"),
      "load_tensors: offloaded 49/49 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size =  1001.58 MiB",
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result?.main).toEqual({ gpuMib: 0, cpuMib: 1001.58, gpu_layers_exact: 0 });
  });

  it("clamps the exact count to the claim when a build over-assigns defensively", () => {
    const stderr = [
      ...layerLines(40, (i) => (i < 35 ? "CUDA0" : "CPU")),
      "load_tensors: offloaded 33/33 layers to GPU",
      "load_tensors:        CUDA0 model buffer size =  100.00 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)?.main?.gpu_layers_exact).toBe(33);
  });

  it("keeps base and draft exact counts separate on an MTP item", () => {
    const stderr = [
      ...layerLines(36, (i) => (i < 30 ? "Vulkan" : "CPU")), // base: 30 GPU
      "load_tensors: offloaded 31/37 layers to GPU",
      "load_tensors:      Vulkan model buffer size =  1290.62 MiB",
      "load_tensors:   CPU_Mapped model buffer size =   500.00 MiB",
      ...layerLines(5, () => "CPU"), // draft: 0 GPU
      "load_tensors: offloaded 5/6 layers to GPU",
      "load_tensors:   CPU_Mapped model buffer size =    68.00 MiB",
    ].join("\n");
    const result = parseModelBufferSizes(stderr);
    expect(result).toEqual({
      main: { gpuMib: 1290.62, cpuMib: 500, gpu_layers_exact: 30 },
      draft: { gpuMib: 0, cpuMib: 68, gpu_layers_exact: 0 },
    });
  });

  it("tolerates a logger prefix and a missing is_swa suffix on layer lines", () => {
    const stderr = [
      "12.345.678 I load_tensors: layer   0 assigned to device CUDA0, is_swa = 0",
      "12.345.679 I load_tensors: layer   1 assigned to device CPU",
      "load_tensors: offloaded 2/3 layers to GPU",
      "load_tensors:        CUDA0 model buffer size =   64.00 MiB",
    ].join("\n");
    expect(parseModelBufferSizes(stderr)?.main).toEqual({ gpuMib: 64, cpuMib: 0, gpu_layers_exact: 1 });
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
