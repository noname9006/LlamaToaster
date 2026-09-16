import { describe, expect, it } from "vitest";
import { measureGpuSpill, measurePhasedGpuSpill, UNMEASURED_GPU_SPILL } from "./gpuSpill.js";

// Real loads, 2026-09-14: Qwen3.6-35B-A3B-UD-IQ4_NL on a Radeon RX 6600 XT
// (8176MiB), Vulkan, llama.cpp b10956. deviceMib and contextMib are sums of
// llama.cpp's own "<device> <kind> buffer size" lines; dedicated is the
// process's WDDM Dedicated Usage once loaded.
const LOADS: { load: string; deviceMib: number; contextMib: number; dedicated: number; cause: "layers" | "cache" | null }[] = [
  { load: "1024 tok, 0 layers", deviceMib: 274.5, contextMib: 274.5, dedicated: 279.75, cause: null },
  { load: "1024 tok, 4 layers", deviceMib: 2001.57, contextMib: 263.01, dedicated: 2017.33, cause: null },
  { load: "1024 tok, 8 layers", deviceMib: 3634.7, contextMib: 215.51, dedicated: 2806.27, cause: "layers" },
  { load: "1024 tok, 10 layers", deviceMib: 4439.46, contextMib: 217.51, dedicated: 3102.77, cause: "layers" },
  { load: "1024 tok, 12 layers", deviceMib: 5251.33, contextMib: 217.51, dedicated: 4463.16, cause: "layers" },
  { load: "1024 tok, 15 layers", deviceMib: 6462.03, contextMib: 219.51, dedicated: 4623.1, cause: "layers" },
  { load: "16384 tok, 15 layers", deviceMib: 6607.52, contextMib: 365, dedicated: 4743.73, cause: "layers" },
  { load: "32768 tok, 15 layers", deviceMib: 6783.52, contextMib: 541, dedicated: 4773.48, cause: "layers" },
  { load: "65536 tok, 15 layers", deviceMib: 7135.52, contextMib: 893, dedicated: 4870.98, cause: "layers" },
  { load: "262144 tok, 15 layers", deviceMib: 9359.52, contextMib: 3117, dedicated: 5056.43, cause: "layers" },
  { load: "262144 tok, 0 layers", deviceMib: 1069, contextMib: 1069, dedicated: 822.35, cause: "cache" },
  { load: "262144 tok, 4 layers", deviceMib: 3319.56, contextMib: 1581, dedicated: 3085.23, cause: "cache" },
];

describe("measureGpuSpill", () => {
  it.each(LOADS)("$load", ({ deviceMib, contextMib, dedicated, cause }) => {
    const v = measureGpuSpill({ buffers: { deviceMib, contextMib }, dedicatedMib: dedicated, dedicatedJitterMib: 0 });
    expect(v.measured).toBe(true);
    expect(v.spilled).toBe(cause != null);
    expect(v.cause).toBe(cause);
  });

  it("reports the signed difference, so a clean load reads below zero", () => {
    const v = measureGpuSpill({ buffers: { deviceMib: 2001.57, contextMib: 263.01 }, dedicatedMib: 2017.33, dedicatedJitterMib: 0 });
    expect(v.inSystemRamMib!).toBeCloseTo(-15.76, 2);
  });

  // The only tolerance is what the counter itself could not resolve on this load.
  it("does not call a difference within the counter's own movement a spill", () => {
    const buffers = { deviceMib: 2000, contextMib: 263 };
    expect(measureGpuSpill({ buffers, dedicatedMib: 1950, dedicatedJitterMib: 60 })).toMatchObject({ spilled: false, jitterMib: 60 });
    expect(measureGpuSpill({ buffers, dedicatedMib: 1950, dedicatedJitterMib: 40 })).toMatchObject({ spilled: true, cause: "cache" });
  });

  it("files a spill under the layers only once it outgrows every context-sized buffer", () => {
    const buffers = { deviceMib: 5000, contextMib: 1000 };
    expect(measureGpuSpill({ buffers, dedicatedMib: 3999, dedicatedJitterMib: 0 }).cause).toBe("layers");
    expect(measureGpuSpill({ buffers, dedicatedMib: 4001, dedicatedJitterMib: 0 }).cause).toBe("cache");
  });

  // A real rerun of 8 layers at 1,024 tokens through the worker's own code put
  // only 189MiB in system RAM -- less than that context's 216MiB of buffers --
  // but 1,024 tokens is the smallest context the probe tries.
  it("files any spill at the smallest context under the layers", () => {
    const input = { buffers: { deviceMib: 3634.7, contextMib: 215.51 }, dedicatedMib: 3446, dedicatedJitterMib: 2 };
    expect(measureGpuSpill(input).cause).toBe("cache");
    expect(measureGpuSpill({ ...input, atSmallestContext: true })).toMatchObject({ spilled: true, cause: "layers" });
  });

  it("does not turn a clean load at the smallest context into a spill", () => {
    const input = { buffers: { deviceMib: 2001.57, contextMib: 263.01 }, dedicatedMib: 2017, dedicatedJitterMib: 0 };
    expect(measureGpuSpill({ ...input, atSmallestContext: true })).toMatchObject({ spilled: false, cause: null });
  });

  it("treats a missing or negative jitter as a counter that did not move", () => {
    const buffers = { deviceMib: 2000, contextMib: 263 };
    expect(measureGpuSpill({ buffers, dedicatedMib: 1990, dedicatedJitterMib: null })).toMatchObject({ spilled: true, jitterMib: 0 });
    expect(measureGpuSpill({ buffers, dedicatedMib: 1990, dedicatedJitterMib: -5 })).toMatchObject({ spilled: true, jitterMib: 0 });
  });

  it.each([
    ["no buffer report", { buffers: null, dedicatedMib: 2000, dedicatedJitterMib: 0 }],
    ["no per-process VRAM reading", { buffers: { deviceMib: 3634.7, contextMib: 215.51 }, dedicatedMib: null, dedicatedJitterMib: null }],
    ["nothing placed on a GPU", { buffers: { deviceMib: 0, contextMib: 0 }, dedicatedMib: 0, dedicatedJitterMib: 0 }],
  ])("is unmeasured, never clean, with %s", (_label, input) => {
    expect(measureGpuSpill(input)).toEqual(UNMEASURED_GPU_SPILL);
  });
});

describe("measurePhasedGpuSpill", () => {
  const buffers = { deviceMib: 2408, contextMib: 263 };
  const at = (atMs: number, dedicatedMib: number | null) => ({ atMs, dedicatedMib });
  const ready = { fromMs: 1000, toMs: 4500 };
  const work = { fromMs: 4500, toMs: 30_000 };

  it("judges the hold on its last reading, so memory still paging in right after load is not a spill", () => {
    const readings = [at(2000, 2100), at(3000, 2300), at(4000, 2420), ...[6000, 12_000, 20_000, 29_000].map((t) => at(t, 2421))];
    const v = measurePhasedGpuSpill({ buffers, readings, ready, work });
    expect(v.readyPhase).toMatchObject({ measured: true, spilled: false });
    expect(v).toMatchObject({ measured: true, spilled: false });
  });

  it("catches memory moved to system RAM only once the model is used", () => {
    const readings = [at(2000, 2420), at(3000, 2421), at(4000, 2420), ...[6000, 12_000, 20_000, 29_000].map((t) => at(t, 2249))];
    const v = measurePhasedGpuSpill({ buffers, readings, ready, work });
    expect(v.readyPhase.spilled).toBe(false);
    expect(v.workPhase).toMatchObject({ spilled: true, inSystemRamMib: 159 });
    expect(v).toMatchObject({ spilled: true, inSystemRamMib: 159 });
  });

  it("judges work on the median, so one reading mid-reshuffle cannot decide it", () => {
    const readings = [at(4000, 2420), at(6000, 2420), at(8000, 2200), at(10_000, 2420), at(12_000, 2421)];
    const v = measurePhasedGpuSpill({ buffers, readings, ready, work });
    // The dip widens the tolerance instead of deciding the verdict.
    expect(v.workPhase).toMatchObject({ spilled: false, jitterMib: 221 });
  });

  it("never counts a reading taken after the first request toward the ready hold", () => {
    // The only hold reading lands 500ms into the prompt: it belongs to work only.
    const readings = [at(5000, 2249)];
    const v = measurePhasedGpuSpill({ buffers, readings, ready, work });
    expect(v.readyPhase.measured).toBe(false);
    expect(v.workPhase).toMatchObject({ measured: true, inSystemRamMib: 159 });
  });

  it("counts a reading that lands up to a second after the work window", () => {
    const readings = [at(30_900, 2249)];
    expect(measurePhasedGpuSpill({ buffers, readings, ready: null, work }).workPhase.measured).toBe(true);
  });

  it("is unmeasured with no readings in either phase", () => {
    const v = measurePhasedGpuSpill({ buffers, readings: [at(500, 2400)], ready, work });
    expect(v).toMatchObject({ measured: false, spilled: false });
  });
});
