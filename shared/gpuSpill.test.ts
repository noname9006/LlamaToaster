import { describe, expect, it } from "vitest";
import {
  buildSpillAnchor,
  measureGpuSpill,
  measureGrowthSpill,
  UNMEASURED_GPU_SPILL,
  withAnchorControl,
  type MeasuredLoad,
  type MemoryReading,
} from "./gpuSpill.js";

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

describe("measureGrowthSpill", () => {
  // Real loads, 2026-09-16: Qwen3.8-27B-UD-Q5_K_M on the same RX 6600 XT, Vulkan,
  // llama.cpp b11009, 1,024 tokens unless noted. Claim is llama.cpp's buffer
  // report; dedicated and shared are the process's WDDM counters.
  const ready = { fromMs: 1000, toMs: 4500 };
  const work = { fromMs: 4500, toMs: 30_000 };
  const READY_AT = [2000, 3000, 4000];
  const WORK_AT = [6000, 10_000, 14_000, 18_000, 22_000, 26_000, 29_000];

  function load(
    claim: number,
    contextMib: number,
    dedicated: number,
    shared: number | null,
    over: { ready?: [number, number | null][]; work?: [number, number | null][] } = {}
  ): MeasuredLoad {
    const readings: MemoryReading[] = [
      ...(over.ready ?? READY_AT.map(() => [dedicated, shared] as [number, number | null])).map(([d, sh], i) => ({ atMs: READY_AT[i], dedicatedMib: d, sharedMib: sh })),
      ...(over.work ?? WORK_AT.map(() => [dedicated, shared] as [number, number | null])).map(([d, sh], i) => ({ atMs: WORK_AT[i], dedicatedMib: d, sharedMib: sh })),
    ];
    return { buffers: { deviceMib: claim, contextMib }, readings, ready, work };
  }

  const A = load(1233, 240, 1005, 1272);
  const anchor = withAnchorControl(buildSpillAnchor(A)!, A);

  it("reads a load that grew in VRAM and not in shared as clean", () => {
    expect(measureGrowthSpill({ ...load(1841, 244, 1616, 1272), atSmallestContext: true }, anchor)).toMatchObject({
      measured: true,
      spilled: false,
      sharedGrowthMib: 0,
      unlandedGrowthMib: -3,
    });
  });

  it("fails a load whose shared growth and unlanded claim agree, filed under the layers at the smallest context", () => {
    const v = measureGrowthSpill({ ...load(3330, 246, 2972, 1417), atSmallestContext: true }, anchor);
    expect(v).toMatchObject({ spilled: true, sharedGrowthMib: 145, unlandedGrowthMib: 130, inSystemRamMib: 130, cause: "layers" });
    // The two accounts disagree by 15: that is the whole tolerance on steady readings.
    expect(v.jitterMib).toBe(15);
  });

  it("does not call overhead growing in shared a spill while dedicated took the whole claim", () => {
    expect(measureGrowthSpill(load(2228, 240, 2008, 2267), anchor)).toMatchObject({ spilled: false, sharedGrowthMib: 995, unlandedGrowthMib: -8 });
  });

  it("stays clean when a buffer moved back into VRAM at a larger context", () => {
    // 262,144 tokens, 5 layers: the compute buffer is resident here, d goes negative.
    expect(measureGrowthSpill(load(4323, 3452, 4347, 1292), anchor)).toMatchObject({ spilled: false, sharedGrowthMib: 20, unlandedGrowthMib: -252 });
  });

  it("files a spill no larger than the context's own growth over the anchor under the cache", () => {
    expect(measureGrowthSpill(load(6239, 3452, 5503, 2062), anchor)).toMatchObject({ spilled: true, inSystemRamMib: 508, cause: "cache" });
  });

  it("widens every tolerance by how far the anchor's second load differed", () => {
    const noisy = withAnchorControl(buildSpillAnchor(A)!, load(1233, 240, 999, 1272));
    expect(noisy.work?.noiseMib).toBe(6);
    const v = measureGrowthSpill(load(1841, 244, 1616, 1275), noisy);
    // s +3 against d -3 disagree by 6, plus the anchor noise of 6.
    expect(v.jitterMib).toBe(6 + 6);
  });

  it("judges the hold on its last reading, so memory still paging in after load is not a spill", () => {
    const paging = load(1841, 244, 1616, 1272, { ready: [[1300, 1500], [1500, 1300], [1616, 1272]] });
    expect(measureGrowthSpill(paging, anchor).readyPhase).toMatchObject({ measured: true, spilled: false, unlandedGrowthMib: -3 });
  });

  it("judges work on the median of paired readings, so a jump for a minority of it widens the tolerance", () => {
    const jump: [number, number][] = [[1616, 1272], [1616, 1272], [1616, 1272], [1616, 1272], [1616, 1272], [1400, 1488], [1400, 1488]];
    const v = measureGrowthSpill(load(1841, 244, 1616, 1272, { work: jump }), anchor).workPhase;
    expect(v).toMatchObject({ spilled: false, unlandedGrowthMib: -3, sharedGrowthMib: 0 });
    // The two accounts disagree by 3 at the median; the jump moved both by 216.
    expect(v.jitterMib).toBe(3 + 216);
  });

  it("uses the unlanded claim alone where there is no shared counter", () => {
    const noShared = withAnchorControl(buildSpillAnchor(load(1233, 240, 1005, null))!, load(1233, 240, 1005, null));
    expect(measureGrowthSpill(load(3330, 246, 2972, null), noShared)).toMatchObject({ spilled: true, sharedGrowthMib: null, inSystemRamMib: 130, jitterMib: 0 });
  });

  it("is unmeasured without an anchor, and there is no anchor without a buffer report", () => {
    expect(measureGrowthSpill(load(3330, 246, 2972, 1417), null)).toMatchObject({ measured: false, spilled: false });
    expect(buildSpillAnchor({ ...A, buffers: null })).toBeNull();
  });
});
