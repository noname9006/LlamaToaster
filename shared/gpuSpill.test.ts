import { describe, expect, it } from "vitest";
import {
  buildSpillAnchor,
  measureGpuSpill,
  measureGrowthSpill,
  spillBaselineFromAnchor,
  spillBaselineKey,
  UNMEASURED_GPU_SPILL,
  withAnchorControl,
  type MeasuredLoad,
  type MemoryReading,
  type SpillAnchor,
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

describe("measureGrowthSpill under --load-mode mlock", () => {
  // Measured 2026-10-31: RX 6600 XT, Vulkan, b11840, gemma-4-12b Q8_0, 1,024 tokens.
  // With mmap off the weights left on the CPU sit in the GPU's pinned host buffer,
  // which WDDM counts as shared, so shared barely moves while real spill grows.
  const load = (claim: number, pin: number, ded: number, shared: number): MeasuredLoad => ({
    buffers: { deviceMib: claim, contextMib: 200, hostPinnedMib: pin },
    readings: [{ atMs: 1500, dedicatedMib: ded, sharedMib: shared }],
    ready: { fromMs: 1000, toMs: 2000 },
    work: null,
  });
  const anchor = buildSpillAnchor(load(1170, 12101, 1227.6, 12101.7));

  it.each([
    { layers: 15, claim: 4492, pin: 8866, ded: 1649.4, shared: 11797, spill: 2900 },
    { layers: 25, claim: 6854, pin: 6577, ded: 1950.9, shared: 11570.4, spill: 4960 },
    { layers: 42, claim: 10881, pin: 2661, ded: 2069.7, shared: 11570.9, spill: 8868 },
  ])("still sees the spill at $layers layers", ({ claim, pin, ded, shared, spill }) => {
    const v = measureGrowthSpill(load(claim, pin, ded, shared), anchor);
    expect(v.spilled).toBe(true);
    expect(v.inSystemRamMib!).toBeCloseTo(spill, -1);
  });

  it("does not call pinned host memory leaving shared a spill, or hide one behind it", () => {
    // Uncorrected, s here would read -304 and min(s, d) could never fire.
    expect(measureGrowthSpill(load(4492, 8866, 1649.4, 11797), anchor).sharedGrowthMib!).toBeCloseTo(2930, 0);
    // All of the claim landed in VRAM: no spill, whatever moved in shared.
    expect(measureGrowthSpill(load(4492, 8866, 4560, 8870), anchor).spilled).toBe(false);
  });

  it("changes nothing when no host buffer moved", () => {
    const plain = (claim: number, ded: number, shared: number): MeasuredLoad => ({
      buffers: { deviceMib: claim, contextMib: 200 },
      readings: [{ atMs: 1500, dedicatedMib: ded, sharedMib: shared }],
      ready: { fromMs: 1000, toMs: 2000 },
      work: null,
    });
    const v = measureGrowthSpill(plain(4492, 1614.8, 2963.3), buildSpillAnchor(plain(1170, 1180.4, 33.8)));
    expect(v.sharedGrowthMib!).toBeCloseTo(2929.5, 1);
    expect(v.spilled).toBe(true);
  });
});

// Real loads, 2026-11-02: a 7.5B gemma4 Q2_K file (42 layers) on a GeForce MX150
// (2GiB), CUDA, llama.cpp b12040, llama-bench with a 512-token prompt. Claim is
// llama.cpp's buffer report (model + KV + compute), dedicated and shared the
// process's WDDM counters at peak. Weights past layer 24 land in system RAM:
// shared grows by about one layer (41MiB) per layer while dedicated stops at
// 1953MiB, yet the buffers-minus-dedicated difference reads below zero until 43.
describe("a baseline outside llama.cpp's buffers", () => {
  const CUDA_NEMOTRON: { ngl: number; claim: number; contextMib: number; dedicated: number; spilled: boolean }[] = [
    { ngl: 2, claim: 337, contextMib: 17, dedicated: 1013, spilled: false },
    { ngl: 14, claim: 820, contextMib: 17, dedicated: 1495, spilled: false },
    { ngl: 20, claim: 1066, contextMib: 20, dedicated: 1739, spilled: false },
    { ngl: 24, claim: 1232, contextMib: 26, dedicated: 1905, spilled: false },
    { ngl: 26, claim: 1324, contextMib: 31, dedicated: 1931, spilled: true },
    { ngl: 32, claim: 1582, contextMib: 41, dedicated: 1953, spilled: true },
    { ngl: 40, claim: 1923, contextMib: 55, dedicated: 1954, spilled: true },
    { ngl: 43, claim: 2101, contextMib: 59, dedicated: 1946, spilled: true },
  ];
  // The anchor: 2 layers, 1013MiB dedicated against a 337MiB claim.
  const anchor: SpillAnchor = {
    claimMib: 337,
    contextMib: 17,
    ready: { dedicatedMib: 1013, sharedMib: 110, movementMib: 1, noiseMib: 2 },
    work: null,
  };

  it("reads 40 layers with 634MiB in system RAM as clean against zero -- the blind spot", () => {
    const row = CUDA_NEMOTRON.find((r) => r.ngl === 40)!;
    const v = measureGpuSpill({ buffers: { deviceMib: row.claim, contextMib: row.contextMib }, dedicatedMib: row.dedicated, dedicatedJitterMib: 1 });
    expect(v).toMatchObject({ spilled: false, inSystemRamMib: -31 });
  });

  it.each(CUDA_NEMOTRON)("judges $ngl layers against the anchor's baseline", ({ claim, contextMib, dedicated, spilled }) => {
    const baseline = spillBaselineFromAnchor(anchor)!;
    const v = measureGpuSpill({
      buffers: { deviceMib: claim, contextMib },
      dedicatedMib: dedicated,
      dedicatedJitterMib: 1 + baseline.toleranceMib,
      baselineMib: baseline.mib,
    });
    expect(v.spilled).toBe(spilled);
  });

  it("measures how much is in system RAM once the baseline is added back", () => {
    const v = measureGpuSpill({ buffers: { deviceMib: 1923, contextMib: 55 }, dedicatedMib: 1954, dedicatedJitterMib: 0, baselineMib: 676 });
    expect(v).toMatchObject({ spilled: true, inSystemRamMib: 645, cause: "layers" });
  });

  it("changes nothing without a baseline, or with a negative one", () => {
    const input = { buffers: { deviceMib: 2000, contextMib: 263 }, dedicatedMib: 1950, dedicatedJitterMib: 0 };
    const bare = measureGpuSpill(input);
    expect(measureGpuSpill({ ...input, baselineMib: null })).toEqual(bare);
    expect(measureGpuSpill({ ...input, baselineMib: -40 })).toEqual(bare);
  });

  describe("spillBaselineFromAnchor", () => {
    it("is the dedicated VRAM the anchor held beyond its claim, with its movement and noise as tolerance", () => {
      expect(spillBaselineFromAnchor(anchor)).toEqual({ mib: 676, toleranceMib: 3 });
    });

    it("falls back to the work phase when the ready hold was not measured", () => {
      const work = { dedicatedMib: 1020, sharedMib: null, movementMib: 4, noiseMib: 0 };
      expect(spillBaselineFromAnchor({ ...anchor, ready: null, work })).toEqual({ mib: 683, toleranceMib: 4 });
    });

    it("is never negative", () => {
      const low = { ...anchor, ready: { dedicatedMib: 300, sharedMib: null, movementMib: 0, noiseMib: 0 } };
      expect(spillBaselineFromAnchor(low)).toEqual({ mib: 0, toleranceMib: 0 });
    });

    it("is null with no anchor, or an anchor with no reading", () => {
      expect(spillBaselineFromAnchor(null)).toBeNull();
      expect(spillBaselineFromAnchor({ ...anchor, ready: null, work: null })).toBeNull();
    });
  });

  it("keys a baseline by file, backend, GPU and build", () => {
    const parts = { modelPath: "C:\models\a.gguf", backend: "cuda", gpuName: "GeForce MX150", buildTag: "b12040" };
    const key = spillBaselineKey(parts);
    expect(spillBaselineKey({ ...parts })).toBe(key);
    for (const other of [{ modelPath: "C:\models\b.gguf" }, { backend: "vulkan" }, { gpuName: "UHD 620" }, { gpuName: null }, { buildTag: "b12038" }]) {
      expect(spillBaselineKey({ ...parts, ...other })).not.toBe(key);
    }
  });
});
