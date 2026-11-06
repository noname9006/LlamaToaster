import { describe, expect, it } from "vitest";
import {
  DEEP_CHUNK_BYTES,
  candidateThreadCounts,
  deepFromChunks,
  median,
  planDeepChunks,
  splitRange,
  summarize,
} from "./memSpeed.js";
import { allocSharedMemory, buildKernelModuleBytes, instantiateKernels } from "./memSpeedKernels.mjs";

const GIB = 1024 ** 3;

describe("candidateThreadCounts", () => {
  it("sweeps 1, 2, 4, physical and logical cores on a 6c/12t part", () => {
    expect(candidateThreadCounts(6, 12)).toEqual([1, 2, 4, 6, 12]);
  });
  it("never exceeds the logical count and dedupes", () => {
    expect(candidateThreadCounts(null, 1)).toEqual([1]);
    expect(candidateThreadCounts(2, 2)).toEqual([1, 2]);
    expect(candidateThreadCounts(4, 4)).toEqual([1, 2, 4]);
  });
});

describe("median / summarize", () => {
  it("takes the middle value, ignoring one wild outlier", () => {
    expect(median([5, 100, 6, 5.5, 6.2])).toBe(6);
    expect(median([1, 2, 3, 4])).toBe(2.5);
    expect(median([])).toBe(0);
  });
  it("reports median, min and max", () => {
    expect(summarize([3, 1, 2])).toEqual({ medianGBs: 2, minGBs: 1, maxGBs: 3 });
    expect(summarize([])).toEqual({ medianGBs: 0, minGBs: 0, maxGBs: 0 });
  });
});

describe("splitRange", () => {
  it("makes equal aligned slices that stay inside the range", () => {
    const slices = splitRange(512 * 1024 * 1024, 12);
    expect(slices).toHaveLength(12);
    const len = slices[0].end - slices[0].start;
    expect(len % 4096).toBe(0);
    for (let i = 0; i < slices.length; i++) {
      expect(slices[i].end - slices[i].start).toBe(len);
      if (i > 0) expect(slices[i].start).toBe(slices[i - 1].end);
    }
    expect(slices[11].end).toBeLessThanOrEqual(512 * 1024 * 1024);
  });
  it("refuses a region too small for the thread count", () => {
    expect(() => splitRange(4096, 2)).toThrow();
  });
});

describe("planDeepChunks", () => {
  it("keeps the larger of 2 GiB and 15% of free memory as a margin", () => {
    const plan = planDeepChunks(40 * GIB);
    expect("chunks" in plan).toBe(true);
    if ("chunks" in plan) {
      expect(plan.reserveBytes).toBe(6 * GIB); // 15% of 40 GiB beats 2 GiB
      expect(plan.chunks).toBe(Math.floor((34 * GIB) / DEEP_CHUNK_BYTES));
    }
    const small = planDeepChunks(8 * GIB);
    if ("chunks" in small) expect(small.reserveBytes).toBe(2 * GIB);
  });
  it("skips the scan when too little is free", () => {
    const plan = planDeepChunks(2.5 * GIB);
    expect("skipped" in plan).toBe(true);
  });
  it("caps the scan at 128 GiB", () => {
    const plan = planDeepChunks(1024 * GIB);
    if ("chunks" in plan) expect(plan.chunks).toBe(256);
  });
});

describe("deepFromChunks", () => {
  it("counts chunks under 85% of the median read as slow", () => {
    const read = [40, 40, 40, 40, 20, 20];
    const deep = deepFromChunks(512, read, read.map((v) => v / 2), read);
    expect(deep.chunks).toBe(6);
    expect(deep.testedMiB).toBe(3072);
    expect(deep.read.medianGBs).toBe(40);
    expect(deep.read.minGBs).toBe(20);
    expect(deep.slowChunks).toBe(2);
    expect(deep.chunkReadGBs).toEqual([40, 40, 40, 40, 20, 20]);
  });
  it("finds no slow chunks in a uniform scan", () => {
    const read = [39.1, 38.7, 39.4, 38.9];
    expect(deepFromChunks(512, read, read, read).slowChunks).toBe(0);
  });
});

// The hand-assembled WebAssembly kernels: correctness, not speed (this is the
// code a typo in an opcode would silently break -- a bare 0x40 once decoded as
// -64 and made the loop exit after one iteration).
describe("memSpeedKernels", () => {
  it("assembles a valid module", () => {
    const wasm = (globalThis as unknown as { WebAssembly: { validate(bytes: Uint8Array): boolean } }).WebAssembly;
    expect(wasm.validate(buildKernelModuleBytes())).toBe(true);
  });

  it("read sums every i64 lane, honours reps, and does nothing for reps = 0", () => {
    const mem = allocSharedMemory(1 << 20);
    const k = instantiateKernels(mem);
    const u64 = new BigUint64Array(mem.buffer, 0, 1024);
    let expected = 0n;
    for (let i = 0; i < 1024; i++) {
      u64[i] = BigInt(i * 7 + 3);
      expected += u64[i];
    }
    expect(k.r(0, 8192, 1)).toBe(expected);
    expect(k.r(0, 8192, 3)).toBe(expected * 3n);
    expect(k.r(0, 64, 0)).toBe(0n);
  });

  it("write fills exactly [ptr, end)", () => {
    const mem = allocSharedMemory(1 << 20);
    const k = instantiateKernels(mem);
    const u8 = new Uint8Array(mem.buffer);
    u8.fill(0, 0, 1024);
    k.w(64, 64 + 256, 1);
    expect(u8[63]).toBe(0);
    expect(u8[64]).toBe(0xef);
    expect(u8[64 + 255]).toBe(0xfe);
    expect(u8[64 + 256]).toBe(0);
  });

  it("copy moves [ptr, end) to ptr + delta and leaves the byte after it alone", () => {
    const mem = allocSharedMemory(1 << 20);
    const k = instantiateKernels(mem);
    const u8 = new Uint8Array(mem.buffer);
    for (let i = 0; i < 4096; i++) u8[i] = (i * 13 + 5) & 255;
    u8.fill(0, 1 << 19, (1 << 19) + 8192);
    k.c(0, 4096, 1 << 19, 1);
    for (let i = 0; i < 4096; i++) expect(u8[(1 << 19) + i]).toBe((i * 13 + 5) & 255);
    expect(u8[(1 << 19) + 4096]).toBe(0);
  });
});
