import { describe, expect, it } from "vitest";
import { fitGrade, fitScale, isFastUnified } from "./fitGrade";
import { isBreadMarkStyle, DEFAULT_BREAD_MARK_STYLE, sliceOffsets, stackShades, gradeSvg } from "./breadMarks";

const point = (gpu: number, total: number, overflow = 0) => ({ layers_gpu: gpu, layers_total: total, overflow_layers: overflow });

describe("fitGrade, gpu scale", () => {
  it("5 when every layer fits at the maximum context, 4 when only at the context asked", () => {
    expect(fitGrade({ level: "fits", point: point(37, 37), scale: "gpu", fitsAtTrained: true })).toBe(5);
    expect(fitGrade({ level: "fits", point: point(37, 37), scale: "gpu", fitsAtTrained: false })).toBe(4);
  });
  it("3 for all layers with MoE experts in RAM", () => {
    expect(fitGrade({ level: "partial", point: point(31, 31, 17), scale: "gpu", fitsAtTrained: false })).toBe(3);
  });
  it("2 at half the layers or more, 1 under half, including none", () => {
    expect(fitGrade({ level: "partial", point: point(16, 32), scale: "gpu", fitsAtTrained: false })).toBe(2);
    expect(fitGrade({ level: "partial", point: point(15, 32), scale: "gpu", fitsAtTrained: false })).toBe(1);
    expect(fitGrade({ level: "partial", point: point(0, 32), scale: "gpu", fitsAtTrained: false })).toBe(1);
  });
  it("partial layers with experts also offloaded grade by layer share, not 3", () => {
    expect(fitGrade({ level: "partial", point: point(20, 31, 20), scale: "gpu", fitsAtTrained: false })).toBe(2);
  });
  it("0 for not enough memory, null for a failed check", () => {
    expect(fitGrade({ level: "no_fit", point: point(0, 32), scale: "gpu", fitsAtTrained: false })).toBe(0);
    expect(fitGrade({ level: null, point: null, scale: "gpu", fitsAtTrained: false })).toBeNull();
  });
});

describe("fitGrade, ram scale", () => {
  it("2 at the maximum context, 1 at a shorter one, 0 when it won't fit", () => {
    expect(fitGrade({ level: "fits", point: null, scale: "ram", fitsAtTrained: true })).toBe(2);
    expect(fitGrade({ level: "fits", point: null, scale: "ram", fitsAtTrained: false })).toBe(1);
    expect(fitGrade({ level: "partial", point: point(10, 32), scale: "ram", fitsAtTrained: false })).toBe(1);
    expect(fitGrade({ level: "no_fit", point: null, scale: "ram", fitsAtTrained: false })).toBe(0);
  });
});

describe("fitScale", () => {
  const dev = (name: string, description: string) => ({ name, description, total_mib: 8192, free_mib: 8000 });
  it("ram with no GPU", () => expect(fitScale({ devices: [], unified: false })).toBe("ram"));
  it("gpu for dedicated VRAM", () => expect(fitScale({ devices: [dev("Vulkan0", "AMD Radeon RX 6600 XT")], unified: false })).toBe("gpu"));
  it("gpu for fast unified memory", () => {
    expect(fitScale({ devices: [dev("MTL0", "Apple M2 Max")], unified: true })).toBe("gpu");
    expect(fitScale({ devices: [dev("Vulkan0", "AMD Radeon(TM) 8060S Graphics")], unified: true })).toBe("gpu");
  });
  it("ram for an integrated GPU on ordinary DDR", () => {
    expect(fitScale({ devices: [dev("Vulkan0", "Intel(R) UHD Graphics 770")], unified: true })).toBe("ram");
    expect(fitScale({ devices: [dev("Vulkan0", "AMD Radeon(TM) Graphics")], unified: true })).toBe("ram");
  });
  it("does not mistake model numbers for Apple chips", () => {
    expect(isFastUnified(["NVIDIA Quadro M2000", "GeForce GTX 960M"])).toBe(false);
  });
});

describe("breadMarks", () => {
  it("validates styles", () => {
    expect(isBreadMarkStyle(DEFAULT_BREAD_MARK_STYLE)).toBe(true);
    expect(isBreadMarkStyle({ ...DEFAULT_BREAD_MARK_STYLE, step: 99 })).toBe(false);
    expect(isBreadMarkStyle({ ...DEFAULT_BREAD_MARK_STYLE, golden: "jam" })).toBe(false);
  });
  it("growing gaps keep the even total width", () => {
    const off = sliceOffsets(5, 3.5, 0.8);
    expect(off[4]).toBeCloseTo(14);
    expect(off[4] - off[3]).toBeGreaterThan(off[1] - off[0]);
  });
  it("stacks: gpu grade n has n slices, ram tops out at deep + golden", () => {
    expect(stackShades("gpu", 5)).toEqual([1, 2, 3, 4, 5]);
    expect(stackShades("gpu", 0)).toEqual([0]);
    expect(stackShades("ram", 2)).toEqual([4, 5]);
    expect(stackShades("ram", 1)).toEqual([4]);
    expect(gradeSvg("gpu", 3, { px: 16 })).toMatch(/^<svg[^>]+height="16"/);
  });
});
