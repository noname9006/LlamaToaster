import { describe, expect, it } from "vitest";
import { fillBoundaries, fillTarget, groupFillCurves, validateFillCurveSpec, type FillCurveRow } from "./fillCurve.js";

describe("fillTarget", () => {
  it("stops 0.5 % short of the context", () => {
    expect(fillTarget(32_768)).toBe(32_604);
    expect(fillTarget(4_096)).toBe(4_075);
  });
});

describe("fillBoundaries", () => {
  it("is strictly increasing and ends exactly at the fill", () => {
    const b = fillBoundaries(32_604, 16);
    expect(b).toHaveLength(16);
    expect(b[b.length - 1]).toBe(32_604);
    for (let i = 1; i < b.length; i++) expect(b[i]).toBeGreaterThan(b[i - 1]);
  });

  it("collapses to fewer steps when the fill has fewer tokens than steps", () => {
    expect(fillBoundaries(3, 16)).toEqual([1, 2, 3]);
  });
});

describe("validateFillCurveSpec", () => {
  it("accepts a context within the trained one", () => {
    expect(validateFillCurveSpec({ ctx: 8_192 }, 32_768)).toBeNull();
  });
  it("refuses a context beyond the trained one", () => {
    expect(validateFillCurveSpec({ ctx: 65_536 }, 32_768)).toMatch(/trained context/);
  });
  it("refuses out-of-range steps", () => {
    expect(validateFillCurveSpec({ ctx: 8_192, steps: 1 }, null)).toMatch(/steps/);
  });
});

describe("groupFillCurves", () => {
  const row = (idx: number, depth: number, n: number, tps: number, method = 6): FillCurveRow => ({
    idx,
    test_type: "pp",
    n_prompt: n,
    n_depth: depth,
    avg_tps: tps,
    stddev_tps: 0,
    n_gpu_layers: idx === 0 ? 33 : 20,
    cache_type_k: "f16",
    cache_type_v: "f16",
    method_version: method,
  });

  it("builds one curve per item with x = depth + slice, in fill order", () => {
    const curves = groupFillCurves([row(0, 1000, 1000, 800), row(0, 0, 1000, 1000), row(1, 0, 1000, 500)], 6);
    expect(curves.map((c) => c.ngl)).toEqual([33, 20]);
    expect(curves[0].points.map((p) => p.fill)).toEqual([1000, 2000]);
    expect(curves[0].points.map((p) => p.tps)).toEqual([1000, 800]);
  });

  it("ignores rows of another method version", () => {
    expect(groupFillCurves([row(0, 0, 1000, 1000, 1)], 6)).toEqual([]);
  });
});
