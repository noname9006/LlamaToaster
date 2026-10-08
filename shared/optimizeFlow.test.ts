import { describe, expect, it } from "vitest";
import type { FitPoint } from "./fitParams.js";
import type { KvSupportRow } from "./kvSupport.js";
import { detectMoe } from "./moePlacement.js";
import {
  buildFitSeries,
  defaultFitSelection,
  fitInsight,
  machineClassOf,
  pointY,
  proposeCandidates,
  targetStop,
  validateFitMapSpec,
} from "./optimizeFlow.js";

// Real step-1 / step-5 numbers for Qwen3.8-27B Q5_K_M on the RX 6600 XT
// (2026-10-07), layers per ctx 4k..256k.
const LAYERS: Record<string, number[]> = {
  "f16|f16|on": [17, 17, 17, 16, 14, 11, 7],
  "f16|f16|off": [18, 17, 15, 12, 14, 12, 10],
  "q8_0|q8_0|on": [18, 18, 17, 17, 16, 14, 10],
  "q4_0|q4_0|on": [18, 18, 18, 17, 17, 16, 14],
};
const STOPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

function pt(ctk: string, ctv: string, fa: "on" | "off", ctx: number, layers: number): FitPoint {
  const host = 20000 - layers * 260;
  return {
    ctx, fa, ctk, ctv, margin_mib: 1024, verdict: "partial", reason: null, ngl: layers, layers_gpu: layers, layers_total: 65,
    overflow_layers: 0, ot: null, dev_used_mib: 6150, dev_free_mib: 1080, host_used_mib: host, total_mib: 6150 + host,
    kv_mib: 2000, compute_mib: 500, need_all_gpu_mib: 20500, splits: 2, fa_disabled: false, n_expert: 0, n_expert_used: 0,
    inferred: false, raw_args: null, error: null,
  };
}
const points: FitPoint[] = Object.entries(LAYERS).flatMap(([k, ls]) => {
  const [ctk, ctv, fa] = k.split("|");
  return ls.map((l, i) => pt(ctk, ctv, fa as "on" | "off", STOPS[i], l));
});
const ok = (ctk: string, ctv: string, fa: "on" | "off" = "on"): KvSupportRow => ({ ctk, ctv, fa, status: "ok", splits: 2, baseline_splits: 2, detail: null });
const kvRows = [ok("f16", "f16"), ok("q8_0", "q8_0"), ok("q4_0", "q4_0"), ok("iq4_nl", "q4_0"), ok("q4_0", "iq4_nl"), ok("iq4_nl", "iq4_nl")];

describe("machineClassOf", () => {
  it("classifies", () => {
    expect(machineClassOf({ hasGpu: false, unified: false })).toBe("cpu");
    expect(machineClassOf({ hasGpu: true, unified: true })).toBe("unified");
    expect(machineClassOf({ hasGpu: true, unified: false })).toBe("dgpu");
  });
});

describe("validateFitMapSpec", () => {
  it("accepts the default and rejects out-of-range stops", () => {
    const spec = { margin_mib: 1024, ctx_stops: STOPS, machine: "dgpu", kv_detect: true, kv_fit: true };
    expect(validateFitMapSpec(spec, 262144)).toBeNull();
    expect(validateFitMapSpec({ ...spec, ctx_stops: [2048] }, 262144)).toMatch(/4096/);
    expect(validateFitMapSpec({ ...spec, margin_mib: 700 }, 262144)).toMatch(/margin/);
    expect(validateFitMapSpec({ ...spec, kv_detect: false }, 262144)).toMatch(/kv_detect/);
  });
});

describe("buildFitSeries", () => {
  it("f16 paths first, then one path per KV size with its pairs listed", () => {
    const s = buildFitSeries(points, kvRows);
    expect(s.map((x) => x.id)).toEqual(["f16-on", "f16-off", "kv-q8_0-q8_0-on", "kv-q4_0-q4_0-on"]);
    const q4 = s.find((x) => x.id === "kv-q4_0-q4_0-on")!;
    expect(q4.pairs).toHaveLength(4);
    expect(q4.label).toBe("q4_0 / q4_0 +3");
    expect(q4.rampIndex).toBe(1);
    expect(q4.points.map((p) => p.ctx)).toEqual(STOPS);
  });
});

describe("defaultFitSelection", () => {
  it("target ctx, most layers, tie -> less RAM", () => {
    const s = buildFitSeries(points, kvRows);
    expect(defaultFitSelection(s, 32768, null)).toEqual({ seriesId: "kv-q8_0-q8_0-on", ctx: 32768 });
    // At 256k FA off (10) beats FA on (7); q4_0 (14) beats both.
    expect(defaultFitSelection(s.slice(0, 2), 262144, null)).toEqual({ seriesId: "f16-off", ctx: 262144 });
    expect(defaultFitSelection(s, 262144, null)?.seriesId).toBe("kv-q4_0-q4_0-on");
  });
  it("snaps a non-stop target to the next stop", () => {
    expect(targetStop(STOPS, 20000)).toBe(32768);
    expect(targetStop(STOPS, 999999)).toBe(262144);
  });
});

describe("fitInsight", () => {
  it("compares against the default at the same context", () => {
    const s = buildFitSeries(points, kvRows);
    const text = fitInsight({ series: s, selection: { seriesId: "kv-q4_0-q4_0-on", ctx: 262144 }, moe: null, machine: "dgpu" });
    expect(text).toMatch(/^At 256k, q4_0 \/ q4_0 \+3 fits 7 more layers than the default and saves 1\.8 GB of RAM\./);
    expect(text).toMatch(/slightly reduce output quality/);
    const def = fitInsight({ series: s, selection: { seriesId: "f16-on", ctx: 32768 }, moe: null, machine: "dgpu" });
    expect(def).toMatch(/default cache fits 16 of 65 layers on the GPU/);
  });
});

describe("pointY for MoE", () => {
  const moe = detectMoe({ dense: Array(40).fill(1), moe: Array(40).fill(100), embed: 0, output: 0, other: 0 });
  it("expert phase counts layers keeping their experts", () => {
    const p = { ...pt("f16", "f16", "on", 32768, 41), ngl: 41, layers_total: 41, overflow_layers: 33 };
    expect(pointY(p, moe)).toEqual({ band: "main", value: 8 });
  });
  it("dense phase falls in the lower band", () => {
    const p = { ...pt("f16", "f16", "on", 262144, 5), ngl: 5, layers_total: 41 };
    expect(pointY(p, moe)).toEqual({ band: "dense", value: 5 });
  });
});

describe("proposeCandidates", () => {
  it("best f16, best quantized and longest-context (most layers at the longest context)", () => {
    const s = buildFitSeries(points, kvRows);
    const c = proposeCandidates({ series: s, targetCtx: 32768, moe: null, threads: null });
    expect(c.map((x) => [x.source, x.ctk, x.fa, x.ngl])).toEqual([
      ["fit", "f16", "on", 16],
      ["fit_kv", "q8_0", "on", 17],
      // q4_0/q4_0 holds the most layers at 256k, so it is the long-context pick.
      ["fit_ctx", "q4_0", "on", 17],
    ]);
  });
});
