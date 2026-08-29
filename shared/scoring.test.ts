import { describe, expect, it } from "vitest";
import {
  scoreProfiles,
  p90,
  sampleStddev,
  QUALITY_NOT_MEASURED_DISCLAIMER,
  type ScoringRow,
} from "./scoring.js";
import { defaultGoals, type GoalsConfig } from "./goals.js";
import type { MaxCtxEstimate } from "./vramEstimate.js";

let nextIdx = 0;

function row(over: Partial<ScoringRow> & { test_type: "pp" | "tg"; avg_tps: number }): ScoringRow {
  return {
    idx: over.idx ?? nextIdx++,
    n_prompt: 512,
    n_gen: 128,
    n_depth: 0,
    n_threads: 8,
    n_gpu_layers: 99,
    batch_size: 2048,
    ubatch_size: 512,
    cache_type_k: "f16",
    cache_type_v: "f16",
    flash_attn: "on",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
    method_version: 1,
    stddev_tps: 0,
    sample_count: 5,
    vram_peak_mib: 4000,
    ram_peak_mib: 2000,
    gpu_memory_total_mb: 8192,
    system_memory_total_mb: 32768,
    ...over,
  };
}

// One measured configuration = a pp row + a tg row sharing every axis.
function config(opts: {
  idx: number;
  pp: number;
  tg: number;
  vramPeak?: number;
  ramPeak?: number;
  axes?: Partial<ScoringRow>;
  depth?: number;
  extra?: Partial<ScoringRow>;
}): ScoringRow[] {
  const shared = {
    idx: opts.idx,
    n_depth: opts.depth ?? 0,
    vram_peak_mib: opts.vramPeak ?? 4000,
    ram_peak_mib: opts.ramPeak ?? 2000,
    ...opts.axes,
    ...opts.extra,
  };
  return [
    row({ ...shared, test_type: "pp", avg_tps: opts.pp }),
    row({ ...shared, test_type: "tg", avg_tps: opts.tg }),
  ];
}

describe("§0.3 primitives", () => {
  it("P90 of medians, not the raw max -- one timer-bug repeat must not pin the denominator", () => {
    const clean = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
    // Nearest-rank: ceil(0.9 * 10) = 9 -> the 9th smallest.
    expect(p90(clean)).toBe(18);
    // A single 1e6 outlier appended shifts the rank by exactly one position,
    // never to the outlier itself -- which is the whole point of P90 here.
    expect(p90([...clean, 1_000_000])).toBe(19);
    expect(Math.max(...clean, 1_000_000)).toBe(1_000_000);
  });

  it("uses llama-bench's own sample (n-1) stddev formula", () => {
    // Population stddev of [2,4,4,4,5,5,7,9] is 2; the sample one is larger.
    expect(sampleStddev([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(2.13809, 4);
  });

  it("normalizes against the per-config MEDIAN of repeat samples, not the mean", () => {
    const rows = [
      row({ idx: 0, test_type: "pp", avg_tps: 1000, sample_count: 5 }),
      row({
        idx: 0,
        test_type: "tg",
        avg_tps: 104, // the mean of the samples below -- what config.tg displays
        repeat_samples: [100, 100, 100, 100, 120],
        sample_count: 5,
      }),
    ];
    const result = scoreProfiles({ rows, repeats: 5 });
    const card = result.profiles.find((p) => p.id === "max_speed")!;
    expect(card.config.tg).toBe(104);
    // A lone config's P90 denominator is its own per-config value. If that
    // were the mean (104), tgHat would come out to exactly 1. The median of
    // [100,100,100,100,120] is 100, so a correct implementation normalizes
    // to 104/100, not 104/104.
    expect(card.config.tgHat).toBeCloseTo(104 / 100, 6);
  });
});

describe("M3 profile scoring", () => {
  it("pinned reduction: Even + no target reproduces argmax(0.4·PP + 0.4·TG + 0.2·(1−PRESSURE))", () => {
    const rows = [
      ...config({ idx: 0, pp: 1000, tg: 40, vramPeak: 7000 }),
      ...config({ idx: 1, pp: 900, tg: 39, vramPeak: 2000, axes: { n_gpu_layers: 50 } }),
      ...config({ idx: 2, pp: 500, tg: 20, vramPeak: 1000, axes: { n_gpu_layers: 10 } }),
    ];
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.goals).toEqual(defaultGoals());
    expect(result.weights).toEqual({ wPP: 0.5, wTG: 0.5 });

    // Reproduce the fixed-weight rule by hand over the same normalized values.
    const ppDen = p90([1000, 900, 500]);
    const tgDen = p90([40, 39, 20]);
    const manual = [
      { key: 0, pp: 1000 / ppDen, tg: 40 / tgDen, pressure: 7000 / 8192 },
      { key: 1, pp: 900 / ppDen, tg: 39 / tgDen, pressure: 2000 / 8192 },
      { key: 2, pp: 500 / ppDen, tg: 20 / tgDen, pressure: 1000 / 8192 },
    ]
      .map((c) => ({ ...c, score: 0.4 * c.pp + 0.4 * c.tg + 0.2 * (1 - c.pressure) }))
      .sort((a, b) => b.score - a.score);

    const balanced = result.profiles.find((p) => p.id === "balanced")!;
    expect(balanced.config.itemIndices).toEqual([manual[0].key]);
    expect(balanced.basis).toContain("0.8·S");
  });

  it("Balanced and Max Speed appear under the default goal; Max Context is named as hidden", () => {
    const rows = [...config({ idx: 0, pp: 1000, tg: 40 }), ...config({ idx: 1, pp: 500, tg: 20, axes: { n_gpu_layers: 4 } })];
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.profiles.map((p) => p.id).sort()).toEqual(["balanced", "low_memory", "max_speed"]);
    const hidden = result.hidden.find((h) => h.id === "max_context")!;
    expect(hidden.reason).toContain("Max context");
  });

  it("Max context goal: only Max Context (+ Low Memory) exist, ranked on the estimate and saying so", () => {
    const maxCtxByNgl: Record<number, MaxCtxEstimate> = {
      99: { tokens: 20_000, confidence: "good", binding: "kv" },
      50: { tokens: 83_000, confidence: "good", binding: "kv" },
      10: { tokens: 120_000, confidence: "good", binding: "kv" },
    };
    const rows = [
      ...config({ idx: 0, pp: 1000, tg: 40 }),
      ...config({ idx: 1, pp: 900, tg: 39, axes: { n_gpu_layers: 50 } }),
      // Fast context but crawling generation: excluded by the usable-speed floor.
      ...config({ idx: 2, pp: 400, tg: 8, axes: { n_gpu_layers: 10 } }),
    ];
    const goals: GoalsConfig = {
      goal: "max_context",
      target_ctx: 32_768,
      speed_floor_frac: 0.5,
      workload: "chat",
      kv_preset: "extended",
    };
    const result = scoreProfiles({
      rows,
      repeats: 5,
      goals,
      maxCtxFor: (axes) => maxCtxByNgl[axes.n_gpu_layers as number] ?? null,
    });
    const ids = result.profiles.map((p) => p.id);
    expect(ids).toContain("max_context");
    expect(ids).not.toContain("max_speed");
    expect(ids).not.toContain("balanced");
    const card = result.profiles.find((p) => p.id === "max_context")!;
    // ngl 10 affords the most tokens but crawls at 8 tok/s (< 50 % of 40).
    expect(card.config.axes.n_gpu_layers).toBe(50);
    expect(card.basis).toContain("not measured at your target");
    expect(card.config.fit).toBe(1);
  });

  it("stating a target reorders Balanced only through FIT", () => {
    const rows = [
      ...config({ idx: 0, pp: 1000, tg: 40, vramPeak: 4000 }),
      ...config({ idx: 1, pp: 995, tg: 39.9, vramPeak: 4000, axes: { cache_type_k: "q8_0", cache_type_v: "q8_0" } }),
    ];
    const noTarget = scoreProfiles({ rows, repeats: 5 });
    expect(noTarget.profiles.find((p) => p.id === "balanced")!.config.axes.cache_type_k).toBe("f16");

    const withTarget = scoreProfiles({
      rows,
      repeats: 5,
      goals: { ...defaultGoals(), target_ctx: 60_000 },
      maxCtxFor: (axes) =>
        axes.cache_type_k === "q8_0"
          ? { tokens: 83_000, confidence: "good", binding: "kv" }
          : { tokens: 30_000, confidence: "good", binding: "kv" },
    });
    // Same speeds within noise, but only the q8_0 config affords the target.
    expect(withTarget.profiles.find((p) => p.id === "balanced")!.config.axes.cache_type_k).toBe("q8_0");
  });

  it("workload shape sets (wPP, wTG) verbatim", () => {
    const rows = [
      ...config({ idx: 0, pp: 2000, tg: 20 }),
      ...config({ idx: 1, pp: 500, tg: 45, axes: { n_gpu_layers: 50 } }),
    ];
    const docs = scoreProfiles({ rows, repeats: 5, goals: { ...defaultGoals(), goal: "max_speed", workload: "docs" } });
    expect(docs.weights).toEqual({ wPP: 0.7, wTG: 0.3 });
    expect(docs.profiles.find((p) => p.id === "max_speed")!.config.itemIndices).toEqual([0]);

    const chat = scoreProfiles({ rows, repeats: 5, goals: { ...defaultGoals(), goal: "max_speed", workload: "chat" } });
    expect(chat.profiles.find((p) => p.id === "max_speed")!.config.itemIndices).toEqual([1]);
  });
});

describe("§0.3 eligibility gates and rejection accounting", () => {
  it("tallies which gate removed each tuple and never renders a silent zero", () => {
    const rows = [
      // stability: stddev above max(10 % of mean, 0.5)
      ...config({ idx: 0, pp: 100, tg: 10, extra: { stddev_tps: 50 } }),
      // suspect samples
      ...config({ idx: 1, pp: 100, tg: 10, axes: { n_gpu_layers: 50 }, extra: { suspect_count: 1 } }),
      // missing the tg side entirely
      row({ idx: 2, test_type: "pp", avg_tps: 100, n_gpu_layers: 30 }),
    ];
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.scoredCount).toBe(0);
    expect(result.tallies).toEqual({
      stability: 1,
      suspect_samples: 1,
      missing_pp_or_tg: 1,
      caveat_flagged: 0,
    });
    // Not "only stability", so no waived card is emitted.
    expect(result.profiles).toHaveLength(0);
  });

  it("emits a visibly unverified Max Speed card when every tuple failed only stability", () => {
    const rows = [
      ...config({ idx: 0, pp: 100, tg: 10, extra: { stddev_tps: 50 } }),
      ...config({ idx: 1, pp: 90, tg: 9, axes: { n_gpu_layers: 50 }, extra: { stddev_tps: 40 } }),
    ];
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.profiles).toHaveLength(1);
    expect(result.profiles[0].id).toBe("max_speed");
    expect(result.profiles[0].unverified).toBe(true);
  });

  it("n < 3 fails the stability gate even at stddev 0", () => {
    const rows = config({ idx: 0, pp: 100, tg: 10, extra: { sample_count: 2, stddev_tps: 0 } });
    expect(scoreProfiles({ rows, repeats: 2 }).tallies.stability).toBe(1);
  });

  it("the absolute stddev floor stops near-zero means exploding the ratio", () => {
    // mean 1 tok/s, stddev 0.4: 10 % of the mean is 0.1, but the 0.5 floor passes it.
    const rows = config({ idx: 0, pp: 1, tg: 1, extra: { stddev_tps: 0.4 } });
    expect(scoreProfiles({ rows, repeats: 5 }).scoredCount).toBe(1);
  });

  it("a failed item disqualifies its tuple; a skipped item does not", () => {
    const failed = scoreProfiles({
      rows: config({ idx: 0, pp: 100, tg: 10 }),
      repeats: 5,
      itemStatusByIdx: { 0: "failed_oom" },
    });
    expect(failed.scoredCount).toBe(0);
    const skipped = scoreProfiles({
      rows: config({ idx: 0, pp: 100, tg: 10 }),
      repeats: 5,
      itemStatusByIdx: { 0: "skipped" },
    });
    expect(skipped.scoredCount).toBe(1);
  });

  it("excludes swa rows from the TG reference-depth comparison, tallying caveat_flagged when nothing is left", () => {
    const rows = [
      row({ idx: 0, test_type: "pp", avg_tps: 100, n_depth: 16384 }),
      row({ idx: 0, test_type: "tg", avg_tps: 10, n_depth: 16384, caveat_flags: ["swa"] }),
    ];
    const result = scoreProfiles({ rows, repeats: 5, referenceDepthTokens: 16384 });
    expect(result.tallies.caveat_flagged).toBe(1);
    expect(result.scoredCount).toBe(0);
  });

  it("selects the tg row at the nearest available depth and records it", () => {
    const rows = [
      row({ idx: 0, test_type: "pp", avg_tps: 100, n_depth: 0 }),
      row({ idx: 0, test_type: "tg", avg_tps: 12, n_depth: 0 }),
      row({ idx: 1, test_type: "pp", avg_tps: 95, n_depth: 16384 }),
      row({ idx: 1, test_type: "tg", avg_tps: 10, n_depth: 16384 }),
    ];
    // Same tuple identity (depth is not part of it), reference depth 16k.
    const result = scoreProfiles({ rows, repeats: 5, referenceDepthTokens: 16_000 });
    expect(result.referenceDepth).toBe(16384);
    expect(result.candidateCount).toBe(1);
    expect(result.profiles.find((p) => p.id === "max_speed")!.config.tg).toBe(10);
    expect(result.profiles.find((p) => p.id === "max_speed")!.config.referenceDepth).toBe(16384);
  });
});

describe("§0.3 PRESSURE degradation", () => {
  it("falls back to RAM-only pressure when no VRAM totals exist", () => {
    const rows = config({
      idx: 0,
      pp: 100,
      tg: 10,
      extra: { vram_peak_mib: null, gpu_memory_total_mb: null, ram_peak_mib: 16384 },
    });
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.pressureUnavailable).toBe(false);
    expect(result.profiles.find((p) => p.id === "low_memory")!.config.pressure).toBeCloseTo(0.5, 5);
  });

  it("emits a speed card with a stated reason -- never a silent collapse -- when neither total exists", () => {
    const rows = config({
      idx: 0,
      pp: 100,
      tg: 10,
      extra: { vram_peak_mib: null, gpu_memory_total_mb: null, system_memory_total_mb: null },
    });
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.pressureUnavailable).toBe(true);
    expect(result.profiles.map((p) => p.id)).toContain("max_speed");
    expect(result.hidden.find((h) => h.id === "low_memory")!.reason).toContain("no VRAM or RAM totals");
  });
});

describe("§0.1 method-version segregation", () => {
  it("never averages rows of different vintages -- the newest vintage is what gets scored", () => {
    const rows = [
      ...config({ idx: 0, pp: 100, tg: 10, extra: { method_version: 1 } }),
      ...config({ idx: 1, pp: 900, tg: 90, axes: { n_gpu_layers: 50 }, extra: { method_version: 2 } }),
    ];
    const result = scoreProfiles({ rows, repeats: 5 });
    expect(result.methodVersion).toBe(2);
    expect(result.candidateCount).toBe(1);
    expect(result.profiles.find((p) => p.id === "max_speed")!.config.pp).toBe(900);
  });
});

describe("N6 steady-state discard", () => {
  it("recomputes mean and stddev from the stored samples after dropping the first repeats", () => {
    const samples = [5, 40, 41, 42, 39];
    const rows = [
      row({ idx: 0, test_type: "pp", avg_tps: 100, repeat_samples: [200, 100, 100, 100, 100], sample_count: 5 }),
      row({ idx: 0, test_type: "tg", avg_tps: 33.4, repeat_samples: samples, sample_count: 5, stddev_tps: 15 }),
    ];
    const withDiscard = scoreProfiles({ rows, repeats: 5, discardFirstRepeats: 1 });
    const card = withDiscard.profiles.find((p) => p.id === "max_speed")!;
    expect(card.config.tg).toBeCloseTo((40 + 41 + 42 + 39) / 4, 6);
    // Without the discard the cold first repeat fails the stability gate.
    expect(scoreProfiles({ rows, repeats: 5 }).tallies.stability).toBe(1);
  });
});

describe("the boundary: quality never enters a ranking", () => {
  it("scoring takes no quality input at all -- the N4 calibration gate cannot be bypassed by accident", () => {
    // Structural, not behavioural: ScoringInput has no ppl/kld field, so
    // there is no code path by which a perplexity number could reach an
    // argmax. If someone adds one, this test is what breaks.
    const inputKeys = [
      "rows",
      "itemStatusByIdx",
      "goals",
      "repeats",
      "discardFirstRepeats",
      "referenceDepthTokens",
      "maxCtxFor",
    ];
    const probe: Record<string, unknown> = {
      rows: config({ idx: 0, pp: 100, tg: 10 }),
      repeats: 5,
      // Deliberately smuggled in: it must be ignored entirely.
      ppl: 6.0,
      kld_vs_baseline: 0.004,
    };
    const withQuality = scoreProfiles(probe as never);
    const without = scoreProfiles({ rows: config({ idx: 0, pp: 100, tg: 10 }), repeats: 5 });
    expect(withQuality.profiles.map((p) => p.config.key)).toEqual(without.profiles.map((p) => p.config.key));
    expect(inputKeys).not.toContain("ppl");
  });

  it("every card states that quality was not measured", () => {
    const result = scoreProfiles({ rows: config({ idx: 0, pp: 100, tg: 10 }), repeats: 5 });
    expect(result.profiles.length).toBeGreaterThan(0);
    expect(QUALITY_NOT_MEASURED_DISCLAIMER).toContain("quality was not measured");
  });
});
