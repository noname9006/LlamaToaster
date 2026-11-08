import { describe, expect, it } from "vitest";
import { incompleteRepeatsNote, llamaBenchSampleRates, meanAndSampleStddev, reconcileRowStats, statsAgree } from "./rowStats.js";

// Captured 2026-11-08 from llama-bench b9503 (Vulkan, RX 6600 XT), Gemma 4 E2B
// Q4, `-p 64 -n 16 -r 3 -o json`: the figures it printed, verbatim.
const PP = { n_prompt: 64, n_gen: 0, avg_ts: 1225.338233, stddev_ts: 4.097138, samples_ns: [52430400, 52158400, 52103800], samples_ts: [1220.67, 1227.03, 1228.32] };
const TG = { n_prompt: 0, n_gen: 16, avg_ts: 83.57616, stddev_ts: 0.210803, samples_ns: [191999400, 191204900, 191124600], samples_ts: [83.3336, 83.6799, 83.715] };

describe("meanAndSampleStddev", () => {
  it("uses the n-1 formula", () => {
    const { mean, stddev } = meanAndSampleStddev([1, 2, 3, 4]);
    expect(mean).toBe(2.5);
    expect(stddev).toBeCloseTo(Math.sqrt(5 / 3), 12);
  });
  it("gives 0 spread for one value and zeros for none", () => {
    expect(meanAndSampleStddev([7])).toEqual({ mean: 7, stddev: 0 });
    expect(meanAndSampleStddev([])).toEqual({ mean: 0, stddev: 0 });
  });
});

describe("llamaBenchSampleRates", () => {
  for (const [name, e] of [["pp", PP], ["tg", TG]] as const) {
    it(`reproduces llama-bench's own ${name} avg_ts and stddev_ts from samples_ns`, () => {
      const rates = llamaBenchSampleRates(e.n_prompt, e.n_gen, e.samples_ns)!;
      const { mean, stddev } = meanAndSampleStddev(rates);
      expect(statsAgree(mean, e.avg_ts)).toBe(true);
      expect(statsAgree(stddev, e.stddev_ts)).toBe(true);
    });
  }
  it("negative control: the rounded samples_ts do not reproduce avg_ts", () => {
    const { mean } = meanAndSampleStddev(PP.samples_ts);
    expect(statsAgree(mean, PP.avg_ts)).toBe(false);
  });
  it("refuses malformed timings rather than guessing", () => {
    expect(llamaBenchSampleRates(64, 0, undefined)).toBeUndefined();
    expect(llamaBenchSampleRates(64, 0, [])).toBeUndefined();
    expect(llamaBenchSampleRates(64, 0, [100, 0])).toBeUndefined();
    expect(llamaBenchSampleRates(64, 0, [100, "x"])).toBeUndefined();
    expect(llamaBenchSampleRates(0, 0, [100])).toBeUndefined();
  });
});

describe("reconcileRowStats", () => {
  it("agrees with a worker that used the same formula", () => {
    const clean = [10, 11, 12];
    const { mean, stddev } = meanAndSampleStddev(clean);
    const r = reconcileRowStats({ avg_tps: mean, stddev_tps: stddev, clean_samples: clean, repeats_requested: 3 });
    expect(r).toMatchObject({ avg_tps: mean, stddev_tps: stddev, stats_mismatch: 0, worker_avg_tps: null, sample_count: 3, repeats_complete: 1 });
  });

  it("replaces a population stddev and keeps the worker's figure", () => {
    const clean = [10, 11, 12];
    const populationStddev = Math.sqrt(2 / 3);
    const r = reconcileRowStats({ avg_tps: 11, stddev_tps: populationStddev, clean_samples: clean });
    expect(r.stats_mismatch).toBe(1);
    expect(r.stddev_tps).toBeCloseTo(1, 12);
    expect(r.worker_stddev_tps).toBe(populationStddev);
    expect(r.worker_avg_tps).toBe(11);
  });

  it("excludes suspect readings from the mean but counts them as readings", () => {
    const r = reconcileRowStats({
      avg_tps: 20,
      stddev_tps: 0,
      clean_samples: [20, 20],
      suspect_samples: [1e6],
      sample_count: 3,
      suspect_count: 1,
      repeats_requested: 3,
    });
    expect(r).toMatchObject({ avg_tps: 20, stats_mismatch: 0, sample_count: 3, suspect_count: 1, repeats_complete: 1 });
  });

  it("keeps an all-suspect row's own figures", () => {
    const r = reconcileRowStats({ avg_tps: 5e5, stddev_tps: 1, clean_samples: [], suspect_samples: [5e5, 5e5], repeats_requested: 2 });
    expect(r).toMatchObject({ avg_tps: 5e5, stddev_tps: 1, stats_mismatch: null, sample_count: 2, repeats_complete: 1 });
  });

  it("marks a row whose repeats did not all produce a reading", () => {
    const r = reconcileRowStats({ avg_tps: 10, stddev_tps: 0, clean_samples: [10, 10, 10], repeats_requested: 5 });
    expect(r.repeats_complete).toBe(0);
    expect(r.sample_count).toBe(3);
  });

  it("leaves an old worker's row as it was", () => {
    const r = reconcileRowStats({ avg_tps: 42, stddev_tps: 1, sample_count: 5, suspect_count: 0 });
    expect(r).toEqual({
      avg_tps: 42,
      stddev_tps: 1,
      sample_count: 5,
      suspect_count: 0,
      worker_avg_tps: null,
      worker_stddev_tps: null,
      stats_mismatch: null,
      repeats_complete: null,
    });
  });
});

describe("incompleteRepeatsNote", () => {
  it("says how many repeats produced a reading", () => {
    expect(incompleteRepeatsNote({ repeats_complete: false, repeats_requested: 5, sample_count: 3 })).toMatch(/^3 of 5/);
  });
  it("is silent for complete or unknown rows", () => {
    expect(incompleteRepeatsNote({ repeats_complete: true, repeats_requested: 5, sample_count: 5 })).toBeUndefined();
    expect(incompleteRepeatsNote({ repeats_complete: null })).toBeUndefined();
    expect(incompleteRepeatsNote({})).toBeUndefined();
  });
});
