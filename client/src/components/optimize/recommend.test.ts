import { describe, expect, it } from "vitest";
import { recommend } from "./SpeedSections";
import type { SpeedRunView } from "./flowData";
import { baselineCandidate, type SpeedCandidate } from "../../../../shared/speedRun.js";

const cand: SpeedCandidate = { label: "Most layers", source: "fit", ngl: 17, ot: null, ctk: "q8_0", ctv: "q8_0", fa: "on", threads: null };

function view(base: [number, number, number], other: [number, number, number], spill = 0): SpeedRunView {
  const outcome = (r: [number, number, number], idx: number, shared = 0) => ({
    idx,
    status: "done" as const,
    error: null,
    pp: { mean: 1000, stddev: 1, samples: r[2] },
    tg: { mean: r[0], stddev: r[1], samples: r[2] },
    sharedPeakMib: shared,
    vramPeakMib: 6000,
    ramPeakMib: 1000,
  });
  return {
    spec: { target_ctx: 32768, prompt_tokens: 4096, n_gen: 512, repeats: base[2], candidates: [baselineCandidate(), cand] },
    rows: [
      { candidate: baselineCandidate(), outcome: outcome(base, 0) },
      { candidate: cand, outcome: outcome(other, 1, spill) },
    ],
  };
}

describe("recommend", () => {
  it("a clear winner replaces the default", () => {
    const r = recommend(view([20, 0.5, 3], [30, 0.5, 3]))!;
    expect(r.row.candidate.label).toBe("Most layers");
    expect(r.tgDelta).toBe("+50% over defaults");
    expect(r.why).toMatch(/quantized KV/);
  });
  it("inside the spread the default stays", () => {
    const r = recommend(view([20, 2, 3], [21, 2, 3]))!;
    expect(r.row.candidate.source).toBe("baseline");
    expect(r.why).toMatch(/within the run-to-run spread/);
  });
  it("one repeat never calls a winner and says why", () => {
    const r = recommend(view([20, 0, 1], [40, 0, 1]))!;
    expect(r.row.candidate.source).toBe("baseline");
    expect(r.why).toMatch(/no spread/);
  });
  it("a run that spilled is never recommended", () => {
    const r = recommend(view([20, 0.5, 3], [40, 0.5, 3], 4096))!;
    expect(r.row.candidate.source).toBe("baseline");
  });
});
