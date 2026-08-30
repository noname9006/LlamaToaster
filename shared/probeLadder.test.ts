import { describe, expect, it } from "vitest";
import {
  bestLadderResult,
  computeCtxStops,
  nextAnchoredCandidate,
  nextLadderRung,
  PROBE_LADDER_MIN_CTX,
  PROBE_MAX_LOADS,
  snapToSafeCtx,
  type AnchoredOutcome,
  type LadderAttempt,
  type ProbeGranularity,
  type ProbeMode,
} from "./probeLadder.js";

const TRAINED = 262_144;
const NGL_MAX = 48;
// A KV-cache-shaped constraint: more layers on GPU leaves less room for
// context. The TRUE boundary at ngl is FIT_LIMIT / (ngl + 4).
const FIT_LIMIT = 900_000;
const fits = (r: { ctx: number; ngl: number }): boolean => r.ctx * (r.ngl + 4) <= FIT_LIMIT;

// A deliberately IMPERFECT estimate -- real ones can be wrong (see the
// gemma4 SWA case this ladder was built to be robust against): consistently
// ~30% optimistic on ngl, ~25% optimistic on ctx, so tests prove the search
// still converges on the TRUE boundary via the bracket rule, not merely on
// whatever the estimate said.
function calculateNgl(): number {
  return 26; // true safe ngl at ctx=1024 is floor(900000/1024)-4 = 875; layers only go to 48, so "safe" just means "most of them"
}
function calculateCtx(pinnedNgl: number): number {
  const trueMax = Math.floor(FIT_LIMIT / (pinnedNgl + 4));
  return Math.round(trueMax * 1.25); // optimistic by construction
}

function runLadder(
  opts: { mode: ProbeMode; granularity: ProbeGranularity; candidateCtx: number; candidateNgl: number; maxCtx?: number; nglMax?: number },
  fitsFn: (rung: { ctx: number; ngl: number }) => boolean = fits
): LadderAttempt[] {
  const history: LadderAttempt[] = [];
  for (let guard = 0; guard <= PROBE_MAX_LOADS + 2; guard++) {
    const next = nextLadderRung({
      mode: opts.mode,
      granularity: opts.granularity,
      candidateCtx: opts.candidateCtx,
      candidateNgl: opts.candidateNgl,
      nglMax: opts.nglMax ?? NGL_MAX,
      maxCtx: opts.maxCtx ?? TRAINED,
      history,
      calculateNgl,
      calculateCtx,
    });
    if (next === null) return history;
    history.push({ ...next, ok: fitsFn(next) });
  }
  throw new Error("ladder did not terminate");
}

describe("nextAnchoredCandidate", () => {
  const base = { min: 0, max: 100, tolerance: 1 };

  it("does nothing with an empty history -- the seed is the caller's job", () => {
    expect(nextAnchoredCandidate({ ...base, history: [], growTarget: 80, shrinkTarget: 10 })).toBeNull();
  });

  it("only failures so far: bisects toward shrinkTarget", () => {
    const history: AnchoredOutcome[] = [{ value: 48, ok: false }];
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 20 })).toBe(34);
  });

  it("repeated failures keep bisecting toward shrinkTarget using the latest failure", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 34, ok: false },
    ];
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 20 })).toBe(27);
  });

  it("only successes so far: bisects toward growTarget", () => {
    const history: AnchoredOutcome[] = [{ value: 1024, ok: true }];
    expect(nextAnchoredCandidate({ min: 1024, max: 200_000, tolerance: 64, history, growTarget: 32_768, shrinkTarget: 1024 })).toBe(
      16_896
    );
  });

  it("a seed that IS its own growTarget converges in one success, no second candidate", () => {
    const history: AnchoredOutcome[] = [{ value: 48, ok: true }];
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 20 })).toBeNull();
  });

  it("once a bracket exists, bisects the two real values directly", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 34, ok: false },
      { value: 27, ok: true },
    ];
    // bracket [27 ok, 34 bad] -> mean 30.5 -> rounds to 31 (matches the plan's worked example)
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 20 })).toBe(31);
  });

  it("stops once the bracket is within tolerance", () => {
    const history: AnchoredOutcome[] = [
      { value: 27, ok: true },
      { value: 28, ok: false },
    ];
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 20 })).toBeNull();
  });

  it("ignores a flaky failure below an already-proven success", () => {
    const history: AnchoredOutcome[] = [
      { value: 30, ok: true },
      { value: 20, ok: false }, // below the proven success -- must not drag the bracket down
    ];
    const next = nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 20 });
    // Still only ONE real success and no qualifying failure above it -> keeps growing toward growTarget.
    expect(next).toBe(39);
  });

  it("tests the shrink target itself before giving up, even when already within tolerance", () => {
    // A failure that's already numerically close to shrinkTarget must not
    // short-circuit to "nothing fits" without ever trying shrinkTarget --
    // that would report failure based on proximity to an UNTESTED value.
    const history: AnchoredOutcome[] = [{ value: 27, ok: false }];
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBe(26);
  });

  it("once the shrink target itself has been tried and also failed, stops for real", () => {
    const history: AnchoredOutcome[] = [
      { value: 27, ok: false },
      { value: 26, ok: false },
    ];
    expect(nextAnchoredCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBeNull();
  });

  it("tests the grow target itself before stopping, symmetric to the shrink case", () => {
    const history: AnchoredOutcome[] = [{ value: 95, ok: true }];
    expect(nextAnchoredCandidate({ min: 0, max: 100, tolerance: 5, history, growTarget: 99, shrinkTarget: 0 })).toBe(99);
  });

  it("clamps a candidate to [min, max] and refuses to repeat an already-tried value", () => {
    const history: AnchoredOutcome[] = [{ value: 95, ok: true }];
    // mean(95, 200) would be 147.5, clamped to max=100 -- but 100 wasn't tried yet, so it's offered once.
    expect(nextAnchoredCandidate({ min: 0, max: 100, tolerance: 1, history, growTarget: 200, shrinkTarget: 0 })).toBe(100);
    const historyAtCeiling: AnchoredOutcome[] = [
      { value: 95, ok: true },
      { value: 100, ok: true },
    ];
    // Now every candidate the rule could produce clamps straight back to 100, already tried -> null.
    expect(
      nextAnchoredCandidate({ min: 0, max: 100, tolerance: 1, history: historyAtCeiling, growTarget: 200, shrinkTarget: 0 })
    ).toBeNull();
  });
});

describe("the ladder as a whole", () => {
  it("custom performs exactly one load at the user's exact values, pass or fail", () => {
    const passing = runLadder({ mode: "custom", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, () => true);
    expect(passing).toEqual([{ ctx: 32_768, ngl: 27, ok: true }]);

    const failing = runLadder({ mode: "custom", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, () => false);
    expect(failing).toEqual([{ ctx: 32_768, ngl: 27, ok: false }]);
    expect(bestLadderResult(failing)).toBeNull();
  });

  it.each<[ProbeMode, ProbeGranularity]>([
    ["max_gpu", "basic"],
    ["max_gpu", "fine"],
    ["max_context", "basic"],
    ["max_context", "fine"],
    ["balanced", "basic"],
    ["balanced", "fine"],
    ["keep_context", "basic"],
    ["keep_context", "fine"],
    ["fixed_offload", "basic"],
    ["fixed_offload", "fine"],
  ])("%s/%s terminates within budget, stays in bounds, and never repeats a rung", (mode, granularity) => {
    const rungs = runLadder({ mode, granularity, candidateCtx: 32_768, candidateNgl: 27 });
    expect(rungs.length).toBeGreaterThan(0);
    expect(rungs.length).toBeLessThanOrEqual(PROBE_MAX_LOADS);
    for (const rung of rungs) {
      expect(rung.ctx).toBeGreaterThanOrEqual(PROBE_LADDER_MIN_CTX);
      expect(rung.ctx).toBeLessThanOrEqual(TRAINED);
      expect(rung.ngl).toBeGreaterThanOrEqual(0);
      expect(rung.ngl).toBeLessThanOrEqual(NGL_MAX);
    }
    const seen = rungs.map((r) => `${r.ctx}:${r.ngl}`);
    expect(new Set(seen).size).toBe(seen.length);
  });

  it("max_gpu skips the layer phase entirely when every layer already fits at the floor context", () => {
    // Loosen the constraint so (ngl=48, ctx=1024) itself fits.
    const generous = (r: { ctx: number; ngl: number }): boolean => r.ctx * (r.ngl + 4) <= 10_000_000;
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, generous);
    // First (and only "layer-phase") rung is the seed itself, at full layers.
    expect(rungs[0]).toMatchObject({ ngl: NGL_MAX, ok: true });
    // No other rung ever revisits ngl=48 at a different context via a failed
    // layer-phase probe -- everything after the first rung is the ctx phase
    // (ngl stays pinned at NGL_MAX throughout).
    expect(rungs.slice(1).every((r) => r.ngl === NGL_MAX)).toBe(true);
  });

  it("max_gpu runs the layer phase first when the floor context doesn't fit at max layers, then grows context", () => {
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    expect(rungs[0]).toMatchObject({ ngl: NGL_MAX, ctx: PROBE_LADDER_MIN_CTX });
    const best = bestLadderResult(rungs);
    expect(best).not.toBeNull();
    // The true boundary at whatever ngl it settled on.
    const trueMax = Math.floor(FIT_LIMIT / (best!.ngl + 4));
    expect(best!.ctx).toBeLessThanOrEqual(trueMax);
    expect(best!.ctx).toBeGreaterThan(PROBE_LADDER_MIN_CTX);
  });

  it("keep_context never moves the pinned context, regardless of the estimate's own accuracy", () => {
    const rungs = runLadder({ mode: "keep_context", granularity: "basic", candidateCtx: 16_384, candidateNgl: 40 });
    for (const rung of rungs) expect(rung.ctx).toBe(16_384);
    const best = bestLadderResult(rungs);
    expect(best).not.toBeNull();
    expect(fits(best!)).toBe(true);
  });

  it("fixed_offload never moves the pinned layer count", () => {
    const rungs = runLadder({ mode: "fixed_offload", granularity: "basic", candidateCtx: 16_384, candidateNgl: 20 });
    for (const rung of rungs) expect(rung.ngl).toBe(20);
    const best = bestLadderResult(rungs);
    expect(best).not.toBeNull();
    expect(fits(best!)).toBe(true);
  });

  it("max_context's second phase reuses the first phase's own rung instead of spending a load on it", () => {
    const rungs = runLadder({ mode: "max_context", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    // Phase 2 seeds at the user's ngl (27) at the resolved ctx -- a combo
    // phase 1 (which never moves ngl) already tested. If dedup works, that
    // exact (resolvedCtx, 27) pair appears at most once in the whole trace.
    const resolvedCtxAfterPhase1 = rungs.find((r) => r.ngl === 27)?.ctx;
    if (resolvedCtxAfterPhase1 != null) {
      const repeats = rungs.filter((r) => r.ctx === resolvedCtxAfterPhase1 && r.ngl === 27).length;
      expect(repeats).toBe(1);
    }
  });

  it("balanced starts exactly at the user's own (ngl, ctx), not the estimate or an extreme", () => {
    const rungs = runLadder({ mode: "balanced", granularity: "basic", candidateCtx: 8192, candidateNgl: 30 });
    expect(rungs[0]).toMatchObject({ ctx: 8192, ngl: 30 });
  });

  it("fine granularity converges tighter than basic, at the cost of more or equal loads", () => {
    const basicRungs = runLadder({ mode: "fixed_offload", granularity: "basic", candidateCtx: 16_384, candidateNgl: 20 });
    const fineRungs = runLadder({ mode: "fixed_offload", granularity: "fine", candidateCtx: 16_384, candidateNgl: 20 });
    const basicBest = bestLadderResult(basicRungs)!;
    const fineBest = bestLadderResult(fineRungs)!;
    const trueMax = Math.floor(FIT_LIMIT / 24);
    // Fine must land at least as close to the true boundary as basic.
    expect(trueMax - fineBest.ctx).toBeLessThanOrEqual(trueMax - basicBest.ctx);
  });

  it("reports nothing when even the shrink target fails outright", () => {
    // Nothing fits, ever.
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, () => false);
    expect(bestLadderResult(rungs)).toBeNull();
    expect(rungs.length).toBeGreaterThan(0);
    expect(rungs.length).toBeLessThanOrEqual(PROBE_MAX_LOADS);
  });

  it("stops immediately once history already meets the load budget", () => {
    const history: LadderAttempt[] = Array.from({ length: PROBE_MAX_LOADS }, (_, i) => ({
      ctx: 1024 * (i + 1),
      ngl: 20,
      ok: true,
    }));
    expect(
      nextLadderRung({
        mode: "fixed_offload",
        granularity: "fine",
        candidateCtx: 16_384,
        candidateNgl: 20,
        nglMax: NGL_MAX,
        maxCtx: TRAINED,
        history,
        calculateNgl,
        calculateCtx,
      })
    ).toBeNull();
  });
});

describe("the safe-value rule", () => {
  it("snaps a value between slider stops DOWN, never up", () => {
    const safe = snapToSafeCtx(43_581, TRAINED);
    expect(safe).toBeLessThanOrEqual(43_581);
    expect(computeCtxStops(TRAINED)).toContain(safe === PROBE_LADDER_MIN_CTX ? safe : safe);
  });

  it("leaves a value that is already a stop untouched", () => {
    expect(snapToSafeCtx(32_768, TRAINED)).toBe(32_768);
  });

  it("never returns more than was actually verified", () => {
    for (const verified of [1024, 5000, 32_768, 43_581, 200_000, TRAINED]) {
      expect(snapToSafeCtx(verified, TRAINED)).toBeLessThanOrEqual(verified);
    }
  });
});

describe("bestLadderResult", () => {
  it("picks the largest passing context, breaking ties on placement", () => {
    const best = bestLadderResult([
      { ctx: 8192, ngl: 40, ok: true },
      { ctx: 16_384, ngl: 20, ok: true },
      { ctx: 16_384, ngl: 27, ok: true },
      { ctx: 32_768, ngl: 27, ok: false },
    ]);
    expect(best).toEqual({ ctx: 16_384, ngl: 27, ok: true });
  });
});
