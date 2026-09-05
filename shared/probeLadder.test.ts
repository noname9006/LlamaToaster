import { describe, expect, it } from "vitest";
import {
  bestLadderResult,
  bestNglAtContext,
  bestNglForMaxContext,
  computeCtxStops,
  ctxLadderStops,
  nextAnchoredCandidate,
  nextDirectCandidate,
  nextLadderRung,
  nextSliderCandidate,
  nextSliderRefineCandidate,
  PROBE_LADDER_MIN_CTX,
  PROBE_MAX_LOADS,
  snapToSafeCtx,
  type AnchoredOutcome,
  type LadderAttempt,
  type ProbeGranularity,
  type ProbeMode,
} from "./probeLadder.js";

it("PROBE_MAX_LOADS defaults to 24, not the old 8 -- the new basic-ladder + fine-refine flow needs more budget to converge", () => {
  expect(PROBE_MAX_LOADS).toBe(24);
});

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

describe("nextDirectCandidate", () => {
  const base = { min: 0, max: 48, tolerance: 1 };

  it("jumps straight to shrinkTarget on the seed's first failure, not a midpoint", () => {
    const history: AnchoredOutcome[] = [{ value: 48, ok: false }];
    expect(nextDirectCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBe(26);
  });

  it("once shrinkTarget itself is tried, forms a real bracket with the seed and bisects normally", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 26, ok: true },
    ];
    // bracket [26 ok, 48 fail] -> mean 37
    expect(nextDirectCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBe(37);
  });

  it("continues bracket bisection on later steps exactly like nextAnchoredCandidate", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 26, ok: true },
      { value: 37, ok: false },
    ];
    // bracket [26 ok, 37 fail] -> mean 31.5 -> 32
    expect(nextDirectCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBe(32);
  });

  it("retargets toward min once shrinkTarget itself has also failed", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 26, ok: false },
    ];
    // shrinkTarget (26) already tried and failed -- no more trusted value to
    // jump to, so bisect toward the hard floor instead: mean(26, 0) = 13.
    expect(nextDirectCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBe(13);
  });

  it("tests min itself before giving up, symmetric to the shrink-target rule", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 26, ok: false },
      { value: 1, ok: false },
    ];
    expect(nextDirectCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBe(0);
  });

  it("reports nothing once even min has failed", () => {
    const history: AnchoredOutcome[] = [
      { value: 48, ok: false },
      { value: 26, ok: false },
      { value: 1, ok: false },
      { value: 0, ok: false },
    ];
    expect(nextDirectCandidate({ ...base, history, growTarget: 48, shrinkTarget: 26 })).toBeNull();
  });

  it("mirrors the direct jump on the grow side", () => {
    const history: AnchoredOutcome[] = [{ value: 10, ok: true }];
    expect(nextDirectCandidate({ min: 0, max: 100, tolerance: 1, history, growTarget: 80, shrinkTarget: 0 })).toBe(80);
  });
});

describe("nextSliderCandidate", () => {
  const stops = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

  it("walks up one stop at a time while every stop succeeds", () => {
    expect(nextSliderCandidate({ history: [{ value: 16384, ok: true }], stops })).toBe(32768);
  });

  it("walks down one stop at a time while every stop fails", () => {
    expect(nextSliderCandidate({ history: [{ value: 16384, ok: false }], stops })).toBe(8192);
  });

  it("stops the instant the walk reverses -- the two stops are already adjacent", () => {
    const history: AnchoredOutcome[] = [
      { value: 16384, ok: true },
      { value: 32768, ok: false },
    ];
    expect(nextSliderCandidate({ history, stops })).toBeNull();
  });

  it("stops at the top of the ladder instead of walking off the end", () => {
    expect(nextSliderCandidate({ history: [{ value: 262144, ok: true }], stops })).toBeNull();
  });

  it("stops at the bottom of the ladder instead of walking off the end", () => {
    expect(nextSliderCandidate({ history: [{ value: 1024, ok: false }], stops })).toBeNull();
  });
});

describe("ctxLadderStops", () => {
  it("always includes the hard floor and the model's own ceiling", () => {
    const stops = ctxLadderStops(262_144);
    expect(stops[0]).toBe(PROBE_LADDER_MIN_CTX);
    expect(stops[stops.length - 1]).toBe(262_144);
  });

  it("is ascending with no duplicates", () => {
    const stops = ctxLadderStops(262_144);
    for (let i = 1; i < stops.length; i++) expect(stops[i]).toBeGreaterThan(stops[i - 1]);
  });

  it("is pure power-of-two doublings from the floor, per the user's own spec (1024, 2048, 4096, 8192, ...)", () => {
    expect(ctxLadderStops(262_144)).toEqual([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]);
  });

  it("still includes the exact ceiling even when it isn't a power of two", () => {
    expect(ctxLadderStops(50_000)).toEqual([1024, 2048, 4096, 8192, 16384, 32768, 50_000]);
  });

  it("collapses to just the floor when maxCtx is at or below it", () => {
    expect(ctxLadderStops(1024)).toEqual([1024]);
    expect(ctxLadderStops(500)).toEqual([1024]);
  });
});

describe("nextSliderRefineCandidate", () => {
  const stops = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

  it("basic: behaves exactly like nextSliderCandidate, converging once the walk reverses", () => {
    const history: AnchoredOutcome[] = [
      { value: 16384, ok: true },
      { value: 32768, ok: false },
    ];
    expect(nextSliderRefineCandidate({ history, stops, granularity: "basic", min: 1024, max: 262144 })).toBeNull();
  });

  it("fine: continues past basic's convergence, refining within the bracket the walk found", () => {
    const history: AnchoredOutcome[] = [
      { value: 16384, ok: true },
      { value: 32768, ok: false },
    ];
    // bracket width = 32768-16384 = 16384; mean = 24576, the first refine candidate.
    expect(nextSliderRefineCandidate({ history, stops, granularity: "fine", min: 1024, max: 262144 })).toBe(24576);
  });

  it("fine: refine tolerance is fixed at 1/16 of the ORIGINAL slider bracket, not recomputed smaller each step", () => {
    const history: AnchoredOutcome[] = [
      { value: 16384, ok: true },
      { value: 32768, ok: false },
      { value: 24576, ok: true }, // first refine step, narrows the live bracket to [24576, 32768] (width 8192)
    ];
    // If tolerance were recomputed from the NEW (narrower) bracket, it would
    // shrink to 8192/16=512. Fixed at the original 16384/16=1024 instead:
    // next candidate is mean(24576,32768)=28672, and convergence should
    // still require the ORIGINAL bracket's own 1024-token tolerance.
    expect(nextSliderRefineCandidate({ history, stops, granularity: "fine", min: 1024, max: 262144 })).toBe(28672);
  });

  it("fine: converges once the bracket is within 1/16 of the original slider-stop gap", () => {
    // Original bracket [16384 ok, 32768 fail], width 16384, tolerance 1024.
    const history: AnchoredOutcome[] = [
      { value: 16384, ok: true },
      { value: 32768, ok: false },
      { value: 24576, ok: true },
      { value: 28672, ok: false },
      { value: 26624, ok: true },
      { value: 27648, ok: false }, // bracket [26624, 27648], width 1024 <= tolerance 1024
    ];
    expect(nextSliderRefineCandidate({ history, stops, granularity: "fine", min: 1024, max: 262144 })).toBeNull();
  });

  it("fine: nothing to refine when the walk never bracketed at all", () => {
    const allOk: AnchoredOutcome[] = [{ value: 262144, ok: true }];
    expect(nextSliderRefineCandidate({ history: allOk, stops, granularity: "fine", min: 1024, max: 262144 })).toBeNull();
    const allFail: AnchoredOutcome[] = [{ value: 1024, ok: false }];
    expect(nextSliderRefineCandidate({ history: allFail, stops, granularity: "fine", min: 1024, max: 262144 })).toBeNull();
  });

  it("fine: once an off-grid refine value has been tested, never falls back to nextSliderCandidate's nearest-stop snap again", () => {
    // Regression: 49152 sits EXACTLY equidistant between the stops 32768 and
    // 65536 (16384 either way). Before the fix, nextSliderRefineCandidate
    // re-ran nextSliderCandidate on the full (mixed) history every call --
    // its nearest-stop tie-break (favors the lower index) treated 49152 as
    // "closest to 32768", then walked DOWN one more ladder notch to 16384,
    // completely abandoning the bisection instead of continuing it.
    const history: AnchoredOutcome[] = [
      { value: 32768, ok: true },
      { value: 65536, ok: false },
      { value: 49152, ok: false }, // refine step 1: mean(32768, 65536)
    ];
    // Correct next step is bracket bisection continuing from the NARROWEST
    // known bracket [32768 ok, 49152 fail]: mean = 40960. NOT 16384.
    expect(nextSliderRefineCandidate({ history, stops, granularity: "fine", min: 1024, max: 262144 })).toBe(40960);
  });
});

describe("bestNglForMaxContext", () => {
  it("finds ngl=0 when context strictly decreases as layers are added (the naive VRAM-scarce case)", () => {
    const calculateCtx = (ngl: number) => 100_000 - ngl * 1000;
    expect(bestNglForMaxContext(48, calculateCtx)).toBe(0);
  });

  it("climbs to an interior peak when partial offload frees more room than either extreme", () => {
    // A clean tent, symmetric, peaking exactly at ngl=20.
    const calculateCtx = (ngl: number) => 50_000 - Math.abs(ngl - 20) * 100;
    expect(bestNglForMaxContext(48, calculateCtx)).toBe(20);
  });

  it("the coarse (step-2) pass alone would miss an odd peak just past it -- the bidirectional refine must find it", () => {
    // Same tent (coarse alone lands on 20), but the ODD value 21 is secretly
    // even better -- the coarse pass, stepping only by 2, would never test
    // 21 at all; refine must search upward from the coarse peak, not just
    // back down through the values coarse skipped.
    const tent = (ngl: number) => 50_000 - Math.abs(ngl - 20) * 100;
    const calculateCtx = (ngl: number) => (ngl === 21 ? 99_999 : tent(ngl));
    expect(bestNglForMaxContext(48, calculateCtx)).toBe(21);
  });

  it("checks the exact ceiling when nglMax is odd (the coarse step-2 pass never lands on it directly)", () => {
    const calculateCtx = (ngl: number) => ngl; // strictly increasing -- true best is nglMax itself
    expect(bestNglForMaxContext(49, calculateCtx)).toBe(49);
  });

  it("nglMax=0 trivially returns 0 without evaluating anything impossible", () => {
    expect(bestNglForMaxContext(0, () => 12_345)).toBe(0);
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

  it("max_gpu/basic: a failed full-layer attempt jumps straight to the precalculated ngl, not a midpoint", () => {
    // Only ngl <= 30 "fits" -- excludes the (48+26)/2 = 37 midpoint the OLD
    // gradual-bisection engine would have tested next, so landing on 26
    // (calculateNgl's fixed return) at rungs[1] proves the direct jump.
    const onlyLowNgl = (r: { ctx: number; ngl: number }): boolean => r.ngl <= 30;
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, onlyLowNgl);
    expect(rungs[0]).toMatchObject({ ngl: NGL_MAX, ok: false });
    expect(rungs[1]).toMatchObject({ ngl: 26, ok: true });
  });

  it("max_gpu/basic: the context phase only ever tests real slider stops, never an arbitrary bisected number", () => {
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    const stops = new Set(ctxLadderStops(TRAINED));
    // rungs[0] is the layer phase (its ctx field is just the pinned floor,
    // not a slider probe) -- everything from rungs[1] on is the ctx phase.
    for (const rung of rungs.slice(1)) expect(stops.has(rung.ctx)).toBe(true);
  });

  it("max_gpu/basic: the context phase seeds at the slider stop nearest the pre-flight estimate", () => {
    const generous = (r: { ctx: number; ngl: number }): boolean => r.ctx * (r.ngl + 4) <= 10_000_000;
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, generous);
    // Layer phase converges in 1 load (nglMax fits outright, per the
    // "skips the layer phase entirely" test above) -- rungs[1] is the ctx
    // phase's own seed. calculateCtx(NGL_MAX) = round(17307 * 1.25) = 21634;
    // ctxLadderStops(TRAINED) is powers of two here, and 16384 is nearer to
    // 21634 than 32768 is (5250 vs 11134).
    expect(rungs[1]).toMatchObject({ ctx: 16_384, ngl: NGL_MAX });
  });

  it("max_gpu/fine: the context phase starts with the SAME slider walk as basic, then extends past it with refinement", () => {
    const generous = (r: { ctx: number; ngl: number }): boolean => r.ctx * (r.ngl + 4) <= 10_000_000;
    const basicRungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, generous);
    const fineRungs = runLadder({ mode: "max_gpu", granularity: "fine", candidateCtx: 32_768, candidateNgl: 27 }, generous);
    // Same seed -- fine is the SAME walk, not a different engine (the old
    // engine seeded fine's ctx phase at the 1024 floor instead).
    expect(fineRungs[1]).toMatchObject({ ctx: 16_384, ngl: NGL_MAX });
    // Fine does strictly more work once it has a bracket to refine.
    expect(fineRungs.length).toBeGreaterThan(basicRungs.length);
    // At least one fine-phase ctx value is off the power-of-two grid --
    // proof it actually refined, not just re-walked the same stops.
    const stops = new Set(ctxLadderStops(TRAINED));
    expect(fineRungs.slice(1).some((r) => !stops.has(r.ctx))).toBe(true);
    const fineBest = bestLadderResult(fineRungs)!;
    const basicBest = bestLadderResult(basicRungs)!;
    expect(fineBest.ctx).toBeGreaterThanOrEqual(basicBest.ctx);
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

  it("max_context ignores the user's own ngl for its starting pin -- this fixture's calculateCtx is strictly decreasing in ngl, so bestNglForMaxContext anchors at 0, not the user's 27", () => {
    expect(bestNglForMaxContext(NGL_MAX, calculateCtx)).toBe(0);
    const rungs = runLadder({ mode: "max_context", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    expect(rungs[0].ngl).toBe(0);
  });

  it("max_context's second phase reuses phase 1's own resolved rung instead of spending a load re-proving it", () => {
    const rungs = runLadder({ mode: "max_context", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    const resolvedNgl = rungs[0].ngl; // whatever bestNglForMaxContext pinned phase 0 to
    const resolvedCtxAfterPhase1 = bestLadderResult(rungs.filter((r) => r.ngl === resolvedNgl))?.ctx;
    if (resolvedCtxAfterPhase1 != null) {
      const repeats = rungs.filter((r) => r.ctx === resolvedCtxAfterPhase1 && r.ngl === resolvedNgl).length;
      expect(repeats).toBe(1);
    }
  });

  it("max_context never gives back context to fit more layers -- ctx only ever grows or holds across the whole trace", () => {
    const rungs = runLadder({ mode: "max_context", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    const best = bestLadderResult(rungs);
    expect(best).not.toBeNull();
    // Every ok rung's ctx is <= the best ok ctx found anywhere in the trace --
    // phase 1 (growing ngl) never trades away context for a passing layer count.
    for (const r of rungs.filter((r) => r.ok)) expect(r.ctx).toBeLessThanOrEqual(best!.ctx);
  });

  it("max_context genuinely searches for the ngl that maximizes context, not just ngl=0 or the user's own -- a real interior peak", () => {
    // A hand-built model where partial offload (ngl=10) affords MORE context
    // than either extreme: weights compete with KV for VRAM, but this
    // fixture pretends 10 layers' worth of offloaded weights frees more
    // system RAM pressure than it costs in VRAM, up to a point. Clean tent,
    // peaking exactly at ngl=10. runLadder() is locked to the module-level
    // calculateCtx fixture, so this drives nextLadderRung directly with its
    // own matching calculateCtx (the estimate) and fits rule (the truth).
    const peakCtx = (ngl: number) => 40_000 - Math.abs(ngl - 10) * 500;
    const history: LadderAttempt[] = [];
    for (let guard = 0; guard <= PROBE_MAX_LOADS + 2; guard++) {
      const next = nextLadderRung({
        mode: "max_context",
        granularity: "basic",
        candidateCtx: 32_768,
        candidateNgl: 27,
        nglMax: 40,
        maxCtx: TRAINED,
        history,
        calculateNgl,
        calculateCtx: peakCtx,
      });
      if (next === null) break;
      history.push({ ...next, ok: next.ctx <= peakCtx(next.ngl) });
    }
    expect(history[0].ngl).toBe(10); // bestNglForMaxContext's answer for this curve
  });

  it("balanced snaps its seed to the nearest power-of-two stop, not the user's raw ctx", () => {
    // 8192 is already a stop -- pick a value that ISN'T, to actually prove
    // the snap (a value that happens to already be a stop can't tell the
    // difference between "snapped" and "used raw").
    const rungs = runLadder({ mode: "balanced", granularity: "basic", candidateCtx: 10_000, candidateNgl: 30 });
    expect(rungs[0]).toMatchObject({ ctx: 8192, ngl: 30 }); // nearest stop to 10,000 is 8192 (1808 away) over 16384 (6384 away)
  });

  it("fixed_offload snaps its seed to the nearest power-of-two stop too", () => {
    const rungs = runLadder({ mode: "fixed_offload", granularity: "basic", candidateCtx: 10_000, candidateNgl: 20 });
    expect(rungs[0]).toMatchObject({ ctx: 8192, ngl: 20 });
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

describe("bestNglAtContext", () => {
  it("names the most layers whose estimate still affords the target context", () => {
    // Estimate halves roughly with every 8 layers: 40000, 32000, ... 0.
    const estimate = (ngl: number) => Math.max(0, 40_000 - ngl * 1000);
    expect(bestNglAtContext(48, 30_000, estimate)).toBe(10);
    expect(bestNglAtContext(48, 40_000, estimate)).toBe(0);
  });

  it("never returns below the floor, so a grow phase's own seed stays its lower bound", () => {
    const estimate = (ngl: number) => Math.max(0, 40_000 - ngl * 1000);
    // Nothing at or above 20 affords 30,000 -- the floor itself comes back,
    // which callers read as "no room to grow".
    expect(bestNglAtContext(48, 30_000, estimate, 20)).toBe(20);
  });

  it("scans rather than climbing, so a non-monotonic estimate can't end it early", () => {
    // A dip at ngl=5 that a climb-until-worse loop would stop on.
    const estimate = (ngl: number) => (ngl === 5 ? 0 : 50_000 - ngl * 100);
    expect(bestNglAtContext(20, 49_000, estimate)).toBe(10);
  });
});

describe("grow-layers phases are bounded by the estimate, not by nglMax", () => {
  it("max_context skips its layer phase entirely when the estimate affords no extra layer", () => {
    // A fine ladder pushes ctx to the boundary at ngl=0, where the estimate
    // (1.25x the truth) allows at most ngl=1 -- so the layer phase is one
    // load, not a descent through every integer down from nglMax.
    const rungs = runLadder({ mode: "max_context", granularity: "fine", candidateCtx: 32_768, candidateNgl: 27 });
    const ctxResolved = bestLadderResult(rungs)!.ctx;
    const layerRungs = rungs.filter((r) => r.ctx === ctxResolved && r.ngl > 0);
    expect(layerRungs.length).toBeLessThanOrEqual(1);
  });

  it("a bounded grow phase never probes a layer count above what the estimate affords", () => {
    const rungs = runLadder({ mode: "balanced", granularity: "fine", candidateCtx: 32_768, candidateNgl: 16 });
    const ctxResolved = bestLadderResult(rungs)!.ctx;
    const ceiling = bestNglAtContext(NGL_MAX, ctxResolved, calculateCtx, 16);
    for (const rung of rungs.filter((r) => r.ctx === ctxResolved)) {
      expect(rung.ngl).toBeLessThanOrEqual(ceiling);
    }
    expect(ceiling).toBeLessThan(NGL_MAX);
  });

  it("costs strictly fewer loads than growing toward every layer did, for the same verdict", () => {
    const rungs = runLadder({ mode: "max_context", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    // The unbounded version of this search took 7; bounding it takes 5 and
    // still lands on the same rung.
    expect(rungs.length).toBe(5);
    expect(bestLadderResult(rungs)).toEqual({ ctx: 131_072, ngl: 2, ok: true });
  });
});

describe("max_gpu backs off when its context collapses to the floor", () => {
  // Weights dominate: every layer above ~20 leaves room for almost no KV, so
  // the layer phase wins layers the context phase then can't use.
  const tight = (r: { ctx: number; ngl: number }): boolean => r.ctx * (r.ngl + 4) <= 100_000;

  it("gives layers back to reach a context above the floor", () => {
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, tight);
    const best = bestLadderResult(rungs)!;
    expect(best.ctx).toBeGreaterThan(PROBE_LADDER_MIN_CTX);
    // The floor-context phase found more layers than the verdict keeps --
    // that is the trade the back-off exists to make.
    const bestAtFloor = Math.max(...rungs.filter((r) => r.ok && r.ctx === PROBE_LADDER_MIN_CTX).map((r) => r.ngl));
    expect(best.ngl).toBeLessThan(bestAtFloor);
  });

  it("never gives back a layer when the context phase already cleared the floor", () => {
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 });
    const best = bestLadderResult(rungs)!;
    expect(best.ctx).toBeGreaterThan(PROBE_LADDER_MIN_CTX);
    // Every rung sits at the layer count the layer phase resolved: no back-off.
    expect(new Set(rungs.map((r) => r.ngl)).size).toBe(1);
  });

  it("backs off under fine too -- a refined context below the second stop is still 'at the floor'", () => {
    // fine refines INSIDE the [1024, 2048] bracket, landing on something like
    // 1920: larger than the floor exactly, and just as unusable. Gating the
    // back-off on "> the floor" would have let fine slip past it.
    const rungs = runLadder({ mode: "max_gpu", granularity: "fine", candidateCtx: 32_768, candidateNgl: 27 }, tight);
    const best = bestLadderResult(rungs)!;
    expect(best.ctx).toBeGreaterThanOrEqual(2048);
    const bestAtFloor = Math.max(...rungs.filter((r) => r.ok && r.ctx < 2048).map((r) => r.ngl));
    expect(best.ngl).toBeLessThan(bestAtFloor);
  });

  it("re-walks the context ladder from the rung the back-off proved, not from the estimate", () => {
    const rungs = runLadder({ mode: "max_gpu", granularity: "basic", candidateCtx: 32_768, candidateNgl: 27 }, tight);
    const best = bestLadderResult(rungs)!;
    // Everything tested at the backed-off placement starts at 2048 and steps
    // UP the ladder; re-seeding on this fixture's (wildly optimistic)
    // estimate would instead walk back down through 16384/8192/4096.
    const atBackoff = rungs.filter((r) => r.ngl === best.ngl).map((r) => r.ctx);
    expect(Math.min(...atBackoff)).toBe(2048);
    expect(atBackoff.filter((c) => c > 4096)).toHaveLength(0);
  });

  it("stays within budget and reports the largest context that actually loaded", () => {
    const rungs = runLadder({ mode: "max_gpu", granularity: "fine", candidateCtx: 32_768, candidateNgl: 27 }, tight);
    expect(rungs.length).toBeLessThanOrEqual(PROBE_MAX_LOADS);
    const best = bestLadderResult(rungs)!;
    expect(Math.max(...rungs.filter((r) => r.ok).map((r) => r.ctx))).toBe(best.ctx);
  });
});

describe("the safe-value rule", () => {
  it("snaps a value between slider stops DOWN, never up", () => {
    const safe = snapToSafeCtx(43_581, TRAINED);
    expect(safe).toBeLessThanOrEqual(43_581);
    expect(ctxLadderStops(TRAINED)).toContain(safe);
  });

  it("leaves a value that is already a stop untouched", () => {
    expect(snapToSafeCtx(32_768, TRAINED)).toBe(32_768);
  });

  it("never returns more than was actually verified", () => {
    for (const verified of [1024, 5000, 32_768, 43_581, 200_000, TRAINED]) {
      expect(snapToSafeCtx(verified, TRAINED)).toBeLessThanOrEqual(verified);
    }
  });

  it("snaps against the probe's own power-of-two grid, not the client slider's fraction grid", () => {
    // 50,000 isn't a power of two and isn't a "round fraction" of itself
    // either -- ctxLadderStops and computeCtxStops now deliberately diverge.
    expect(ctxLadderStops(50_000)).toEqual([1024, 2048, 4096, 8192, 16384, 32768, 50_000]);
    expect(computeCtxStops(50_000)).not.toEqual(ctxLadderStops(50_000));
    expect(snapToSafeCtx(40_000, 50_000)).toBe(32_768);
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

// Regression: max_context used to stop after resolving context, never testing
// a single GPU layer. Reproduced with this machine's real estimator shape,
// where affordable context decreases monotonically in ngl -- so
// bestNglForMaxContext pins ngl 0, the context walk reaches the trained
// ceiling, and bestNglAtContext(thatCeiling) then answers 0 for every layer
// count. The mode measured exactly one placement and reported it.
describe("max_context grows layers even when the estimate says nothing fits", () => {
  const NGL_MAX = 41;
  const MAXCTX = 262144;
  // Monotonically decreasing, and 0 above ngl 12 -- i.e. the estimate insists
  // no offload can hold a large context.
  const calculateCtx = (ngl: number) => (ngl >= 12 ? 1024 : Math.max(1024, 72192 - ngl * 5400));

  function walk(): { ctx: number; ngl: number; ok: boolean }[] {
    const history: { ctx: number; ngl: number; ok: boolean }[] = [];
    for (let i = 0; i < 24; i++) {
      const rung = nextLadderRung({
        mode: "max_context",
        granularity: "basic",
        candidateCtx: 4096,
        candidateNgl: 20,
        nglMax: NGL_MAX,
        maxCtx: MAXCTX,
        history,
        calculateNgl: () => 6,
        calculateCtx,
      });
      if (!rung) break;
      // Context succeeds everywhere; layers fail above 10, which is the shape
      // the reference machine actually has.
      history.push({ ...rung, ok: rung.ngl <= 10 });
    }
    return history;
  }

  it("measures more than one layer count", () => {
    const tried = new Set(walk().map((h) => h.ngl));
    expect(tried.size).toBeGreaterThan(1);
  });

  it("finds a real offload rather than reporting ngl 0", () => {
    const history = walk();
    const best = history.filter((h) => h.ok).reduce((a, b) => (b.ctx > a.ctx || (b.ctx === a.ctx && b.ngl > a.ngl) ? b : a));
    expect(best.ngl).toBeGreaterThan(0);
    expect(best.ctx).toBe(MAXCTX);
  });

  it("still ends when there is nowhere left to grow", () => {
    const history: { ctx: number; ngl: number; ok: boolean }[] = [];
    for (let i = 0; i < 40; i++) {
      const rung = nextLadderRung({
        mode: "max_context", granularity: "basic", candidateCtx: 4096, candidateNgl: 20,
        nglMax: NGL_MAX, maxCtx: MAXCTX, history, calculateNgl: () => 6, calculateCtx,
      });
      if (!rung) break;
      history.push({ ...rung, ok: rung.ngl <= 10 });
    }
    expect(history.length).toBeLessThanOrEqual(24);
  });
});

// Regression, reproduced from a real production run (probe dbe15026, worker
// log 2026-09-05T17:27): balanced seeded at the user's own 41 layers, on a
// machine where 41 layers is a silent host-backed placement at EVERY context.
// The context phase walked 32768 -> 1024, failed all six loads, and the ladder
// reported total failure -- never once trying fewer layers, on a box that runs
// the same model at 10 layers.
describe("balanced rescues itself when its pinned placement never fits", () => {
  const NGL_MAX = 41;
  const MAXCTX = 262144;
  const calculateCtx = (ngl: number) => (ngl >= 12 ? 1024 : Math.max(1024, 72192 - ngl * 5400));

  function walk(opts: { hostBacked: boolean }): { ctx: number; ngl: number; ok: boolean; hostBacked?: boolean }[] {
    const history: { ctx: number; ngl: number; ok: boolean; hostBacked?: boolean }[] = [];
    for (let i = 0; i < 24; i++) {
      const rung = nextLadderRung({
        mode: "balanced", granularity: "basic", candidateCtx: 32768, candidateNgl: NGL_MAX,
        nglMax: NGL_MAX, maxCtx: MAXCTX, history, calculateNgl: () => 6, calculateCtx,
      });
      if (!rung) break;
      const ok = rung.ngl <= 10;
      history.push({ ...rung, ok, hostBacked: opts.hostBacked ? !ok : undefined });
    }
    return history;
  }

  it("finds a working placement instead of giving up", () => {
    const history = walk({ hostBacked: true });
    expect(history.some((h) => h.ok)).toBe(true);
    const best = history.filter((h) => h.ok).reduce((a, b) => (b.ctx > a.ctx || (b.ctx === a.ctx && b.ngl > a.ngl) ? b : a));
    expect(best.ngl).toBe(10);
    expect(best.ctx).toBe(MAXCTX);
  });

  it("stops walking context down once a failure is known to be host-backed", () => {
    // The layers are in system RAM; a smaller context cannot change that, so
    // the context phase must not spend the whole ladder proving it.
    const atSeedNgl = walk({ hostBacked: true }).filter((h) => h.ngl === NGL_MAX);
    expect(atSeedNgl).toHaveLength(1);
  });

  it("still walks the context ladder when the failure reason is unknown", () => {
    // A worker that reports no reason (older build, or a genuine capacity
    // failure) keeps the original behaviour -- context really might be the
    // problem, so it is still worth walking down.
    const atSeedNgl = walk({ hostBacked: false }).filter((h) => h.ngl === NGL_MAX);
    expect(atSeedNgl.length).toBeGreaterThan(1);
  });

  it("recovers even without the reason, just more expensively", () => {
    const history = walk({ hostBacked: false });
    expect(history.some((h) => h.ok)).toBe(true);
  });
});
