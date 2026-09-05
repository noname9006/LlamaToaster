// BENCHMARKING_PLAN_V8.md N2 -- the probe's search strategy, as pure
// functions so the whole ladder is testable without loading a model. The
// process/HTTP plumbing that consumes this lives in worker/src/index.ts, the
// same split loadDriver.ts has against runtimeBench.ts.
//
// Three ideas carry the whole design:
//
//  1. THE CTX AXIS ALWAYS WALKS A REAL GRID, NEVER LANDS ON AN ARBITRARY
//     NUMBER. Every ctx-axis phase, at BOTH granularities, walks the same
//     power-of-two ladder (1024, 2048, 4096, ...) one notch at a time from a
//     seed near the mode's own target (nextSliderCandidate). "basic" is
//     exactly that walk, converged. "fine" is the SAME walk, continued: once
//     it converges on an adjacent-stops bracket, refine WITHIN that bracket
//     via ordinary bisection, down to a minimum step of 1/16 of the
//     bracket's own (fixed, not re-shrinking) width (nextSliderRefineCandidate).
//     The ngl axis has no such grid -- every integer layer count is already
//     individually selectable -- so it keeps the original continuous
//     bisection (nextAnchoredCandidate): a failure bisects toward a SHRINK
//     target (typically the safe-ngl estimate), a success toward a GROW
//     target (typically the axis's own hard ceiling), and once both a
//     success and a failure are known, it bisects the two real values
//     directly. One phase trusts its shrink target enough to skip the
//     approach entirely: max_gpu's layer phase (nextDirectCandidate) tests
//     the safe-ngl estimate outright on the seed's first failure instead of
//     stepping toward it.
//
//  2. A MODE IS A SEQUENCE OF SINGLE-AXIS PHASES. Rather than moving two axes
//     at once (which has no well-defined bracket), each mode resolves one
//     axis, pins it, then resolves the next. A phase whose seed already IS
//     its own grow/shrink target (e.g. max_gpu's layer phase seeded at every
//     layer, with nowhere higher to go) converges in exactly one load when
//     that seed works -- this is what makes "if it succeeds outright, skip
//     straight to the next phase" fall out of the general rule rather than
//     needing its own special case.
//
//  3. max_context's "highest possible context is the top priority" PINS ITS
//     OWN STARTING ngl BY SEARCHING FOR IT, not by assuming it. Before its
//     phase 0 even runs, bestNglForMaxContext climbs calculateCtx(ngl) over
//     every candidate layer count -- pure estimate evaluations, zero real
//     loads spent -- since the VRAM/RAM dual-pool split is not necessarily
//     monotonic in ngl. Phase 1 then only ever ADDS layers on top of
//     whatever ctx phase 0 found, never sacrificing ctx to do it.
//
// Two rules keep a "grow more layers" phase from spending its budget on a
// staircase that cannot succeed, and keep max_gpu from reporting a context
// nobody can use:
//
//  4. EVERY GROW-LAYERS PHASE IS BOUNDED BY THE ESTIMATE, NOT BY nglMax.
//     bestNglAtContext answers "the most layers whose estimate still affords
//     the context this phase has pinned" -- again pure calculateCtx()
//     evaluations, zero loads. That answer becomes the phase's growTarget
//     (jumped to directly, not halved toward), so a failure brackets against
//     it instead of against every layer; and when it says no additional
//     layer fits at all, a pure grow phase (max_context/balanced phase 1) is
//     skipped outright rather than descending one integer at a time proving
//     the estimate right.
//
//  5. max_gpu BACKS OFF WHEN ITS CONTEXT COLLAPSES TO THE FLOOR. Its layer
//     phase deliberately pins ctx at PROBE_LADDER_MIN_CTX, so "every layer
//     that fits" can be an offload that leaves room for nothing else. When
//     the context phase then converges at that same floor, two more phases
//     run: give back the fewest layers that let the NEXT stop up load
//     (phases 2), then re-walk the context ladder from there (phase 3). The
//     mode still reports the largest context that loaded, so the back-off
//     only ever wins when it actually bought context.

// --- Public vocabulary ------------------------------------------------------

export type ProbeMode = "max_gpu" | "max_context" | "keep_context" | "balanced" | "fixed_offload" | "custom";
export type ProbeGranularity = "basic" | "fine";

export const PROBE_MODES: readonly ProbeMode[] = [
  "max_gpu",
  "max_context",
  "keep_context",
  "balanced",
  "fixed_offload",
  "custom",
];
export const PROBE_GRANULARITIES: readonly ProbeGranularity[] = ["basic", "fine"];

export function isProbeMode(value: unknown): value is ProbeMode {
  return typeof value === "string" && (PROBE_MODES as readonly string[]).includes(value);
}

export function isProbeGranularity(value: unknown): value is ProbeGranularity {
  return typeof value === "string" && (PROBE_GRANULARITIES as readonly string[]).includes(value);
}

/**
 * Nothing is ever probed below this. Deliberately ABOVE the server's absolute
 * MIN_PROBE_CTX (256, routes/measurements.ts): that bound stays where it is so
 * rows written before this ladder existed still validate, while the ladder
 * itself refuses to spend a real model load proving a context nobody would
 * run.
 */
export const PROBE_LADDER_MIN_CTX = 1024;

/**
 * Hard ceiling on real model loads per probe, across every phase. The server
 * enforces the same number independently (MAX_PROBE_ATTEMPTS) so a
 * misbehaving worker cannot report an unbounded ladder. Admin-configurable
 * at runtime (AppSettings.probeMaxLoads, propagated over the worker
 * heartbeat) -- this is only the default a caller falls back to when no
 * live setting has reached it yet (LadderInput.maxLoads overrides it).
 */
export const PROBE_MAX_LOADS = 24;

/** ngl converges once a bracket is this narrow -- every integer is already
 * individually selectable, so there is no coarser "basic" grid to round to
 * the way context has. */
const NGL_TOLERANCE = 1;

// --- The anchored bisection primitive ---------------------------------------

export interface AnchoredOutcome {
  value: number;
  ok: boolean;
}

export interface AnchoredBisectInput {
  /** This phase's own outcomes only, in the order they were tested. */
  history: AnchoredOutcome[];
  /** Where to bisect toward while every outcome so far has succeeded. */
  growTarget: number;
  /** Where to bisect toward while every outcome so far has failed. */
  shrinkTarget: number;
  min: number;
  max: number;
  /** Stop once the live bracket (or the gap to a target) is this narrow. */
  tolerance: number;
}

/**
 * The next value to try on one axis, or null once this phase has converged.
 * The seed itself is NOT produced here -- the caller already knows it (it's
 * the phase's own definition), so this only ever answers "given what has
 * already been tried, what's next."
 *
 * Both axes are monotone in the same direction -- a smaller context, or
 * fewer GPU layers, is never harder to fit than a larger one -- so "fits" is
 * a downward-closed set and the boundary is a single crossing point.
 *
 * Decision order:
 *   1. A bracket exists (a success AND a failure are both known) -> bisect
 *      their mean directly -- both ends are real, already-tested values, so
 *      converging here needs no further verification.
 *   2. Only successes so far -> bisect toward growTarget. A seed that IS
 *      already its own growTarget (nothing higher to try, e.g. every GPU
 *      layer) converges here after exactly one success.
 *   3. Only failures so far -> bisect toward shrinkTarget, the same rule
 *      pointed the other way.
 *
 * Cases 2 and 3 bisect toward a TARGET the caller supplied but this function
 * has never itself tested. Once the live gap to that target closes to within
 * tolerance, the honest next move is to test the target exactly -- NOT to
 * declare victory on proximity alone. Skipping that would mean, in the
 * all-failures case, reporting "nothing fits" without ever having tried the
 * one value (shrinkTarget) that might actually work. Only once the target
 * itself has been tried (win or lose) does this return null.
 *
 * A stray result on the wrong side of an already-proven value (a flaky load,
 * or another process taking memory mid-probe) is ignored: only a failure
 * ABOVE the best success, or a success BELOW the worst qualifying failure,
 * ever defines the bracket.
 */
export function nextAnchoredCandidate(input: AnchoredBisectInput): number | null {
  const { history, min, max, tolerance } = input;
  if (history.length === 0) return null;

  const tried = new Set(history.map((h) => h.value));
  const finalize = (raw: number): number | null => {
    const candidate = clamp(Math.round(raw), min, max);
    return tried.has(candidate) ? null : candidate;
  };

  const goods = history.filter((h) => h.ok).map((h) => h.value);
  const bads = history.filter((h) => !h.ok).map((h) => h.value);
  const bestGood = goods.length > 0 ? Math.max(...goods) : null;
  const badsAbove = bads.filter((b) => bestGood === null || b > bestGood);
  const worstBad = badsAbove.length > 0 ? Math.min(...badsAbove) : null;

  if (bestGood !== null && worstBad !== null) {
    if (worstBad - bestGood <= tolerance) return null;
    return finalize((bestGood + worstBad) / 2);
  }
  if (bestGood !== null) {
    if (Math.abs(input.growTarget - bestGood) <= tolerance) return finalize(input.growTarget);
    return finalize((bestGood + input.growTarget) / 2);
  }
  const worstTried = bads.length > 0 ? Math.min(...bads) : null;
  if (worstTried === null) return null;
  if (Math.abs(worstTried - input.shrinkTarget) <= tolerance) return finalize(input.shrinkTarget);
  return finalize((worstTried + input.shrinkTarget) / 2);
}

/**
 * Variant of nextAnchoredCandidate for phases where growTarget/shrinkTarget
 * is not just a direction to average toward but a trusted precalculated
 * estimate worth testing outright: the FIRST probe on either side of the
 * seed jumps straight to that target (shrinkTarget after a failure,
 * growTarget after a success) instead of nextAnchoredCandidate's gradual
 * halving toward it. Once the target itself has ALSO been tried, this falls
 * back to ordinary bracket bisection -- and if it was the target that
 * failed (the estimate was wrong), retargets toward the axis's own hard
 * bound (min on the shrink side, max on the grow side) rather than getting
 * stuck re-offering the same disproven value forever.
 *
 * Used by max_gpu's layer phase (BENCHMARKING_PLAN_V8.md N2): when every
 * layer fails to fit, the very next load should be the safe-estimate layer
 * count, not an arbitrary point between it and the ceiling.
 */
export function nextDirectCandidate(input: AnchoredBisectInput): number | null {
  const { history, min, max, tolerance } = input;
  if (history.length === 0) return null;

  const tried = new Set(history.map((h) => h.value));
  const finalize = (raw: number): number | null => {
    const candidate = clamp(Math.round(raw), min, max);
    return tried.has(candidate) ? null : candidate;
  };

  const goods = history.filter((h) => h.ok).map((h) => h.value);
  const bads = history.filter((h) => !h.ok).map((h) => h.value);
  const bestGood = goods.length > 0 ? Math.max(...goods) : null;
  const badsAbove = bads.filter((b) => bestGood === null || b > bestGood);
  const worstBad = badsAbove.length > 0 ? Math.min(...badsAbove) : null;

  if (bestGood !== null && worstBad !== null) {
    if (worstBad - bestGood <= tolerance) return null;
    return finalize((bestGood + worstBad) / 2);
  }

  if (bestGood !== null) {
    const growCandidate = clamp(Math.round(input.growTarget), min, max);
    if (!tried.has(growCandidate)) return finalize(input.growTarget);
    // growTarget itself already tried (and it also succeeded, or every
    // candidate this rule could offer clamps onto it) -- fall back to
    // bisecting the remaining room up to the hard ceiling.
    if (Math.abs(bestGood - max) <= tolerance) return finalize(max);
    return finalize((bestGood + max) / 2);
  }

  const shrinkCandidate = clamp(Math.round(input.shrinkTarget), min, max);
  if (!tried.has(shrinkCandidate)) return finalize(input.shrinkTarget);
  // The trusted estimate itself failed -- there is no more "trusted" value
  // left to jump to, so retarget toward the hard floor via ordinary
  // bisection, same as nextAnchoredCandidate's own shrink-target rule.
  const worstTried = Math.min(...bads);
  if (Math.abs(worstTried - min) <= tolerance) return finalize(min);
  return finalize((worstTried + min) / 2);
}

export interface SliderStepInput {
  /** This phase's own outcomes only, in the order they were tested. */
  history: AnchoredOutcome[];
  /** The full discrete ladder this phase may test, ascending, deduped. */
  stops: readonly number[];
}

/**
 * The next slider-aligned candidate for a "basic"-granularity context phase:
 * one notch up the ladder on a success, one notch down on a failure -- so
 * every value this phase ever tests is a real slider stop a user could
 * actually pick, unlike nextAnchoredCandidate's continuous bisection (which
 * happily lands on numbers like 260104 that mean nothing to the UI).
 * Converges the instant the walk reverses (a failure immediately above a
 * success, or vice versa): those two stops are already adjacent on the
 * ladder, so there is nothing coarser left to try between them.
 */
export function nextSliderCandidate(input: SliderStepInput): number | null {
  const { history, stops } = input;
  if (history.length === 0 || stops.length === 0) return null;
  const last = history[history.length - 1];
  const anchorIdx = stops.indexOf(last.value);
  const idx = anchorIdx >= 0 ? anchorIdx : nearestStopIndex(stops, last.value);
  const nextIdx = last.ok ? idx + 1 : idx - 1;
  if (nextIdx < 0 || nextIdx >= stops.length) return null;
  const candidate = stops[nextIdx];
  return history.some((h) => h.value === candidate) ? null : candidate;
}

function nearestStopIndex(stops: readonly number[], target: number): number {
  let best = 0;
  let bestDist = Math.abs(target - stops[0]);
  for (let i = 1; i < stops.length; i++) {
    const d = Math.abs(target - stops[i]);
    if (d < bestDist) {
      best = i;
      bestDist = d;
    }
  }
  return best;
}

export interface SliderRefineInput {
  /** This phase's own outcomes only, in the order they were tested. */
  history: AnchoredOutcome[];
  /** The full discrete ladder this phase may test, ascending, deduped. */
  stops: readonly number[];
  granularity: ProbeGranularity;
  min: number;
  max: number;
}

/**
 * The ctx axis's whole search, at BOTH granularities: walk the power-of-two
 * ladder (nextSliderCandidate) until it converges on an adjacent-stops
 * bracket -- that IS "basic"'s complete answer, and every value tested along
 * the way is a real stop the UI can express. "fine" is not a different
 * search, it is the SAME walk with extra steps afterward: once the walk
 * converges, refine WITHIN that specific bracket via ordinary bracket
 * bisection (nextAnchoredCandidate), down to a minimum step of 1/16 of the
 * bracket's OWN width. That width is fixed once, from the literal
 * slider-stop pair the walk itself found (identified by filtering history
 * down to values that are actual stops) -- not recomputed smaller on every
 * subsequent refine step, which would otherwise quietly turn "never finer
 * than a 16th" into a 16th of an already-narrowed gap a few loads later.
 *
 * Only ever consults nextSliderCandidate while EVERY outcome so far is a
 * real stop -- the instant refine has tested one off-grid value, that value
 * is neither in `stops` nor found by indexOf, so nextSliderCandidate's own
 * nearest-stop fallback (built for snapping an odd SEED, not a bisection
 * midpoint) would otherwise silently reinterpret it as "closest to
 * whichever stop is nearest" and jump the walk sideways instead of
 * continuing the bisection.
 */
export function nextSliderRefineCandidate(input: SliderRefineInput): number | null {
  const { history, stops, granularity, min, max } = input;
  const stopSet = new Set(stops);
  const stillWalking = history.every((h) => stopSet.has(h.value));
  if (stillWalking) {
    const sliderNext = nextSliderCandidate({ history, stops });
    if (sliderNext !== null) return sliderNext;
  }
  if (granularity !== "fine") return null;

  const stopOutcomes = history.filter((h) => stopSet.has(h.value));
  const stopGoods = stopOutcomes.filter((h) => h.ok).map((h) => h.value);
  const stopBads = stopOutcomes.filter((h) => !h.ok).map((h) => h.value);
  const bestStopGood = stopGoods.length > 0 ? Math.max(...stopGoods) : null;
  const stopBadsAbove = stopBads.filter((b) => bestStopGood === null || b > bestStopGood);
  const worstStopBad = stopBadsAbove.length > 0 ? Math.min(...stopBadsAbove) : null;
  // The walk never bracketed at all (ran off the top or bottom of the
  // ladder without ever seeing both an ok and a fail) -- nothing to refine.
  if (bestStopGood === null || worstStopBad === null) return null;

  const bracketWidth = worstStopBad - bestStopGood;
  const tolerance = Math.max(1, Math.round(bracketWidth / 16));
  return nextAnchoredCandidate({ history, growTarget: worstStopBad, shrinkTarget: bestStopGood, min, max, tolerance });
}

/**
 * max_context's own pre-step, entirely calculateCtx() evaluations -- zero
 * real loads spent, since this is just re-running a cheap formula -- to find
 * which layer count actually maximizes affordable context. "Highest possible
 * context is the top priority" means phase 0 must not just assume the
 * ceiling sits at the user's own ngl, or even at ngl=0: the VRAM/RAM
 * dual-pool split is not necessarily monotonic in ngl (a little GPU offload
 * can sometimes free more room than either extreme), so this climbs the
 * curve rather than assuming a shape for it. Coarse step-2 climb from 0
 * until the estimate stops improving, then a step-1 refine from that coarse
 * peak -- BOTH directions, not just back down through the skipped odd
 * values below it, since the true (odd-resolution) peak can sit just above
 * the coarse peak too (between it and the next even point the coarse pass
 * rejected). Each direction keeps stepping while it keeps improving.
 */
export function bestNglForMaxContext(nglMax: number, calculateCtx: (ngl: number) => number): number {
  const ceiling = Math.max(0, Math.floor(nglMax));

  let bestNgl = 0;
  let bestCtx = calculateCtx(0);
  for (let ngl = 2; ngl <= ceiling; ngl += 2) {
    const ctx = calculateCtx(ngl);
    if (ctx <= bestCtx) break;
    bestNgl = ngl;
    bestCtx = ctx;
  }

  for (const step of [1, -1] as const) {
    for (let ngl = bestNgl + step; ngl >= 0 && ngl <= ceiling; ngl += step) {
      const ctx = calculateCtx(ngl);
      if (ctx <= bestCtx) break;
      bestNgl = ngl;
      bestCtx = ctx;
    }
  }

  return bestNgl;
}

/**
 * The most layers whose ESTIMATE still affords `targetCtx`, never below
 * `floorNgl` -- the grow-side counterpart to bestNglForMaxContext, and like
 * it, pure calculateCtx() evaluations with zero real loads spent.
 *
 * Every "now add layers at the context we just resolved" phase used to grow
 * toward nglMax, which is almost never reachable: phase 0 has just pushed
 * context to that placement's real boundary, so the layer counts between the
 * seed and nglMax are exactly the ones that cannot fit. Bisecting that range
 * spends the whole remaining budget walking a descending staircase of
 * failures. Bounding the target by the estimate instead turns the same
 * search into "test the estimate, then bracket against it".
 *
 * Deliberately a full scan rather than a climb-until-worse: calculateCtx is
 * not guaranteed monotonic in ngl (the same dual-pool reason
 * bestNglForMaxContext scans), and nglMax is at most a few hundred.
 * Returns floorNgl when nothing above it affords the target -- callers read
 * "the estimate says there is no room to grow" from `result <= seed`.
 */
export function bestNglAtContext(
  nglMax: number,
  targetCtx: number,
  calculateCtx: (ngl: number) => number,
  floorNgl = 0
): number {
  const ceiling = Math.max(0, Math.floor(nglMax));
  const floor = clamp(Math.floor(floorNgl), 0, ceiling);
  let best = floor;
  for (let ngl = floor + 1; ngl <= ceiling; ngl++) {
    if (calculateCtx(ngl) >= targetCtx) best = ngl;
  }
  return best;
}

// --- Modes as phase sequences ------------------------------------------------

export type LadderAxis = "ctx" | "ngl";

interface PhaseSpec {
  axis: LadderAxis;
  seed: number;
  growTarget: number;
  shrinkTarget: number;
  /**
   * "bisect" (default ngl engine): nextAnchoredCandidate, continuous.
   * "direct" (max_gpu's layer phase only): nextDirectCandidate -- jump
   * straight to the estimate on the first probe, bisect afterward.
   * "slider_refine" (every ctx-axis phase, both granularities):
   * nextSliderRefineCandidate -- walk the power-of-two ctx ladder, then
   * (fine only) refine within the bracket it converges on.
   */
  searchStyle: "bisect" | "direct" | "slider_refine";
  /**
   * Re-pin the OTHER axis at this value for the duration of this phase,
   * instead of inheriting whatever the previous phase resolved it to. Only
   * max_gpu's back-off phases use it: giving layers back is only meaningful
   * measured against a context above the floor the layer phase pinned, so
   * that phase has to move ctx before it can search ngl again.
   */
  pinOther?: number;
  /**
   * Marks the "the context axis found nothing at this placement, back the
   * layers off and try again" phase. The mode after it needs to know the
   * rescue actually ran, since it re-walks context at whatever placement the
   * rescue settled on.
   */
  isRescue?: boolean;
}

/**
 * The `phaseIndex`'th phase a mode runs, given whatever the OTHER axis is
 * currently pinned to (the previous phase's own resolution, or the caller's
 * starting candidate before any phase has run) -- null once the mode has no
 * more phases. `calculateNgl`/`calculateCtx` are the caller's real estimator
 * calls (estimateSafeNgl / maxAffordableContext in shared/vramEstimate.ts);
 * this module never imports that file directly, so a phase whose target
 * depends on the estimate asks for it fresh, at the axis value THIS phase
 * actually pins the other one to.
 *
 *  * `max_gpu`      -- phase 0: every layer, shrinking toward the safe
 *                      estimate on failure (nowhere higher to grow -- grow
 *                      target IS the seed, so a working seed converges in one
 *                      load). phase 1: the ctx stop nearest the max-affordable
 *                      estimate at the now-resolved layers, walking the
 *                      power-of-two ladder from there. phases 2+3 run ONLY
 *                      when phase 1 converged at the floor context the layer
 *                      phase had pinned: give back the fewest layers that let
 *                      the next stop up load (phase 2), then re-walk the ctx
 *                      ladder at that placement (phase 3). Whichever rung
 *                      reached the larger context wins the verdict, so the
 *                      back-off can only ever help.
 *  * `max_context`  -- phase 0: the ctx stop nearest the max-affordable
 *                      estimate, pinned at whichever ngl bestNglForMaxContext
 *                      found maximizes that estimate (NOT necessarily the
 *                      user's own ngl, or 0 -- "highest possible context" is
 *                      searched for, not assumed). phase 1: THAT SAME pinned
 *                      layer count -- not the user's own -- already proven at
 *                      the resolved context by phase 0's own search (phase 0
 *                      never moves ngl), so the seed always comes from
 *                      history rather than a real load, growing toward
 *                      whatever bestNglAtContext says the resolved context
 *                      still affords (see growLayersPhase), never shrinking
 *                      ctx back down to make room.
 *  * `balanced`     -- same two phases as max_context, seeded at the user's
 *                      OWN current (ctx, ngl) instead of the estimate/every
 *                      layer -- no ngl pre-step, "trades context against
 *                      layers" is the whole point of this mode.
 *  * `keep_context` -- ("Fixed context") one phase: the user's own layers,
 *                      growing toward whatever the estimate still affords at
 *                      that fixed context, shrinking toward the safe-ngl
 *                      estimate -- jumping directly to whichever of the two
 *                      the seed's own verdict points at.
 *  * `fixed_offload`-- one phase: the ctx stop nearest the user's own ctx,
 *                      walking the power-of-two ladder at the user's fixed
 *                      layers.
 *  * `custom`       -- no phases; handled directly in nextLadderRung.
 */
interface PhaseState {
  /** The phase just before this one produced no successes at all. */
  lastPhaseAllFailed: boolean;
  /** A rescue phase has already run in this mode. */
  rescueRan: boolean;
}

function phaseSpecFor(
  mode: ProbeMode,
  phaseIndex: number,
  pinnedCtx: number,
  pinnedNgl: number,
  nglMax: number,
  maxCtx: number,
  calculateNgl: (pinnedCtx: number) => number,
  calculateCtx: (pinnedNgl: number) => number,
  state: PhaseState
): PhaseSpec | null {
  const clampNgl = (v: number): number => clamp(Math.round(v), 0, nglMax);
  const clampCtx = (v: number): number => clamp(Math.round(v), PROBE_LADDER_MIN_CTX, maxCtx);
  // Every ctx-axis phase below walks the SAME power-of-two ladder, seeded at
  // whichever real value the mode cares about (an estimate, or the user's
  // own ctx) snapped to the nearest stop -- growTarget/shrinkTarget are
  // vestigial for these (nextSliderRefineCandidate drives the walk off the
  // stops list alone), kept only so PhaseSpec has one shape for both axes.
  const ctxPhase = (seedTarget: number): PhaseSpec => ({
    axis: "ctx",
    seed: nearestStop(ctxLadderStops(maxCtx), clampCtx(seedTarget)),
    growTarget: maxCtx,
    shrinkTarget: PROBE_LADDER_MIN_CTX,
    searchStyle: "slider_refine",
  });
  /**
   * "Now add layers at the context we just resolved" -- max_context's and
   * balanced's phase 1. The seed is always a rung phase 0 already loaded
   * (phase 0 never moves ngl), so it costs no load; what this decides is how
   * far above it to look. That ceiling is the ESTIMATE's answer, not nglMax:
   * phase 0 has just pushed context to this placement's real boundary, so
   * growing toward every layer means bisecting a range whose every member is
   * known-hopeless. Null -- no phase at all -- when the estimate says not
   * even one more layer affords this context, which is the common case after
   * a `fine` refine and used to cost five or six guaranteed failures.
   */
  /**
   * "The context axis found nothing at this placement" -- back the layers off
   * and search again at the cheapest context.
   *
   * Without this, a mode whose pinned layer count is unusable at EVERY context
   * simply reports total failure: confirmed live, a balanced probe seeded at
   * 41 layers walked 32768 -> 1024 and failed all six loads, never once trying
   * fewer layers, on a machine that runs the same model happily at 10. That is
   * the opposite of what "trades context against layers" promises, and it got
   * much more likely once silent host-backed placements started failing
   * honestly instead of passing with a warning.
   *
   * Shrink-only and pinned at the floor context: the point is to find ANY
   * workable placement cheaply, after which the mode re-walks context from
   * there. Jumps straight to the safe-ngl estimate rather than bisecting
   * toward it, same reasoning as max_gpu's own layer phase.
   */
  const rescueLayersPhase = (nglNow: number): PhaseSpec | null => {
    if (nglNow <= 0) return null;
    const ceiling = clampNgl(nglNow - 1);
    const safe = clampNgl(calculateNgl(PROBE_LADDER_MIN_CTX));
    return {
      axis: "ngl",
      pinOther: PROBE_LADDER_MIN_CTX,
      seed: Math.min(ceiling, safe),
      growTarget: ceiling,
      shrinkTarget: 0,
      searchStyle: "direct",
      isRescue: true,
    };
  };

  const growLayersPhase = (ctxNow: number, nglNow: number): PhaseSpec | null => {
    if (nglNow >= nglMax) return null;
    const estimateSaysGrow = bestNglAtContext(nglMax, ctxNow, calculateCtx, nglNow);
    // When the estimate says not even one more layer affords this context,
    // that used to end the mode. It must not, because the condition is
    // self-inflicted: phase 0 walks context up until a real load FAILS, so a
    // machine that outperforms the estimate lands on a ctxNow above
    // calculateCtx(nglNow) by construction -- and then no ngl at all
    // "affords" it, every time, and the layer axis is never measured. On a
    // GPU whose driver silently backs overcommitted allocations with system
    // RAM, the estimate is not what decides placement anyway (measured: the
    // driver stopped taking weights with 1.9GiB of VRAM still free), so
    // treating it as a veto over whether to MEASURE is exactly backwards.
    //
    // So the estimate now only chooses the target to jump to, never whether
    // to run at all: its own safe-ngl answer when it has one, and otherwise
    // a single step up, which costs one load and brackets immediately if it
    // fails. "direct" jumps straight to the target rather than climbing, so
    // a hopeless target costs one load, not a staircase of them.
    const grow = estimateSaysGrow > nglNow ? estimateSaysGrow : Math.min(nglMax, nglNow + 1);
    return { axis: "ngl", seed: nglNow, growTarget: grow, shrinkTarget: nglNow, searchStyle: "direct" };
  };

  switch (mode) {
    case "max_gpu":
      if (phaseIndex === 0) {
        return {
          axis: "ngl",
          seed: nglMax,
          growTarget: nglMax,
          shrinkTarget: clampNgl(calculateNgl(PROBE_LADDER_MIN_CTX)),
          // Nowhere higher than nglMax to jump to, so "direct" only ever
          // bites on the shrink side: a failed full-offload attempt goes
          // straight to the safe estimate next, not a midpoint toward it.
          searchStyle: "direct",
        };
      }
      if (phaseIndex === 1) return ctxPhase(calculateCtx(pinnedNgl));
      // Back-off. The layer phase pinned ctx at the floor while it searched,
      // so "the most layers that fit" can be a placement with room for
      // nothing else -- and phase 1 converging AT that same floor is exactly
      // that outcome. Rather than report a context nobody can run, give back
      // the fewest layers that let the next stop up load, then re-walk the
      // ladder from there. pinnedNgl is known to FAIL at backoffCtx (phase 1
      // just proved it, which is why it converged at the floor), so this
      // phase's own ceiling is one layer below it.
      if (phaseIndex === 2) {
        const backoffCtx = clampCtx(PROBE_LADDER_MIN_CTX * 2);
        // "Still at the floor" is measured against the NEXT stop up, not
        // against the floor exactly: a `fine` context phase refines inside
        // the [floor, next stop] bracket, so it lands on something like 1920
        // -- larger than the floor, and just as unusable. Anything below the
        // stop the back-off is trying to reach counts as not having reached
        // it.
        if (pinnedCtx >= backoffCtx || pinnedNgl < 1 || backoffCtx <= PROBE_LADDER_MIN_CTX) return null;
        const ceiling = clampNgl(pinnedNgl - 1);
        return {
          axis: "ngl",
          pinOther: backoffCtx,
          seed: Math.min(ceiling, bestNglAtContext(nglMax, backoffCtx, calculateCtx)),
          growTarget: ceiling,
          shrinkTarget: 0,
          searchStyle: "bisect",
        };
      }
      // Only reachable when phase 2 actually ran (a null phase ends the
      // mode), so the pin is already the backed-off placement.
      // Seeded at the context phase 2 just PROVED at this placement, not at
      // the estimate: the estimate is what sent the layer phase past a
      // usable context in the first place, and re-seeding on it walks the
      // ladder back down through failures to get here anyway. The proven
      // rung costs no load (history reuse), so this walks up from it.
      if (phaseIndex === 3) return ctxPhase(pinnedCtx);
      return null;

    case "max_context":
      // pinnedNgl here is ALREADY bestNglForMaxContext's answer -- set once,
      // upstream in nextLadderRung, before any phase runs.
      if (phaseIndex === 0) return ctxPhase(calculateCtx(pinnedNgl));
      if (phaseIndex === 1) {
        return state.lastPhaseAllFailed ? rescueLayersPhase(pinnedNgl) : growLayersPhase(pinnedCtx, pinnedNgl);
      }
      // Only after a rescue: re-walk context at the placement it settled on.
      if (phaseIndex === 2 && state.rescueRan) return ctxPhase(calculateCtx(pinnedNgl));
      return null;

    case "balanced":
      if (phaseIndex === 0) return ctxPhase(pinnedCtx);
      if (phaseIndex === 1) {
        return state.lastPhaseAllFailed ? rescueLayersPhase(pinnedNgl) : growLayersPhase(pinnedCtx, pinnedNgl);
      }
      if (phaseIndex === 2 && state.rescueRan) return ctxPhase(calculateCtx(pinnedNgl));
      return null;

    case "keep_context":
      if (phaseIndex === 0) {
        const shrink = clampNgl(calculateNgl(pinnedCtx));
        // Unlike a pure grow phase this one can never be skipped -- it is the
        // mode's ONLY phase, and a seed that fails still has to find its way
        // down. So the estimate bounds the grow direction (never below the
        // seed, which would close it entirely) and the shrink target stays
        // the safe-ngl estimate; "direct" jumps to whichever of the two the
        // seed's own verdict points at.
        const grow = Math.max(pinnedNgl, bestNglAtContext(nglMax, pinnedCtx, calculateCtx, pinnedNgl));
        return { axis: "ngl", seed: pinnedNgl, growTarget: grow, shrinkTarget: shrink, searchStyle: "direct" };
      }
      return null;

    case "fixed_offload":
      if (phaseIndex === 0) return ctxPhase(pinnedCtx);
      return null;

    case "custom":
      return null;
  }
}

// --- The whole ladder --------------------------------------------------------

export interface LadderRung {
  ctx: number;
  ngl: number;
}

export interface LadderAttempt extends LadderRung {
  ok: boolean;
  /**
   * This rung failed because its weights were measurably host-backed, rather
   * than because it did not fit. The distinction matters to the search: a
   * smaller context cannot fix a placement whose LAYERS are in system RAM, so
   * a context phase that hits this stops immediately instead of walking the
   * whole ladder down proving it. Observed live: six loads spent walking
   * 32768 -> 1024 at 41 layers, every one failing for the same reason.
   */
  hostBacked?: boolean;
}

export interface LadderInput {
  mode: ProbeMode;
  granularity: ProbeGranularity;
  candidateCtx: number;
  candidateNgl: number;
  nglMax: number;
  /** The model's trained context -- the ladder never probes above it. */
  maxCtx: number;
  maxLoads?: number;
  /** Every rung already loaded, in order. */
  history: LadderAttempt[];
  /** The estimate's own predicted safe layer count at a given pinned context. */
  calculateNgl: (pinnedCtx: number) => number;
  /** The estimate's own predicted max-affordable context at a given pinned ngl. */
  calculateCtx: (pinnedNgl: number) => number;
}

/**
 * The next rung to load, or null when the ladder is finished.
 *
 * Stateless by design: the worker hands back the full history each time, so
 * a probe's decisions can be replayed from its stored probe_attempts rows.
 * The budget is a single pool shared across every phase (not pre-split): a
 * phase that converges in one load leaves the rest for whichever phase needs
 * it, rather than wasting a fixed share on a phase that didn't need it.
 */
export function nextLadderRung(input: LadderInput): LadderRung | null {
  const maxLoads = input.maxLoads ?? PROBE_MAX_LOADS;
  if (input.history.length >= maxLoads) return null;

  const nglMax = Math.max(0, Math.floor(input.nglMax));
  const maxCtx = Math.max(PROBE_LADDER_MIN_CTX, Math.floor(input.maxCtx));

  if (input.mode === "custom") {
    const ctx = clamp(Math.floor(input.candidateCtx), PROBE_LADDER_MIN_CTX, maxCtx);
    const ngl = clamp(Math.floor(input.candidateNgl), 0, nglMax);
    return input.history.length === 0 ? { ctx, ngl } : null;
  }

  // max_gpu's whole definition is "start at every layer and the floor
  // context" -- it ignores the caller's own candidate entirely, the same way
  // "custom" above ignores everything BUT the caller's candidate. max_context
  // similarly ignores the user's own ngl for its STARTING pin -- "highest
  // possible context is top priority" means searching for whichever layer
  // count actually maximizes the estimate (bestNglForMaxContext, pure
  // estimate evaluations, zero real loads), not assuming it's the user's own.
  let pinnedCtx =
    input.mode === "max_gpu" ? PROBE_LADDER_MIN_CTX : clamp(Math.floor(input.candidateCtx), PROBE_LADDER_MIN_CTX, maxCtx);
  let pinnedNgl =
    input.mode === "max_gpu"
      ? nglMax
      : input.mode === "max_context"
      ? bestNglForMaxContext(nglMax, input.calculateCtx)
      : clamp(Math.floor(input.candidateNgl), 0, nglMax);
  // Forward replay: history is a flat list with no phase markers, so the only
  // way to know which rung belonged to which phase is to re-run the same
  // deterministic decisions and consume history in lockstep.
  let cursor = 0;
  const state: PhaseState = { lastPhaseAllFailed: false, rescueRan: false };

  for (let phaseIdx = 0; ; phaseIdx++) {
    const spec = phaseSpecFor(
      input.mode,
      phaseIdx,
      pinnedCtx,
      pinnedNgl,
      nglMax,
      maxCtx,
      input.calculateNgl,
      input.calculateCtx,
      state
    );
    if (!spec) break;
    if (spec.isRescue) state.rescueRan = true;
    // A phase that re-pins the other axis (max_gpu's back-off) must do so
    // BEFORE its first rung is built, and the new pin has to persist into
    // the phases after it -- phase 3 re-walks the ctx ladder at exactly the
    // placement phase 2 settled on.
    if (spec.pinOther != null) {
      if (spec.axis === "ngl") pinnedCtx = clamp(Math.round(spec.pinOther), PROBE_LADDER_MIN_CTX, maxCtx);
      else pinnedNgl = clamp(Math.round(spec.pinOther), 0, nglMax);
    }

    const min = spec.axis === "ctx" ? PROBE_LADDER_MIN_CTX : 0;
    const max = spec.axis === "ctx" ? maxCtx : nglMax;
    const tolerance = NGL_TOLERANCE; // only "bisect"/"direct" (ngl-axis) phases consume this -- ctx-axis phases compute their own
    const stops = spec.axis === "ctx" ? ctxLadderStops(maxCtx) : null;

    const outcomes: AnchoredOutcome[] = [];
    let candidate: number | null = spec.seed;

    while (candidate !== null) {
      const rung: LadderRung = spec.axis === "ctx" ? { ctx: candidate, ngl: pinnedNgl } : { ctx: pinnedCtx, ngl: candidate };

      // A phase's own seed is frequently a rung an earlier phase already
      // loaded (max_context/balanced's layer phase starts exactly where the
      // context phase already proved it). Reuse that verdict instead of
      // spending a real model load re-proving it.
      const prior = input.history.slice(0, cursor).find((h) => h.ctx === rung.ctx && h.ngl === rung.ngl);
      let ok: boolean;
      let hostBacked = false;
      if (prior) {
        ok = prior.ok;
        hostBacked = prior.hostBacked === true;
      } else if (cursor < input.history.length) {
        ok = input.history[cursor].ok;
        hostBacked = input.history[cursor].hostBacked === true;
        cursor++;
      } else {
        if (cursor >= maxLoads) return null; // out of budget entirely
        return rung; // not loaded yet: run it now
      }

      outcomes.push({ value: candidate, ok });
      if (ok) {
        if (spec.axis === "ctx") pinnedCtx = candidate;
        else pinnedNgl = candidate;
      }

      // A context phase cannot fix a host-backed PLACEMENT: the layers are in
      // system RAM regardless of how small the context gets. Stop the walk on
      // the first such failure and let the next phase move the other axis --
      // this is the difference between one wasted load and six.
      if (spec.axis === "ctx" && !ok && hostBacked) break;

      candidate =
        spec.searchStyle === "slider_refine"
          ? nextSliderRefineCandidate({ history: outcomes, stops: stops!, granularity: input.granularity, min, max })
          : spec.searchStyle === "direct"
          ? nextDirectCandidate({ history: outcomes, growTarget: spec.growTarget, shrinkTarget: spec.shrinkTarget, min, max, tolerance })
          : nextAnchoredCandidate({ history: outcomes, growTarget: spec.growTarget, shrinkTarget: spec.shrinkTarget, min, max, tolerance });
    }

    // Nothing on this axis fit at all. On the NGL axis that is terminal --
    // the search already went as low as 0 layers. On the CTX axis it is not:
    // a later phase moving the other axis genuinely can rescue it, and
    // treating it as terminal is what made a balanced probe give up after six
    // failures without ever trying fewer layers. The mode decides, via
    // PhaseState.lastPhaseAllFailed; if it offers no rescue, the loop ends at
    // the next phaseSpecFor returning null anyway.
    state.lastPhaseAllFailed = outcomes.length > 0 && !outcomes.some((o) => o.ok);
    if (state.lastPhaseAllFailed && spec.axis === "ngl") return null;
  }
  return null;
}

/**
 * The rung a probe should report as its verdict: the largest context that
 * passed, preferring the placement that achieved it.
 */
export function bestLadderResult(history: LadderAttempt[]): LadderAttempt | null {
  const passing = history.filter((h) => h.ok);
  if (passing.length === 0) return null;
  return passing.reduce((best, h) => (h.ctx > best.ctx || (h.ctx === best.ctx && h.ngl > best.ngl) ? h : best));
}

// --- The probe's ctx ladder + the client's OWN separate slider grid --------
//
// Two different discrete grids on purpose:
//  - ctxLadderStops (below): pure power-of-two doublings from the hard floor
//    (1024, 2048, 4096, 8192, ...) up to the model's own ceiling. This is
//    what the basic/fine engine walks and what snapToSafeCtx rounds down to
//    -- every value `basic` probes is one of these; `fine` additionally
//    tests values strictly between two adjacent ones.
//  - computeCtxStops (further below): a fraction-of-trained-ctx ladder
//    (100/75/50/25/12.5/...%) for the CLIENT's own context-target slider
//    display. Unrelated to what the probe itself tests; kept here (not the
//    client) only so nothing else has to duplicate it.

/**
 * The discrete ctx ladder the probe's walk steps along, and the value the
 * apply-to-slider snap rounds down to: doublings from the hard floor up to
 * the model's own ceiling, which is always included exactly even when it
 * isn't itself a power of two -- otherwise the true ceiling could never be
 * reached or reported.
 */
export function ctxLadderStops(maxCtx: number): number[] {
  const ceiling = Math.max(PROBE_LADDER_MIN_CTX, Math.floor(maxCtx));
  const stops: number[] = [];
  for (let v = PROBE_LADDER_MIN_CTX; v < ceiling; v *= 2) stops.push(v);
  stops.push(ceiling);
  return stops;
}

// Roughly 100/75/50/25/12.5/7.5/5/2.5/1% of the model's trained context, each
// rounded to the nearest power of two and forced strictly below the stop
// before it so two fractions can never collide on the same tick once
// rounded. The client's own context-target slider grid ONLY -- see the note
// above; the probe search itself no longer uses this.
const CTX_STOP_FRACTIONS = [1, 0.75, 0.5, 0.25, 0.125, 0.075, 0.05, 0.025, 0.01];

export function computeCtxStops(maxCtx: number): number[] {
  const out: number[] = [];
  let prev = Infinity;
  for (const frac of CTX_STOP_FRACTIONS) {
    let val = frac === 1 ? maxCtx : Math.pow(2, Math.round(Math.log2(Math.max(maxCtx * frac, 1))));
    val = Math.min(val, maxCtx);
    if (val >= prev) val = prev / 2;
    val = Math.max(1, Math.round(val));
    out.push(val);
    prev = val;
  }
  return out.reverse();
}

function nearestStop(stops: readonly number[], target: number): number {
  return stops[nearestStopIndex(stops, target)];
}

/**
 * A verified context rounded to something a slider can actually express.
 *
 * A `fine` search converges on a context BETWEEN two stops. That number is
 * the real measurement and is what gets stored (model_machine_limits'
 * verified_ctx_tokens) and shown -- snapping it at that point was a mistake
 * this once made: refinement only ever happens inside one stop interval, so
 * snapping down always landed back on the stop `basic` had already found,
 * making every extra `fine` load incapable of changing the stored result.
 *
 * The snap belongs where a slider value is genuinely needed instead -- the
 * Benchmark page's "apply this card" action. Down, never up: rounding a
 * verified ceiling upward would claim a context that was never loaded.
 */
export function snapToSafeCtx(verifiedCtx: number, maxCtx: number): number {
  const stops = ctxLadderStops(maxCtx);
  const below = stops.filter((s) => s <= verifiedCtx);
  return below.length > 0 ? Math.max(...below) : Math.min(verifiedCtx, PROBE_LADDER_MIN_CTX);
}

// --- helpers ------------------------------------------------------------

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
