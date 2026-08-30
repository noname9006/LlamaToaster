// BENCHMARKING_PLAN_V8.md N2 -- the probe's search strategy, as pure
// functions so the whole ladder is testable without loading a model. The
// process/HTTP plumbing that consumes this lives in worker/src/index.ts, the
// same split loadDriver.ts has against runtimeBench.ts.
//
// Two ideas carry the whole design:
//
//  1. EVERY SEARCH IS ANCHORED ON THE PRE-FLIGHT ESTIMATE. Rather than
//     doubling blindly outward from a seed, each axis search bisects toward a
//     `calculated` target -- estimateSafeNgl for layers, maxAffordableContext
//     for context -- supplied by the caller (see nextAnchoredCandidate). A
//     failure bisects toward a SHRINK target (typically the estimate, a
//     smaller and presumably-safe value); a success bisects toward a GROW
//     target (typically the estimate or the axis's own hard ceiling, a larger
//     value); once both a success and a failure are known, it bisects between
//     THOSE two real numbers directly. No grid, no index arithmetic -- means
//     of real ctx/ngl values.
//
//  2. A MODE IS A SEQUENCE OF SINGLE-AXIS PHASES. Rather than moving two axes
//     at once (which has no well-defined bracket), each mode resolves one
//     axis, pins it, then resolves the next. A phase whose seed already IS
//     its own grow/shrink target (e.g. max_gpu's layer phase seeded at every
//     layer, with nowhere higher to go) converges in exactly one load when
//     that seed works -- this is what makes "if it succeeds outright, skip
//     straight to the next phase" fall out of the general rule rather than
//     needing its own special case.

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
 * misbehaving worker cannot report an unbounded ladder.
 */
export const PROBE_MAX_LOADS = 8;

/** ngl converges once a bracket is this narrow -- every integer is already
 * individually selectable, so there is no coarser "basic" grid to round to
 * the way context has. */
const NGL_TOLERANCE = 1;

/** ctx converges once the bracket is this fraction of the phase's own seed --
 * Fine tune narrows four times tighter than Basic. */
const CTX_TOLERANCE_FRACTION: Record<ProbeGranularity, number> = { basic: 1 / 16, fine: 1 / 64 };

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

// --- Modes as phase sequences ------------------------------------------------

export type LadderAxis = "ctx" | "ngl";

interface PhaseSpec {
  axis: LadderAxis;
  seed: number;
  growTarget: number;
  shrinkTarget: number;
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
 *                      load). phase 1: the floor context, growing toward the
 *                      max-affordable estimate at the now-resolved layers.
 *  * `max_context`  -- phase 0: the max-affordable estimate itself (grow
 *                      target IS the seed; a failure retreats toward the
 *                      floor). phase 1: the user's own layers -- already
 *                      proven at the resolved context by phase 0's own
 *                      search (phase 0 never moves ngl), so this always
 *                      converges via history reuse before spending a real
 *                      load -- growing toward every layer.
 *  * `balanced`     -- same two phases as max_context, seeded at the user's
 *                      OWN current (ctx, ngl) instead of the estimate/every
 *                      layer.
 *  * `keep_context` -- ("Fixed context") one phase: the user's own layers,
 *                      growing toward every layer, shrinking toward the safe
 *                      estimate at the user's fixed context.
 *  * `fixed_offload`-- one phase: the user's own context, growing toward the
 *                      max-affordable estimate at the user's fixed layers,
 *                      shrinking toward the floor.
 *  * `custom`       -- no phases; handled directly in nextLadderRung.
 */
function phaseSpecFor(
  mode: ProbeMode,
  phaseIndex: number,
  pinnedCtx: number,
  pinnedNgl: number,
  nglMax: number,
  maxCtx: number,
  calculateNgl: (pinnedCtx: number) => number,
  calculateCtx: (pinnedNgl: number) => number
): PhaseSpec | null {
  const clampNgl = (v: number): number => clamp(Math.round(v), 0, nglMax);
  const clampCtx = (v: number): number => clamp(Math.round(v), PROBE_LADDER_MIN_CTX, maxCtx);

  switch (mode) {
    case "max_gpu":
      if (phaseIndex === 0) return { axis: "ngl", seed: nglMax, growTarget: nglMax, shrinkTarget: clampNgl(calculateNgl(PROBE_LADDER_MIN_CTX)) };
      if (phaseIndex === 1) {
        const grow = clampCtx(calculateCtx(pinnedNgl));
        return { axis: "ctx", seed: PROBE_LADDER_MIN_CTX, growTarget: grow, shrinkTarget: PROBE_LADDER_MIN_CTX };
      }
      return null;

    case "max_context":
      if (phaseIndex === 0) {
        const seed = clampCtx(calculateCtx(pinnedNgl));
        return { axis: "ctx", seed, growTarget: seed, shrinkTarget: PROBE_LADDER_MIN_CTX };
      }
      if (phaseIndex === 1) return { axis: "ngl", seed: pinnedNgl, growTarget: nglMax, shrinkTarget: pinnedNgl };
      return null;

    case "balanced":
      if (phaseIndex === 0) {
        const grow = clampCtx(calculateCtx(pinnedNgl));
        return { axis: "ctx", seed: pinnedCtx, growTarget: grow, shrinkTarget: PROBE_LADDER_MIN_CTX };
      }
      if (phaseIndex === 1) return { axis: "ngl", seed: pinnedNgl, growTarget: nglMax, shrinkTarget: pinnedNgl };
      return null;

    case "keep_context":
      if (phaseIndex === 0) {
        const shrink = clampNgl(calculateNgl(pinnedCtx));
        return { axis: "ngl", seed: pinnedNgl, growTarget: nglMax, shrinkTarget: shrink };
      }
      return null;

    case "fixed_offload":
      if (phaseIndex === 0) {
        const grow = clampCtx(calculateCtx(pinnedNgl));
        return { axis: "ctx", seed: pinnedCtx, growTarget: grow, shrinkTarget: PROBE_LADDER_MIN_CTX };
      }
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
  // "custom" above ignores everything BUT the caller's candidate.
  let pinnedCtx =
    input.mode === "max_gpu" ? PROBE_LADDER_MIN_CTX : clamp(Math.floor(input.candidateCtx), PROBE_LADDER_MIN_CTX, maxCtx);
  let pinnedNgl = input.mode === "max_gpu" ? nglMax : clamp(Math.floor(input.candidateNgl), 0, nglMax);
  // Forward replay: history is a flat list with no phase markers, so the only
  // way to know which rung belonged to which phase is to re-run the same
  // deterministic decisions and consume history in lockstep.
  let cursor = 0;

  for (let phaseIdx = 0; ; phaseIdx++) {
    const spec = phaseSpecFor(input.mode, phaseIdx, pinnedCtx, pinnedNgl, nglMax, maxCtx, input.calculateNgl, input.calculateCtx);
    if (!spec) break;

    const min = spec.axis === "ctx" ? PROBE_LADDER_MIN_CTX : 0;
    const max = spec.axis === "ctx" ? maxCtx : nglMax;
    const tolerance =
      spec.axis === "ctx" ? Math.max(1, Math.round(spec.seed * CTX_TOLERANCE_FRACTION[input.granularity])) : NGL_TOLERANCE;

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
      if (prior) {
        ok = prior.ok;
      } else if (cursor < input.history.length) {
        ok = input.history[cursor].ok;
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

      candidate = nextAnchoredCandidate({
        history: outcomes,
        growTarget: spec.growTarget,
        shrinkTarget: spec.shrinkTarget,
        min,
        max,
        tolerance,
      });
    }

    // Nothing on this axis fit at all -- a later phase cannot rescue that.
    if (outcomes.length > 0 && !outcomes.some((o) => o.ok)) return null;
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

// --- The basic (slider-stop) grid, kept only for the safe-value snap and the
// client's own context-target slider -- the search path above no longer
// steps through a grid at all. -------------------------------------------

// Roughly 100/75/50/25/12.5/7.5/5/2.5/1% of the model's trained context, each
// rounded to the nearest power of two and forced strictly below the stop
// before it so two fractions can never collide on the same tick once rounded.
//
// Lives here (not the client) so the slider and the probe's stored ceiling
// cannot disagree about what a legal context is.
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

function dedupeAscending(values: number[]): number[] {
  return [...new Set(values.filter((v) => Number.isFinite(v) && v > 0))].sort((a, b) => a - b);
}

/**
 * The SAFE context to store as a verified ceiling.
 *
 * A `fine` search can converge on a context that sits between two slider
 * stops. That number is real and stays visible in probe_attempts, but it is
 * not something the rest of the app can express or a user can pick, so what
 * gets written to model_machine_limits -- the value later benchmark runs
 * actually consume -- is snapped DOWN to the nearest slider stop. Down, never
 * up: rounding a verified ceiling upward would claim a context that was
 * never loaded.
 */
export function snapToSafeCtx(verifiedCtx: number, maxCtx: number): number {
  const ceiling = Math.max(PROBE_LADDER_MIN_CTX, Math.floor(maxCtx));
  const stops = dedupeAscending([
    PROBE_LADDER_MIN_CTX,
    ...computeCtxStops(ceiling).filter((v) => v >= PROBE_LADDER_MIN_CTX),
    ceiling,
  ]);
  const below = stops.filter((s) => s <= verifiedCtx);
  return below.length > 0 ? Math.max(...below) : Math.min(verifiedCtx, PROBE_LADDER_MIN_CTX);
}

// --- helpers ------------------------------------------------------------

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
