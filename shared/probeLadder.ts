// The context test's search, as pure functions of the loads already made, so
// the whole ladder is testable without a GPU and a stored probe can be replayed
// from its probe_attempts rows. The process/HTTP plumbing that consumes this
// lives in worker/src/index.ts.
//
// Two answers come out of the same loads (docs/CONTEXT_TEST_REDESIGN.md §10):
//
//   target 2, FITS FREE VRAM -- the most layers whose llama.cpp GPU claim stays
//     below what `llama-server --list-devices` reported free before the first
//     load. A claim is buffer sizes, which only grow with layers and with
//     context, so this boundary bisects -- and after two loads at one context
//     the per-layer claim predicts it outright (predictFitNgl).
//
//   target 1, NO SPILL -- the most layers whose GPU memory all stayed in VRAM.
//     Spill does NOT grow steadily with layers: on the RX 6600 XT at 1,024
//     tokens 5 layers spilled 159MiB while 6 spilled nothing, the same at four
//     contexts. So the search finds where spill starts, then loads the two
//     layer counts above the first spilling one, and repeats from any that
//     comes back clean (cleanStep).
//
// Every scenario runs target 2 before target 1, because target 2 caps it: a
// claim larger than free VRAM cannot all sit in VRAM. Nor does spill grow
// steadily with context -- 8 layers spilled at 1k, 2k and 8k and not at 16k --
// so no verdict is carried between contexts. Instead a larger context may never
// use more layers than the smaller one's no-spill answer (the Wizard's ceiling).
//
// Each no-spill answer is loaded a second time (the control), because the first
// load of a session has measured differently from later ones. In the Wizard the
// control runs after the next context's target 2, so other loads sit between
// the two and the next context's ceiling is never taken from an unconfirmed
// answer.

// --- Public vocabulary ------------------------------------------------------

/**
 * frontier -- the Wizard: both targets at every context stop, smallest first.
 * keep_context -- Targets with context pinned: both targets on the layer axis.
 * fixed_offload -- Targets with layers pinned: both targets on the context axis.
 * custom -- Targets with both pinned: one load.
 */
export type ProbeMode = "frontier" | "keep_context" | "fixed_offload" | "custom";

/** Still accepted on the wire so older clients and stored configs validate;
 * the search no longer has a finer setting and ignores it. */
export type ProbeGranularity = "basic" | "fine";

export const PROBE_MODES: readonly ProbeMode[] = ["frontier", "keep_context", "fixed_offload", "custom"];
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
 * itself refuses to spend a real model load proving a context nobody would run.
 */
export const PROBE_LADDER_MIN_CTX = 1024;

/**
 * Default ceiling on loads per probe. Admin-configurable at runtime
 * (AppSettings.probeMaxLoads, propagated over the worker heartbeat); the server
 * enforces the live value independently. Raised from 24 when controls and the
 * per-context target 1 search arrived. A full Wizard curve took 46 loads on the
 * RX 6600 XT with Qwen3.6-35B-A3B (docs/CONTEXT_TEST_REDESIGN.md §10).
 */
export const PROBE_MAX_LOADS = 40;

/**
 * The anchor's layer count: every spill verdict is growth over a load of the
 * model at this many layers and the smallest context (shared/gpuSpill.ts
 * measureGrowthSpill). One layer, never zero: the process's shared memory jumps
 * by about a gigabyte with the first layer on the GPU, overhead no later load
 * should be charged with.
 */
export const PROBE_ANCHOR_NGL = 1;

/**
 * What one full load actually exercises, regardless of the context it
 * allocates: every load runs this prompt and generates this many tokens, so
 * the speeds a rung reports describe roughly PROBE_EXERCISED_TOKENS of context
 * and NOT the context in its row.
 */
export const PROBE_PROMPT_TOKENS = 256;
export const PROBE_GEN_TOKENS = 256;
export const PROBE_EXERCISED_TOKENS = PROBE_PROMPT_TOKENS + PROBE_GEN_TOKENS;

export interface LadderRung {
  ctx: number;
  ngl: number;
}

export interface LadderAttempt extends LadderRung {
  /** Target 1's verdict for this load: it loaded, generated, and its measured
   * spill stayed within the counter's own movement. */
  ok: boolean;
  /**
   * Whether this load's GPU memory placement was measured (llama.cpp's buffer
   * report against a per-process VRAM reading). A clean verdict without it
   * means "it loaded" and nothing about where the memory went.
   */
  placementJudged?: boolean;
  /**
   * Target 2's verdict: llama.cpp's GPU claim stayed below the free VRAM
   * --list-devices reported (claimFitsFree). Null or absent where either figure
   * is missing.
   */
  fitsFree?: boolean | null;
  /** Everything llama.cpp reported placing on GPU devices, MiB -- what
   * predictFitNgl extrapolates from. */
  claimedMib?: number | null;
  /**
   * The load failed for a reason that says nothing about memory: the server
   * never became ready in time, or died without an out-of-memory signature.
   * Target 2 ignores it (no fit or miss is carried from it); target 1 still
   * treats it as not clean, since nothing clean was observed there.
   */
  inconclusive?: boolean;
}

/**
 * Target 2's verdict for one load. `loaded` is whether the load ran far enough
 * to report its buffers at all: one that died first claimed more than it could
 * get, so it does not fit; one that ran without printing buffers (an older
 * build) is unknown rather than a failure.
 */
export function claimFitsFree(input: {
  claimedMib: number | null | undefined;
  freeMib: number | null | undefined;
  loaded: boolean;
}): boolean | null {
  if (input.freeMib == null) return null;
  if (input.claimedMib == null) return input.loaded ? null : false;
  return input.claimedMib < input.freeMib;
}

/**
 * Target 2's verdict device by device: every device llama.cpp put buffers on
 * must have claimed less than its own free memory. Summing instead lets one
 * device's spare room hide another's overflow -- most often an integrated GPU,
 * whose "free" is mostly shared system RAM, masking a discrete card that is
 * already full. Null when either side is missing, or a used device was not
 * listed by --list-devices.
 */
export function claimFitsFreeByDevice(
  claimByDevice: Record<string, number> | null | undefined,
  freeByDevice: Record<string, number> | null | undefined
): boolean | null {
  if (!claimByDevice || !freeByDevice) return null;
  const used = Object.entries(claimByDevice).filter(([, mib]) => mib > 0);
  if (used.length === 0) return null;
  for (const [device, mib] of used) {
    const free = freeByDevice[device];
    if (free == null) return null;
    if (mib >= free) return false;
  }
  return true;
}

/** How one boundary at one context was settled. */
export interface BoundaryState {
  /** The answer: a layer count (or a context, on fixed_offload's axis). Null
   * when nothing qualifies -- or, while unresolved, nothing known to yet. */
  value: number | null;
  resolved: boolean;
  /** "measured" -- loads at this context decided it. "implied" -- settled
   * without one (a ceiling of 0 from the context below, or nothing fitting
   * there). "unmeasured" -- still open when the probe stopped. */
  source: "measured" | "implied" | "unmeasured";
}

/** Target 1 plus the state of its control load. */
export interface CleanState extends BoundaryState {
  /** "confirmed" -- a second load of the answer was clean too. "pending" --
   * only loaded once so far. "none" -- no answer to control. */
  control: "confirmed" | "pending" | "none";
  /** The answer's placement was measured on every load of it (see
   * LadderAttempt.placementJudged). False means "it loaded" only. */
  judged: boolean;
}

export interface LadderInput {
  mode: ProbeMode;
  /**
   * Spill is judged as growth over an anchor: before any search, every mode
   * loads the anchor point (PROBE_ANCHOR_NGL layers, the smallest context)
   * twice. A probe recorded before anchors existed leaves this unset, so its
   * rows still resolve the way they were searched.
   */
  anchored?: boolean;
  candidateCtx: number;
  candidateNgl: number;
  /** llama.cpp's -ngl ceiling for this model (n_layer + the output layer). */
  nglMax: number;
  /** The model's trained context -- the ladder never probes above it. */
  maxCtx: number;
  maxLoads?: number;
  /** Every load already made, in order. */
  history: readonly LadderAttempt[];
  /**
   * What `llama-server --list-devices` reported free on the devices this probe
   * uses, read once before the first load. Null or absent: target 2 cannot be
   * judged, so target 1 is searched alone with every layer as its cap.
   */
  freeVramMib?: number | null;
  /** The estimator's safe layer count at a context -- only ever the first load's
   * layer count, never a verdict. Absent where only display is wanted. */
  calculateNgl?: (ctx: number) => number;
}

/** One context stop of the Wizard's result. */
export interface CurveStop {
  ctx: number;
  /** Null when the probe had no --list-devices reading. */
  fit: BoundaryState | null;
  clean: CleanState;
}

export interface ProbeOutcome {
  /** The next load to make, or null when the search is finished. Ignores the
   * budget -- nextLadderRung applies that. */
  next: LadderRung | null;
  /** Target 1's answer as a placement; null when none was found. */
  clean: (LadderRung & { control: CleanState["control"]; judged: boolean }) | null;
  /** Target 2's answer as a placement; null when nothing fits or no reading. */
  fit: LadderRung | null;
  /** The Wizard's per-context result; absent for the other modes. */
  curve?: CurveStop[];
  /** The Wizard stopped because nothing fits at this context (or anywhere above). */
  nothingFitsFrom?: number | null;
  /** `next` is an anchor load: the first or second load of the anchor point. */
  nextIsAnchor?: boolean;
  /**
   * `next` can only ever answer target 2: it is above every layer count target 1
   * may still accept at its context -- the stop below's no-spill answer, which a
   * control can only lower, or no answer at all. The load stops once its claim
   * is known to fit; generating would judge a spill nothing reads.
   */
  nextClaimOnly?: boolean;
}

/** The next load, or null when the ladder is finished or out of budget. */
export function nextLadderRung(input: LadderInput): LadderRung | null {
  const maxLoads = input.maxLoads ?? PROBE_MAX_LOADS;
  if (input.history.length >= maxLoads) return null;
  return probeOutcome(input).next;
}

/**
 * Everything a probe has established so far and what to load next. The search
 * (nextLadderRung) and every display of a probe's result call this same
 * function, so they cannot disagree about what was found.
 */
export function probeOutcome(input: LadderInput): ProbeOutcome {
  const nglMax = Math.max(0, Math.floor(input.nglMax));
  const maxCtx = Math.max(PROBE_LADDER_MIN_CTX, Math.floor(input.maxCtx));
  const history = input.history.filter((h) => h.ngl >= 0 && h.ngl <= nglMax);
  const free = input.freeVramMib ?? null;

  if (input.anchored) {
    const gate = anchorGate(history, input.mode, nglMax, maxCtx, free);
    if (gate) return gate;
  }

  switch (input.mode) {
    case "custom": {
      const ctx = clamp(Math.floor(input.candidateCtx), PROBE_LADDER_MIN_CTX, maxCtx);
      const ngl = clamp(Math.floor(input.candidateNgl), 0, nglMax);
      const loads = loadsAt(history, ctx, ngl);
      if (loads.length === 0) return { next: { ctx, ngl }, clean: null, fit: null };
      const clean = loads.every((h) => h.ok);
      return {
        next: null,
        clean: clean ? { ctx, ngl, control: "none", judged: loads.every((h) => h.placementJudged === true) } : null,
        fit: free != null && loads.some((h) => fitVerdict(h)) ? { ctx, ngl } : null,
      };
    }
    case "keep_context":
      return keepContextOutcome(history, input, nglMax, maxCtx, free);
    case "fixed_offload":
      return fixedOffloadOutcome(history, input, nglMax, maxCtx, free);
    case "frontier":
      return frontierOutcome(history, input, nglMax, maxCtx, free);
  }
}

/**
 * The anchor before anything else. Both loads are needed: the first is what
 * every later load is measured against, the second how far two identical loads
 * differ. A failed anchor ends the probe -- nothing else can be judged -- and
 * one whose claim did not fit free VRAM means nothing fits at any placement.
 * Null once both loads are in and clean: the mode's own search takes over, and
 * counts them as the anchor point's load and control.
 */
function anchorGate(
  history: readonly LadderAttempt[],
  mode: ProbeMode,
  nglMax: number,
  maxCtx: number,
  free: number | null
): ProbeOutcome | null {
  const anchor = { ctx: PROBE_LADDER_MIN_CTX, ngl: Math.min(PROBE_ANCHOR_NGL, nglMax) };
  const loads = loadsAt(history, anchor.ctx, anchor.ngl);
  const failed = loads.find((h) => !h.ok);
  if (failed) {
    const nothingFits = free != null && !failed.inconclusive && fitVerdict(failed) === false;
    if (!nothingFits || mode !== "frontier") return { next: null, clean: null, fit: null };
    const curve: CurveStop[] = ctxLadderStops(maxCtx).map((ctx) => ({
      ctx,
      fit: { value: null, resolved: true, source: ctx === anchor.ctx ? "measured" : "implied" },
      clean: { value: null, resolved: true, source: "implied", control: "none", judged: false },
    }));
    return summarise(curve, null, anchor.ctx);
  }
  if (loads.length < 2) return { next: anchor, clean: null, fit: null, nextIsAnchor: true };
  return null;
}

// --- Verdicts ---------------------------------------------------------------

function loadsAt(history: readonly LadderAttempt[], ctx: number, ngl: number): LadderAttempt[] {
  return history.filter((h) => h.ctx === ctx && h.ngl === ngl);
}

/** Target 2's verdict. A load with no claim reading (a build that printed no
 * buffer sizes) falls back to its spill verdict rather than stalling. */
function fitVerdict(h: LadderAttempt): boolean {
  return h.fitsFree ?? h.ok;
}

/** Target 1's verdict at one point: clean only when EVERY load there was clean,
 * so a control that spills overturns the first reading. Null when never loaded. */
function cleanAt(history: readonly LadderAttempt[], ctx: number, ngl: number): boolean | null {
  const loads = loadsAt(history, ctx, ngl);
  return loads.length === 0 ? null : loads.every((h) => h.ok);
}

// --- Target 2 on the layer axis ---------------------------------------------

/**
 * Target 2 at one context. A load that fits carries DOWN the context axis and
 * one that does not carries UP it, because a claim only grows with context --
 * and `ceiling` (the answer at the context below) caps it from above.
 *
 * Next load, in order of preference: the claim prediction inside what is still
 * open; the caller's opener when nothing has been loaded here yet; bisection.
 */
function fitStep(input: {
  history: readonly LadderAttempt[];
  ctx: number;
  nglMax: number;
  free: number;
  ceiling: number | null;
  opener: number | null;
}): { state: BoundaryState; next: number | null } {
  const { ctx, nglMax, free } = input;
  // An inconclusive load is still "tried" (it is never offered again), but it
  // decides nothing about the claim.
  const history = input.history.filter((h) => !h.inconclusive);
  const fits = history.filter((h) => h.ctx >= ctx && fitVerdict(h)).map((h) => h.ngl);
  const misses = history.filter((h) => h.ctx <= ctx && !fitVerdict(h)).map((h) => h.ngl);
  if (input.ceiling != null) misses.push(input.ceiling + 1);
  const knownFit = fits.length > 0 ? Math.max(...fits) : null;
  const missesAbove = misses.filter((m) => knownFit == null || m > knownFit);
  const firstMiss = missesAbove.length > 0 ? Math.min(...missesAbove) : null;

  const hereDecides = (n: number | null) => history.some((h) => h.ctx === ctx && (n == null || h.ngl === n));
  if (firstMiss === 0) {
    return { state: { value: null, resolved: true, source: hereDecides(0) ? "measured" : "implied" }, next: null };
  }
  if (knownFit != null && (knownFit >= nglMax || firstMiss === knownFit + 1)) {
    const measured = history.some((h) => h.ctx === ctx && (h.ngl === knownFit || h.ngl === firstMiss));
    return { state: { value: knownFit, resolved: true, source: measured ? "measured" : "implied" }, next: null };
  }

  const tried = new Set(input.history.filter((h) => h.ctx === ctx).map((h) => h.ngl));
  const lo = knownFit ?? -1;
  const hi = Math.min(firstMiss ?? nglMax + 1, nglMax + 1);
  const open = (n: number) => n > lo && n < hi && !tried.has(n);
  const state: BoundaryState = { value: knownFit, resolved: false, source: "unmeasured" };

  const guided = claimGuidedNgl(predictFitNgl(history, ctx, free), knownFit, firstMiss, nglMax, tried);
  if (guided != null) return { state, next: guided };
  if (tried.size === 0 && input.opener != null) {
    const opener = clamp(Math.round(input.opener), lo + 1, hi - 1);
    if (open(opener)) return { state, next: opener };
  }
  const mid = clamp(Math.floor((lo + hi) / 2), lo + 1, hi - 1);
  if (open(mid)) return { state, next: mid };
  for (let n = hi - 1; n > lo; n--) if (open(n)) return { state, next: n };
  return { state, next: null };
}

/**
 * Target 2's boundary at one context, predicted from claims already measured:
 * the most layers whose claim would stay below the free VRAM, or -1 when not
 * even 0 would. Null until two loads at one context give a per-layer cost.
 *
 * A claim is buffer sizes, so it is close to linear in layers -- ~405MiB a layer
 * on the RX 6600 XT sweep, a little more for the output layer. The prediction
 * extrapolates from the measured rung nearest the free line, so every load
 * sharpens it where it matters. Between two measured contexts the fixed part of
 * the claim (KV cache, compute buffer) is interpolated linearly; beyond them it
 * is extrapolated, and with only one context measured it is assumed not to grow,
 * which errs toward too many layers -- one failing load, never a missed one.
 *
 * Only ever a seed (claimGuidedNgl): every boundary it names is still loaded.
 */
export function predictFitNgl(history: readonly LadderAttempt[], ctx: number, freeVramMib: number): number | null {
  const byCtx = new Map<number, { ngl: number; claim: number }[]>();
  for (const h of history) {
    if (h.claimedMib == null || !(h.claimedMib > 0)) continue;
    byCtx.set(h.ctx, [...(byCtx.get(h.ctx) ?? []), { ngl: h.ngl, claim: h.claimedMib }]);
  }
  let perLayer: number | null = null;
  let span = 0;
  for (const rungs of byCtx.values()) {
    const low = rungs.reduce((a, b) => (b.ngl < a.ngl ? b : a));
    const high = rungs.reduce((a, b) => (b.ngl > a.ngl ? b : a));
    if (high.ngl - low.ngl > span) {
      span = high.ngl - low.ngl;
      perLayer = (high.claim - low.claim) / span;
    }
  }
  if (perLayer == null) {
    // One load only: its whole claim over its layer count overstates the cost
    // of a layer (the claim also holds the KV cache and compute buffer), so this
    // answer errs LOW -- the next load then fits, and two loads give the slope.
    const one = history.find((h) => h.ctx === ctx && h.ngl > 0 && h.claimedMib != null && h.claimedMib > 0);
    return one ? Math.ceil((freeVramMib * one.ngl) / one.claimedMib!) - 1 : null;
  }
  if (!(perLayer > 0)) return null;
  const slope = perLayer;

  const fixedPartAt = (c: number): number => {
    const rungs = byCtx.get(c)!;
    const nearest = rungs.reduce((a, b) => (Math.abs(freeVramMib - b.claim) < Math.abs(freeVramMib - a.claim) ? b : a));
    return nearest.claim - slope * nearest.ngl;
  };
  const measured = [...byCtx.keys()].sort((a, b) => a - b);
  let fixed: number;
  if (byCtx.has(ctx)) {
    fixed = fixedPartAt(ctx);
  } else {
    const below = measured.filter((c) => c < ctx);
    const above = measured.filter((c) => c > ctx);
    const pair =
      below.length > 0 && above.length > 0
        ? [below[below.length - 1], above[0]]
        : below.length >= 2
          ? below.slice(-2)
          : above.length >= 2
            ? above.slice(0, 2)
            : null;
    if (pair) {
      const [c1, c2] = pair;
      const f1 = fixedPartAt(c1);
      fixed = f1 + ((fixedPartAt(c2) - f1) * (ctx - c1)) / (c2 - c1);
    } else {
      fixed = fixedPartAt(measured[0]);
    }
  }
  return Math.ceil((freeVramMib - fixed) / slope) - 1;
}

/**
 * The layer count a prediction says to load next, inside what is still open:
 * above the most layers known to fit, below the fewest known not to. Once a
 * predicted boundary has been loaded and fit, the open interval starts one above
 * it, so the next move tests that neighbour and closes the bracket in two loads.
 * A prediction more than a layer outside the open interval has been contradicted
 * by real loads, and the caller bisects instead.
 */
function claimGuidedNgl(
  predicted: number | null,
  knownFit: number | null,
  knownMiss: number | null,
  nglMax: number,
  tried: Set<number>
): number | null {
  if (predicted == null) return null;
  const lo = (knownFit ?? -1) + 1;
  const hi = Math.min(nglMax, (knownMiss ?? nglMax + 1) - 1);
  if (hi < lo || predicted < lo - 1 || predicted > hi + 1) return null;
  const target = clamp(predicted, lo, hi);
  return tried.has(target) ? null : target;
}

// --- Target 1 ----------------------------------------------------------------

/**
 * Target 1 on one axis, over the positions 0..cap (layer counts, or indices
 * into the context stops). `cleanOf(p)` is the verdict at a position: true
 * clean, false spilled, null never loaded.
 *
 * With A the highest clean position loaded and S the lowest spilling one above
 * it: bisect between them until S is A+1 -- where spill starts -- then load
 * A+2 and A+3. If one comes back clean it becomes A and the same rule runs
 * again from there; once A+1..A+3 (those not above the cap) all spilled, A is
 * the answer. A clean island more than two positions above spill is missed on
 * purpose: finding every one means loading everything below the cap.
 *
 * Nothing loaded yet: the opener if given, else the cap itself -- which in the
 * Wizard is the context below's answer, usually clean again in one load.
 */
function cleanStep(input: {
  cap: number | null;
  cleanOf: (p: number) => boolean | null;
  opener?: number | null;
}): { value: number | null; resolved: boolean; next: number | null } {
  const { cap, cleanOf } = input;
  if (cap == null || cap < 0) return { value: null, resolved: true, next: null };

  let highestClean = -1;
  let anyLoaded = false;
  for (let p = 0; p <= cap; p++) {
    const v = cleanOf(p);
    if (v != null) anyLoaded = true;
    if (v === true) highestClean = p;
  }
  if (!anyLoaded) {
    const opener = input.opener != null ? clamp(Math.round(input.opener), 0, cap) : cap;
    return { value: null, resolved: false, next: opener };
  }
  const value = highestClean >= 0 ? highestClean : null;
  if (highestClean === cap) return { value, resolved: true, next: null };

  let lowestSpillAbove: number | null = null;
  for (let p = highestClean + 1; p <= cap; p++) {
    if (cleanOf(p) === false) {
      lowestSpillAbove = p;
      break;
    }
  }
  if (lowestSpillAbove == null) {
    // Nothing loaded above the highest clean position: grow toward the cap.
    return { value, resolved: false, next: Math.ceil((highestClean + cap + 1) / 2) };
  }
  if (lowestSpillAbove > highestClean + 1) {
    return { value, resolved: false, next: Math.floor((highestClean + lowestSpillAbove) / 2) };
  }
  for (const p of [highestClean + 2, highestClean + 3]) {
    if (p <= cap && cleanOf(p) == null) return { value, resolved: false, next: p };
  }
  return { value, resolved: true, next: null };
}

function controlOf(history: readonly LadderAttempt[], rung: LadderRung | null): CleanState["control"] {
  if (rung == null) return "none";
  return loadsAt(history, rung.ctx, rung.ngl).length >= 2 ? "confirmed" : "pending";
}

function judgedAt(history: readonly LadderAttempt[], rung: LadderRung | null): boolean {
  if (rung == null) return false;
  const loads = loadsAt(history, rung.ctx, rung.ngl);
  return loads.length > 0 && loads.every((h) => h.placementJudged === true);
}

// --- keep_context: context pinned --------------------------------------------

function keepContextOutcome(
  history: readonly LadderAttempt[],
  input: LadderInput,
  nglMax: number,
  maxCtx: number,
  free: number | null
): ProbeOutcome {
  const ctx = clamp(Math.floor(input.candidateCtx), PROBE_LADDER_MIN_CTX, maxCtx);
  const seed = clamp(Math.floor(input.candidateNgl), 0, nglMax);
  const here = history.filter((h) => h.ctx === ctx);

  let cap = nglMax;
  let fit: LadderRung | null = null;
  if (free != null) {
    const step = fitStep({ history: here, ctx, nglMax, free, ceiling: null, opener: seed });
    if (!step.state.resolved) return { next: step.next == null ? null : { ctx, ngl: step.next }, clean: null, fit: null };
    if (step.state.value == null) return { next: null, clean: null, fit: null };
    cap = step.state.value;
    fit = { ctx, ngl: cap };
  }

  const clean = cleanStep({
    cap,
    cleanOf: (n) => cleanAt(here, ctx, n),
    opener: free == null ? seed : null,
  });
  if (!clean.resolved) return { next: clean.next == null ? null : { ctx, ngl: clean.next }, clean: null, fit };
  const answer = clean.value == null ? null : { ctx, ngl: clean.value };
  const control = controlOf(here, answer);
  return {
    next: control === "pending" ? answer : null,
    clean: answer ? { ...answer, control, judged: judgedAt(here, answer) } : null,
    fit,
  };
}

// --- fixed_offload: layers pinned --------------------------------------------

function fixedOffloadOutcome(
  history: readonly LadderAttempt[],
  input: LadderInput,
  nglMax: number,
  maxCtx: number,
  free: number | null
): ProbeOutcome {
  const ngl = clamp(Math.floor(input.candidateNgl), 0, nglMax);
  const stops = ctxLadderStops(maxCtx);
  const here = history.filter((h) => h.ngl === ngl);
  const seedIdx = nearestStopIndex(stops, clamp(Math.floor(input.candidateCtx), PROBE_LADDER_MIN_CTX, maxCtx));
  const loadsAtIdx = (i: number) => here.filter((h) => h.ctx === stops[i]);

  let capIdx = stops.length - 1;
  let fit: LadderRung | null = null;
  if (free != null) {
    // A claim only grows with context: a fit carries down the stops, a miss up.
    let knownFit = -1;
    const misses: number[] = [];
    stops.forEach((_, i) => {
      const loads = loadsAtIdx(i).filter((h) => !h.inconclusive);
      if (loads.length === 0) return;
      if (loads.some(fitVerdict)) knownFit = Math.max(knownFit, i);
      else misses.push(i);
    });
    // A miss at or below a known fit contradicts it (a flaky load): trust the
    // fit and ignore that miss, but keep every miss above it.
    const firstMiss = Math.min(stops.length, ...misses.filter((i) => i > knownFit));
    const resolved = firstMiss === 0 || knownFit === stops.length - 1 || firstMiss === knownFit + 1;
    if (!resolved) {
      const tried = new Set(here.map((h) => h.ctx));
      let nextIdx: number;
      if (knownFit < 0 && firstMiss === stops.length) nextIdx = seedIdx;
      else if (firstMiss === stops.length) nextIdx = Math.ceil((knownFit + stops.length) / 2);
      else nextIdx = Math.floor((knownFit + firstMiss) / 2);
      nextIdx = clamp(nextIdx, knownFit + 1, firstMiss - 1);
      const next = tried.has(stops[nextIdx]) ? null : { ctx: stops[nextIdx], ngl };
      return { next, clean: null, fit: null };
    }
    if (firstMiss === 0) return { next: null, clean: null, fit: null };
    capIdx = knownFit;
    fit = { ctx: stops[capIdx], ngl };
  }

  const cleanOfIdx = (i: number) => {
    const loads = loadsAtIdx(i);
    return loads.length === 0 ? null : loads.every((h) => h.ok);
  };
  const clean = cleanStep({ cap: capIdx, cleanOf: cleanOfIdx, opener: free == null ? seedIdx : null });
  if (!clean.resolved) return { next: clean.next == null ? null : { ctx: stops[clean.next], ngl }, clean: null, fit };
  const answer = clean.value == null ? null : { ctx: stops[clean.value], ngl };
  const control = controlOf(here, answer);
  return {
    next: control === "pending" ? answer : null,
    clean: answer ? { ...answer, control, judged: judgedAt(here, answer) } : null,
    fit,
  };
}

// --- frontier: the Wizard ------------------------------------------------------

/**
 * Both targets at every context stop, smallest first:
 *
 *   1. target 2 here, capped by target 2 at the stop below. Nothing fits here
 *      means nothing fits above either -- after the pending control below, the
 *      probe ends.
 *   2. the control of the stop below's no-spill answer.
 *   3. target 1 here, capped by target 2 here and by the stop below's no-spill
 *      answer -- a larger context never tries more layers than a smaller one
 *      held clean.
 *
 * The first load at the smallest stop is the estimate's layer count, never
 * every layer: after two loads the claim prediction takes over, and a
 * full-offload load of a model larger than VRAM is the slowest load there is.
 */
function frontierOutcome(
  history: readonly LadderAttempt[],
  input: LadderInput,
  nglMax: number,
  maxCtx: number,
  free: number | null
): ProbeOutcome {
  const stops = ctxLadderStops(maxCtx);
  const curve: CurveStop[] = [];
  const unmeasured = (ctx: number): CurveStop => ({
    ctx,
    fit: free == null ? null : { value: null, resolved: false, source: "unmeasured" },
    clean: { value: null, resolved: false, source: "unmeasured", control: "none", judged: false },
  });
  const finish = (next: LadderRung | null, from: number, nextClaimOnly = false): ProbeOutcome => {
    for (let j = from; j < stops.length; j++) curve.push(unmeasured(stops[j]));
    return { ...summarise(curve, next, null), nextClaimOnly: next != null && nextClaimOnly };
  };
  const estimate = (ctx: number): number | null => {
    if (!input.calculateNgl) return null;
    const safe = clamp(Math.round(input.calculateNgl(ctx)), 0, nglMax);
    return safe;
  };

  for (let i = 0; i < stops.length; i++) {
    const ctx = stops[i];
    const below = i > 0 ? curve[i - 1] : null;
    let fitState: BoundaryState | null = null;

    if (free != null) {
      const ceiling = below?.fit?.value ?? null;
      const step = fitStep({
        history,
        ctx,
        nglMax,
        free,
        ceiling,
        opener: i === 0 ? estimate(ctx) : ceiling,
      });
      if (!step.state.resolved) {
        curve.push({ ctx, fit: step.state, clean: unmeasured(ctx).clean });
        // Every stop below is resolved by now, so its no-spill answer already
        // bounds target 1 here (step 3's cap).
        const claimOnly =
          below != null && step.next != null && (below.clean.value == null || step.next > below.clean.value);
        return finish(step.next == null ? null : { ctx, ngl: step.next }, i + 1, claimOnly);
      }
      fitState = step.state;
    }

    if (below && below.clean.control === "pending" && below.clean.value != null) {
      curve.push({ ctx, fit: fitState, clean: unmeasured(ctx).clean });
      return finish({ ctx: below.ctx, ngl: below.clean.value }, i + 1);
    }

    if (fitState && fitState.value == null) {
      curve.push({
        ctx,
        fit: fitState,
        clean: { value: null, resolved: true, source: "implied", control: "none", judged: false },
      });
      for (let j = i + 1; j < stops.length; j++) {
        curve.push({
          ctx: stops[j],
          fit: { value: null, resolved: true, source: "implied" },
          clean: { value: null, resolved: true, source: "implied", control: "none", judged: false },
        });
      }
      return summarise(curve, null, ctx);
    }

    const cap = below ? (below.clean.value == null ? null : Math.min(below.clean.value, fitState?.value ?? nglMax)) : (fitState?.value ?? nglMax);
    const clean = cleanStep({
      cap,
      cleanOf: (n) => cleanAt(history, ctx, n),
      opener: i === 0 && free == null ? estimate(ctx) : null,
    });
    if (!clean.resolved) {
      curve.push({
        ctx,
        fit: fitState,
        clean: { value: clean.value, resolved: false, source: "unmeasured", control: "none", judged: false },
      });
      return finish(clean.next == null ? null : { ctx, ngl: clean.next }, i + 1);
    }
    const answer = clean.value == null ? null : { ctx, ngl: clean.value };
    curve.push({
      ctx,
      fit: fitState,
      clean: {
        value: clean.value,
        resolved: true,
        source: history.some((h) => h.ctx === ctx) ? "measured" : "implied",
        control: controlOf(history, answer),
        judged: judgedAt(history, answer),
      },
    });
  }

  const last = curve[curve.length - 1];
  const next = last && last.clean.control === "pending" && last.clean.value != null ? { ctx: last.ctx, ngl: last.clean.value } : null;
  return summarise(curve, next, null);
}

/** The Wizard's headline answers, plus the whole curve. Target 1's headline --
 * the stored ceiling -- is the largest context whose answer a control confirmed;
 * only when none was confirmed (the budget ran out first) does an unconfirmed
 * one stand in, labelled by its `control`. Target 2 needs no control. */
function summarise(curve: CurveStop[], next: LadderRung | null, nothingFitsFrom: number | null): ProbeOutcome {
  let confirmed: ProbeOutcome["clean"] = null;
  let unconfirmed: ProbeOutcome["clean"] = null;
  let fit: LadderRung | null = null;
  for (const stop of curve) {
    if (stop.clean.resolved && stop.clean.value != null) {
      const answer = { ctx: stop.ctx, ngl: stop.clean.value, control: stop.clean.control, judged: stop.clean.judged };
      if (stop.clean.control === "confirmed") confirmed = answer;
      else unconfirmed = answer;
    }
    if (stop.fit?.resolved && stop.fit.value != null) fit = { ctx: stop.ctx, ngl: stop.fit.value };
  }
  return { next, clean: confirmed ?? unconfirmed, fit, curve, nothingFitsFrom };
}

// --- The context grids -------------------------------------------------------
//
// Two different discrete grids on purpose:
//  - ctxLadderStops: power-of-two doublings from the hard floor up to the
//    model's own ceiling. Every context the probe loads is one of these.
//  - computeCtxStops: a fraction-of-trained-ctx ladder for the CLIENT's own
//    context-target slider display. Unrelated to what the probe tests; kept here
//    only so nothing else has to duplicate it.

/**
 * The discrete context ladder the probe loads: doublings from the hard floor up
 * to the model's own ceiling, which is always included exactly even when it is
 * not itself a power of two -- otherwise the true ceiling could never be reached.
 */
export function ctxLadderStops(maxCtx: number): number[] {
  const ceiling = Math.max(PROBE_LADDER_MIN_CTX, Math.floor(maxCtx));
  const stops: number[] = [];
  for (let v = PROBE_LADDER_MIN_CTX; v < ceiling; v *= 2) stops.push(v);
  stops.push(ceiling);
  return stops;
}

// Roughly 100/75/50/25/12.5/7.5/5/2.5/1% of the model's trained context, each
// rounded to the nearest power of two and forced strictly below the stop before
// it so two fractions can never collide on the same tick once rounded. The
// client's own context-target slider grid ONLY.
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

/**
 * A verified context rounded down to something a slider can express. Every
 * context the ladder loads is already a stop; this exists for results stored
 * before `fine` was removed, which could land between two. Down, never up:
 * rounding a verified ceiling upward would claim a context that was never loaded.
 */
export function snapToSafeCtx(verifiedCtx: number, maxCtx: number): number {
  const stops = ctxLadderStops(maxCtx);
  const below = stops.filter((s) => s <= verifiedCtx);
  return below.length > 0 ? Math.max(...below) : Math.min(verifiedCtx, PROBE_LADDER_MIN_CTX);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
