// BENCHMARKING_PLAN_V8.md M2 -- "What are you optimizing for?", captured
// BEFORE the grid is built. Three visible questions plus one collapsed
// optional, and SKIPPABLE BY CONSTRUCTION: accepting every default produces a
// trigger payload byte-identical to the previous behavior (the caller passes
// `goals` only when the user actually touched something).
//
// Accessibility (the plan's own conformance block): the whole thing is a
// labelled fieldset; goal/floor/workload/KV-preset chips carry role="radio" +
// aria-checked; the floor reveal and the KV-preset section are disclosures
// with aria-expanded; the target clamp announces via aria-live="polite".

import { useEffect, useId, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  KV_PRESET_PAIRS,
  KV_PRESETS,
  SPEED_FLOOR_CHOICES,
  WORKLOAD_WEIGHTS,
  type GoalKind,
  type GoalsConfig,
  type KvPreset,
  type WorkloadShape,
} from "../goals";
import {
  maxAffordableContext,
  computeDualPoolFit,
  suggestPlacementConfigs,
  type MaxCtxEstimate,
  type PoolReading,
  type PoolFit,
  type DualPoolFit,
  type SuggestionResult,
  type TensorLayerBreakdown,
} from "../vramEstimate";
import {
  computeCtxStops,
  PROBE_GRANULARITIES,
  PROBE_LADDER_MIN_CTX,
  PROBE_MAX_LOADS,
  type ProbeGranularity,
  type ProbeMode,
} from "../../../shared/probeLadder";

export interface GoalQuestionnaireProps {
  goals: GoalsConfig;
  onChange: (goals: GoalsConfig) => void;
  /** Clamps the target inline; undefined when the model's header never said. */
  trainedCtx?: number | null;
  /** Inputs for M1's live feasibility readout. Absent => the readout says so. */
  affordability?: {
    totalMib: number | null;
    weightsMib: number | null;
    nLayer?: number;
    nHeadKv?: number;
    headDimK?: number;
    headDimV?: number;
    nEmbd?: number;
    nHead?: number;
    slidingWindow?: number;
  };
  /** True when these values came from a preset that predates M5's goals block. */
  unset?: boolean;
  /**
   * Whether the exact-number entry beside the context slider is shown.
   * The Benchmark console hides it (the slider's own stops are the only
   * contexts its chain is meant to run), while New Run keeps it for hand-built
   * grids. The "don't know" reset lives OUTSIDE this block deliberately --
   * hiding it too would leave the Benchmark page no way back to
   * `target_ctx: null`, which is the target_unverified marker M3 depends on.
   */
  showCtxNumberInput?: boolean;
  /**
   * How finely the context slider is subdivided. "coarse" is the shared stop
   * list every other surface uses; "fine" interpolates between those stops.
   * Only affects what the SLIDER can express -- a probe's own granularity is
   * chosen separately on the Tested-configurations row.
   */
  ctxGranularity?: "coarse" | "fine";
  /**
   * The offload (GPU/CPU layer split) matrix paired with the context slider
   * above -- absent until a real model+worker pairing exists (matching
   * `affordability`'s own null-until-computable posture). Placement is
   * machine-specific, never part of `goals`/presets -- see Benchmark.tsx's
   * own placementStorageKey and M5's principle for why.
   */
  placement?: {
    ngl: number;
    onNglChange: (ngl: number) => void;
    /** Total layers including the output layer -- the slider's own max. */
    nglMax: number;
    /** Raw n_layer (no +1) -- only transformer layers carry a KV cache. */
    kvLayerCount: number;
    modelSizeBytes: number | null;
    /**
     * Real per-tensor weight-byte breakdown (see shared/vramEstimate.ts's
     * TensorLayerBreakdown/placeWeightBytes) -- drives an accurate fit check
     * instead of the flat per-layer average when available. Absent/null for
     * a model registered before this existed.
     */
    tensorBreakdown?: TensorLayerBreakdown | null;
    /** Non-null when the slider is locked and pinned -- see the lock rules. */
    locked: "cpu" | "unified" | null;
    vram: PoolReading;
    ram: PoolReading;
    unifiedPool: boolean;
    noGpu: boolean;
    /** A recent failed/failed_oom verify's safety margin -- see Benchmark.tsx. */
    poolHaircutFrac: number;
    onVerify: (ngl: number, ctx: number, mode: ProbeMode, granularity: ProbeGranularity) => void;
    /** Clears one card's own result back to untested -- purely local state. */
    onReset: (mode: ProbeMode) => void;
    /** Keyed by mode -- every card keeps its own independent result so Test
     * All can run every mode concurrently without one overwriting another. */
    verifyResults: Partial<Record<ProbeMode, PlacementVerifyResult>>;
  };
}

export interface PlacementVerifyResult {
  ngl: number;
  ctx: number;
  runId: string;
  status: "pending" | "verified" | "failed" | "failed_oom" | "error";
  detail?: string;
  verifiedCtxTokens?: number | null;
  mode?: ProbeMode;
  /** The winning rung's own placement and real measured usage -- see
   * Benchmark.tsx's poll effect for where this comes from. */
  measuredNgl?: number | null;
  measuredVramPeakMib?: number | null;
  measuredRamPeakMib?: number | null;
  /** When this card's Test was clicked -- epoch ms, shown compactly next to
   * the run link so a tested card names both what it found and when. */
  testedAt?: number;
}

const GOAL_CHOICES: { value: GoalKind; label: string }[] = [
  { value: "balanced", label: "Balanced" },
  { value: "max_speed", label: "Max tok/s" },
  { value: "max_context", label: "Max context" },
];

const WORKLOAD_CHOICES: { value: WorkloadShape; label: string }[] = [
  { value: "chat", label: "Chat · gen-heavy" },
  { value: "docs", label: "Documents · prompt-heavy" },
  { value: "even", label: "Even" },
];

export const KV_PRESET_LABEL: Record<KvPreset, string> = {
  compact: "Compact",
  basic: "Basic",
  extended: "Extended",
  comprehensive: "Comprehensive",
};

const KV_PRESET_BLURB: Record<KvPreset, string> = {
  compact: "Fast sanity check — baseline, mild, and the popular long-context setup.",
  basic: "Compact + BF16, to check it behaves like F16.",
  extended: "Basic + isolates K vs V impact and an aggressive-quantization data point. Default.",
  comprehensive: "Extended + every remaining corner: asymmetric quantization, a mid bit-width rung, and an alternative 4-bit codec.",
};

function formatTokens(tokens: number): string {
  if (tokens >= 1000) return `${Math.round(tokens / 1000)} k`;
  return String(tokens);
}

function formatMib(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${Math.round(mib)} MiB`;
}

// A recent verify failure haircuts the free-mib figure fed into the fit
// check (not the displayed live reading) -- see suggestPlacementConfigs'
// own doc comment for why: the estimate was proven wrong at that point, so
// the next round shouldn't trust the same number at full confidence.
function haircut(freeMib: number | null, frac: number): number | null {
  return freeMib == null ? null : freeMib * (1 - frac);
}

// N2's six search goals, as the user meets them. Each card is a real probe
// mode (shared/probeLadder.ts) rather than a static suggestion: the estimate
// only decides where each one STARTS, and pressing Test measures the rest.
const MODE_LABEL: Record<ProbeMode, string> = {
  max_gpu: "Max GPU speed",
  max_context: "Max context",
  keep_context: "Fixed context",
  balanced: "Balanced",
  fixed_offload: "Fixed offload",
  custom: "Custom",
};

const MODE_BLURB: Record<ProbeMode, string> = {
  max_gpu: "Every layer on the GPU, then as much context as still fits.",
  max_context: "Your placement, pushed to the largest context this machine can actually hold, then more layers if there's room.",
  keep_context: "Your context is fixed; the layer split moves to make it fit.",
  balanced: "Starts where you are and trades context against layers.",
  fixed_offload: "Your layer split is fixed; context moves to make it fit.",
  custom: "One load at exactly the settings above — no search.",
};

const GRANULARITY_LABEL: Record<ProbeGranularity, string> = {
  basic: "Basic",
  fine: "Fine tune",
};

const GRANULARITY_BLURB: Record<ProbeGranularity, string> = {
  basic: "Converges on a value within about 1/16th of the search range.",
  fine: "Converges four times tighter — more precise, more loads.",
};

// The context-target slider stops now live in shared/probeLadder.ts, so the
// slider and the probe cannot disagree about what a legal context is --
// "Basic" granularity is defined as exactly this list.
const FINE_SLIDER_DIVISIONS = 4;

/**
 * The slider's own stops. "coarse" is the shared list unchanged; "fine" adds
 * evenly spaced values between each pair so the New Run page can express a
 * context the Benchmark page's coarser slider cannot.
 */
function sliderStops(maxCtx: number, granularity: "coarse" | "fine"): number[] {
  const base = computeCtxStops(maxCtx);
  if (granularity === "coarse" || base.length < 2) return base;
  const out: number[] = [];
  for (let i = 0; i < base.length - 1; i++) {
    const from = base[i];
    const step = (base[i + 1] - from) / FINE_SLIDER_DIVISIONS;
    for (let d = 0; d < FINE_SLIDER_DIVISIONS; d++) out.push(Math.round(from + step * d));
  }
  out.push(base[base.length - 1]);
  return [...new Set(out.filter((v) => v > 0))].sort((a, b) => a - b);
}

export function GoalQuestionnaire({
  goals,
  onChange,
  trainedCtx,
  affordability,
  unset,
  placement,
  showCtxNumberInput = true,
  ctxGranularity = "coarse",
}: GoalQuestionnaireProps) {
  const [kvPresetOpen, setKvPresetOpen] = useState(false);
  const [targetText, setTargetText] = useState(goals.target_ctx != null ? String(goals.target_ctx) : "");
  const kvPresetId = useId();
  const clampId = useId();

  const clamped =
    goals.target_ctx != null && trainedCtx != null && trainedCtx > 0 && Number(targetText) > trainedCtx;

  // M2's stated default: min(32k, trained_ctx). Shown as a placeholder, not
  // written into goals.target_ctx, so an untouched field still stores null
  // (target_unverified) -- M3's pinned reduction requires FIT null under
  // every default, and "skippable" means the questionnaire never fabricates
  // an answer the user didn't give.
  const suggestedTarget = Math.min(32_768, trainedCtx != null && trainedCtx > 0 ? trainedCtx : 32_768);

  // The slider only exists once there's a real ceiling to fraction up to --
  // without a trained context, "how full" has nothing to be a fraction OF,
  // so the field stays the plain number entry below instead of inventing
  // stops against a house number.
  const ctxStops = useMemo(
    () => (trainedCtx != null && trainedCtx > 0 ? sliderStops(trainedCtx, ctxGranularity) : []),
    [trainedCtx, ctxGranularity]
  );
  // Never written to state on its own -- this only positions the thumb.
  // Unset (target_ctx == null) previews at the suggested default, same as
  // the number field's placeholder, so neither control fabricates an answer
  // by rendering.
  const ctxActiveValue = goals.target_ctx ?? suggestedTarget;
  const ctxActiveIndex = useMemo(() => {
    if (ctxStops.length === 0) return -1;
    let best = 0;
    let bestDiff = Infinity;
    ctxStops.forEach((stop, i) => {
      const diff = Math.abs(stop - ctxActiveValue);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = i;
      }
    });
    return best;
  }, [ctxStops, ctxActiveValue]);

  function setTarget(raw: string): void {
    setTargetText(raw);
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed) || parsed <= 0) {
      onChange({ ...goals, target_ctx: null });
      return;
    }
    // Clamped by trained context INLINE -- typing 65 536 against a 32 768 cap
    // shows the clamp immediately rather than at submit time.
    const value = trainedCtx != null && trainedCtx > 0 ? Math.min(Math.floor(parsed), trainedCtx) : Math.floor(parsed);
    onChange({ ...goals, target_ctx: value });
  }

  // M1's live feasibility readout, priced for the two KV pairs the
  // questionnaire actually talks about.
  function affordabilityFor(cacheType: string): MaxCtxEstimate | null {
    if (!affordability || affordability.totalMib == null) return null;
    return maxAffordableContext({
      totalMib: affordability.totalMib,
      weightsMib: affordability.weightsMib,
      nLayer: affordability.nLayer ?? 0,
      nHeadKv: affordability.nHeadKv ?? 0,
      headDimK: affordability.headDimK,
      headDimV: affordability.headDimV,
      nEmbd: affordability.nEmbd,
      nHead: affordability.nHead,
      slidingWindow: affordability.slidingWindow,
      cacheTypeK: cacheType,
      cacheTypeV: cacheType,
    });
  }

  const f16 = affordabilityFor("f16");
  const q8 = affordabilityFor("q8_0");
  const feasible = f16 && f16.confidence !== "unknown" ? f16 : null;

  // The dual-pool (VRAM + RAM) fit matrix -- same posture as affordabilityFor
  // above (plain consts, not memoized: the arithmetic is cheap and every
  // other estimate on this page is computed the same way), anchored on the
  // same f16/f16 pair and the same ctxActiveValue the slider itself shows.
  const dualFit = placement
    ? computeDualPoolFit({
        modelSizeBytes: placement.modelSizeBytes ?? 0,
        totalModelLayers: placement.nglMax,
        kvLayerCount: placement.kvLayerCount,
        ngl: placement.ngl,
        ctxTokens: ctxActiveValue,
        nHeadKv: affordability?.nHeadKv ?? 0,
        headDimK: affordability?.headDimK,
        headDimV: affordability?.headDimV,
        nEmbd: affordability?.nEmbd,
        nHead: affordability?.nHead,
        cacheTypeK: "f16",
        cacheTypeV: "f16",
        slidingWindow: affordability?.slidingWindow,
        vram: { freeMib: haircut(placement.vram.freeMib, placement.poolHaircutFrac), totalMib: placement.vram.totalMib },
        ram: { freeMib: haircut(placement.ram.freeMib, placement.poolHaircutFrac), totalMib: placement.ram.totalMib },
        unifiedPool: placement.unifiedPool,
        tensorBreakdown: placement.tensorBreakdown,
      })
    : null;

  // Only computed once the live check says the current combo doesn't fit --
  // an "ok" verdict has nothing to suggest against.
  const suggestions =
    placement && dualFit?.fits === false
      ? suggestPlacementConfigs({
          modelSizeBytes: placement.modelSizeBytes ?? 0,
          totalModelLayers: placement.nglMax,
          kvLayerCount: placement.kvLayerCount,
          currentNgl: placement.ngl,
          currentCtx: ctxActiveValue,
          trainedCtx: trainedCtx ?? null,
          nHeadKv: affordability?.nHeadKv ?? 0,
          headDimK: affordability?.headDimK,
          headDimV: affordability?.headDimV,
          nEmbd: affordability?.nEmbd,
          nHead: affordability?.nHead,
          cacheTypeK: "f16",
          cacheTypeV: "f16",
          slidingWindow: affordability?.slidingWindow,
          vram: { freeMib: haircut(placement.vram.freeMib, placement.poolHaircutFrac), totalMib: placement.vram.totalMib },
          ram: { freeMib: haircut(placement.ram.freeMib, placement.poolHaircutFrac), totalMib: placement.ram.totalMib },
          unifiedPool: placement.unifiedPool,
          noGpu: placement.noGpu,
          tensorBreakdown: placement.tensorBreakdown,
        })
      : null;

  return (
    <fieldset className="rounded-xl border border-border bg-surface p-4">
      <legend className="flex flex-wrap items-center gap-2 px-1 text-sm font-semibold text-fg">
        What are you optimizing for?
        <span className="rounded-full bg-surface-raised px-2 py-0.5 text-[11px] font-normal text-muted">
          skippable — “just optimize” keeps every default
        </span>
        {unset && (
          <span className="rounded-full border border-dashed border-border px-2 py-0.5 text-[11px] font-normal text-muted">
            unset — this preset predates goals
          </span>
        )}
      </legend>

      <div className="mt-3 flex flex-col gap-2.5">
        {/* Priority ------------------------------------------------------------ */}
        <div className="rounded-lg border border-border bg-surface-raised p-3.5">
          <span className="text-[13.5px] text-fg">
            A · Priority · What matters most?
            <small className="mt-0.5 block text-[11px] font-normal text-muted">
              drives ranking and which cards appear
            </small>
          </span>
          <div className="mt-2 inline-flex overflow-hidden rounded-lg border border-border" role="radiogroup" aria-label="Goal">
            {GOAL_CHOICES.map((choice) => (
              <button
                key={choice.value}
                type="button"
                role="radio"
                aria-checked={goals.goal === choice.value}
                onClick={() => onChange({ ...goals, goal: choice.value })}
                className={
                  goals.goal === choice.value
                    ? "bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg"
                    : "bg-surface px-4 py-1.5 text-sm text-muted hover:text-fg"
                }
              >
                {choice.label}
              </button>
            ))}
          </div>

          {/* The usable-speed floor is a DISCLOSURE revealed by Max context:
              without it, argmax-context degenerates into "quantize
              everything and crawl". */}
          <div
            className={goals.goal === "max_context" ? "mt-2.5 rounded-lg border border-dashed border-border p-2.5" : "hidden"}
            aria-expanded={goals.goal === "max_context"}
          >
            <span className="text-[12px] text-muted">
              <b className="text-fg">Usable-speed floor</b> — without it, argmax-context degenerates into
              “quantize everything and crawl”:
            </span>
            <span className="ml-2 inline-flex gap-1.5" role="radiogroup" aria-label="Usable-speed floor">
              {SPEED_FLOOR_CHOICES.map((frac) => (
                <button
                  key={frac}
                  type="button"
                  role="radio"
                  aria-checked={(goals.speed_floor_frac ?? 0.5) === frac}
                  onClick={() => onChange({ ...goals, speed_floor_frac: frac })}
                  className={
                    (goals.speed_floor_frac ?? 0.5) === frac
                      ? "rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 text-[12px] font-semibold text-accent"
                      : "rounded-full border border-border bg-surface px-2.5 py-0.5 text-[12px] text-muted"
                  }
                >
                  ≥ {Math.round(frac * 100)} % of best TG
                </button>
              ))}
            </span>
          </div>
        </div>

        {/* Context target ------------------------------------------------------ */}
        <div className="rounded-lg border border-border bg-surface-raised p-3.5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="text-[13.5px] text-fg">
              B · Context target · How full should the window be?
              <small className="mt-0.5 block text-[11px] font-normal text-muted">anchors depth and Test B sizes</small>
            </span>
            <span
              className={`font-mono text-[13.5px] font-bold ${goals.target_ctx != null ? "text-accent" : "text-muted"}`}
            >
              {ctxActiveValue.toLocaleString()} tokens
            </span>
          </div>

          {/* The slider snaps to common fractions of this model's trained
              context; the exact number field below it stays the source of
              truth (its value is what's stored), so a drag and a typed
              number never disagree about what will be sent. */}
          {ctxStops.length > 0 && (
            <div className="mt-3 px-0.5">
              <input
                type="range"
                min={0}
                max={ctxStops.length - 1}
                step={1}
                value={ctxActiveIndex}
                onChange={(e) => setTarget(String(ctxStops[Number(e.target.value)]))}
                aria-label="Target context, snapped to common fractions of the trained context"
                aria-describedby={clampId}
                className="w-full cursor-pointer"
              />
              <div className="relative mt-1.5 h-[12px] font-mono text-[9.5px] font-semibold">
                {ctxStops.map((stop, i) => {
                  const last = ctxStops.length - 1;
                  // A fine slider has several times as many stops as there is
                  // room to label, so only every Nth gets text -- the ends
                  // always do, since they name the range. Every stop is still
                  // selectable; this is purely what the eye can read.
                  const labelEvery = Math.max(1, Math.ceil((last + 1) / 9));
                  if (i !== 0 && i !== last && i % labelEvery !== 0) return null;
                  // Interior ticks sit exactly under where the native
                  // thumb's center lands for that stop (inset from the
                  // edges by half its width), so label and thumb agree.
                  // The first/last ticks stay flush against the track's
                  // own ends instead of matching the thumb, which can
                  // never fully reach them without overflowing.
                  if (i === 0) {
                    return (
                      <span
                        key={stop}
                        className={`absolute left-0 ${i === ctxActiveIndex ? "text-accent" : "text-muted"}`}
                      >
                        {stop.toLocaleString()}
                      </span>
                    );
                  }
                  if (i === last) {
                    return (
                      <span
                        key={stop}
                        className={`absolute right-0 ${i === ctxActiveIndex ? "text-accent" : "text-muted"}`}
                      >
                        {stop.toLocaleString()}
                      </span>
                    );
                  }
                  const frac = i / last;
                  return (
                    <span
                      key={stop}
                      style={{ left: `calc(4px + ${frac} * (100% - 8px))`, transform: "translateX(-50%)" }}
                      className={`absolute ${i === ctxActiveIndex ? "text-accent" : "text-muted"}`}
                    >
                      {stop.toLocaleString()}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* The "don't know" reset and the clamp chip sit OUTSIDE the number
              input's own conditional: hiding the input must not also remove
              the only route back to target_ctx: null, which is the
              target_unverified marker M3's pinned reduction depends on. */}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {showCtxNumberInput && (
              <input
                type="number"
                min={1}
                value={targetText}
                onChange={(e) => setTarget(e.target.value)}
                placeholder={String(suggestedTarget)}
                aria-label="Target context in tokens"
                aria-describedby={clampId}
                className="w-32 rounded-lg border border-border bg-bg px-2.5 py-1 font-mono text-sm text-fg outline-none focus:border-accent"
              />
            )}
            <button
              type="button"
              role="radio"
              aria-checked={goals.target_ctx == null}
              onClick={() => {
                setTargetText("");
                onChange({ ...goals, target_ctx: null });
              }}
              className={
                goals.target_ctx == null
                  ? "rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 text-[12px] font-semibold text-accent"
                  : "rounded-full border border-border bg-surface px-2.5 py-0.5 text-[12px] text-muted"
              }
            >
              don’t know
            </button>
            {clamped && trainedCtx != null && (
              <span className="rounded-full bg-surface px-2 py-0.5 text-[11px] text-muted">
                clamped → {trainedCtx.toLocaleString()}
              </span>
            )}
          </div>
          <p id={clampId} aria-live="polite" className="mt-1.5 text-[11px] leading-relaxed text-muted">
            {trainedCtx != null
              ? `Clamped by trained context (${trainedCtx.toLocaleString()}). `
              : "This model's header never reported a trained context, so nothing clamps the target. "}
            {goals.target_ctx == null ? (
              <>
                Suggested: {suggestedTarget.toLocaleString()} (min of 32 k and this model's trained context) — left
                blank, this stores a <span className="font-mono">target_unverified</span> marker instead of a fit
                line.
              </>
            ) : (
              <>Stored with the run, so scoring is reproducible without today’s defaults.</>
            )}
            {feasible && q8 && (
              <>
                {" "}
                Live feasibility: this card affords about <b className="text-fg">{formatTokens(feasible.tokens)}</b>{" "}
                tokens at f16/f16 and <b className="text-fg">{formatTokens(q8.tokens)}</b> at q8_0/q8_0 · confidence{" "}
                <b className="text-fg">{feasible.confidence}</b>.
              </>
            )}
            {!feasible && (
              <> Affordability is unavailable for this model+machine — the estimate never fabricates a number.</>
            )}
          </p>

          {placement && (
            <PlacementMatrix
              placement={placement}
              dualFit={dualFit}
              suggestions={suggestions}
              ctx={ctxActiveValue}
              trainedCtx={trainedCtx ?? null}
              affordability={affordability}
              onApplyConfig={(ngl, ctx) => {
                placement.onNglChange(ngl);
                setTarget(String(ctx));
              }}
            />
          )}
        </div>

        {/* Workload shape ------------------------------------------------------- */}
        <div className="rounded-lg border border-border bg-surface-raised p-3.5">
          <span className="text-[13.5px] text-fg">
            C · Workload shape · Chat, docs, or even?
            <small className="mt-0.5 block text-[11px] font-normal text-muted">sets the pp/tg scoring weights</small>
          </span>
          <div className="mt-2 inline-flex overflow-hidden rounded-lg border border-border" role="radiogroup" aria-label="Workload shape">
            {WORKLOAD_CHOICES.map((choice) => (
              <button
                key={choice.value}
                type="button"
                role="radio"
                aria-checked={goals.workload === choice.value}
                onClick={() => onChange({ ...goals, workload: choice.value })}
                className={
                  goals.workload === choice.value
                    ? "bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg"
                    : "bg-surface px-4 py-1.5 text-sm text-muted hover:text-fg"
                }
              >
                {choice.label}
              </button>
            ))}
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
            Weights shown as they will be applied:{" "}
            <span className="font-mono">
              (wPP {WORKLOAD_WEIGHTS[goals.workload].wPP.toFixed(2)} · wTG{" "}
              {WORKLOAD_WEIGHTS[goals.workload].wTG.toFixed(2)})
            </span>
            . Answers are stored as configuration under the root run’s <span className="font-mono">goals</span> key
            — reproducible, never re-measured.
          </p>
        </div>

        {/* KV cache preset -- collapsed by default, understated rather than
            boxed: an advanced knob most runs never touch. Four curated,
            strictly nested grids (Compact ⊂ Basic ⊂ Extended ⊂
            Comprehensive) replace the old prune-a-tolerance knob -- each tier
            IS the exact set of (K,V) pairs the sweep stage runs, not a
            cross-product superset of it. ------------------------------- */}
        <div>
          <button
            type="button"
            onClick={() => setKvPresetOpen((open) => !open)}
            aria-expanded={kvPresetOpen}
            aria-controls={kvPresetId}
            className="text-left text-xs text-muted underline decoration-dotted hover:text-fg"
          >
            KV cache preset (advanced) — collapsed, picks the KV grid the sweep stage runs
          </button>
          <div id={kvPresetId} hidden={!kvPresetOpen} className="mt-2">
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="KV cache preset">
              {KV_PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  role="radio"
                  aria-checked={(goals.kv_preset ?? "extended") === preset}
                  onClick={() => onChange({ ...goals, kv_preset: preset })}
                  className={
                    (goals.kv_preset ?? "extended") === preset
                      ? "rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 text-[12px] font-semibold text-accent"
                      : "rounded-full border border-border bg-surface-raised px-2.5 py-0.5 text-[12px] text-muted"
                  }
                >
                  {KV_PRESET_LABEL[preset]} ({KV_PRESET_PAIRS[preset].length})
                </button>
              ))}
            </div>
            <KvPresetPreview preset={goals.kv_preset ?? "extended"} />
          </div>
        </div>
      </div>
    </fieldset>
  );
}

// The Step-2 fit matrix: an offload slider paired with the context slider
// above it, live dual-pool (VRAM+RAM) indicators, an inaccuracy warning, and
// the Tested-configurations row -- one card per N2 search mode, each able to
// fire a real probe at its own starting point.
function PlacementMatrix({
  placement,
  dualFit,
  suggestions,
  ctx,
  trainedCtx,
  affordability,
  onApplyConfig,
}: {
  placement: NonNullable<GoalQuestionnaireProps["placement"]>;
  dualFit: DualPoolFit | null;
  suggestions: SuggestionResult | null;
  ctx: number;
  trainedCtx: number | null;
  affordability: GoalQuestionnaireProps["affordability"];
  onApplyConfig: (ngl: number, ctx: number) => void;
}) {
  const [granularity, setGranularity] = useState<ProbeGranularity>("basic");
  // Which card was explicitly clicked/tested last. Several modes
  // (keep_context/balanced/fixed_offload/custom) all start from the SAME
  // (ngl, ctx) as the user's current sliders by definition, so comparing
  // start to the live placement would highlight all of them at once -- and
  // permanently, since there'd be nothing to move away from. Tracking the
  // clicked mode instead makes selection exclusive, and it still clears
  // naturally once the sliders are dragged somewhere that mode's start no
  // longer matches (see `selected` below).
  const [activeMode, setActiveMode] = useState<ProbeMode | null>(null);
  // "Test all"'s own queue -- NOT six simultaneous triggers. The server's
  // §0.5 duplicate-trigger guard refuses more than one non-terminal probe
  // per (model, worker) at a time regardless of which card asked for it, so
  // firing all six at once leaves five of them rejected with a 409. This
  // drains one mode at a time instead, advancing only once nothing for this
  // pairing (from this queue or a manual Test click) is still pending.
  const [testQueue, setTestQueue] = useState<ProbeMode[]>([]);

  // Where each mode STARTS. The estimate only picks the starting point; the
  // probe measures the rest, which is the whole reason these are called
  // tested rather than suggested configurations. Modes that search from the
  // user's own placement (everything but max_gpu/max_context) all start at
  // exactly the same (ngl, ctx) shown above -- what differs between them is
  // which axis the search moves, not where it begins; shared/probeLadder.ts's
  // own estimate (estimateSafeNgl/maxAffordableContext) supplies the target
  // those searches bisect TOWARD, computed worker-side once a probe is
  // actually running.
  const modeStarts = useMemo<Record<ProbeMode, { ngl: number; ctx: number }>>(() => {
    const affordable = maxAffordableContext({
      totalMib: affordability?.totalMib ?? 0,
      weightsMib: affordability?.weightsMib ?? null,
      nLayer: placement.kvLayerCount,
      nHeadKv: affordability?.nHeadKv ?? 0,
      headDimK: affordability?.headDimK,
      headDimV: affordability?.headDimV,
      nEmbd: affordability?.nEmbd,
      nHead: affordability?.nHead,
      cacheTypeK: "f16",
      cacheTypeV: "f16",
      slidingWindow: affordability?.slidingWindow,
      trainedCtx,
    });
    // A zero/unknown affordability must not become a zero-token start: fall
    // back to what the user is already looking at rather than a fake number.
    const maxCtxStart = affordable.tokens > 0 ? Math.min(affordable.tokens, trainedCtx ?? affordable.tokens) : ctx;
    return {
      max_gpu: { ngl: placement.nglMax, ctx: PROBE_LADDER_MIN_CTX },
      max_context: { ngl: placement.ngl, ctx: maxCtxStart },
      keep_context: { ngl: placement.ngl, ctx },
      balanced: { ngl: placement.ngl, ctx },
      fixed_offload: { ngl: placement.ngl, ctx },
      custom: { ngl: placement.ngl, ctx },
    };
  }, [placement, affordability, trainedCtx, ctx]);

  useEffect(() => {
    if (testQueue.length === 0) return;
    const anyPending = Object.values(placement.verifyResults).some((r) => r?.status === "pending");
    if (anyPending) return;
    const [next, ...rest] = testQueue;
    setTestQueue(rest);
    const start = modeStarts[next];
    setActiveMode(next);
    placement.onVerify(start.ngl, start.ctx, next, granularity);
  }, [testQueue, placement, modeStarts, granularity]);

  const lockedReason =
    placement.locked === "cpu"
      ? "This machine has no GPU — every layer runs on CPU. Nothing to trade here."
      : placement.locked === "unified"
        ? "Metal / shared-memory GPU — VRAM and RAM are the same physical pool, so offload placement costs nothing and buys nothing. Every layer runs on GPU for speed."
        : null;

  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[13.5px] text-fg">
          Placement · GPU ↔ CPU layers
          <small className="mt-0.5 block text-[11px] font-normal text-muted">
            paired with the context above — together they decide what fits
          </small>
        </span>
        <span className="font-mono text-[13.5px] font-bold text-fg">
          {placement.ngl} / {placement.nglMax} layers on GPU
        </span>
      </div>

      <div className="mt-3 px-0.5">
        <input
          type="range"
          min={0}
          max={placement.nglMax}
          step={1}
          value={placement.ngl}
          disabled={placement.locked != null}
          onChange={(e) => placement.onNglChange(Number(e.target.value))}
          aria-label="GPU layers offloaded, CPU/RAM the rest"
          aria-disabled={placement.locked != null}
          className={`w-full ${placement.locked != null ? "cursor-not-allowed opacity-40" : "cursor-pointer"}`}
        />
      </div>
      {lockedReason && <p className="mt-1.5 text-[11px] leading-relaxed text-muted">{lockedReason}</p>}

      {dualFit && (
        <div className="mt-3 flex flex-col gap-2">
          {dualFit.unifiedPool ? (
            <PoolBar label="Unified memory (GPU + CPU)" fit={dualFit.gpu} />
          ) : (
            <>
              <PoolBar label="VRAM" fit={dualFit.gpu} />
              <PoolBar label="RAM" fit={dualFit.cpu} />
            </>
          )}
        </div>
      )}

      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        <b className="text-fg">Estimate only, not a guarantee.</b> Flat per-layer average — weakest for
        Mixture-of-Experts models — and doesn't account for the real compute/scratch buffer, which varies with batch
        size and architecture. Verify below before trusting it for a real run.
        {placement.poolHaircutFrac > 0 && (
          <>
            {" "}
            A recent verify came back short, so free memory above is shown with a{" "}
            {Math.round(placement.poolHaircutFrac * 100)}% safety margin until the next successful verify.
          </>
        )}
      </p>

      {dualFit?.fits === false && suggestions?.outcome === "cannot_run" && (
        <p className="mt-3 rounded-lg border border-dashed border-danger/40 p-3 text-[12px] leading-relaxed text-danger">
          <b>This model cannot run on this machine</b> — no combination of offload placement and context length fits
          VRAM and RAM together.
        </p>
      )}
      {dualFit?.fits === false && suggestions?.outcome === "unknown" && (
        <p className="mt-3 rounded-lg border border-dashed border-border p-3 text-[12px] leading-relaxed text-muted">
          Not enough data to suggest a fix yet — waiting on a live memory reading from this machine, or this model's
          KV geometry.
        </p>
      )}

      {/* Tested configurations -- always shown, not only when the current
          combo fails to fit. Each card is a real search mode: clicking the
          card moves the sliders to its starting point (highlighting it while
          it's the active selection), Test measures what actually fits from
          there. */}
      <div className="mt-3 border-t border-border pt-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <span className="text-[13.5px] text-fg">
            Tested configurations
            <small className="mt-0.5 block text-[11px] font-normal text-muted">
              the estimate picks where each one starts — testing measures the rest
            </small>
          </span>
          <div className="flex items-center gap-2">
            <div className="inline-flex overflow-hidden rounded-lg border border-border" role="radiogroup" aria-label="Test granularity">
              {PROBE_GRANULARITIES.map((value) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={granularity === value}
                  title={GRANULARITY_BLURB[value]}
                  onClick={() => setGranularity(value)}
                  className={
                    granularity === value
                      ? "bg-accent px-2.5 py-1 text-[11px] font-semibold text-accent-fg"
                      : "bg-surface px-2.5 py-1 text-[11px] text-muted hover:text-fg"
                  }
                >
                  {GRANULARITY_LABEL[value]}
                </button>
              ))}
            </div>
            <button
              type="button"
              title="Runs every card below in turn -- only one probe can be in flight for this model+machine at a time"
              onClick={() => setTestQueue(Object.keys(MODE_LABEL) as ProbeMode[])}
              className="rounded-lg border border-accent px-2.5 py-1 text-[11px] font-semibold text-accent hover:bg-accent/10"
            >
              Test all
            </button>
          </div>
        </div>
        <p className="mt-1.5 text-[11px] leading-relaxed text-muted">{GRANULARITY_BLURB[granularity]}</p>

        <div className="mt-2 flex flex-wrap gap-2">
          {(Object.keys(MODE_LABEL) as ProbeMode[]).map((mode) => {
            const start = modeStarts[mode];
            const result = placement.verifyResults[mode] ?? null;
            return (
              <ModeCard
                key={mode}
                mode={mode}
                start={start}
                selected={activeMode === mode && placement.ngl === start.ngl && ctx === start.ctx}
                busy={result?.status === "pending"}
                result={result}
                onApply={() => {
                  setActiveMode(mode);
                  onApplyConfig(start.ngl, start.ctx);
                }}
                onTest={() => {
                  setActiveMode(mode);
                  placement.onVerify(start.ngl, start.ctx, mode, granularity);
                }}
                onReset={() => placement.onReset(mode)}
              />
            );
          })}
        </div>

        <p className="mt-2 text-[11px] leading-relaxed text-muted">
          Testing really loads the model on this machine — up to {PROBE_MAX_LOADS} loads, narrowing in on the boundary
          rather than walking to it. Results land on the run’s own page, and every load is kept.
        </p>
      </div>
    </div>
  );
}

function formatMibShort(mib: number | null | undefined): string {
  if (mib == null) return "—";
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${Math.round(mib)} MiB`;
}

// Compact enough to sit on the same line as the run link: "08/30 14:07",
// never a relative "2h ago" that would go stale while the card sits there.
function formatTestedAt(ms: number): string {
  const d = new Date(ms);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const min = String(d.getMinutes()).padStart(2, "0");
  return `${mm}/${dd} ${hh}:${min}`;
}

// Fixed height sized for the fullest state (blurb + start values + a result
// line + a measured-needs line + a run-link/tested-at line + the button row)
// so all 6 cards line up regardless of which of those lines the current one
// actually has.
const MODE_CARD_HEIGHT = "h-[204px]";

function ModeCard({
  mode,
  start,
  selected,
  busy,
  result,
  onApply,
  onTest,
  onReset,
}: {
  mode: ProbeMode;
  start: { ngl: number; ctx: number };
  selected: boolean;
  busy: boolean;
  result: PlacementVerifyResult | null;
  onApply: () => void;
  onTest: () => void;
  onReset: () => void;
}) {
  const running = result?.status === "pending";
  const failed = result?.status === "failed" || result?.status === "failed_oom";
  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={onApply}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onApply();
        }
      }}
      className={`flex ${MODE_CARD_HEIGHT} w-[188px] shrink-0 cursor-pointer flex-col gap-1 rounded-lg border p-2.5 transition-colors ${
        selected ? "border-accent bg-accent/10" : "border-border bg-surface hover:border-accent/40"
      }`}
    >
      <div className="flex items-start justify-between gap-1">
        <span className="text-[11.5px] font-semibold text-fg">{MODE_LABEL[mode]}</span>
        {result && !running && (
          <button
            type="button"
            title="Clear this card's result"
            aria-label={`Reset ${MODE_LABEL[mode]}`}
            onClick={(e) => {
              e.stopPropagation();
              onReset();
            }}
            className="shrink-0 rounded text-[13px] leading-none text-muted hover:text-fg"
          >
            ×
          </button>
        )}
      </div>
      <span className="text-[10.5px] leading-relaxed text-muted">{MODE_BLURB[mode]}</span>
      <span className="font-mono text-[11px] text-muted">
        from {start.ngl} layers · {start.ctx.toLocaleString()} tokens
      </span>
      <div className="flex-1">
        {result?.status === "verified" && (
          <span className="block font-mono text-[11px] font-semibold text-success">
            ✓ {(result.verifiedCtxTokens ?? result.ctx).toLocaleString()} tokens
            {result.measuredNgl != null ? ` · ${result.measuredNgl} layers` : ""}
          </span>
        )}
        {failed && <span className="block font-mono text-[11px] font-semibold text-danger">✗ didn’t fit</span>}
        {result?.status === "error" && <span className="block text-[10.5px] text-danger">Error — {result.detail ?? "unknown"}</span>}
        {result?.status === "verified" && (result.measuredVramPeakMib != null || result.measuredRamPeakMib != null) && (
          <span className="mt-0.5 block font-mono text-[10.5px] text-muted">
            needed: {formatMibShort(result.measuredVramPeakMib)} VRAM · {formatMibShort(result.measuredRamPeakMib)} RAM
          </span>
        )}
        {result?.runId && (
          <span className="mt-0.5 block font-mono text-[10.5px] text-muted">
            <Link to={`/runs/${result.runId}`} onClick={(e) => e.stopPropagation()} className="text-accent hover:underline">
              Run ↗
            </Link>
            {result.testedAt != null ? ` · ${formatTestedAt(result.testedAt)}` : ""}
          </span>
        )}
      </div>
      <div className="mt-1 flex gap-1.5">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onTest();
          }}
          disabled={busy}
          className="flex-1 rounded-md bg-accent px-2 py-1 text-[11px] font-bold text-accent-fg disabled:opacity-40"
        >
          {running ? "Testing…" : "Test"}
        </button>
      </div>
    </div>
  );
}

function PoolBar({ label, fit }: { label: string; fit: PoolFit }) {
  const pct = fit.freeMib != null && fit.freeMib > 0 ? Math.min(100, (fit.neededMib / fit.freeMib) * 100) : null;
  const tone = fit.fits == null ? "text-muted" : fit.fits ? "text-success" : "text-danger";
  return (
    <div>
      <div className="flex items-baseline justify-between text-[11.5px]">
        <span className="text-muted">{label}</span>
        <span className={`font-mono font-semibold ${tone}`}>
          {formatMib(fit.neededMib)} needed
          {fit.freeMib != null ? ` / ${formatMib(fit.freeMib)} free` : " · no live reading yet"}
        </span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface">
        <div
          className={`h-full ${fit.fits === false ? "bg-danger" : "bg-accent"}`}
          style={{ width: `${pct ?? 0}%` }}
        />
      </div>
    </div>
  );
}

function KvPresetPreview({ preset }: { preset: KvPreset }) {
  const pairs = KV_PRESET_PAIRS[preset];
  return (
    <div className="mt-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {pairs.map(([k, v]) => (
          <span
            key={`${k}/${v}`}
            className="rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 font-mono text-[11px] text-accent"
          >
            {k} / {v}
          </span>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] leading-relaxed text-muted">{KV_PRESET_BLURB[preset]}</p>
    </div>
  );
}
