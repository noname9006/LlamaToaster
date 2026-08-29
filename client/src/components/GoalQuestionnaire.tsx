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

import { useId, useMemo, useState } from "react";
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
  type SuggestedConfig,
  type SuggestedConfigLabel,
  type SuggestionResult,
} from "../vramEstimate";

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
    /** Non-null when the slider is locked and pinned -- see the lock rules. */
    locked: "cpu" | "unified" | null;
    vram: PoolReading;
    ram: PoolReading;
    unifiedPool: boolean;
    noGpu: boolean;
    /** A recent failed/failed_oom verify's safety margin -- see Benchmark.tsx. */
    poolHaircutFrac: number;
    onVerify: (ngl: number, ctx: number) => void;
    verifying: boolean;
    verifyResult: {
      ngl: number;
      ctx: number;
      runId: string;
      status: "pending" | "verified" | "failed" | "failed_oom" | "error";
      detail?: string;
      verifiedCtxTokens?: number | null;
    } | null;
  };
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

const SUGGESTION_LABEL: Record<SuggestedConfigLabel, string> = {
  target_ctx_reduce_offload: "Keep your context, reduce offload",
  max_offload_reduce_ctx: "Max GPU speed, reduce context",
  balanced: "Balanced",
  minimum_viable: "Minimum viable",
};

// The context-target slider's stops: roughly 100/75/50/25/12.5/7.5/5/2.5/1%
// of the model's trained context, each rounded to the nearest power of two
// and forced strictly below the stop before it so two fractions can never
// collide on the same tick once rounded.
const CTX_STOP_FRACTIONS = [1, 0.75, 0.5, 0.25, 0.125, 0.075, 0.05, 0.025, 0.01];

function computeCtxStops(maxCtx: number): number[] {
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

export function GoalQuestionnaire({
  goals,
  onChange,
  trainedCtx,
  affordability,
  unset,
  placement,
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
    () => (trainedCtx != null && trainedCtx > 0 ? computeCtxStops(trainedCtx) : []),
    [trainedCtx]
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

          <div className="mt-2 flex flex-wrap items-center gap-2">
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
              onApplyConfig={(cfg) => {
                placement.onNglChange(cfg.ngl);
                setTarget(String(cfg.ctx));
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
// above it, live dual-pool (VRAM+RAM) indicators, an inaccuracy warning, up
// to three auto-suggested configs when the current combo doesn't fit, and a
// Verify button that fires a real N2 probe at the exact placement shown.
function PlacementMatrix({
  placement,
  dualFit,
  suggestions,
  ctx,
  onApplyConfig,
}: {
  placement: NonNullable<GoalQuestionnaireProps["placement"]>;
  dualFit: DualPoolFit | null;
  suggestions: SuggestionResult | null;
  ctx: number;
  onApplyConfig: (config: SuggestedConfig) => void;
}) {
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

      {dualFit?.fits === false && suggestions && (
        <div className="mt-3 rounded-lg border border-dashed border-border p-3">
          {suggestions.outcome === "cannot_run" && (
            <p className="text-[12px] leading-relaxed text-danger">
              <b>This model cannot run on this machine</b> — no combination of offload placement and context length
              fits VRAM and RAM together.
            </p>
          )}
          {suggestions.outcome === "unknown" && (
            <p className="text-[12px] leading-relaxed text-muted">
              Not enough data to suggest a fix yet — waiting on a live memory reading from this machine, or this
              model's KV geometry.
            </p>
          )}
          {suggestions.outcome === "ok" && (
            <>
              <span className="text-[12px] font-semibold text-fg">Suggested configurations</span>
              <div className="mt-2 flex flex-wrap gap-2">
                {suggestions.configs.map((cfg) => (
                  <SuggestionCard
                    key={`${cfg.label}:${cfg.ngl}:${cfg.ctx}`}
                    config={cfg}
                    onApply={() => onApplyConfig(cfg)}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3">
        <button
          type="button"
          disabled={placement.verifying}
          onClick={() => placement.onVerify(placement.ngl, ctx)}
          className="rounded-md border border-accent px-3 py-1 text-[11.5px] font-semibold text-accent disabled:opacity-40"
        >
          {placement.verifying ? "Verifying…" : "Verify with a probe"}
        </button>
        <span className="text-[11px] leading-relaxed text-muted">
          Actually loads the model on this machine at the exact placement above and confirms it fits — up to 3 real
          loads.
        </span>
      </div>
      {placement.verifyResult && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-muted">{verifyResultText(placement.verifyResult)}</p>
      )}
    </div>
  );
}

function verifyResultText(result: NonNullable<NonNullable<GoalQuestionnaireProps["placement"]>["verifyResult"]>): string {
  const at = `${result.ngl} layers / ${result.ctx.toLocaleString()} tokens`;
  switch (result.status) {
    case "pending":
      return `Verifying at ${at}…`;
    case "verified":
      return `✓ Verified — fits at ${at}${
        result.verifiedCtxTokens != null ? ` (confirmed ceiling ~${result.verifiedCtxTokens.toLocaleString()} tokens)` : ""
      }.`;
    case "failed":
      return `✗ Doesn't fit at ${at} — the machine couldn't load this configuration.${result.detail ? ` ${result.detail}` : ""}`;
    case "failed_oom":
      return `✗ Doesn't fit at ${at} — ran out of memory.${result.detail ? ` ${result.detail}` : ""}`;
    case "error":
      return `Error verifying ${at}: ${result.detail ?? "unknown error"}`;
  }
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

function SuggestionCard({ config, onApply }: { config: SuggestedConfig; onApply: () => void }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface p-2.5">
      <span className="text-[11.5px] font-semibold text-fg">{SUGGESTION_LABEL[config.label]}</span>
      <span className="font-mono text-[11px] text-muted">
        {config.ngl} layers · {config.ctx.toLocaleString()} tokens
      </span>
      <button
        type="button"
        onClick={onApply}
        className="mt-1 rounded-md bg-accent px-2 py-1 text-[11px] font-bold text-accent-fg"
      >
        Apply
      </button>
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
