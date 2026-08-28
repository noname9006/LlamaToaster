// BENCHMARKING_PLAN_V8.md M2 -- "What are you optimizing for?", captured
// BEFORE the grid is built. Three visible questions plus one collapsed
// optional, and SKIPPABLE BY CONSTRUCTION: accepting every default produces a
// trigger payload byte-identical to the previous behavior (the caller passes
// `goals` only when the user actually touched something).
//
// Accessibility (the plan's own conformance block): the whole thing is a
// labelled fieldset; goal/floor/workload chips carry role="radio" +
// aria-checked; the floor reveal and the KV-tolerance section are disclosures
// with aria-expanded; the target clamp announces via aria-live="polite"; and
// tolerance-pruned KV chips stay in the accessibility tree, struck through
// with aria-disabled and the reason in an accessible description.

import { useId, useMemo, useState } from "react";
import {
  DEFAULT_RECOMMENDED_KV_PAIRS,
  KV_TOLERANCES,
  SPEED_FLOOR_CHOICES,
  WORKLOAD_WEIGHTS,
  pairAllowedUnderTolerance,
  type GoalKind,
  type GoalsConfig,
  type KvTolerance,
  type WorkloadShape,
} from "../goals";
import { maxAffordableContext, type MaxCtxEstimate } from "../vramEstimate";

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

const TOLERANCE_LABEL: Record<KvTolerance, string> = {
  q4_0_ok: "q4_0 ok",
  q8_0_ok: "q8_0 ok",
  f16_only: "f16 only",
};

function formatTokens(tokens: number): string {
  if (tokens >= 1000) return `${Math.round(tokens / 1000)} k`;
  return String(tokens);
}

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
}: GoalQuestionnaireProps) {
  const [toleranceOpen, setToleranceOpen] = useState(false);
  const [targetText, setTargetText] = useState(goals.target_ctx != null ? String(goals.target_ctx) : "");
  const toleranceId = useId();
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

        {/* KV tolerance -- collapsed by default, understated rather than boxed:
            an advanced knob most runs never touch. -------------------------- */}
        <div>
          <button
            type="button"
            onClick={() => setToleranceOpen((open) => !open)}
            aria-expanded={toleranceOpen}
            aria-controls={toleranceId}
            className="text-left text-xs text-muted underline decoration-dotted hover:text-fg"
          >
            KV quality tolerance (advanced) — collapsed, prunes the grid before it runs
          </button>
          <div id={toleranceId} hidden={!toleranceOpen} className="mt-2">
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="KV quality tolerance">
              {KV_TOLERANCES.map((tolerance) => (
                <button
                  key={tolerance}
                  type="button"
                  role="radio"
                  aria-checked={(goals.kv_tolerance ?? "q4_0_ok") === tolerance}
                  onClick={() => onChange({ ...goals, kv_tolerance: tolerance })}
                  className={
                    (goals.kv_tolerance ?? "q4_0_ok") === tolerance
                      ? "rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 text-[12px] font-semibold text-accent"
                      : "rounded-full border border-border bg-surface-raised px-2.5 py-0.5 text-[12px] text-muted"
                  }
                >
                  {TOLERANCE_LABEL[tolerance]}
                </button>
              ))}
            </div>
            <KvAxisPreview tolerance={goals.kv_tolerance ?? "q4_0_ok"} />
          </div>
        </div>
      </div>
    </fieldset>
  );
}

// M4 -- the pruned pairs stay in the accessibility tree: struck through
// visually, aria-disabled with the reason in an accessible description,
// exactly like the profile-source rule. The count above already reflects the
// pruning, because pruning happens at grid-BUILD time, before expansion.
function KvAxisPreview({ tolerance }: { tolerance: KvTolerance }) {
  const reasonId = useId();
  const removed = DEFAULT_RECOMMENDED_KV_PAIRS.filter(([k, v]) => !pairAllowedUnderTolerance(k, v, tolerance));
  return (
    <div className="mt-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {DEFAULT_RECOMMENDED_KV_PAIRS.map(([k, v]) => {
          const kept = pairAllowedUnderTolerance(k, v, tolerance);
          return (
            <span
              key={`${k}/${v}`}
              aria-disabled={!kept}
              aria-describedby={kept ? undefined : reasonId}
              className={
                kept
                  ? "rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 font-mono text-[11px] text-accent"
                  : "rounded-full border border-border bg-surface-raised px-2.5 py-0.5 font-mono text-[11px] text-muted line-through opacity-50"
              }
            >
              {k} / {v}
            </span>
          );
        })}
      </div>
      <p id={reasonId} className="mt-1.5 text-[11px] leading-relaxed text-muted">
        {removed.length > 0 ? (
          <>
            Removed by “{TOLERANCE_LABEL[tolerance]}”, before expansion — the live count and cost estimate below
            already reflect it.{" "}
          </>
        ) : (
          <>Nothing is pruned at this tolerance. </>
        )}
        One unquantized pair always survives so flash-attention-off keeps something to vary against.
      </p>
    </div>
  );
}
