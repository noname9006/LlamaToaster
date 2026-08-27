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

import { useId, useState } from "react";
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
        <span className="rounded-full bg-raised px-2 py-0.5 text-[11px] font-normal text-muted">
          skippable — “just optimize” keeps every default
        </span>
        {unset && (
          <span className="rounded-full border border-dashed border-border px-2 py-0.5 text-[11px] font-normal text-muted">
            unset — this preset predates goals
          </span>
        )}
      </legend>

      <div className="mt-3 flex flex-col gap-4">
        {/* Q1 ---------------------------------------------------------------- */}
        <div className="grid gap-2 sm:grid-cols-[150px_1fr] sm:items-start">
          <span className="text-sm text-fg">
            Q1 · What matters most?
            <small className="mt-0.5 block text-[11px] font-normal text-muted">
              drives ranking and which cards appear
            </small>
          </span>
          <div>
            <div className="inline-flex overflow-hidden rounded-lg border border-border" role="radiogroup" aria-label="Goal">
              {GOAL_CHOICES.map((choice) => (
                <button
                  key={choice.value}
                  type="button"
                  role="radio"
                  aria-checked={goals.goal === choice.value}
                  onClick={() => onChange({ ...goals, goal: choice.value })}
                  className={
                    goals.goal === choice.value
                      ? "bg-accent px-4 py-1.5 text-sm font-semibold text-bg"
                      : "bg-raised px-4 py-1.5 text-sm text-muted hover:text-fg"
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
              className={goals.goal === "max_context" ? "mt-2 rounded-lg border border-dashed border-border p-2.5" : "hidden"}
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
                        : "rounded-full border border-border bg-raised px-2.5 py-0.5 text-[12px] text-muted"
                    }
                  >
                    ≥ {Math.round(frac * 100)} % of best TG
                  </button>
                ))}
              </span>
            </div>
          </div>
        </div>

        {/* Q2 ---------------------------------------------------------------- */}
        <div className="grid gap-2 sm:grid-cols-[150px_1fr] sm:items-start">
          <span className="text-sm text-fg">
            Q2 · Target context?
            <small className="mt-0.5 block text-[11px] font-normal text-muted">anchors depth and Test B sizes</small>
          </span>
          <div>
            <div className="flex flex-wrap items-center gap-2">
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
                    : "rounded-full border border-border bg-raised px-2.5 py-0.5 text-[12px] text-muted"
                }
              >
                don’t know
              </button>
              {clamped && trainedCtx != null && (
                <span className="rounded-full bg-raised px-2 py-0.5 text-[11px] text-muted">
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
        </div>

        {/* Q3 ---------------------------------------------------------------- */}
        <div className="grid gap-2 sm:grid-cols-[150px_1fr] sm:items-start">
          <span className="text-sm text-fg">
            Q3 · Workload shape?
            <small className="mt-0.5 block text-[11px] font-normal text-muted">sets the pp/tg scoring weights</small>
          </span>
          <div>
            <div className="inline-flex overflow-hidden rounded-lg border border-border" role="radiogroup" aria-label="Workload shape">
              {WORKLOAD_CHOICES.map((choice) => (
                <button
                  key={choice.value}
                  type="button"
                  role="radio"
                  aria-checked={goals.workload === choice.value}
                  onClick={() => onChange({ ...goals, workload: choice.value })}
                  className={
                    goals.workload === choice.value
                      ? "bg-accent px-4 py-1.5 text-sm font-semibold text-bg"
                      : "bg-raised px-4 py-1.5 text-sm text-muted hover:text-fg"
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
        </div>

        {/* KV tolerance -- collapsed by default -------------------------------- */}
        <div className="grid gap-2 sm:grid-cols-[150px_1fr] sm:items-start">
          <button
            type="button"
            onClick={() => setToleranceOpen((open) => !open)}
            aria-expanded={toleranceOpen}
            aria-controls={toleranceId}
            className="text-left text-sm text-fg"
          >
            KV quality tolerance
            <small className="mt-0.5 block text-[11px] font-normal text-muted">
              collapsed by default — prunes the grid before it runs
            </small>
          </button>
          <div id={toleranceId} hidden={!toleranceOpen}>
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
                      : "rounded-full border border-border bg-raised px-2.5 py-0.5 text-[12px] text-muted"
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
                  : "rounded-full border border-border bg-raised px-2.5 py-0.5 font-mono text-[11px] text-muted line-through opacity-50"
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
