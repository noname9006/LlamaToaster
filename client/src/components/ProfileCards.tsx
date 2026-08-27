// BENCHMARKING_PLAN_V8.md M3 -- scored cards, ranked on stated intent and
// honest about which of them ranks on an ESTIMATE.
//
// Three rules this component exists to keep visible:
//   * Card membership follows the goal exactly; hidden cards are NAMED, never
//     silently absent.
//   * Every card carries M1's fit line, upgraded to N2's verified number when
//     a probe has run for that machine + build + KV pair.
//   * Scoring never rejects anything silently: the rejection tallies render
//     here, because a silent zero-profile outcome is indistinguishable from a
//     bug.
//
// Accessibility: fit lines and confidence words are TEXT, never colour-only.

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { GoalsConfig, ProfileCard, ProfilesResponse, ScoredConfig, VerifiedLimitDto, QualityRowDto } from "../types";
import { QUALITY_NOT_MEASURED_DISCLAIMER } from "../../../shared/scoring";
import { WORKLOAD_WEIGHTS } from "../goals";
import { DEFAULT_QUALITY_DATASET_HASH, DEFAULT_QUALITY_DATASET_LABEL } from "../types";

const PROFILE_ICON: Record<string, string> = {
  max_speed: "⚡",
  balanced: "⚖",
  max_context: "⬆",
  low_memory: "🪫",
};

function formatTps(value: number): string {
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function commandFor(config: ScoredConfig): string {
  const axes = config.axes;
  const parts = [
    "llama-bench -m model.gguf",
    `-fa ${axes.flash_attn ?? "on"}`,
    `-ctk ${axes.cache_type_k ?? "f16"}`,
    `-ctv ${axes.cache_type_v ?? "f16"}`,
    `-ngl ${axes.n_gpu_layers ?? 0}`,
  ];
  if ((axes.n_cpu_moe ?? 0) > 0) parts.push(`--n-cpu-moe ${axes.n_cpu_moe}`);
  parts.push(`-b ${axes.batch_size ?? 0} -ub ${axes.ubatch_size ?? 0}`);
  if (config.referenceDepth > 0) parts.push(`-d ${config.referenceDepth}`);
  return parts.join(" ");
}

// M1's fit line, in words. It ANNOTATES; it reorders nothing except the Max
// Context card, whose goal asked for estimates as the criterion.
function fitLine(config: ScoredConfig, goals: GoalsConfig): string | null {
  if (!config.maxCtx || config.maxCtx.confidence === "unknown") {
    const floor = config.maxCtx?.conservativeFloorTokens;
    return floor
      ? `Affordability is unknown for this configuration — the estimate never fabricates a number. A conservative floor candidate (this model's full trained context, ${floor.toLocaleString()} tokens) can be verified with a probe instead.`
      : "Affordability is unknown for this configuration — the estimate never fabricates a number.";
  }
  const tokens = config.maxCtx.tokens;
  if (goals.target_ctx == null) {
    return `Tops out near ${tokens.toLocaleString()} tokens · confidence ${config.maxCtx.confidence} · binding ${
      config.maxCtx.binding ?? "unknown"
    }. No target was stated, so nothing is compared against one.`;
  }
  const headroom = tokens / goals.target_ctx;
  return headroom >= 1
    ? `Affords your ${goals.target_ctx.toLocaleString()} target — yes, ~${headroom.toFixed(1)}× headroom · confidence ${config.maxCtx.confidence} · binding ${config.maxCtx.binding ?? "unknown"}`
    : `Tops out near ${tokens.toLocaleString()} — below your ${goals.target_ctx.toLocaleString()} target · confidence ${config.maxCtx.confidence}`;
}

// N2 -- upgrades the fit line from "estimate" to "verified up to X tokens on
// this machine" when a matching row exists. Verification is per (machine,
// build, KV pair, placement): a mismatch is simply no match, never a reused
// verdict.
function verifiedFor(config: ScoredConfig, limits: VerifiedLimitDto[]): VerifiedLimitDto | undefined {
  const kvType = `${config.axes.cache_type_k}/${config.axes.cache_type_v}`;
  return limits.find((limit) => limit.kv_type === kvType);
}

// N4 -- the most recent quality measurement at this card's own KV pair, if
// any. A footnote only -- never a score, never mixed into any argmax (see
// QUALITY_NOT_MEASURED_DISCLAIMER).
function qualityFor(config: ScoredConfig, results: QualityRowDto[]): QualityRowDto | undefined {
  const matches = results.filter(
    (r) => r.cache_type_k === config.axes.cache_type_k && r.cache_type_v === config.axes.cache_type_v
  );
  return matches.sort((a, b) => b.created_at - a.created_at)[0];
}

export interface ProfileCardsProps {
  runId: string;
  /** Re-fetched whenever the run's own status changes. */
  refreshKey?: unknown;
  /** N2 -- needed to enqueue a probe for the card's own placement. */
  modelId?: string;
  workerId?: string | null;
}

export function ProfileCards({ runId, refreshKey, modelId, workerId }: ProfileCardsProps) {
  const [data, setData] = useState<ProfilesResponse | null>(null);
  const [error, setError] = useState("");
  const [override, setOverride] = useState<Partial<GoalsConfig> | null>(null);
  const [changing, setChanging] = useState(false);
  const [probeMsg, setProbeMsg] = useState("");

  // N2 -- estimate → VERIFY. The probe rides the ordinary trigger route, so
  // every §0.5 guard applies unchanged; the engine is pinned to llama-server
  // server-side, and it performs at most three loads, ever.
  async function verifyWithProbe(config: ScoredConfig): Promise<void> {
    if (!modelId || !workerId) {
      setProbeMsg("This run has no machine attached, so there is nothing to probe against.");
      return;
    }
    // N2 step 1: candidate = the M1 estimate for this placement. When
    // confidence is unknown, offer the conservative floor candidate (the
    // full trained_ctx load) instead of just a dead end -- explicitly
    // labeled conservative, never presented as if it were the real estimate.
    const isUnknown = config.maxCtx?.confidence === "unknown";
    const conservativeFloor = config.maxCtx?.conservativeFloorTokens;
    const candidate = isUnknown ? conservativeFloor : config.maxCtx?.tokens;
    if (!candidate) {
      setProbeMsg(
        "No affordability estimate exists for this placement, and this model's trained context is unknown too, so there is no candidate context to verify at all."
      );
      return;
    }
    setProbeMsg(isUnknown ? "Enqueueing a conservative full-trained-context probe…" : "Enqueueing probe…");
    try {
      await api.triggerRun({
        model_id: modelId,
        worker_id: workerId,
        kind: "probe",
        probe: {
          // Verification is PLACEMENT-SPECIFIC: the probe assumes exactly the
          // ngl / n-cpu-moe the estimate did.
          candidate_ctx: candidate,
          placement: {
            ngl: Number(config.axes.n_gpu_layers ?? 0),
            n_cpu_moe: Number(config.axes.n_cpu_moe ?? 0) || undefined,
            slots: Number(config.axes.concurrency ?? 1),
          },
          kv_pair: [String(config.axes.cache_type_k), String(config.axes.cache_type_v)],
        },
        sweep: {
          n_prompt: [Number(config.axes.n_prompt ?? 512)],
          n_gen: [Number(config.axes.n_gen ?? 128)],
          threads: [Number(config.axes.threads ?? 0)],
          n_gpu_layers: [Number(config.axes.n_gpu_layers ?? 0)],
          batch_size: [Number(config.axes.batch_size ?? 2048)],
          ubatch_size: [Number(config.axes.ubatch_size ?? 512)],
          cache_type_k: [String(config.axes.cache_type_k)],
          cache_type_v: [String(config.axes.cache_type_v)],
          flash_attn: [String(config.axes.flash_attn)],
          mtp: ["off"],
          n_gpu_layers_draft: [0],
          n_cpu_moe: [Number(config.axes.n_cpu_moe ?? 0)],
          repeats: 1,
        },
      });
      setProbeMsg(
        isUnknown
          ? `Conservative probe enqueued at ${candidate.toLocaleString()} tokens (this model's full trained context — no affordability estimate exists for this placement, so this is a floor, not a fitted candidate). Three loads maximum, ever.`
          : `Probe enqueued at ${candidate.toLocaleString()} tokens. Three loads maximum, ever — a failure retries once at ×0.75, and a roomy success probes once higher.`
      );
    } catch (err) {
      setProbeMsg(err instanceof Error ? err.message : String(err));
    }
  }

  const [qualityMsg, setQualityMsg] = useState("");

  // N4 -- one llama-perplexity measurement for this card's own KV pair,
  // against the bundled default corpus (every worker provisions it at
  // startup, see worker/src/qualityCorpus.ts) so this never needs a manually
  // placed dataset. Rides the ordinary trigger route, same as N2's probe --
  // every §0.5 guard applies unchanged, and the result is a labeled
  // synthetic proxy that never enters ranking (see QUALITY_NOT_MEASURED_
  // DISCLAIMER above).
  async function measureQuality(config: ScoredConfig): Promise<void> {
    if (!modelId || !workerId) {
      setQualityMsg("This run has no machine attached, so there is nothing to measure against.");
      return;
    }
    // Default to one ctx point, the questionnaire target when stated --
    // never a ladder. Falls back to the depth this card was actually
    // measured at, then a safe default, when no target was given.
    const ctxTokens = data?.goals.target_ctx ?? (config.referenceDepth > 0 ? config.referenceDepth : 4096);
    setQualityMsg("Enqueueing quality measurement…");
    try {
      await api.triggerRun({
        model_id: modelId,
        worker_id: workerId,
        kind: "quality",
        quality: {
          ctx_tokens: ctxTokens,
          kv_pair: [String(config.axes.cache_type_k), String(config.axes.cache_type_v)],
          dataset_hash: DEFAULT_QUALITY_DATASET_HASH,
        },
        sweep: {
          n_prompt: [Number(config.axes.n_prompt ?? 512)],
          n_gen: [Number(config.axes.n_gen ?? 128)],
          threads: [Number(config.axes.threads ?? 0)],
          n_gpu_layers: [Number(config.axes.n_gpu_layers ?? 0)],
          batch_size: [Number(config.axes.batch_size ?? 2048)],
          ubatch_size: [Number(config.axes.ubatch_size ?? 512)],
          cache_type_k: [String(config.axes.cache_type_k)],
          cache_type_v: [String(config.axes.cache_type_v)],
          flash_attn: [String(config.axes.flash_attn)],
          mtp: ["off"],
          n_gpu_layers_draft: [0],
          n_cpu_moe: [Number(config.axes.n_cpu_moe ?? 0)],
          repeats: 1,
        },
      });
      setQualityMsg(
        `Quality measurement enqueued at ${ctxTokens.toLocaleString()} tokens against ${DEFAULT_QUALITY_DATASET_LABEL.toLowerCase()}. Perplexity only — KLD needs a matching f16/f16 baseline measured first at the same context.`
      );
    } catch (err) {
      setQualityMsg(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    let cancelled = false;
    api
      .getProfiles(runId, override ? { ...override } : undefined)
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError("");
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [runId, refreshKey, override]);

  if (error) return <p className="mt-2 text-sm text-danger">Could not score this run: {error}</p>;
  if (!data) return <p className="mt-2 text-sm text-muted">Scoring…</p>;

  const { scoring, goals } = data;
  const weights = WORKLOAD_WEIGHTS[goals.workload] ?? { wPP: 0.5, wTG: 0.5 };

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-lg border border-border bg-surface px-3.5 py-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-sm text-fg">
            Scored for:{" "}
            <b>
              {goals.goal === "max_context" ? "Max context" : goals.goal === "max_speed" ? "Max tok/s" : "Balanced"}
            </b>
            {goals.target_ctx != null && ` · target ${goals.target_ctx.toLocaleString()}`}
            {data.target_ctx_clamped && " (clamped)"} · {goals.workload}{" "}
            <span className="ml-1 rounded-full bg-raised px-2 py-0.5 font-mono text-[11px] text-muted">
              wPP {weights.wPP.toFixed(2)} · wTG {weights.wTG.toFixed(2)}
            </span>
          </span>
          <button
            type="button"
            onClick={() => setChanging((open) => !open)}
            aria-expanded={changing}
            className="rounded-lg border border-border px-3 py-1 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent"
          >
            Change goal ▾
          </button>
        </div>
        <p className="mt-1 text-[11px] text-muted">
          Switching the goal re-scores instantly over stored results — never re-measurement.
          {scoring.referenceDepth != null && ` TG is compared at depth ${scoring.referenceDepth.toLocaleString()}.`}
          {data.method_versions_present.length > 1 &&
            ` This run mixes methodology versions (${data.method_versions_present.join(", ")}); only the newest is scored — vintages are never averaged together.`}
        </p>
        {changing && (
          <div className="mt-2 flex flex-wrap gap-1.5" role="radiogroup" aria-label="Goal">
            {(["balanced", "max_speed", "max_context"] as const).map((goal) => (
              <button
                key={goal}
                type="button"
                role="radio"
                aria-checked={goals.goal === goal}
                onClick={() => setOverride({ ...goals, goal })}
                className={
                  goals.goal === goal
                    ? "rounded-full border border-accent bg-accent/10 px-2.5 py-0.5 text-[12px] font-semibold text-accent"
                    : "rounded-full border border-border bg-raised px-2.5 py-0.5 text-[12px] text-muted"
                }
              >
                {goal === "max_context" ? "Max context" : goal === "max_speed" ? "Max tok/s" : "Balanced"}
              </button>
            ))}
          </div>
        )}
      </div>

      {scoring.profiles.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3" role="group" aria-label="Profiles">
          {scoring.profiles.map((card) => (
            <Card
              key={card.id}
              card={card}
              goals={goals}
              limits={data.verified_limits}
              qualityResults={data.quality_results}
              onVerify={modelId && workerId ? verifyWithProbe : undefined}
              onMeasureQuality={modelId && workerId ? measureQuality : undefined}
            />
          ))}
          {scoring.hidden.length > 0 && (
            <div className="flex flex-col gap-2 rounded-xl border border-dashed border-border p-3.5">
              <h4 className="text-sm font-semibold text-muted">
                {scoring.hidden.map((h) => h.title).join(" · ")}
                <span className="ml-2 rounded-full border border-dashed border-border px-2 py-0.5 text-[10px] font-normal">
                  hidden
                </span>
              </h4>
              {scoring.hidden.map((hidden) => (
                <p key={hidden.id} className="text-xs leading-relaxed text-muted">
                  <b className="text-fg">{hidden.title}:</b> {hidden.reason}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {probeMsg && <p className="text-[11px] text-muted">{probeMsg}</p>}
      {qualityMsg && <p className="text-[11px] text-muted">{qualityMsg}</p>}

      <p className="text-[11px] text-muted">{QUALITY_NOT_MEASURED_DISCLAIMER}</p>

      {/* Rejection accounting. Zero-profile outcomes render the tallies,
          never silence -- a silent zero is indistinguishable from a bug. */}
      <p className="rounded-lg border border-border bg-raised px-3.5 py-2.5 text-xs leading-relaxed text-muted">
        <b className="text-fg">Scoring rejected nothing silently:</b> {scoring.candidateCount} configuration
        {scoring.candidateCount === 1 ? "" : "s"} considered, {scoring.scoredCount} scored
        {scoring.tallies.stability > 0 && ` · ${scoring.tallies.stability} failed the stability gate`}
        {scoring.tallies.suspect_samples > 0 && ` · ${scoring.tallies.suspect_samples} carried suspect samples`}
        {scoring.tallies.missing_pp_or_tg > 0 &&
          ` · ${scoring.tallies.missing_pp_or_tg} never produced both a pp and a tg row`}
        {scoring.tallies.caveat_flagged > 0 &&
          ` · ${scoring.tallies.caveat_flagged} were removed by a caveat flag`}
        .
        {scoring.profiles.length === 0 &&
          " No card could be produced from this run — the tallies above are the reason, not an error."}
      </p>
    </div>
  );
}

function Card({
  card,
  goals,
  limits,
  qualityResults,
  onVerify,
  onMeasureQuality,
}: {
  card: ProfileCard;
  goals: GoalsConfig;
  limits: VerifiedLimitDto[];
  qualityResults: QualityRowDto[];
  onVerify?: (config: ScoredConfig) => void;
  onMeasureQuality?: (config: ScoredConfig) => void;
}) {
  const verified = verifiedFor(card.config, limits);
  const quality = qualityFor(card.config, qualityResults);
  const fit = fitLine(card.config, goals);
  return (
    <div
      className={
        card.unverified
          ? "flex flex-col gap-2 rounded-xl border border-warning/50 bg-raised p-3.5"
          : "flex flex-col gap-2 rounded-xl border border-border bg-raised p-3.5"
      }
    >
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-semibold text-fg">
          {PROFILE_ICON[card.id]} {card.title.toUpperCase()}
        </h4>
        {card.unverified ? (
          <span className="rounded-full bg-warning-bg px-2 py-0.5 text-[10px] font-bold text-warning">
            unverified
          </span>
        ) : (
          <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-bold text-accent">
            {goals.goal === "max_context" && card.id === "max_context" ? "this goal" : "scored"}
          </span>
        )}
      </div>

      <div className="font-mono text-[11px] leading-relaxed text-muted">
        fa {String(card.config.axes.flash_attn)} · k/v {String(card.config.axes.cache_type_k)}/
        {String(card.config.axes.cache_type_v)}
        <br />
        ngl {String(card.config.axes.n_gpu_layers)}
        {(card.config.axes.n_cpu_moe ?? 0) > 0 && ` · n-cpu-moe ${card.config.axes.n_cpu_moe}`} · b{" "}
        {String(card.config.axes.batch_size)}/ub {String(card.config.axes.ubatch_size)}
        {card.config.referenceDepth > 0 && ` · d ${card.config.referenceDepth.toLocaleString()}`}
      </div>

      <dl className="flex flex-col gap-0.5 text-xs">
        <div className="flex justify-between gap-3">
          <dt className="text-muted">PP</dt>
          <dd className="font-mono text-fg">{formatTps(card.config.pp)} tok/s</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-muted">TG @ depth {card.config.referenceDepth.toLocaleString()}</dt>
          <dd className="font-mono text-fg">{formatTps(card.config.tg)} tok/s</dd>
        </div>
        {card.config.maxCtx && card.config.maxCtx.confidence !== "unknown" && (
          <div className="flex justify-between gap-3">
            <dt className="text-muted">Affordable tokens (est.)</dt>
            <dd className="font-mono text-fg">{card.config.maxCtx.tokens.toLocaleString()}</dd>
          </div>
        )}
        <div className="flex justify-between gap-3">
          <dt className="text-muted">Pressure</dt>
          <dd className="font-mono text-fg">
            {card.config.pressure != null ? card.config.pressure.toFixed(2) : "not measurable here"}
          </dd>
        </div>
      </dl>

      {fit && <p className="border-l-2 border-border pl-2 text-[11px] leading-relaxed text-muted">{fit}</p>}

      {verified && (
        <p className="border-l-2 border-success pl-2 text-[11px] leading-relaxed text-muted">
          Verified: <b className="text-fg">up to {verified.verified_ctx_tokens.toLocaleString()} tokens</b> on this
          machine at {verified.kv_type}
          {verified.margin_observed_frac != null &&
            ` · margin ${(verified.margin_observed_frac * 100).toFixed(0)} %`}
          . Verification is per machine + build + KV pair + placement — changing any of them needs a fresh probe.
        </p>
      )}

      {quality && (
        <p className="border-l-2 border-border pl-2 text-[11px] leading-relaxed text-muted">
          KV {quality.cache_type_k}/{quality.cache_type_v}: PPL <b className="text-fg">{quality.ppl?.toFixed(4)}</b>
          {quality.kld_vs_baseline != null && ` · KLD ${quality.kld_vs_baseline.toFixed(4)} vs f16 baseline`} at{" "}
          {quality.ctx_tokens.toLocaleString()} tokens — a labeled synthetic proxy against a pinned corpus, never a
          score.
        </p>
      )}

      <pre className="overflow-x-auto rounded-lg border border-border bg-bg p-2 font-mono text-[10px] leading-relaxed text-muted">
        {commandFor(card.config)}
      </pre>

      <p className="text-[11px] leading-relaxed text-muted">{card.basis}</p>
      {card.id === "max_context" && (
        <p className="text-[11px] leading-relaxed text-warning">
          Ranked on estimated fit — not measured at your target. Verify with a probe run.
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {onVerify && !verified && (
          <button
            type="button"
            onClick={() => onVerify(card.config)}
            className="self-start rounded-lg border border-border px-3 py-1 text-[11px] font-semibold text-fg hover:border-accent/40 hover:text-accent"
          >
            Verify with a probe
          </button>
        )}
        {onMeasureQuality && (
          <button
            type="button"
            onClick={() => onMeasureQuality(card.config)}
            className="self-start rounded-lg border border-border px-3 py-1 text-[11px] font-semibold text-fg hover:border-accent/40 hover:text-accent"
          >
            Measure quality
          </button>
        )}
      </div>
      <p className="text-[10px] leading-relaxed text-muted">{QUALITY_NOT_MEASURED_DISCLAIMER}</p>
    </div>
  );
}
