import { useState, type ReactNode } from "react";
import { Btn, CopyCommand, Kicker, Notice, Panel, ProgressBar, StatCard, StepTitle, fmtDuration, fmtGb, fmtTps } from "./ui";
import type { FlowContext } from "./flowContext";
import type { FlowStep } from "../../types";
import { SpeedBars } from "./charts";
import { cpuSideMib, isFailed, speedRunView, spilledMib, templateSweep, type SpeedRunView } from "./flowData";
import { isActive, type FlowRun } from "./useOptimizeFlow";
import { ctxLabel } from "../../../../shared/optimizeFlow.js";
import {
  baselineCandidate,
  candidateKey,
  compareToBaseline,
  estimateSpeedRunSeconds,
  MAX_SPEED_CANDIDATES,
  serverCommandLine,
  speedWorkload,
  type SpeedCandidate,
  type SpeedReading,
} from "../../../../shared/speedRun.js";

function readingText(r: SpeedReading | null): string {
  if (!r) return "—";
  return r.samples < 2 ? `${fmtTps(r.mean)}` : `${fmtTps(r.mean)} ± ${fmtTps(r.stddev)}`;
}

function deltaText(cand: SpeedReading | null, base: SpeedReading | null): { text: string; tone: string } {
  const c = compareToBaseline(cand, base);
  if (c.pct == null) return { text: "—", tone: "text-muted" };
  const rounded = Math.round(c.pct);
  const text = rounded === 0 ? "±0%" : `${rounded > 0 ? "+" : ""}${rounded}%`;
  if (c.noSpread) return { text: `${text} · no spread`, tone: "text-fg-2" };
  if (!c.clear) return { text: `${text} · within noise`, tone: "text-muted" };
  return { text, tone: c.pct >= 0 ? "text-accent" : "text-danger" };
}

function statusText(view: SpeedRunView["rows"][number]["outcome"]): string {
  switch (view.status) {
    case "done":
      return spilledMib(view) != null ? `spilled ${fmtGb(spilledMib(view))}` : "ok";
    case "failed_oom":
      return "out of memory";
    case "failed_timeout":
      return "timed out";
    case "failed":
    case "failed_unsupported":
      return "failed";
    case "cancelled":
      return "stopped";
    case "skipped":
      return "skipped";
    case "pending":
    case "queued":
      return "queued";
    default:
      return "running";
  }
}

export function SpeedTable({ view, baseline, refName }: { view: SpeedRunView; baseline: SpeedRunView | null; refName?: string }) {
  const base = baseline?.rows.find((r) => r.candidate.source === "baseline")?.outcome ?? null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-border text-[13px] text-muted">
            <th scope="col" className="px-3 py-2 font-normal">Config</th>
            <th scope="col" className="px-2 py-2 text-right font-normal">pp t/s</th>
            <th scope="col" className="px-2 py-2 text-right font-normal">tg t/s</th>
            {base && (
              <th scope="col" className="px-2 py-2 text-right font-normal">
                tg vs {refName && refName !== "the default settings" ? "current" : "default"}
              </th>
            )}
            <th scope="col" className="px-2 py-2 text-right font-normal">VRAM peak</th>
            <th scope="col" className="px-3 py-2 font-normal">Status</th>
          </tr>
        </thead>
        <tbody>
          {view.rows.map(({ candidate, outcome }) => {
            const d = base && candidate.source !== "baseline" ? deltaText(outcome.tg, base.tg) : null;
            const failed = isFailed(outcome.status);
            const spill = spilledMib(outcome);
            return (
              <tr key={outcome.idx} className="border-b border-border font-mono text-xs">
                <td className="px-3 py-2">
                  <div className="font-sans text-sm font-medium text-fg">{candidate.label}</div>
                  <div className="text-muted [overflow-wrap:anywhere]">
                    -ngl {candidate.ngl} · {candidate.ctk}/{candidate.ctv} · FA {candidate.fa}
                    {candidate.ot ? " · experts on CPU" : ""}
                    {candidate.b != null || candidate.ub != null ? ` · -b ${candidate.b ?? 2048} -ub ${candidate.ub ?? 512}` : ""}
                    {candidate.threads ? ` · -t ${candidate.threads.t} -tb ${candidate.threads.tb}` : ""}
                    {cpuSideMib(outcome) ? ` · ${fmtGb(cpuSideMib(outcome))} on CPU side` : ""}
                  </div>
                  {failed && outcome.error && <div className="mt-1 font-sans text-danger">{outcome.error}</div>}
                </td>
                <td className="px-2 py-2 text-right text-fg">{readingText(outcome.pp)}</td>
                <td className="px-2 py-2 text-right text-fg">{readingText(outcome.tg)}</td>
                {base && <td className={`px-2 py-2 text-right ${d?.tone ?? "text-muted"}`}>{d?.text ?? "reference"}</td>}
                <td className="px-2 py-2 text-right text-fg-2">{fmtGb(outcome.vramPeakMib)}</td>
                <td className={`px-3 py-2 ${failed || spill != null ? "text-danger" : "text-fg-2"}`}>{statusText(outcome)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function runProgress(run: FlowRun): number {
  const total = run.test.items_total ?? run.items.length;
  if (!total) return 5;
  const done = run.items.filter((i) => ["done", "failed", "failed_oom", "failed_timeout", "failed_unsupported", "cancelled", "skipped"].includes(i.status)).length;
  return Math.max(5, (done / total) * 100);
}

export function RunProgress({ run }: { run: FlowRun }) {
  const live = run.items.find((i) => !["done", "failed", "failed_oom", "failed_timeout", "failed_unsupported", "cancelled", "skipped", "queued"].includes(i.status));
  return (
    <div className="flex flex-col gap-1.5" aria-live="polite">
      <ProgressBar pct={runProgress(run)} live />
      <span className="font-mono text-xs text-fg-2">
        {run.test.status === "scheduled" ? "queued on the machine" : live?.detail ?? "running"}
      </span>
    </div>
  );
}

export function BaselineSection({
  ctx,
  run,
  n = "1",
  title = "Baseline — what you get with no tuning",
}: {
  ctx: FlowContext;
  run: FlowRun | null;
  n?: string;
  title?: string;
}) {
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const w = speedWorkload(ctx.targetCtx);
  const active = !!run && isActive(run.test);
  const view = speedRunView(run);
  const outcome = view?.rows[0]?.outcome ?? null;
  const eta = estimateSpeedRunSeconds({ candidates: 1, repeats: ctx.repeats, promptTokens: w.promptTokens, nGen: w.nGen });

  async function start() {
    setError(null);
    setStarting(true);
    try {
      await ctx.trigger("baseline", {
        model_id: ctx.model.id,
        worker_id: ctx.workerId,
        main_gpu: ctx.mainGpu,
        kind: "runtime",
        sweep: templateSweep(ctx.repeats),
        speed_run: {
          target_ctx: ctx.targetCtx,
          prompt_tokens: w.promptTokens,
          n_gen: w.nGen,
          repeats: ctx.repeats,
          candidates: [baselineCandidate()],
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  const spill = outcome ? spilledMib(outcome) : null;
  return (
    <Panel>
      <StepTitle n={n} right={<span className="font-mono text-xs text-muted">llama-server · -ngl 999 · defaults</span>}>
        {title}
      </StepTitle>
      <p className="mb-3 text-sm text-fg-2">
        llama-server at {ctxLabel(ctx.targetCtx)} context{" "}
        {ctx.machine === "dgpu" ? "with every layer on the GPU" : ctx.machine === "unified" ? "in the shared memory pool" : "on the CPU"} and
        llama.cpp&apos;s own defaults: one {w.promptTokens.toLocaleString("en-US")}-token
        prompt, then {w.nGen} generated tokens, {ctx.repeats === 1 ? "once" : `${ctx.repeats} times`}.
        {w.shrunk && ` The prompt is shortened from 4,096 so prompt + generation fit in ${ctxLabel(ctx.targetCtx)}.`}
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <Btn kind={run ? "secondary" : "primary"} onClick={start} disabled={active || starting || ctx.machineOffline}>
          {active ? "Running…" : run ? "Run baseline again" : "Run baseline"}
        </Btn>
        <span className="font-mono text-xs text-muted">{fmtDuration(eta)} incl. model load</span>
      </div>
      {error && <div className="mt-3"><Notice tone="fail">{error}</Notice></div>}
      {run && active && <div className="mt-3"><RunProgress run={run} /></div>}
      {outcome && outcome.status === "done" && (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <StatCard label="Generation · tg" value={readingText(outcome.tg)} sub={`tok/s · ${outcome.tg && outcome.tg.samples < 2 ? "single run, no spread" : `${outcome.tg?.samples ?? 0} runs`}`} />
          <StatCard label="Prompt · pp" value={readingText(outcome.pp)} sub={`tok/s · ${w.promptTokens.toLocaleString("en-US")}-token prompt`} />
          <StatCard label="VRAM peak" value={fmtGb(outcome.vramPeakMib)} sub={spill != null ? `${fmtGb(spill)} lived in system RAM` : "nothing spilled"} />
        </div>
      )}
      {outcome && spill != null && (
        <div className="mt-3">
          <Notice tone="warn">
            It ran, but {fmtGb(spill)} of its GPU memory lived in system RAM — the driver paged it out instead of failing. The context and offload
            step finds placements that keep it on the GPU.
          </Notice>
        </div>
      )}
      {outcome && isFailed(outcome.status) && (
        <div className="mt-3">
          <Notice tone="fail">
            {outcome.status === "failed_oom" ? "Out of memory with every layer on the GPU. " : ""}
            {outcome.error ?? "The baseline failed."}
          </Notice>
        </div>
      )}
      <p className="mt-3 text-xs text-muted">
        Prompt speed through llama-server with flash attention on can read lower than llama-bench on some GPUs; every speed test here uses the same
        harness, so the comparisons stay like for like.
      </p>
    </Panel>
  );
}

export function ConfirmSection({
  ctx,
  candidates,
  onRemove,
  run,
  baselineRun,
  step = "confirm",
  n = "4",
  title = "Confirm the winners",
  intro,
  reference = baselineCandidate(),
  refName = "the default settings",
  emptyText = "Add configs from the fit map (or the suggestions) to measure them here.",
  disabledReason,
  resultTitle = "Recommended config",
}: {
  ctx: FlowContext;
  candidates: SpeedCandidate[];
  onRemove?: (key: string) => void;
  run: FlowRun | null;
  baselineRun: FlowRun | null;
  /** The flow step the run is stored under. */
  step?: FlowStep;
  n?: string;
  title?: string;
  intro?: ReactNode;
  /** Measured alongside every candidate as the comparison row (source "baseline"). */
  reference?: SpeedCandidate;
  refName?: string;
  emptyText?: string;
  /** Set when the run can't start for a reason the section doesn't know. */
  disabledReason?: string | null;
  resultTitle?: string;
}) {
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const w = speedWorkload(ctx.targetCtx);
  const active = !!run && isActive(run.test);
  const view = speedRunView(run);
  const baseView = speedRunView(baselineRun);
  const list = candidates.slice(0, MAX_SPEED_CANDIDATES - 1);
  const eta = estimateSpeedRunSeconds({ candidates: list.length + 1, repeats: ctx.repeats, promptTokens: w.promptTokens, nGen: w.nGen });

  async function start() {
    setError(null);
    setStarting(true);
    try {
      await ctx.trigger(step, {
        model_id: ctx.model.id,
        worker_id: ctx.workerId,
        main_gpu: ctx.mainGpu,
        kind: "runtime",
        sweep: templateSweep(ctx.repeats),
        speed_run: {
          target_ctx: ctx.targetCtx,
          prompt_tokens: w.promptTokens,
          n_gen: w.nGen,
          repeats: ctx.repeats,
          // The reference runs alongside, so the comparison is from the same sitting.
          candidates: [{ ...reference, source: "baseline" }, ...list],
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  const best = view ? recommend(view, refName) : null;
  const carried = best ? null : carryForward(view, refName);
  return (
    <Panel>
      <StepTitle n={n} right={<span className="font-mono text-xs text-muted">same harness as the baseline</span>}>
        {title}
      </StepTitle>
      <p className="mb-3 text-sm text-fg-2">
        {intro ?? (
          <>
            Fit maximises layers, not speed — on some cards generation peaks below full offload. Each config below gets a real load at{" "}
            {ctxLabel(ctx.targetCtx)}, the same {w.promptTokens.toLocaleString("en-US")}-token prompt and {w.nGen} generated tokens, next to{" "}
            {refName}.
          </>
        )}
      </p>
      {list.length === 0 ? (
        <Notice>{emptyText}</Notice>
      ) : (
        <ul className="mb-3 flex flex-col border-t border-border">
          {list.map((c) => (
            <li key={candidateKey(c)} className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border px-3 py-2 last:border-b-0">
              <span className="min-w-0 flex-[1_1_220px]">
                <span className="block text-sm font-medium">{c.label}</span>
                <span className="block font-mono text-xs text-muted [overflow-wrap:anywhere]">
                  -ngl {c.ngl} · {c.ctk}/{c.ctv} · FA {c.fa}
                  {c.ot ? " · -ot (experts on CPU)" : ""}
                  {c.b != null || c.ub != null ? ` · -b ${c.b ?? 2048} -ub ${c.ub ?? 512}` : ""}
                  {c.threads ? ` · -t ${c.threads.t} -tb ${c.threads.tb}` : ""}
                </span>
              </span>
              {onRemove && (
                <Btn kind="ghost" onClick={() => onRemove(candidateKey(c))} disabled={active} aria-label={`Remove ${c.label}`}>
                  Remove
                </Btn>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Btn
          kind="primary"
          onClick={start}
          disabled={active || starting || list.length === 0 || ctx.machineOffline || !!disabledReason}
          aria-describedby={disabledReason || ctx.machineOffline ? `${step}-why` : undefined}
        >
          {active ? "Measuring…" : `Measure ${list.length + 1} config${list.length ? "s" : ""}`}
        </Btn>
        <span className="font-mono text-xs text-muted">
          {fmtDuration(eta)} · {ctx.repeats} repeat{ctx.repeats === 1 ? "" : "s"} each
        </span>
      </div>
      {(disabledReason || ctx.machineOffline) && !active && (
        <p id={`${step}-why`} className="mt-2 text-sm text-muted">
          {disabledReason ?? `${ctx.worker.displayName} is offline — start its worker to measure.`}
        </p>
      )}
      {error && <div className="mt-3"><Notice tone="fail">{error}</Notice></div>}
      {run && active && <div className="mt-3"><RunProgress run={run} /></div>}
      {view && (
        <div className="mt-4 flex flex-col gap-4">
          <SpeedTable view={view} baseline={view.rows.some((r) => r.candidate.source === "baseline") ? view : baseView} refName={refName} />
          <SpeedBars view={view} chosen={best?.row.candidate ?? carried?.candidate ?? null} />
          {best && (
            <div className="relative bg-surface p-5 sm:px-6">
              <Kicker className="text-accent">{resultTitle}</Kicker>
              <div className="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <Kicker>Generation · tg</Kicker>
                  <div className="font-display text-[40px] font-semibold leading-none">{fmtTps(best.row.outcome.tg?.mean)}</div>
                  <div className="text-xs text-fg-2">tok/s · {best.tgDelta}</div>
                </div>
                <div>
                  <Kicker>Prompt · pp</Kicker>
                  <div className="font-display text-[40px] font-semibold leading-none">{fmtTps(best.row.outcome.pp?.mean)}</div>
                  <div className="text-xs text-fg-2">tok/s · {w.promptTokens.toLocaleString("en-US")}-token prompt</div>
                </div>
                <div>
                  <Kicker>Context</Kicker>
                  <div className="font-display text-[40px] font-semibold leading-none">{ctxLabel(ctx.targetCtx)}</div>
                  <div className="text-xs text-fg-2">{spilledMib(best.row.outcome) != null ? "spilled — see the table" : "measured, nothing spilled"}</div>
                </div>
              </div>
              <div className="mt-4">
                <CopyCommand command={serverCommandLine(ctx.modelFile, ctx.targetCtx, best.row.candidate)} />
              </div>
              <p className="mt-3 text-sm text-fg-2">
                <strong className="font-semibold text-fg">Why this one:</strong> {best.why}
              </p>
            </div>
          )}
        </div>
      )}
      {view && !active && !best && view.rows.every((r) => r.outcome.status !== "pending") && (
        <div className="mt-3">
          <Notice tone="warn">
            No config finished cleanly — see the reasons in the table.
            {carried ? ` The next step builds on the fastest one that finished, ${carried.candidate.label}${carried.spilled ? ", which spilled" : ""}.` : ""}
          </Notice>
        </div>
      )}
    </Panel>
  );
}

/** The config the next step builds on: the recommendation, else (when every
 * config spilled) the fastest one that finished, flagged as spilled. */
export function carryForward(view: SpeedRunView | null, refName?: string): { candidate: SpeedCandidate; spilled: boolean } | null {
  if (!view) return null;
  const rec = recommend(view, refName);
  if (rec) return { candidate: rec.row.candidate, spilled: false };
  const done = view.rows.filter((r) => r.outcome.status === "done" && r.outcome.tg);
  if (done.length === 0) return null;
  const fastest = done.reduce((a, b) => (b.outcome.tg!.mean > a.outcome.tg!.mean ? b : a));
  return { candidate: fastest.candidate, spilled: spilledMib(fastest.outcome) != null };
}

/** The recommended config: the fastest generation among clean runs (no
 * failure, nothing spilled), ties to the better prompt speed. It only beats
 * the default when the gap is larger than both spreads. */
export function recommend(
  view: SpeedRunView,
  refName = "the default settings"
): { row: SpeedRunView["rows"][number]; why: string; tgDelta: string } | null {
  const clean = view.rows.filter((r) => r.outcome.status === "done" && r.outcome.tg && spilledMib(r.outcome) == null);
  if (clean.length === 0) return null;
  const base = view.rows.find((r) => r.candidate.source === "baseline")?.outcome ?? null;
  const sorted = [...clean].sort((a, b) => (b.outcome.tg!.mean - a.outcome.tg!.mean) || ((b.outcome.pp?.mean ?? 0) - (a.outcome.pp?.mean ?? 0)));
  let pick = sorted[0];
  const cmp = base ? compareToBaseline(pick.outcome.tg, base.tg) : null;
  // A "winner" inside the noise doesn't replace the default when the default ran clean.
  const baseRow = clean.find((r) => r.candidate.source === "baseline");
  if (baseRow && pick !== baseRow && cmp && !cmp.clear) pick = baseRow;
  const pc = base ? compareToBaseline(pick.outcome.tg, base.tg) : null;
  const signed = (pct: number) => {
    const r = Math.round(pct);
    return r === 0 ? "±0%" : `${r > 0 ? "+" : ""}${r}%`;
  };
  const tgDelta =
    pick.candidate.source === "baseline"
      ? refName
      : pc?.pct != null
        ? `${signed(pc.pct)} over ${refName === "the default settings" ? "defaults" : refName}${pc.noSpread ? " (single run)" : ""}`
        : "measured";
  const faster = sorted[0];
  const why =
    pick.candidate.source === "baseline"
      ? cmp && !cmp.clear && faster !== baseRow
        ? cmp.noSpread
          ? `${faster.candidate.label} read faster, but with one repeat there is no spread to tell it from noise, so ${refName} stays. Raise the repeats to decide.`
          : `${faster.candidate.label} measured faster, but within the run-to-run spread, so ${refName} stays.`
        : `Nothing measured beat ${refName} cleanly.`
      : `Fastest generation of the configs that ran without spilling${pick.candidate.ot ? ", with the experts placed the way fit sized them" : ""}${pick.candidate.ctk !== "f16" || pick.candidate.ctv !== "f16" ? "; the quantized KV cache can slightly reduce output quality" : ""}.`;
  return { row: pick, why, tgDelta };
}
