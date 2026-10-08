import { useMemo, useState } from "react";
import { Btn, Kicker, Notice, Panel, StepTitle, fmtDuration, fmtTps } from "./ui";
import type { FlowContext } from "./flowContext";
import { templateSweep, threadRunView } from "./flowData";
import { RunProgress } from "./SpeedSections";
import { isActive, type FlowRun } from "./useOptimizeFlow";
import {
  defaultThreadPlan,
  pickThreadWinner,
  threadCandidates,
  type ThreadCandidate,
  type ThreadCandidateKind,
} from "../../../../shared/threadPlan.js";
import { THREAD_PP_TOKENS, THREAD_TG_TOKENS, type SpeedCandidate, type ThreadArgs, type ThreadSweepItem } from "../../../../shared/speedRun.js";

export function threadArgsFromPlan(ctx: Pick<FlowContext, "topology">): ThreadArgs {
  const p = defaultThreadPlan(ctx.topology);
  return { t: p.t.threads, tb: p.tb.threads, mask: p.t.mask, maskBatch: p.tb.mask, strict: p.strict };
}

export function ThreadsSection({
  ctx,
  value,
  onChange,
  placement,
  run,
}: {
  ctx: FlowContext;
  value: ThreadArgs;
  onChange: (t: ThreadArgs) => void;
  /** The config the sweep measures (the first speed-test candidate). */
  placement: SpeedCandidate;
  run: FlowRun | null;
}) {
  const plan = useMemo(() => defaultThreadPlan(ctx.topology), [ctx.topology]);
  const tCands = useMemo(() => threadCandidates(ctx.topology, "t"), [ctx.topology]);
  const tbCands = useMemo(() => threadCandidates(ctx.topology, "tb"), [ctx.topology]);
  const [picked, setPicked] = useState<Set<string>>(
    () => new Set([...tCands.filter((c) => c.preselected).map((c) => `t:${c.kind}`), ...tbCands.filter((c) => c.preselected).map((c) => `tb:${c.kind}`)])
  );
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const active = !!run && isActive(run.test);
  const view = threadRunView(run);
  const cpuWork = ctx.machine !== "dgpu" || !!placement.ot || placement.ngl < (ctx.layersTotal ?? 999);
  const physical = ctx.topology.cores.length;
  const custom = value.t !== plan.t.threads || value.tb !== plan.tb.threads;

  const items: ThreadSweepItem[] = [
    ...tCands.filter((c) => picked.has(`t:${c.kind}`) || c.kind === "default").map((c) => toItem("t", c, plan.strict)),
    ...tbCands.filter((c) => picked.has(`tb:${c.kind}`) || c.kind === "default").map((c) => toItem("tb", c, plan.strict)),
  ];
  const eta = items.length * (30 + ctx.repeats * 12);

  async function start() {
    setError(null);
    setStarting(true);
    try {
      await ctx.trigger("threads", {
        model_id: ctx.model.id,
        worker_id: ctx.workerId,
        main_gpu: ctx.mainGpu,
        kind: "runtime",
        sweep: templateSweep(ctx.repeats),
        thread_sweep: {
          placement: { ngl: placement.ngl, ot: placement.ot, ctk: placement.ctk, ctv: placement.ctv, fa: placement.fa === "off" ? "off" : "on" },
          repeats: ctx.repeats,
          items,
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  const tRows = view?.rows.filter((r) => r.item.role === "t") ?? [];
  const tbRows = view?.rows.filter((r) => r.item.role === "tb") ?? [];
  const verdict = (rows: typeof tRows) =>
    pickThreadWinner(rows.filter((r) => r.reading).map((r) => ({ kind: r.item.kind, mean: r.reading!.mean, stddev: r.reading!.stddev, samples: r.reading!.samples })));
  const tWin = view ? verdict(tRows) : null;
  const tbWin = view ? verdict(tbRows) : null;

  function applyWinners() {
    const t = tWin?.kind === "winner" ? tRows.find((r) => r.item.kind === tWin.winner)?.item : null;
    const tb = tbWin?.kind === "winner" ? tbRows.find((r) => r.item.kind === tbWin.winner)?.item : null;
    onChange({
      t: t?.threads ?? plan.t.threads,
      tb: tb?.threads ?? plan.tb.threads,
      mask: t ? t.mask : plan.t.mask,
      maskBatch: tb ? tb.mask : plan.tb.mask,
      strict: plan.strict,
    });
  }

  return (
    <Panel>
      <StepTitle n="3" right={<span className="font-mono text-xs text-muted">-t · -tb · affinity</span>}>
        CPU threads
      </StepTitle>
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
        <div className="min-w-0">
          <p className="font-mono text-xs text-fg-2">
            {ctx.worker.hardware?.cpu.brand || "CPU"} · {plan.summary}
          </p>
          <p className="mt-1 text-sm">
            <span className="font-mono">-t {value.t} -tb {value.tb}</span>
            {value.mask ? <span className="font-mono text-muted"> · -C {value.mask}{value.maskBatch && value.maskBatch !== value.mask ? ` -Cb ${value.maskBatch}` : ""}{value.strict ? " --cpu-strict 1" : ""}</span> : null}
            {custom && <span className="ml-2 text-xs text-accent">custom</span>}
          </p>
          <p className="mt-1 text-sm text-fg-2">{plan.reason}</p>
          {plan.note && <p className="mt-1 text-xs text-muted">{plan.note}</p>}
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1 text-xs text-muted">
            -t
            <input
              type="number"
              min={1}
              max={Math.max(1, ctx.topology.logicalCount)}
              value={value.t}
              onChange={(e) => onChange({ ...value, t: clampInt(e.target.value, 1, physical), mask: null })}
              className="h-10 w-20 border border-border-strong bg-well px-2 font-mono text-sm text-fg"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            -tb
            <input
              type="number"
              min={1}
              max={Math.max(1, ctx.topology.logicalCount)}
              value={value.tb}
              onChange={(e) => onChange({ ...value, tb: clampInt(e.target.value, 1, physical), maskBatch: null })}
              className="h-10 w-20 border border-border-strong bg-well px-2 font-mono text-sm text-fg"
            />
          </label>
          {custom && (
            <Btn kind="ghost" onClick={() => onChange({ t: plan.t.threads, tb: plan.tb.threads, mask: plan.t.mask, maskBatch: plan.tb.mask, strict: plan.strict })}>
              Defaults
            </Btn>
          )}
        </div>
      </div>

      <div className="mt-4 border-t border-border pt-3">
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="text-sm font-medium text-accent hover:text-accent-hover">
          {open ? "Hide" : "Optional:"} measure other thread counts
        </button>
        {open && (
          <div className="mt-3 flex flex-col gap-3">
            {!cpuWork && (
              <Notice>Every layer of this config is on the GPU, so threads have little effect here — the sweep is still available.</Notice>
            )}
            <div className="grid gap-4 sm:grid-cols-2">
              <CandidateList title={`-t → generation (${THREAD_TG_TOKENS} tokens)`} role="t" cands={tCands} picked={picked} setPicked={setPicked} disabled={active} />
              <CandidateList
                title={`-tb → prompt (${THREAD_PP_TOKENS} tokens)`}
                role="tb"
                cands={tbCands}
                picked={picked}
                setPicked={setPicked}
                disabled={active}
                note={ctx.machine === "dgpu" ? "On a GPU box prompt work runs on the GPU, so -tb rarely matters." : undefined}
              />
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Btn onClick={start} disabled={active || starting || ctx.machineOffline}>
                {active ? "Measuring…" : `Measure ${items.length} setting${items.length === 1 ? "" : "s"}`}
              </Btn>
              <span className="font-mono text-xs text-muted">
                llama-bench · {ctx.repeats} repeat{ctx.repeats === 1 ? "" : "s"} · {fmtDuration(eta)}
              </span>
            </div>
            {error && <Notice tone="fail">{error}</Notice>}
            {run && active && <RunProgress run={run} />}
            {view && (
              <div className="grid gap-4 sm:grid-cols-2">
                <ThreadBars title="tg tok/s by -t" rows={tRows} verdict={tWin} />
                <ThreadBars title="pp tok/s by -tb" rows={tbRows} verdict={tbWin} />
              </div>
            )}
            {view && !active && (tWin?.kind === "winner" || tbWin?.kind === "winner") && (
              <div>
                <Btn kind="primary" onClick={applyWinners}>
                  Use the measured winner{tWin?.kind === "winner" && tbWin?.kind === "winner" ? "s" : ""}
                </Btn>
              </div>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}

function clampInt(raw: string, lo: number, hi: number): number {
  const n = Math.round(Number(raw));
  return Number.isFinite(n) ? Math.max(lo, Math.min(Math.max(lo, hi), n)) : lo;
}

function toItem(role: "t" | "tb", c: ThreadCandidate, strict: boolean): ThreadSweepItem {
  return { role, kind: c.kind, label: c.label, threads: c.threads, mask: c.mask, strict: strict && c.mask != null };
}

function CandidateList({
  title,
  role,
  cands,
  picked,
  setPicked,
  disabled,
  note,
}: {
  title: string;
  role: "t" | "tb";
  cands: ThreadCandidate[];
  picked: Set<string>;
  setPicked: (s: Set<string>) => void;
  disabled: boolean;
  note?: string;
}) {
  return (
    <fieldset className="m-0 border border-border p-3">
      <legend className="px-1 font-mono text-xs text-muted">{title}</legend>
      <ul className="flex flex-col gap-1.5">
        {cands.map((c) => {
          const key = `${role}:${c.kind}`;
          const isDefault = c.kind === "default";
          return (
            <li key={key}>
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-[var(--color-accent)]"
                  checked={isDefault || picked.has(key)}
                  disabled={disabled || isDefault}
                  onChange={(e) => {
                    const next = new Set(picked);
                    if (e.target.checked) next.add(key);
                    else next.delete(key);
                    setPicked(next);
                  }}
                />
                <span>{c.label}</span>
                <span className="font-mono text-xs text-muted">
                  {c.threads} thread{c.threads === 1 ? "" : "s"}
                  {isDefault ? " · default, always measured" : ""}
                </span>
              </label>
            </li>
          );
        })}
      </ul>
      {note && <p className="mt-2 text-xs text-muted">{note}</p>}
    </fieldset>
  );
}

export function ThreadBars({
  title,
  rows,
  verdict,
}: {
  title: string;
  rows: { item: ThreadSweepItem; reading: { mean: number; stddev: number; samples: number } | null; outcome: { status: string; error: string | null } }[];
  verdict: ReturnType<typeof pickThreadWinner> | null;
}) {
  if (rows.length === 0) return null;
  const max = Math.max(1, ...rows.map((r) => r.reading?.mean ?? 0));
  const winner: ThreadCandidateKind | null = verdict?.kind === "winner" ? verdict.winner : null;
  return (
    <figure className="m-0 border border-border bg-surface-raised p-3">
      <Kicker>{title}</Kicker>
      <ul className="mt-2 flex flex-col gap-2">
        {rows.map((r) => (
          <li key={`${r.item.role}-${r.item.kind}`} className="grid grid-cols-[96px_1fr_auto] items-center gap-2 font-mono text-xs">
            <span className={r.item.kind === "default" ? "text-fg" : "text-fg-2"}>
              {r.item.threads} · {r.item.label}
            </span>
            <span className="block h-3 border border-border bg-well">
              <span
                className={`block h-full ${r.item.kind === winner ? "bg-accent" : r.item.kind === "default" ? "bg-fg" : "bg-muted"}`}
                style={{ width: `${((r.reading?.mean ?? 0) / max) * 100}%` }}
              />
            </span>
            <span className="text-right text-fg">
              {r.reading ? `${fmtTps(r.reading.mean)}${r.reading.samples > 1 ? ` ±${fmtTps(r.reading.stddev)}` : ""}` : r.outcome.status === "done" ? "—" : r.outcome.status}
            </span>
          </li>
        ))}
      </ul>
      <figcaption className="mt-2 text-xs text-fg-2">
        {verdict == null
          ? ""
          : verdict.kind === "winner"
            ? `${rows.find((r) => r.item.kind === verdict.winner)?.item.label} beats the default by more than both spreads combined.`
            : verdict.reason === "no_spread"
              ? "A single repeat has no spread, so the default is kept."
              : "No clear winner — default kept."}
      </figcaption>
    </figure>
  );
}
