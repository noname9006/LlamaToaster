import { useEffect, useMemo, useState } from "react";
import { FitLegend, FitPathsChart } from "./FitPathsChart";
import { Btn, CopyCommand, Kicker, Notice, Panel, ProgressBar, Seg, StatCard, StepTitle, fmtDuration, fmtGb } from "./ui";
import type { FlowContext } from "./flowContext";
import { templateSweep } from "./flowData";
import { isActive, type FitData, type FlowRun } from "./useOptimizeFlow";
import { DEFAULT_FIT_MARGIN_MIB, FIT_MARGINS_MIB, type FitMarginMib } from "../../../../shared/fitParams.js";
import { isUsableKv, kvSizeGroups, KV_QUALITY_WARNING, pairLabel, type KvSupportRow } from "../../../../shared/kvSupport.js";
import {
  buildFitSeries,
  candidateFromPoint,
  ctxLabel,
  defaultFitMapSpec,
  defaultFitSelection,
  describeFitPlacement,
  estimateFitMapSeconds,
  fitInsight,
  fits,
  pointY,
  usefulSeries,
  type FitSelection,
} from "../../../../shared/optimizeFlow.js";
import { candidateKey, serverCommandLine, type SpeedCandidate } from "../../../../shared/speedRun.js";

const MARGIN_HINT: Record<FitMarginMib, string> = {
  512: "Headless Linux with nothing else on the GPU: usually fine.",
  1024: "The default. Leaves room for the desktop and small apps.",
  1536: "Safer on Windows with a display on the same GPU — the driver's free figure ignores other apps.",
};

export function FitMapSection({
  ctx,
  fitRun,
  fit,
  onTrigger,
  onAddCandidate,
  candidateKeys,
}: {
  ctx: FlowContext;
  fitRun: FlowRun | null;
  fit: FitData | null;
  onTrigger: (payload: Parameters<FlowContext["trigger"]>[1]) => Promise<void>;
  onAddCandidate: (c: SpeedCandidate) => void;
  candidateKeys: Set<string>;
}) {
  const [margin, setMargin] = useState<FitMarginMib>(DEFAULT_FIT_MARGIN_MIB);
  const [withKv, setWithKv] = useState(true);
  const [selection, setSelection] = useState<FitSelection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const running = !!fitRun && isActive(fitRun.test);

  const series = useMemo(() => buildFitSeries(fit?.points ?? [], fit?.kv ?? []), [fit]);
  const [showAll, setShowAll] = useState(false);
  const useful = useMemo(() => usefulSeries(series, ctx.moe), [series, ctx.moe]);
  const target = ctx.targetCtx;
  // The default pick follows the map as it streams in and when the target
  // changes; once the user picks a point it stays put (while it exists).
  const [userPicked, setUserPicked] = useState(false);
  useEffect(() => setUserPicked(false), [target, fit?.testId]);
  useEffect(() => {
    setSelection((cur) => {
      if (userPicked && cur && series.some((s) => s.id === cur.seriesId && s.points.some((p) => p.ctx === cur.ctx))) return cur;
      return defaultFitSelection(series, target, ctx.moe);
    });
  }, [series, target, ctx.moe, userPicked]);
  const pick = (sel: FitSelection) => {
    setUserPicked(true);
    setSelection(sel);
  };

  const selSeries = series.find((s) => s.id === selection?.seriesId) ?? null;
  const visible = showAll ? series : series.filter((s) => useful.has(s.id) || s.id === selection?.seriesId);
  const selPoint = selSeries?.points.find((p) => p.ctx === selection?.ctx) ?? null;
  const stops = defaultFitMapSpec({ trainedCtx: ctx.trainedCtx, machine: ctx.machine, margin, targetCtx: target }).ctx_stops;
  const eta = estimateFitMapSeconds({ stops: stops.length, machine: ctx.machine, kvDetect: withKv, kvFit: withKv, kvKnown: (fit?.kv.length ?? 0) >= 128 });
  const usedMargin = fit?.spec?.margin_mib ?? fit?.points[0]?.margin_mib ?? margin;

  async function run() {
    setError(null);
    setStarting(true);
    try {
      await onTrigger({
        model_id: ctx.model.id,
        worker_id: ctx.workerId,
        main_gpu: ctx.mainGpu,
        kind: "fit",
        sweep: templateSweep(1),
        fit_map: { ...defaultFitMapSpec({ trainedCtx: ctx.trainedCtx, machine: ctx.machine, margin, targetCtx: target }), kv_detect: withKv, kv_fit: withKv },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  const progressItem = fitRun?.items[0];
  const failed = fitRun && !running && (fitRun.test.status === "failed" || fitRun.test.status === "cancelled");
  const selCandidate = selPoint && fits(selPoint) && selSeries
    ? candidateFromPoint(selPoint, `${selSeries.label} · ${ctxLabel(selPoint.ctx)}`, selSeries.kind === "kv" ? "fit_kv" : "fit", ctx.threads)
    : null;
  const added = selCandidate ? candidateKeys.has(candidateKey(selCandidate)) : false;
  const unified = ctx.machine !== "dgpu";
  const moeY = ctx.moe?.isMoe && selPoint && !unified ? pointY(selPoint, ctx.moe) : null;
  const vramShare = selPoint && selPoint.total_mib ? Math.round(((selPoint.dev_used_mib ?? 0) / selPoint.total_mib) * 100) : 0;

  return (
    <Panel>
      <StepTitle
        n="2"
        right={
          <span className="font-mono text-xs text-muted">
            {ctx.machine === "dgpu" ? "context × layers · FA on / off" : "max context · FA on / off"}
          </span>
        }
      >
        Find the best fit
      </StepTitle>
      <p className="mb-3 text-sm text-fg-2">
        llama-fit-params sizes every context from 4k up to {ctxLabel(stops[stops.length - 1])} against this machine&apos;s memory without loading
        the model — {ctx.machine === "dgpu" ? "how many layers fit on the GPU" : ctx.machine === "unified" ? "whether it fits in the shared memory pool" : "whether it fits in system RAM"} with flash attention on and
        off{withKv ? `, then again for every K/V cache type this build ${ctx.machine === "cpu" ? "supports" : "runs on the GPU"}` : ""}.
      </p>

      <div className="mb-3 grid gap-3 sm:grid-cols-[auto_1fr] sm:items-center">
        <Kicker>Safety margin</Kicker>
        <div className="flex flex-col gap-1">
          <Seg
            label="Safety margin"
            value={margin}
            onChange={setMargin}
            disabled={running}
            options={FIT_MARGINS_MIB.map((m) => ({ value: m, label: `${m} MiB${m === DEFAULT_FIT_MARGIN_MIB ? " · default" : ""}` }))}
          />
          <span className="text-xs text-muted">{MARGIN_HINT[margin]}</span>
        </div>
        <Kicker>KV caches</Kicker>
        <label className="flex cursor-pointer items-start gap-2.5 text-sm">
          <input type="checkbox" checked={withKv} disabled={running} onChange={(e) => setWithKv(e.target.checked)} className="mt-0.5 h-[18px] w-[18px] accent-[var(--color-accent)]" />
          <span>
            Detect quantized K/V caches and map each one
            <span className="block text-xs text-muted">{KV_QUALITY_WARNING}</span>
          </span>
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Btn kind="primary" onClick={run} disabled={running || starting || !ctx.canFit || ctx.machineOffline}>
          {running ? "Mapping…" : fit?.points.length ? "Map again" : "Map context × layers"}
        </Btn>
        <span className="font-mono text-xs text-muted">
          {stops.length} contexts · {fmtDuration(eta)} · no model load
        </span>
      </div>
      {!ctx.canFit && <div className="mt-3"><Notice tone="warn">{ctx.canFitReason}</Notice></div>}
      {error && <div className="mt-3"><Notice tone="fail">{error}</Notice></div>}
      {running && (
        <div className="mt-3 flex flex-col gap-1.5" aria-live="polite">
          <ProgressBar pct={progressPct(progressItem?.detail)} live label="Fit map progress" />
          <span className="font-mono text-xs text-fg-2">{progressItem?.detail ?? (fitRun?.test.status === "scheduled" ? "queued on the machine" : "starting")}</span>
        </div>
      )}
      {failed && (
        <div className="mt-3">
          <Notice tone="fail">{fitRun?.items[0]?.error ?? fitRun?.test.error ?? "The fit map did not finish."}</Notice>
        </div>
      )}

      {series.length > 0 && (
        <div className="mt-5 flex flex-col gap-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <FitLegend series={visible} selectedId={selection?.seriesId ?? null} onPick={(id) => {
              const s = series.find((x) => x.id === id);
              const c = selection?.ctx ?? target;
              const p = s?.points.find((x) => x.ctx === c) ?? s?.points.filter(fits).slice(-1)[0];
              if (s && p) pick({ seriesId: s.id, ctx: p.ctx });
            }} />
            <span className="font-mono text-xs text-muted">
              {ctx.modelLabel} · {ctx.layersTotal ?? "?"} layers
              {ctx.machine === "cpu"
                ? ` · CPU only${ctx.ramTotalMib ? ` · ${fmtGb(ctx.ramTotalMib, 0)} RAM` : ""}`
                : `${ctx.gpuName ? ` · ${ctx.gpuName}` : ""}${ctx.freeVramMib != null ? ` · ${fmtGb(ctx.freeVramMib)} free` : ""}`}
              {" "}· margin {usedMargin} MiB
            </span>
          </div>
          {series.length > visible.length || showAll ? (
            <button type="button" onClick={() => setShowAll((v) => !v)} className="self-start text-sm text-accent hover:text-accent-hover">
              {showAll ? "Show only the cache sizes that gain something" : `Show all ${series.length - 2} cache sizes (${series.length - visible.length} more)`}
            </button>
          ) : null}
          <FitPathsChart
            series={visible}
            selection={selection}
            onSelect={pick}
            moe={ctx.moe}
            machine={ctx.machine}
            targetCtx={stopForTarget(series, target)}
            layersTotal={ctx.layersTotal}
          />
          {selPoint && selSeries && (
            <>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
                <StatCard label="Setting" value={`${selSeries.label} · ${ctxLabel(selPoint.ctx)}`} sub={selSeries.subtitle} />
                <StatCard
                  label={unified ? "Fits" : moeY ? (moeY.band === "main" ? "Experts on GPU" : "Dense-only layers") : "Layers on GPU"}
                  value={
                    !fits(selPoint)
                      ? "doesn't fit"
                      : unified
                        ? "yes"
                        : moeY && ctx.moe
                          ? moeY.band === "main"
                            ? `${moeY.value} / ${ctx.moe.moeLayers.length} layers`
                            : `${moeY.value} / ${selPoint.layers_total ?? "?"}`
                          : `${selPoint.layers_gpu} / ${selPoint.layers_total ?? "?"}`
                  }
                  sub={ctx.moe?.isMoe && fits(selPoint) ? describeFitPlacement(selPoint, ctx.moe) : undefined}
                />
                <StatCard label="Total memory" value={selPoint.total_mib != null ? `≈ ${fmtGb(selPoint.total_mib)}` : "—"} sub={selPoint.kv_mib != null ? `KV cache ${fmtGb(selPoint.kv_mib)}` : undefined} wide />
              </div>
              {!unified && selPoint.total_mib != null && selPoint.total_mib > 0 && (
                <div className="flex flex-col gap-1.5">
                  <div className="flex h-[22px] border border-border" aria-hidden="true">
                    <div className="bg-accent" style={{ width: `${vramShare}%` }} />
                    <div className="bg-border-strong" style={{ width: `${100 - vramShare}%` }} />
                  </div>
                  <div className="flex flex-wrap justify-between gap-2 text-[13px]">
                    <span>
                      <strong>{fmtGb(selPoint.dev_used_mib)} VRAM</strong> · {vramShare}%
                    </span>
                    <span>
                      ≈ {fmtGb(selPoint.host_used_mib)} system RAM · {100 - vramShare}%
                    </span>
                  </div>
                </div>
              )}
              <p className="text-[13px] leading-relaxed text-fg-2">{fitInsight({ series, selection: selection!, moe: ctx.moe, machine: ctx.machine })}</p>
              {selCandidate && (
                <div className="flex flex-col gap-2.5">
                  <CopyCommand command={serverCommandLine(ctx.modelFile, selPoint.ctx, selCandidate)} />
                  <div className="flex flex-wrap items-center gap-2.5">
                    <Btn kind="primary" disabled={added} onClick={() => onAddCandidate(selCandidate)}>
                      {added ? "In the speed tests" : "Add to speed tests"}
                    </Btn>
                    {selPoint.ctx !== target && (
                      <span className="text-xs text-muted">Speed tests run at your target context ({ctxLabel(target)}) with this placement.</span>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
          <p className="text-xs text-muted">
            Predicted by llama-fit-params ({ctx.build ?? "this build"}), no model loaded. Total = VRAM + system RAM on a discrete GPU; one shared
            pool on unified memory. {ctx.machine === "dgpu" ? "A real load can still spill when other apps hold VRAM — the confirm step measures it." : ""}
          </p>
        </div>
      )}

      {fit && fit.kv.length > 0 && <KvSupportList rows={fit.kv} onGpu={ctx.machine !== "cpu"} />}
    </Panel>
  );
}

function stopForTarget(series: ReturnType<typeof buildFitSeries>, target: number): number | null {
  const stops = [...new Set(series.flatMap((s) => s.points.map((p) => p.ctx)))].sort((a, b) => a - b);
  return stops.find((s) => s >= target) ?? stops[stops.length - 1] ?? null;
}

function progressPct(detail: string | undefined): number {
  const m = detail ? /(\d+) of ~(\d+) calls/.exec(detail) : null;
  return m ? (Number(m[1]) / Math.max(1, Number(m[2]))) * 100 : 5;
}

function KvSupportList({ rows, onGpu }: { rows: KvSupportRow[]; onGpu: boolean }) {
  const [open, setOpen] = useState(false);
  const onRows = rows.filter((r) => r.fa === "on");
  const usable = rows.filter(isUsableKv);
  const groups = kvSizeGroups(rows);
  const fallbacks = rows.filter((r) => !isUsableKv(r));
  const slow = rows.filter((r) => r.status === "cuda_slow");
  return (
    <div className="mt-5 border-t border-border pt-4">
      <Kicker className="mb-2">K/V cache support on this build</Kicker>
      <p className="text-sm text-fg-2">
        {usable.filter((r) => r.fa === "on").length} of {onRows.length} pairs {onGpu ? "run on the GPU" : "work"} with flash attention on;{" "}
        {usable.filter((r) => r.fa === "off").length} with it off. They size into {groups.length} distinct cache sizes, one path each above.
      </p>
      <div className="mt-2">
        <Notice tone="warn">{KV_QUALITY_WARNING}</Notice>
      </div>
      {slow.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          {slow.length} pair(s) run on the GPU but may be slow on CUDA: llama.cpp converts them to f16 each decode step unless built with GGML_CUDA_FA_QUANTS.
        </p>
      )}
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="mt-2 text-sm text-accent hover:text-accent-hover">
        {open ? "Hide" : "Show"} the {fallbacks.length} combinations that fall back or fail
      </button>
      {open && (
        <ul className="mt-2 grid gap-x-6 gap-y-1 font-mono text-xs text-fg-2 sm:grid-cols-2">
          {fallbacks.map((r) => (
            <li key={`${r.ctk}-${r.ctv}-${r.fa}`}>
              <span className="text-fg">{pairLabel(r.ctk, r.ctv)}</span> · FA {r.fa} ·{" "}
              <span className={r.status === "invalid" ? "text-danger" : "text-muted"}>{r.detail ?? r.status}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
