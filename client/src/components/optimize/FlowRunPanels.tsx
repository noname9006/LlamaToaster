import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useTestView } from "../../api/testView";
import type { Model, ResultRow, Test, TestItem } from "../../types";
import { FitLegend, FitPathsChart } from "./FitPathsChart";
import { SpeedTable } from "./SpeedSections";
import { ThreadBars } from "./ThreadsSection";
import { speedRunView, threadRunView } from "./flowData";
import { Kicker, StatCard, fmtGb } from "./ui";
import type { FitPoint } from "../../../../shared/fitParams.js";
import type { KvSupportRow } from "../../../../shared/kvSupport.js";
import { detectMoe } from "../../../../shared/moePlacement.js";
import {
  buildFitSeries,
  ctxLabel,
  defaultFitSelection,
  describeFitPlacement,
  fitInsight,
  fits,
  pointY,
  usefulSeries,
  type FitMapSpec,
  type FitSelection,
} from "../../../../shared/optimizeFlow.js";
import { pickThreadWinner } from "../../../../shared/threadPlan.js";

// What a single optimization-flow run looks like on its own Test page. The
// New test page shows the same pieces in context; this is the deep link.

export function isFlowRun(run: Test | null): boolean {
  const c = run?.config as { flow_step?: string; fit_map?: unknown; speed_run?: unknown; thread_sweep?: unknown } | undefined;
  return !!c && (!!c.flow_step || !!c.fit_map || !!c.speed_run || !!c.thread_sweep);
}

export function FlowRunPanels({
  run,
  items,
  results,
  model,
  refreshKey,
}: {
  run: Test;
  items: TestItem[];
  results: ResultRow[];
  model: Model | null;
  refreshKey: unknown;
}) {
  const { readOnly } = useTestView();
  const flowId = (run.config as { flow_id?: string }).flow_id;
  const flowRun = { test: run, items, results };
  const speed = speedRunView(flowRun);
  const threads = threadRunView(flowRun);
  return (
    <section className="mt-6 flex flex-col gap-4">
      {flowId && !readOnly && (
        <p className="text-sm text-fg-2">
          Part of an optimization session.{" "}
          <Link to={`/benchmark?flow=${encodeURIComponent(flowId)}`} className="text-accent">
            Open it in New test
          </Link>
        </p>
      )}
      {run.kind === "fit" && <FitRunPanel testId={run.id} model={model} refreshKey={refreshKey} />}
      {speed && (
        <div>
          <Kicker className="mb-2">
            Speed test · {ctxLabel(speed.spec.target_ctx)} context · {speed.spec.prompt_tokens.toLocaleString("en-US")}-token prompt +{" "}
            {speed.spec.n_gen} generated · {speed.spec.repeats} repeat{speed.spec.repeats === 1 ? "" : "s"}
          </Kicker>
          <SpeedTable view={speed} baseline={speed} />
        </div>
      )}
      {threads && (
        <div className="grid gap-4 sm:grid-cols-2">
          {(["t", "tb"] as const).map((role) => {
            const rows = threads.rows.filter((r) => r.item.role === role);
            if (rows.length === 0) return null;
            const verdict = pickThreadWinner(
              rows.filter((r) => r.reading).map((r) => ({ kind: r.item.kind, mean: r.reading!.mean, stddev: r.reading!.stddev, samples: r.reading!.samples }))
            );
            return <ThreadBars key={role} title={role === "t" ? "tg tok/s by -t" : "pp tok/s by -tb"} rows={rows} verdict={verdict} />;
          })}
        </div>
      )}
    </section>
  );
}

function FitRunPanel({ testId, model, refreshKey }: { testId: string; model: Model | null; refreshKey: unknown }) {
  const { api } = useTestView();
  const [data, setData] = useState<{ points: FitPoint[]; kv: KvSupportRow[]; spec: FitMapSpec | null } | null>(null);
  const [selection, setSelection] = useState<FitSelection | null>(null);
  const moe = useMemo(() => (model ? detectMoe(model.metadata.tensor_layer_bytes ?? null) : null), [model]);
  useEffect(() => {
    let cancelled = false;
    api
      .getFitPoints(testId)
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setData({ points: [], kv: [], spec: null }));
    return () => {
      cancelled = true;
    };
  }, [api, testId, refreshKey]);
  const allSeries = useMemo(() => buildFitSeries(data?.points ?? [], data?.kv ?? []), [data]);
  const [showAll, setShowAll] = useState(false);
  const useful = useMemo(() => usefulSeries(allSeries, moe), [allSeries, moe]);
  const series = showAll ? allSeries : allSeries.filter((s) => useful.has(s.id) || s.id === selection?.seriesId);
  const machine = data?.spec?.machine ?? "dgpu";
  const target = data?.spec?.target_ctx ?? null;
  useEffect(() => {
    setSelection((cur) => cur ?? defaultFitSelection(series, target, moe));
  }, [series, target, moe]);
  if (!data) return <p className="text-sm text-muted">Loading the fit map…</p>;
  if (data.points.length === 0) return <p className="text-sm text-muted">No fit answers yet.</p>;
  const selSeries = series.find((s) => s.id === selection?.seriesId);
  const p = selSeries?.points.find((x) => x.ctx === selection?.ctx);
  const layersTotal = data.points.find((x) => x.layers_total != null)?.layers_total ?? null;
  return (
    <div className="flex flex-col gap-4 border border-border bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 font-display text-lg font-semibold">Fit paths</h2>
        <span className="font-mono text-xs text-muted">
          margin {data.spec?.margin_mib ?? data.points[0].margin_mib} MiB · {data.points.length} answers · {data.kv.length} KV checks
        </span>
      </div>
      <FitLegend
        series={series}
        selectedId={selection?.seriesId ?? null}
        onPick={(id) => {
          const s = series.find((x) => x.id === id);
          const pt = s?.points.find((x) => x.ctx === selection?.ctx) ?? s?.points.filter(fits).slice(-1)[0];
          if (s && pt) setSelection({ seriesId: s.id, ctx: pt.ctx });
        }}
      />
      {allSeries.length > series.length || showAll ? (
        <button type="button" onClick={() => setShowAll((v) => !v)} className="self-start text-sm text-accent hover:text-accent-hover">
          {showAll ? "Show only the cache sizes that gain something" : `Show all ${allSeries.length - 2} cache sizes`}
        </button>
      ) : null}
      <FitPathsChart series={series} selection={selection} onSelect={setSelection} moe={moe} machine={machine} targetCtx={target} layersTotal={layersTotal} />
      {p && selSeries && selection && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
            <StatCard label="Setting" value={`${selSeries.label} · ${ctxLabel(p.ctx)}`} sub={selSeries.subtitle} />
            <StatCard
              label={moe?.isMoe && machine === "dgpu" ? (pointY(p, moe).band === "main" ? "Experts on GPU" : "Dense-only layers") : "Layers on GPU"}
              value={
                !fits(p)
                  ? "doesn't fit"
                  : moe?.isMoe && machine === "dgpu" && pointY(p, moe).band === "main"
                    ? `${pointY(p, moe).value} / ${moe.moeLayers.length} layers`
                    : `${p.layers_gpu} / ${p.layers_total ?? "?"}`
              }
              sub={moe?.isMoe && fits(p) ? describeFitPlacement(p, moe) : undefined}
            />
            <StatCard label="Total memory" value={p.total_mib != null ? `≈ ${fmtGb(p.total_mib)}` : "—"} sub={`${fmtGb(p.dev_used_mib)} device + ${fmtGb(p.host_used_mib)} host`} wide />
          </div>
          <p className="text-[13px] text-fg-2">{fitInsight({ series, selection, moe, machine })}</p>
        </>
      )}
      <p className="text-xs text-muted">Predicted by llama-fit-params, no model loaded.</p>
    </div>
  );
}
