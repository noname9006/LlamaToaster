// The Benchmark chain's sweep stage output: prefill speed along the context
// fill, one line per (placement, KV pair). Rows come from shared/fillCurve.ts's
// grouping -- each point is one slice's own prefill rate, plotted at how full
// the context was when that slice finished.

import { useMemo } from "react";
import type { ChartConfiguration } from "chart.js";
import { Chart } from "./Chart";
import { groupFillCurves, type FillCurve } from "../../../shared/fillCurve";
import { FILL_CURVE_METHOD_VERSION, type ResultRow } from "../types";

// Categorical slots in fixed order (dataviz reference palette, dark steps),
// validated against --color-surface #12151c: all six checks pass.
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const TEXT_MUTED = "#8b92a3";
const GRID = "#262b36";

function curveLabel(c: FillCurve): string {
  return `${c.ngl} layers · ${c.kv}`;
}

function formatTokens(n: number): string {
  return n >= 1024 ? `${(n / 1024).toFixed(n % 1024 === 0 ? 0 : 1)}k` : String(n);
}

export function FillCurveChart({ results, ctx }: { results: ResultRow[]; ctx?: number | null }) {
  const curves = useMemo(() => groupFillCurves(results, FILL_CURVE_METHOD_VERSION), [results]);

  const config: ChartConfiguration<"line"> | null = useMemo(() => {
    if (curves.length === 0) return null;
    return {
      type: "line",
      data: {
        datasets: curves.map((c, i) => {
          const color = SERIES[i % SERIES.length];
          return {
            label: curveLabel(c),
            // A slice whose every repeat was excluded has no reading -- a gap,
            // never a zero.
            data: c.points.map((p) => ({ x: p.fill, y: p.tps > 0 ? p.tps : null })),
            borderColor: color,
            backgroundColor: color,
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBorderColor: "#12151c",
            pointBorderWidth: 2,
            spanGaps: false,
            tension: 0,
          };
        }),
      },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        parsing: false,
        interaction: { mode: "nearest", intersect: false, axis: "x" },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: ctx ?? undefined,
            title: { display: true, text: "Context filled (tokens)", color: TEXT_MUTED },
            ticks: {
              color: TEXT_MUTED,
              stepSize: ctx != null ? ctx / 8 : undefined,
              callback: (v) => formatTokens(Number(v)),
            },
            grid: { color: GRID },
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: "Prefill tok/s", color: TEXT_MUTED },
            ticks: { color: TEXT_MUTED },
            grid: { color: GRID },
          },
        },
        plugins: {
          legend: { display: true, labels: { color: TEXT_MUTED, usePointStyle: true, boxHeight: 8 } },
          tooltip: {
            callbacks: {
              label: (item) => {
                const c = curves[item.datasetIndex];
                const p = c?.points[item.dataIndex];
                if (!c || !p) return "";
                return `${curveLabel(c)}: ${p.tps.toFixed(0)} tok/s for tokens ${p.depth.toLocaleString()}–${p.fill.toLocaleString()}`;
              },
            },
          },
        },
      },
    };
  }, [curves, ctx]);

  if (!config) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="relative h-72">
        <Chart config={config as ChartConfiguration} />
      </div>
      {/* The same curves as numbers: where each one starts, where it ends up at
          the full context, and what the fill cost -- so the chart's finding is
          never colour-only. */}
      <table className="w-full text-left text-[12px]">
        <thead className="text-muted">
          <tr>
            <th className="py-1 pr-3 font-medium">Configuration</th>
            <th className="py-1 pr-3 font-medium">First slice</th>
            <th className="py-1 pr-3 font-medium">Last slice</th>
            <th className="py-1 font-medium">Change</th>
          </tr>
        </thead>
        <tbody>
          {curves.map((c) => {
            const measured = c.points.filter((p) => p.tps > 0);
            const first = measured[0];
            const last = measured[measured.length - 1];
            const change = first && last && first !== last ? (last.tps / first.tps - 1) * 100 : null;
            return (
              <tr key={c.idx} className="border-t border-border">
                <td className="py-1 pr-3 font-mono text-fg">{curveLabel(c)}</td>
                <td className="py-1 pr-3 text-fg">{first ? `${first.tps.toFixed(0)} tok/s` : "—"}</td>
                <td className="py-1 pr-3 text-fg">
                  {last ? `${last.tps.toFixed(0)} tok/s @ ${formatTokens(last.fill)}` : "—"}
                </td>
                <td className="py-1 text-muted">{change != null ? `${change > 0 ? "+" : ""}${change.toFixed(0)} %` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
