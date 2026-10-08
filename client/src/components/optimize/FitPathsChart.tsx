import { useEffect, useMemo, useState } from "react";
import type { FitPoint } from "../../../../shared/fitParams.js";
import type { MoeInfo } from "../../../../shared/moePlacement.js";
import { ctxLabel, fits, pointY, type FitSelection, type FitSeries, type MachineClass } from "../../../../shared/optimizeFlow.js";

// The fit map as design 2a "Fit paths" (docs/plans/fit-map-options.dc.html):
// x = total memory a setup needs, y = layers on the GPU, one path per cache
// setup with context as the step along it. Up and to the left is better.
// Hand-written SVG rather than the app's Chart.js wrapper: merged point
// labels, the MoE broken axis and per-point hit targets don't fit Chart.js.

const RAMP = ["var(--color-rank-2)", "var(--color-rank-3)", "var(--color-accent)", "var(--color-rank-4)", "var(--color-rank-5)"];
const DASHES = ["none", "2 3", "6 3", "1 3", "8 3 2 3"];

export function seriesStyle(s: FitSeries): { color: string; dash: string } {
  if (s.kind === "default") return { color: "var(--color-fg)", dash: "none" };
  if (s.kind === "fa_off") return { color: "var(--color-muted)", dash: "5 4" };
  const i = s.rampIndex ?? 0;
  return { color: RAMP[Math.min(RAMP.length - 1, i % RAMP.length)], dash: DASHES[Math.floor(i / RAMP.length + i) % DASHES.length] };
}

function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
}

function ticks(lo: number, hi: number, target: number): number[] {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, target);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

// A callback ref, not useRef: the chart swaps its root element (empty state
// -> chart) as data arrives, and the observer has to follow the element that
// is actually mounted.
function useWidth<T extends HTMLElement>(): [(el: T | null) => void, number] {
  const [el, setEl] = useState<T | null>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!el) return;
    const measure = () => {
      const next = Math.round(el.getBoundingClientRect().width);
      if (next > 0) setW(next);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, w];
}

export interface FitPathsChartProps {
  series: FitSeries[];
  selection: FitSelection | null;
  onSelect: (s: FitSelection) => void;
  moe: MoeInfo | null;
  machine: MachineClass;
  targetCtx: number | null;
  /** "of N" in the y-axis label. */
  layersTotal: number | null;
}

export function FitPathsChart({ series, selection, onSelect, moe, machine, targetCtx, layersTotal }: FitPathsChartProps) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const narrow = width < 560;
  const H = narrow ? 340 : 430;
  const left = narrow ? 34 : 50;
  const right = width - (narrow ? 12 : 16);
  const top = 30;
  const bottom = H - 42;

  const plotted = useMemo(
    () => series.map((s) => ({ s, pts: s.points.filter((p) => p.total_mib != null && p.total_mib > 0) })),
    [series]
  );
  const allPts = plotted.flatMap((x) => x.pts);
  const hasDense = !!moe?.isMoe && allPts.some((p) => pointY(p, moe).band === "dense");
  const mainVals = allPts.filter((p) => pointY(p, moe).band === "main").map((p) => pointY(p, moe).value);
  // The top is the highest point plotted, not the model's layer count: a map
  // that tops out at 17 of 65 layers would otherwise sit in the bottom quarter.
  const maxMain = Math.max(1, ...mainVals);
  const minMain = mainVals.length ? Math.min(...mainVals) : 0;
  // y starts at the lowest point rounded down, so 7 vs 17 isn't squashed
  // against 0..65; layer counts are whole numbers, so are the ticks, and a
  // flat map (every point at all layers) still gets a few rows of room.
  const yStep = Math.max(1, Math.round(niceStep(Math.max(4, maxMain - minMain), 6)));
  const yHi = Math.max(1, Math.ceil(maxMain));
  const yLo = Math.max(0, Math.min(Math.floor(minMain / yStep) * yStep, yHi - 4));
  // CPU-only and unified machines have no layer split to show (every layer
  // runs in the one pool), so the y axis is the context instead: the chart
  // reads as memory needed against context.
  const pool = machine !== "dgpu";
  const ctxs = [...new Set(allPts.map((p) => p.ctx))].sort((a, b) => a - b);
  const cLo = Math.log2(ctxs[0] ?? 4096);
  const cHi = Math.max(cLo + 1, Math.log2(ctxs[ctxs.length - 1] ?? 8192));
  const denseBand = hasDense && !pool ? Math.round((bottom - top) * 0.2) : 0;
  const mainBottom = bottom - denseBand;
  const yOfValue = (v: number): number =>
    pool
      ? mainBottom - ((Math.log2(v) - cLo) / (cHi - cLo)) * (mainBottom - top - 6)
      : mainBottom - ((v - yLo) / (yHi - yLo)) * (mainBottom - top - 6);
  const Y = (p: FitPoint): number => {
    if (pool) return yOfValue(p.ctx);
    const y = pointY(p, moe);
    if (y.band === "dense") {
      const all = p.layers_total ?? (moe ? moe.nBlocks + 1 : 1);
      return bottom - (y.value / Math.max(1, all)) * (denseBand - 12);
    }
    return yOfValue(y.value);
  };
  // Axes follow the setups that fit; a doesn't-fit point far to the right
  // (60 GB of RAM it doesn't have) is pinned to the edge instead of
  // squashing every real answer into a corner.
  const fitPts = allPts.filter(fits);
  const gbs = (fitPts.length ? fitPts : allPts).map((p) => p.total_mib! / 1024);
  const xMinRaw = gbs.length ? Math.min(...gbs) : 0;
  const xMaxRaw = gbs.length ? Math.max(...gbs) : 1;
  const xPad = Math.max(0.2, (xMaxRaw - xMinRaw) * 0.06);
  const xStep = niceStep(Math.max(0.5, xMaxRaw - xMinRaw + 2 * xPad), narrow ? 4 : 7);
  const xLo = Math.max(0, Math.floor((xMinRaw - xPad) / xStep) * xStep);
  const xHi = Math.ceil((xMaxRaw + xPad) / xStep) * xStep;
  const X = (gb: number) => left + ((Math.min(gb, xHi) - xLo) / Math.max(1e-9, xHi - xLo)) * (right - left);

  const xTicks = ticks(xLo, xHi, narrow ? 4 : 7);
  const yTicks: number[] = [];
  if (pool) yTicks.push(...ctxs);
  else {
    for (let v = Math.ceil(yLo / yStep) * yStep; v <= yHi; v += yStep) yTicks.push(v);
    if (yTicks[yTicks.length - 1] !== yHi) yTicks.push(yHi);
  }

  const selId = selection?.seriesId ?? null;
  const ordered = [...plotted].sort((a, b) => (a.s.id === selId ? 1 : 0) - (b.s.id === selId ? 1 : 0));

  // Context labels on the selected path; points closer than 34px merge.
  const labels = useMemo(() => {
    const sel = plotted.find((x) => x.s.id === selId);
    if (!sel) return [];
    const groups: { x: number; y: number; first: FitPoint; last: FitPoint; has: boolean; target: boolean }[] = [];
    for (const p of sel.pts) {
      const x = X(p.total_mib! / 1024);
      const y = Y(p);
      const g = groups[groups.length - 1];
      const isTarget = targetCtx != null && p.ctx === targetCtx;
      if (g && Math.hypot(x - g.x, y - g.y) < 34) {
        g.last = p;
        g.has = g.has || p.ctx === selection?.ctx;
        g.target = g.target || isTarget;
      } else groups.push({ x, y, first: p, last: p, has: p.ctx === selection?.ctx, target: isTarget });
    }
    return groups.map((g) => ({
      x: g.x,
      y: g.y,
      bold: g.has,
      text:
        (g.first === g.last ? ctxLabel(g.first.ctx) : `${ctxLabel(g.first.ctx)}–${ctxLabel(g.last.ctx)}`) +
        (g.target && g.first === g.last ? " · target" : ""),
    }));
    // X/Y close over layout values derived from width; recompute with them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotted, selId, selection?.ctx, targetCtx, width, yLo, yHi, xLo, xHi, denseBand]);

  if (allPts.length === 0) {
    return (
      <div ref={wrapRef} className="border border-border bg-well p-6 text-sm text-muted">
        No answers to plot yet.
      </div>
    );
  }

  const yLabel = pool
    ? "↑ Context"
    : moe?.isMoe
      ? `↑ MoE layers with experts on GPU (of ${moe.moeLayers.length})`
      : `↑ Layers on GPU${layersTotal ? ` (of ${layersTotal})` : ""}`;

  return (
    <div ref={wrapRef} className="w-full">
      <svg
        width={width}
        height={H}
        viewBox={`0 0 ${width} ${H}`}
        role="img"
        aria-label="Fit paths: total memory needed against layers on the GPU, one path per cache setup"
        className="block overflow-visible"
      >
        <text x={left + 4} y={14} className="fill-muted font-mono" fontSize={11} letterSpacing="0.06em">
          {yLabel.toUpperCase()}
        </text>
        {!narrow && (
          <text x={right} y={14} textAnchor="end" className="fill-accent" fontSize={12}>
            {pool ? "← less memory" : "← better: more layers, less memory"}
          </text>
        )}
        {yTicks.map((v) => {
          const y = yOfValue(v);
          return (
            <g key={`y${v}`}>
              <line x1={left} x2={right} y1={y} y2={y} stroke="var(--color-border)" strokeWidth={1} />
              <text x={left - 6} y={y + 4} textAnchor="end" className="fill-muted font-mono" fontSize={11}>
                {pool ? ctxLabel(v) : v}
              </text>
            </g>
          );
        })}
        {xTicks.map((v) => (
          <g key={`x${v}`}>
            <line x1={X(v)} x2={X(v)} y1={top} y2={bottom} stroke="var(--color-border)" strokeWidth={1} />
            <text x={X(v)} y={bottom + 16} textAnchor="middle" className="fill-muted font-mono" fontSize={11}>
              {v % 1 === 0 ? v : v.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={left} y={H - 6} className="fill-muted font-mono" fontSize={11} letterSpacing="0.06em">
          {(machine === "dgpu"
            ? "Total memory needed, GB (VRAM + RAM) →"
            : machine === "unified"
              ? "Total memory needed, GB (unified pool) →"
              : "Memory needed, GB (system RAM) →"
          ).toUpperCase()}
        </text>
        {hasDense && (
          <g>
            <line x1={left} x2={right} y1={mainBottom + 4} y2={mainBottom + 4} stroke="var(--color-border-strong)" strokeDasharray="6 4" />
            <text x={right} y={mainBottom + 16} textAnchor="end" className="fill-muted" fontSize={11}>
              dense layers only (all experts on CPU)
            </text>
          </g>
        )}

        {ordered.map(({ s, pts }) => {
          const st = seriesStyle(s);
          const isSel = s.id === selId;
          // The path ends at the last setup that fits; a doesn't-fit point is
          // still drawn, as a hollow ×.
          const fitting = pts.filter(fits);
          const line = fitting.map((p) => `${X(p.total_mib! / 1024)},${Y(p)}`).join(" ");
          return (
            <g key={s.id} opacity={selId == null || isSel ? 1 : 0.45}>
              {fitting.length > 1 && (
                <polyline points={line} fill="none" stroke={st.color} strokeWidth={isSel ? 3 : 1.5} strokeDasharray={st.dash} />
              )}
              {pts.map((p) => {
                const x = X(p.total_mib! / 1024);
                const y = Y(p);
                const chosen = isSel && p.ctx === selection?.ctx;
                const pick = () => onSelect({ seriesId: s.id, ctx: p.ctx });
                const title = `${s.label} · ${ctxLabel(p.ctx)} · ${p.verdict === "doesnt_fit" ? "doesn't fit" : `${(p.total_mib! / 1024).toFixed(1)} GB`}`;
                return (
                  <g key={p.ctx}>
                    {fits(p) ? (
                      <circle
                        cx={x}
                        cy={y}
                        r={chosen ? 6 : isSel ? 4 : 3}
                        fill={chosen ? st.color : "var(--color-bg)"}
                        stroke={st.color}
                        strokeWidth={1.5}
                        strokeDasharray={p.inferred ? "2 2" : undefined}
                      />
                    ) : (
                      <g stroke="var(--color-muted)" strokeWidth={1.5}>
                        <circle cx={x} cy={y} r={chosen ? 6 : 4.5} fill="var(--color-bg)" />
                        <line x1={x - 2.5} y1={y - 2.5} x2={x + 2.5} y2={y + 2.5} />
                        <line x1={x - 2.5} y1={y + 2.5} x2={x + 2.5} y2={y - 2.5} />
                      </g>
                    )}
                    <circle
                      cx={x}
                      cy={y}
                      r={10}
                      fill="transparent"
                      className="cursor-pointer"
                      role="button"
                      tabIndex={0}
                      aria-label={title}
                      onClick={pick}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          pick();
                        }
                      }}
                    >
                      <title>{title}</title>
                    </circle>
                  </g>
                );
              })}
            </g>
          );
        })}
        {labels.map((l, i) => (
          <text
            key={i}
            x={Math.min(l.x + 10, right - 4)}
            y={Math.max(top + 10, l.y - 10)}
            textAnchor={l.x + 10 > right - 60 ? "end" : "start"}
            className="pointer-events-none fill-fg"
            fontSize={12}
            fontWeight={l.bold ? 700 : 400}
          >
            {l.text}
          </text>
        ))}
      </svg>
    </div>
  );
}

export function FitLegend({
  series,
  selectedId,
  onPick,
}: {
  series: FitSeries[];
  selectedId: string | null;
  onPick: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-x-[18px] gap-y-1.5" role="group" aria-label="Cache setups">
      {series.map((s) => {
        const st = seriesStyle(s);
        const on = s.id === selectedId;
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => onPick(s.id)}
            aria-pressed={on}
            title={s.subtitle}
            className={`flex items-center gap-2 py-1 text-[13px] ${on ? "font-semibold text-fg" : "text-fg-2 opacity-80 hover:opacity-100"}`}
          >
            <svg width="28" height="10" aria-hidden="true">
              <line x1="0" y1="5" x2="28" y2="5" stroke={st.color} strokeWidth={2} strokeDasharray={st.dash} />
            </svg>
            {s.label}
          </button>
        );
      })}
    </div>
  );
}
