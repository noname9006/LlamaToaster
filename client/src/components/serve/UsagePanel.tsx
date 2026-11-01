import { useMemo, useState } from "react";
import type { ChartConfiguration } from "chart.js";
import { Chart } from "../Chart";
import { Btn, Kicker, Panel, StepTitle } from "../optimize/ui";
import { USAGE_SHAPE_LABELS, MIN_PP_TOKENS, MIN_TG_TOKENS } from "../../../../shared/daemonUsage";
import type { DaemonUsageView } from "../../../../shared/daemonTypes";
import { formatShare, formatSpeed, formatTokens, loadSummary } from "./serveData";

// Same dark steps as the rest of the page, validated against --color-surface.
const PP_COLOR = "#3987e5";
const TG_COLOR = "#d95926";
const TEXT_MUTED = "#ab9f8d";
const GRID = "#3d3329";

function SpeedChart({
  title,
  color,
  field,
  view,
  xRange,
}: {
  title: string;
  color: string;
  field: "pp" | "tg";
  view: DaemonUsageView;
  xRange: { min: number; max: number };
}) {
  const config = useMemo((): ChartConfiguration | null => {
    const points = view.points.filter((p) => p[field] != null && p.ctx != null && p.ctx > 0).map((p) => ({ x: p.ctx as number, y: p[field] as number }));
    if (points.length === 0) return null;
    // The band's centre on a log axis, so the median sits among its own points.
    const median = view.curve.filter((c) => c[field] != null).map((c) => ({ x: Math.sqrt(c.from * c.to), y: c[field] as number }));
    return {
      type: "scatter",
      data: {
        datasets: [
          { label: "Median per size band", type: "line", data: median, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 4, pointBorderColor: "#1b1611", pointBorderWidth: 2, tension: 0, order: 1 },
          { label: "One request", data: points, backgroundColor: `${color}66`, borderColor: "transparent", pointRadius: 3, pointHoverRadius: 6, order: 2 },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        responsive: true,
        parsing: false,
        scales: {
          x: {
            type: "logarithmic",
            // The same range on both charts, so a position means the same thing on each.
            min: xRange.min,
            max: xRange.max,
            title: { display: true, text: "Context used (tokens, cached + evaluated)", color: TEXT_MUTED },
            // Powers of two only: the size bands are, and the default log ticks
            // crowd together over a narrow range.
            afterBuildTicks: (axis) => {
              const ticks: { value: number }[] = [];
              for (let v = xRange.min; v <= xRange.max; v *= 2) ticks.push({ value: v });
              const step = Math.max(1, Math.ceil(ticks.length / 8));
              axis.ticks = ticks.filter((_, i) => i % step === 0);
            },
            ticks: { color: TEXT_MUTED, callback: (v) => formatTokens(Number(v)) },
            grid: { color: GRID },
          },
          y: { beginAtZero: true, title: { display: true, text: "tok/s", color: TEXT_MUTED }, ticks: { color: TEXT_MUTED }, grid: { color: GRID } },
        },
        plugins: {
          legend: { display: true, labels: { color: TEXT_MUTED, usePointStyle: true, boxHeight: 8 } },
          tooltip: {
            callbacks: {
              label: (item) => `${formatTokens(item.parsed.x)} tokens of context: ${formatSpeed(item.parsed.y)}`,
            },
          },
        },
      },
    };
  }, [view, field, color, xRange]);

  return (
    <div className="min-w-0 flex-[1_1_360px]">
      <h3 className="m-0 mb-1 font-display text-base font-semibold">{title}</h3>
      {config ? (
        <div className="relative h-64">
          <Chart config={config} />
        </div>
      ) : (
        <p className="m-0 flex h-64 items-center justify-center border border-border text-sm text-muted">No readings yet.</p>
      )}
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-[132px] flex-[1_1_132px] border border-border bg-well px-3 py-2">
      <Kicker>{label}</Kicker>
      <div className="mt-0.5 font-mono text-lg text-fg">{value}</div>
      {sub && <div className="text-xs text-muted">{sub}</div>}
    </div>
  );
}

function TimeSplit({ share }: { share: { prefill: number; generation: number; other: number } }) {
  const parts = [
    { label: "Prefill", value: share.prefill, color: PP_COLOR },
    { label: "Generation", value: share.generation, color: TG_COLOR },
    { label: "Queue and overhead", value: share.other, color: "#7d7264" },
  ];
  return (
    <div>
      <Kicker>Where the wait goes</Kicker>
      <div role="img" aria-label={parts.map((p) => `${p.label} ${formatShare(p.value)}`).join(", ")} className="mt-1 flex h-3 gap-0.5 bg-well">
        {parts.map((p) => (p.value > 0.002 ? <div key={p.label} title={`${p.label}: ${formatShare(p.value)}`} style={{ width: `${p.value * 100}%`, background: p.color }} /> : null))}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 text-sm">
        {parts.map((p) => (
          <span key={p.label} className="flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-2.5 w-2.5" style={{ background: p.color }} />
            <span className="text-fg-2">{p.label}</span>
            <span className="font-mono text-fg">{formatShare(p.value)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export function UsagePanel({
  view,
  modelName,
  loadHash,
  onLoadHash,
  includeConcurrent,
  onIncludeConcurrent,
  onClear,
}: {
  view: DaemonUsageView | null;
  modelName: string | null;
  loadHash: string;
  onLoadHash: (hash: string) => void;
  includeConcurrent: boolean;
  onIncludeConcurrent: (on: boolean) => void;
  onClear: () => void;
}) {
  const [showTable, setShowTable] = useState(false);
  const profile = view?.profile;
  const shape = profile?.shape ? USAGE_SHAPE_LABELS[profile.shape] : null;
  const xRange = useMemo(() => {
    const sizes = (view?.points ?? []).map((p) => p.ctx).filter((c): c is number => c != null && c > 0);
    if (sizes.length === 0) return { min: 1, max: 1024 };
    const min = 2 ** Math.floor(Math.log2(Math.min(...sizes)));
    const max = 2 ** Math.ceil(Math.log2(Math.max(...sizes)));
    return { min, max: max > min ? max : min * 2 };
  }, [view]);
  // Across every load, not just the one in view: after a reload the current settings
  // have no requests yet, and earlier ones must stay reachable through the picker.
  const hasHistory = (profile?.requests ?? 0) > 0 || (view?.loads ?? []).some((l) => l.requests > 0);

  return (
    <Panel>
      <StepTitle
        right={
          view && hasHistory ? (
            <Btn kind="ghost" onClick={onClear}>
              Clear history
            </Btn>
          ) : undefined
        }
      >
        Usage{modelName ? ` · ${modelName}` : ""}
      </StepTitle>
      {!view || !profile || !hasHistory ? (
        <p className="m-0 text-sm text-muted">
          {modelName
            ? "No requests yet. Point an agent at the endpoint and speeds appear here. Only counts and speeds are recorded, never prompts or answers."
            : "Pick a model to see its usage."}
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
            <label className="flex min-w-0 items-center gap-2">
              <span className="text-fg-2">Load settings</span>
              <select value={loadHash} onChange={(e) => onLoadHash(e.target.value)} className="min-h-10 min-w-0 max-w-full border border-border-strong bg-well px-2 text-sm text-fg">
                <option value="">All ({view.loads.length})</option>
                {view.loads.map((l) => (
                  <option key={l.load_hash} value={l.load_hash}>
                    {loadSummary(l)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex min-h-10 items-center gap-2">
              <input type="checkbox" checked={includeConcurrent} onChange={(e) => onIncludeConcurrent(e.target.checked)} />
              <span className="text-fg-2">Include requests that ran at the same time</span>
            </label>
            {!includeConcurrent && view.concurrent_excluded > 0 && <span className="text-xs text-muted">{view.concurrent_excluded} left out: slots share throughput</span>}
          </div>

          {profile.requests === 0 ? (
            <p className="m-0 text-sm text-muted">No requests on these load settings yet. Earlier ones are in the list above.</p>
          ) : (
          <>
          <div className="flex flex-wrap gap-4">
            <SpeedChart title="Prefill (pp)" color={PP_COLOR} field="pp" view={view} xRange={xRange} />
            <SpeedChart title="Generation (tg)" color={TG_COLOR} field="tg" view={view} xRange={xRange} />
          </div>
          <p className="m-0 -mt-2 text-xs text-muted">
            From real requests, so these vary with what agents send. Prefill needs at least {MIN_PP_TOKENS} evaluated tokens to read, generation at least {MIN_TG_TOKENS}.
            {profile.untimedRequests > 0 && ` ${profile.untimedRequests} request${profile.untimedRequests === 1 ? "" : "s"} had no timings to read (/v1/messages, and /v1/responses when not streamed).`}
          </p>
          <div>
            <button type="button" className="border-0 bg-transparent p-0 text-xs text-muted underline hover:text-fg" onClick={() => setShowTable((s) => !s)} aria-expanded={showTable}>
              {showTable ? "Hide" : "Show"} these as a table
            </button>
            {showTable && (
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-muted">
                    <tr>
                      <th className="py-1 pr-3 font-medium">Context used</th>
                      <th className="py-1 pr-3 font-medium">Median pp</th>
                      <th className="py-1 pr-3 font-medium">Requests</th>
                      <th className="py-1 pr-3 font-medium">Median tg</th>
                      <th className="py-1 font-medium">Requests</th>
                    </tr>
                  </thead>
                  <tbody>
                    {view.curve.map((c) => (
                      <tr key={c.from} className="border-t border-border font-mono text-fg">
                        <td className="py-1 pr-3">
                          {formatTokens(c.from)}–{formatTokens(c.to)}
                        </td>
                        <td className="py-1 pr-3">{formatSpeed(c.pp)}</td>
                        <td className="py-1 pr-3">{c.ppCount}</td>
                        <td className="py-1 pr-3">{formatSpeed(c.tg)}</td>
                        <td className="py-1">{c.tgCount}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div>
            <h3 className="m-0 font-display text-base font-semibold">Usage profile</h3>
            <p className="m-0 mt-0.5 text-sm text-fg-2">
              {shape ? (
                <>
                  <strong className="font-semibold text-fg">{shape.title}.</strong> {shape.detail}
                </>
              ) : (
                `Last ${view.window_days} days.`
              )}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Stat label="Tokens in" value={formatTokens(profile.tokens.processed)} sub="evaluated" />
            <Stat label="From cache" value={formatTokens(profile.tokens.cached)} sub={`${formatShare(profile.cacheShare)} of prompt tokens`} />
            <Stat label="Tokens out" value={formatTokens(profile.tokens.generated)} sub="generated" />
            <Stat label="Prompt size" value={formatTokens(profile.promptTokens.p50)} sub={`p95 ${formatTokens(profile.promptTokens.p95)}`} />
            <Stat label="Answer size" value={formatTokens(profile.generatedTokens.p50)} sub={`p95 ${formatTokens(profile.generatedTokens.p95)}`} />
            <Stat label="Speed" value={formatSpeed(profile.speed.tg)} sub={`prefill ${formatSpeed(profile.speed.pp)}`} />
            <Stat label="Requests" value={String(profile.requests)} sub={`${profile.errors} failed · ${profile.cancelled} cancelled`} />
            <Stat label="At once" value={String(profile.concurrency.max ?? 1)} sub="peak requests" />
          </div>
          {profile.timeShare && <TimeSplit share={profile.timeShare} />}
          {profile.observations.length > 0 && (
            <ul className="m-0 flex list-disc flex-col gap-1 pl-5 text-sm text-fg-2">
              {profile.observations.map((o) => (
                <li key={o.kind}>{o.text}</li>
              ))}
            </ul>
          )}
          </>
          )}
        </div>
      )}
    </Panel>
  );
}
