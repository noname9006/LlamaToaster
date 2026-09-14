// The frontier probe's own result: how many layers fit at each context.
//
// Every other probe mode answers with a single placement, which ProbeAttempts
// below already shows rung by rung. This one answers with a staircase, and the
// staircase is the point -- 16 layers at 16k tokens and 12 layers at 128k are
// both true of the same machine, and a user picking between them needs to see
// the trade rather than be handed one end of it.
//
// Nothing here is stored separately. The rungs in probe_attempts ARE the
// measurement; resolveFrontier (shared/probeLadder.ts) turns them into the
// curve, including the stops monotonicity settles without a load. The search
// and this display therefore cannot disagree about what was established --
// they call the same function.

import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { ProbeAttemptDto } from "../types";
import {
  ctxLadderStops,
  resolveFrontier,
  PROBE_EXERCISED_TOKENS,
  type FrontierStop,
  type LadderAttempt,
} from "../../../shared/probeLadder";
import { Chart } from "./Chart";
import { failedForHostBackedLayers } from "./ProbeAttempts";

function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function tokens(value: number): string {
  return value >= 1024 && value % 1024 === 0 ? `${value / 1024}k` : value.toLocaleString();
}

// The stored rows, in the vocabulary the ladder itself reasons in. A row's
// gpu_in_system_ram_mib is the tell for whether its placement was measured at
// all: the worker writes it whenever llama.cpp's buffer report and a
// per-process VRAM reading both existed for that load. Rows from an older
// worker never carry it, so their passes read as unmeasured.
export function toLadderAttempts(rows: ProbeAttemptDto[]): LadderAttempt[] {
  return rows
    .filter((r) => r.ngl != null)
    .map((r) => ({
      ctx: r.candidate_ctx,
      ngl: r.ngl!,
      ok: r.ok === 1,
      hostBacked: failedForHostBackedLayers(r),
      placementJudged: r.gpu_in_system_ram_mib != null,
    }));
}

// The ladder is handed nglMax and the trained context by the worker; neither is
// on an attempt row. Both are recoverable from the rungs themselves, which is
// cheaper than threading the model's metadata through this page: the frontier
// search opens at every layer and reaches for the trained ceiling early, so the
// largest of each is what it was working against.
export function frontierFrom(rows: ProbeAttemptDto[]): FrontierStop[] {
  const history = toLadderAttempts(rows);
  if (history.length === 0) return [];
  const nglMax = Math.max(...history.map((h) => h.ngl));
  const maxCtx = Math.max(...history.map((h) => h.ctx));
  return resolveFrontier({ history, stops: ctxLadderStops(maxCtx), nglMax });
}

const SOURCE_NOTE: Record<FrontierStop["source"], string> = {
  measured: "A rung at this context was actually loaded.",
  implied:
    "Not loaded: two measured stops either side agree, and a larger context can never fit more layers than a smaller one, so this stop is settled without spending a load on it.",
  unmeasured: "The probe ran out of loads before pinning this stop down.",
};

export function ProbeFrontier({ testId, refreshKey }: { testId: string; refreshKey?: unknown }) {
  const [rows, setRows] = useState<ProbeAttemptDto[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getProbeAttempts(testId)
      .then((r) => {
        if (!cancelled) setRows(r.attempts);
      })
      .catch(() => {
        if (!cancelled) setRows([]);
      });
    return () => {
      cancelled = true;
    };
  }, [testId, refreshKey]);

  const frontier = useMemo(() => frontierFrom(rows ?? []), [rows]);
  // The rung that decided a stop, for its measured figures. Implied stops have
  // none by definition -- that is what makes them free.
  const measuredAt = useMemo(() => {
    const byKey = new Map<string, ProbeAttemptDto>();
    for (const r of rows ?? []) if (r.ok === 1 && r.ngl != null) byKey.set(`${r.candidate_ctx}:${r.ngl}`, r);
    return byKey;
  }, [rows]);

  const chartConfig = useMemo(() => {
    const fg = cssVar("--color-fg", "#e8e8e8");
    const muted = cssVar("--color-muted", "#9a9a9a");
    const accent = cssVar("--color-accent", "#6ea8fe");
    const grid = cssVar("--color-border", "#333");
    return {
      type: "line" as const,
      data: {
        labels: frontier.map((s) => tokens(s.ctx)),
        datasets: [
          {
            label: "layers on GPU",
            // Only settled stops are drawn. An unresolved stop knows a layer
            // count that WORKED there, which is a floor rather than the
            // boundary -- plotting it would draw a boundary nobody found, and
            // the gap it leaves instead is the honest shape.
            data: frontier.map((s) => (s.resolved ? s.ngl : null)),
            borderColor: accent,
            backgroundColor: accent,
            // The boundary holds until the next stop is reached, so the line
            // between two stops is a step, not a slope -- drawing it as a slope
            // would claim measurements at contexts nobody tested.
            stepped: "after" as const,
            spanGaps: false,
            pointRadius: frontier.map((s) => (s.source === "measured" ? 4 : 2)),
            pointStyle: frontier.map((s) => (s.source === "measured" ? "circle" : "crossRot")),
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item: { dataIndex: number; parsed: { y: number | null } }) => {
                const stop = frontier[item.dataIndex];
                if (!stop || stop.ngl == null) return "nothing fits at this context";
                return `${stop.ngl} layers · ${stop.source}${stop.unverified ? " · placement not measured" : ""}`;
              },
            },
          },
        },
        scales: {
          x: { title: { display: true, text: "context (tokens)", color: muted }, ticks: { color: muted }, grid: { color: grid } },
          y: {
            beginAtZero: true,
            title: { display: true, text: "layers on GPU", color: muted },
            ticks: { color: fg, precision: 0 },
            grid: { color: grid },
          },
        },
      },
    };
  }, [frontier]);

  if (rows == null) return <p className="text-sm text-muted">Loading…</p>;
  if (frontier.length === 0) return null;

  const anyUnverified = frontier.some((s) => s.unverified);

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h3 className="text-[12.5px] font-semibold text-fg">Layers against context</h3>
      <p className="mt-1 text-[11px] leading-relaxed text-muted">
        The most layers that loaded at each context on this machine. Crossed points were never loaded: a larger context
        can never hold more layers than a smaller one, so stops between two measured ones are settled by the
        measurements either side.
      </p>

      <div className="mt-3 h-56">
        <Chart config={chartConfig} />
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[34rem] border-collapse text-left text-[12px]">
          <thead>
            <tr className="text-muted">
              <th className="py-1 pr-3 font-medium">Context</th>
              <th className="py-1 pr-3 font-medium">Layers</th>
              <th className="py-1 pr-3 font-medium">How</th>
              <th className="py-1 pr-3 font-medium">VRAM peak (llama)</th>
              <th className="py-1 pr-3 font-medium">In system RAM</th>
              <th
                className="py-1 pr-3 font-medium"
                title={`Every rung generates the same ~${PROBE_EXERCISED_TOKENS} tokens whatever context it allocates, so this rate describes a nearly empty cache -- never the speed at the context in its row.`}
              >
                gen tok/s (empty cache)
              </th>
            </tr>
          </thead>
          <tbody>
            {frontier.map((stop) => {
              const row = stop.ngl != null ? measuredAt.get(`${stop.ctx}:${stop.ngl}`) : undefined;
              return (
                <tr key={stop.ctx} className="border-t border-border/60">
                  <td className="py-1 pr-3 tabular-nums text-fg">{stop.ctx.toLocaleString()}</td>
                  <td className="py-1 pr-3 tabular-nums text-fg">
                    {stop.ngl == null ? (
                      <span className="text-muted">{stop.resolved ? "none fit" : "—"}</span>
                    ) : stop.resolved ? (
                      stop.ngl
                    ) : (
                      // The probe proved this many layers load here but ran out
                      // of budget before finding where the boundary actually
                      // is, so this is a floor, not the answer.
                      <span className="text-muted" title="At least this many layers loaded here; the boundary itself was never found.">
                        ≥ {stop.ngl}
                      </span>
                    )}
                    {stop.unverified && (
                      <span
                        className="ml-1.5 text-warning"
                        title="This context loaded, but whether its memory stayed in VRAM was never measured on this worker -- treat it as an allocation ceiling only."
                      >
                        ⚠
                      </span>
                    )}
                  </td>
                  <td className="py-1 pr-3 text-muted" title={SOURCE_NOTE[stop.source]}>
                    {stop.source === "unmeasured" ? "not measured" : stop.source}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-muted">
                    {row?.vram_process_peak_mib != null ? `${Math.round(row.vram_process_peak_mib).toLocaleString()} MiB` : "—"}
                  </td>
                  <td
                    className="py-1 pr-3 tabular-nums text-muted"
                    title="What llama.cpp put on the GPU that this process's dedicated VRAM did not hold."
                  >
                    {row?.gpu_in_system_ram_mib == null
                      ? "—"
                      : row.gpu_in_system_ram_mib > (row.gpu_spill_jitter_mib ?? 0)
                        ? `${Math.round(row.gpu_in_system_ram_mib).toLocaleString()} MiB`
                        : "none"}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-muted">
                    {row?.gen_tps != null ? row.gen_tps.toFixed(1) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {anyUnverified && (
        <p className="mt-2 text-[11px] leading-relaxed text-warning">
          ⚠ Some stops rest on a load whose memory placement was never measured on this worker — the model loaded, but
          part of it may be sitting in system RAM. Those contexts are allocation ceilings, not speeds.
        </p>
      )}
    </div>
  );
}
