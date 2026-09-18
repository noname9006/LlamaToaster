// The Wizard's own result: both answers at every context stop.
//
// Every other probe mode answers with a single placement, which ProbeAttempts
// below already shows load by load. The Wizard answers with two staircases --
// the most layers with no spill, and the most whose llama.cpp claim fits the
// free VRAM -- and the trade between context and layers is the point.
//
// Nothing here is stored separately. The rows in probe_attempts ARE the
// measurement; probeOutcome (shared/probeLadder.ts) resolves them with the same
// function the worker searched with, so the search and this display cannot
// disagree about what was established.

import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { ProbeAttemptDto } from "../types";
import {
  claimFitsFree,
  probeOutcome,
  type CurveStop,
  type LadderAttempt,
  type ProbeMode,
  type ProbeOutcome,
} from "../../../shared/probeLadder";
import { Chart } from "./Chart";

function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function tokens(value: number): string {
  return value >= 1024 && value % 1024 === 0 ? `${value / 1024}k` : value.toLocaleString();
}

// The stored rows, in the vocabulary the ladder itself reasons in.
export function toLadderAttempts(rows: ProbeAttemptDto[]): LadderAttempt[] {
  return rows
    .filter((r) => r.ngl != null)
    .map((r) => ({
      ctx: r.candidate_ctx,
      ngl: r.ngl!,
      ok: r.ok === 1,
      placementJudged: r.gpu_in_system_ram_mib != null,
      claimedMib: r.gpu_buffers_mib,
      inconclusive: r.load_kind === "error",
      // The per-device verdict the worker recorded wins; older rows compare totals.
      fitsFree:
        r.claim_fits_free != null
          ? r.claim_fits_free === 1
          : claimFitsFree({
              claimedMib: r.gpu_buffers_mib,
              freeMib: r.list_devices_free_mib,
              loaded: r.gen_tps != null || r.load_kind === "claim_stop" || r.load_kind === "claim_only",
            }),
    }));
}

/**
 * Re-resolves a stored probe exactly as the worker did. Every row repeats the
 * ladder's own bounds and the --list-devices reading; a row from a worker
 * predating them falls back to the largest layer count and context it loaded.
 * A Targets probe's pinned axis is read off its first load after the anchor,
 * which is always at the pinned value. A probe whose rows carry a spill method
 * was searched with the anchor first, and is re-resolved the same way.
 */
export function outcomeFrom(rows: ProbeAttemptDto[], mode: ProbeMode): ProbeOutcome | null {
  const history = toLadderAttempts(rows);
  if (history.length === 0) return null;
  const first = rows.find((r) => r.ngl != null && r.spill_method !== "anchor") ?? rows.find((r) => r.ngl != null)!;
  return probeOutcome({
    anchored: rows.some((r) => r.spill_method != null),
    mode,
    candidateCtx: first.candidate_ctx,
    candidateNgl: first.ngl!,
    nglMax: rows.find((r) => r.ladder_ngl_max != null)?.ladder_ngl_max ?? Math.max(...history.map((h) => h.ngl)),
    maxCtx: rows.find((r) => r.ladder_max_ctx != null)?.ladder_max_ctx ?? Math.max(...history.map((h) => h.ctx)),
    history,
    freeVramMib: rows.find((r) => r.list_devices_free_mib != null)?.list_devices_free_mib ?? null,
  });
}

const SOURCE_NOTE: Record<CurveStop["clean"]["source"], string> = {
  measured: "Loads at this context decided it.",
  implied:
    "Settled without a load: nothing fits at a smaller context, or the smaller context's no-spill answer was none.",
  unmeasured: "The probe ran out of loads (or was stopped) before settling this context.",
};

function answer(value: number | null, resolved: boolean): string {
  if (!resolved) return value != null ? `≥ ${value}` : "—";
  return value == null ? "none" : String(value);
}

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

  const outcome = useMemo(() => outcomeFrom(rows ?? [], "frontier"), [rows]);
  const curve = outcome?.curve ?? [];
  const hasFit = curve.some((s) => s.fit != null);
  const freeMib = useMemo(() => (rows ?? []).find((r) => r.list_devices_free_mib != null)?.list_devices_free_mib ?? null, [rows]);
  // The full loads at a point, newest last -- figures shown are the first's.
  const loadsAt = (ctx: number, ngl: number | null) =>
    ngl == null ? [] : (rows ?? []).filter((r) => r.candidate_ctx === ctx && r.ngl === ngl && r.load_kind !== "claim_stop" && r.load_kind !== "claim_only");

  const chartConfig = useMemo(() => {
    const fg = cssVar("--color-fg", "#e8e8e8");
    const muted = cssVar("--color-muted", "#9a9a9a");
    const accent = cssVar("--color-accent", "#6ea8fe");
    const warning = cssVar("--color-warning", "#e0a84f");
    const grid = cssVar("--color-border", "#333");
    const line = (label: string, data: (number | null)[], color: string, dashed: boolean) => ({
      label,
      data,
      borderColor: color,
      backgroundColor: color,
      borderDash: dashed ? [4, 3] : [],
      // The boundary holds until the next stop, so the line between two stops is
      // a step: a slope would claim measurements at contexts nobody loaded.
      stepped: "after" as const,
      spanGaps: false,
      pointRadius: 3,
    });
    return {
      type: "line" as const,
      data: {
        labels: curve.map((s) => tokens(s.ctx)),
        datasets: [
          ...(hasFit
            ? [line("claim fits free VRAM", curve.map((s) => (s.fit?.resolved ? s.fit.value : null)), warning, true)]
            : []),
          line("no spill", curve.map((s) => (s.clean.resolved ? s.clean.value : null)), accent, false),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: muted } },
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
  }, [curve, hasFit]);

  if (rows == null) return <p className="text-sm text-muted">Loading…</p>;
  if (!outcome || curve.length === 0) return null;

  const unfinished = outcome.next != null;
  // Recorded before context tests v2: the rows are re-read by today's rules, which
  // did not choose those loads, so an answer can read as unfinished that was not.
  const legacy = (rows ?? []).every((r) => r.ladder_ngl_max == null);
  const anyUnjudged = curve.some((s) => s.clean.resolved && s.clean.value != null && !s.clean.judged);

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h3 className="text-[12.5px] font-semibold text-fg">Layers against context</h3>
      <p className="mt-1 text-[11px] leading-relaxed text-muted">
        Two answers per context, smallest context first. <span className="text-fg">No spill</span>: the most layers whose GPU
        memory all stayed in VRAM, loaded twice to confirm, and never more than the smaller context held.{" "}
        {hasFit && (
          <>
            <span className="text-fg">Claim fits free VRAM</span>: the most layers whose llama.cpp GPU claim stayed below the{" "}
            {freeMib != null ? `${Math.round(freeMib).toLocaleString()} MiB` : "free VRAM"} llama-server --list-devices reported
            before the first load.
          </>
        )}
        {!hasFit && <>This probe had no --list-devices free VRAM reading, so only the no-spill answer was searched.</>}
      </p>
      {outcome.nothingFitsFrom != null && (
        <p className="mt-1.5 text-[11px] font-semibold text-muted">
          Nothing fits free VRAM from {outcome.nothingFitsFrom.toLocaleString()} tokens up, so the probe stopped there.
        </p>
      )}
      {legacy && (
        <p className="mt-1.5 text-[11px] font-semibold text-warning">
          This probe was recorded by an earlier version of the context test. Its loads are shown under today's two-answer
          rules, which chose a different order, so some contexts may read as not measured.
        </p>
      )}
      {unfinished && !legacy && (
        <p className="mt-1.5 text-[11px] font-semibold text-warning">
          The probe stopped before the search finished (its load budget ran out, or it was stopped). Contexts marked "not
          measured" were never settled.
        </p>
      )}

      <div className="mt-3 h-56">
        <Chart config={chartConfig} />
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[40rem] border-collapse text-left text-[12px]">
          <thead>
            <tr className="text-muted">
              <th className="py-1 pr-3 font-medium">Context</th>
              <th className="py-1 pr-3 font-medium">No spill</th>
              {hasFit && <th className="py-1 pr-3 font-medium">Claim fits</th>}
              <th className="py-1 pr-3 font-medium">How</th>
              <th className="py-1 pr-3 font-medium">VRAM peak (llama)</th>
            </tr>
          </thead>
          <tbody>
            {curve.map((stop) => {
              const at = loadsAt(stop.ctx, stop.clean.value)[0];
              return (
                <tr key={stop.ctx} className="border-t border-border/60">
                  <td className="py-1 pr-3 tabular-nums text-fg">{stop.ctx.toLocaleString()}</td>
                  <td className="py-1 pr-3 tabular-nums text-fg">
                    {answer(stop.clean.value, stop.clean.resolved)}
                    {stop.clean.control === "confirmed" && (
                      <span className="ml-1.5 text-success" title="Loaded a second time, clean again.">
                        ✓✓
                      </span>
                    )}
                    {stop.clean.control === "pending" && (
                      <span className="ml-1.5 text-warning" title="Loaded once only -- the confirming second load never ran.">
                        unconfirmed
                      </span>
                    )}
                    {stop.clean.resolved && stop.clean.value != null && !stop.clean.judged && (
                      <span className="ml-1.5 text-warning" title="Where this load's memory went was never measured on this worker.">
                        ⚠
                      </span>
                    )}
                  </td>
                  {hasFit && (
                    <td className="py-1 pr-3 tabular-nums text-fg">{stop.fit ? answer(stop.fit.value, stop.fit.resolved) : "—"}</td>
                  )}
                  <td className="py-1 pr-3 text-muted" title={SOURCE_NOTE[stop.clean.source]}>
                    {stop.clean.source === "unmeasured" ? "not measured" : stop.clean.source}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-muted">
                    {at?.vram_process_peak_mib != null ? `${Math.round(at.vram_process_peak_mib).toLocaleString()} MiB` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {anyUnjudged && (
        <p className="mt-2 text-[11px] leading-relaxed text-warning">
          ⚠ Some answers rest on loads whose memory placement was never measured on this worker — the model loaded, but part
          of it may be sitting in system RAM.
        </p>
      )}
    </div>
  );
}
