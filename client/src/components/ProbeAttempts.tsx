// BENCHMARKING_PLAN_V8.md N2 -- the ladder behind a verified ceiling.
//
// model_machine_limits stores the single context later runs consume; this
// panel shows what was actually loaded to get there. The column that earns
// the panel's existence is needed-vs-free: the estimate and the machine's
// real free memory side by side, per rung. Until this shipped that pair
// only ever reached a worker log line, so an estimate that was wildly off
// (see shared/vramEstimate.ts's own top comment on the sysmem-fallback case)
// was invisible to anyone reading results in the app.

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ProbeAttemptDto } from "../types";

function mib(value: number | null): string {
  return value == null ? "—" : `${Math.round(value).toLocaleString()} MiB`;
}

// The estimate's error against what the machine actually had free is the
// reading a user can act on, so it is stated rather than left to be
// eyeballed across two columns.
function needVsFree(needed: number | null, free: number | null): string | null {
  if (needed == null || free == null || free <= 0) return null;
  return `${Math.round((needed / free) * 100)}% of free`;
}

export interface ProbeAttemptsProps {
  runId: string;
  /** Re-fetched whenever the run's own status changes. */
  refreshKey?: unknown;
}

export function ProbeAttempts({ runId, refreshKey }: ProbeAttemptsProps) {
  const [attempts, setAttempts] = useState<ProbeAttemptDto[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getProbeAttempts(runId)
      .then((res) => {
        if (cancelled) return;
        setAttempts(res.attempts);
        setError("");
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [runId, refreshKey]);

  if (error) return <p className="text-sm text-danger">Could not load this probe's loads: {error}</p>;
  if (!attempts) return <p className="text-sm text-muted">Loading probe loads…</p>;

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">
          Context tests — every load this probe performed
        </span>
        <span className="font-mono text-[11px] text-muted">
          {attempts.length} load{attempts.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-[10px] uppercase tracking-wide text-muted">
              <th className="px-2 py-1.5">#</th>
              <th className="px-2 py-1.5">context</th>
              <th className="px-2 py-1.5 text-right">offload</th>
              <th className="px-2 py-1.5 text-right">VRAM needed</th>
              <th className="px-2 py-1.5 text-right">VRAM free</th>
              <th className="px-2 py-1.5 text-right">VRAM peak</th>
              <th className="px-2 py-1.5 text-right">RAM needed</th>
              <th className="px-2 py-1.5 text-right">RAM free</th>
              <th className="px-2 py-1.5 text-right">gen tok/s</th>
              <th className="px-2 py-1.5">result</th>
            </tr>
          </thead>
          <tbody>
            {attempts.map((a) => {
              const vramRatio = needVsFree(a.vram_needed_mib, a.vram_free_mib);
              const ramRatio = needVsFree(a.ram_needed_mib, a.ram_free_mib);
              return (
                <tr key={a.id} className="border-b border-border/40">
                  <td className="px-2 py-1.5 font-mono text-muted">{a.seq + 1}</td>
                  <td className="px-2 py-1.5 font-mono text-fg">{a.candidate_ctx.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{a.ngl ?? "—"}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted" title={vramRatio ?? undefined}>
                    {mib(a.vram_needed_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_free_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_peak_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted" title={ramRatio ?? undefined}>
                    {mib(a.ram_needed_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.ram_free_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">
                    {a.gen_tps != null ? a.gen_tps.toFixed(1) : "—"}
                  </td>
                  <td className="px-2 py-1.5 text-muted">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {a.ok ? (
                        <>
                          <span className="mr-1 inline-block h-2 w-2 rounded-full bg-success align-middle" /> passed
                        </>
                      ) : (
                        <span
                          title={a.error ?? undefined}
                          className="rounded-full bg-warning-bg px-2 py-0.5 text-[10px] font-bold text-warning"
                        >
                          {a.oom ? "out of memory" : a.spill ? "spilled past VRAM" : "failed"}
                        </span>
                      )}
                      {a.reused_from_run_id && (
                        <Link
                          to={`/runs/${a.reused_from_run_id}`}
                          title="Reused from an earlier batch sibling's own measurement of this exact point -- not reloaded for this run"
                          className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent hover:underline"
                        >
                          ↺ reused
                        </Link>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
            {attempts.length === 0 && (
              <tr>
                <td colSpan={10} className="px-2 py-3 text-muted">
                  This probe recorded no loads. A probe run that never reached the machine leaves no rungs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        <b className="text-fg">Needed</b> is what the estimate predicted for that context and placement;{" "}
        <b className="text-fg">free</b> is what the machine actually had available just before the load;{" "}
        <b className="text-fg">peak</b> is what the load really used. A needed figure far above peak means the
        estimate is over-budgeting for this model — the probe still measured reality, so its verdict stands
        regardless.
      </p>
    </div>
  );
}
