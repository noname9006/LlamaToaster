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

// Claimed-vs-landed: `ngl` is what the ladder asked for and llama.cpp claimed
// to offload; gpu_layers_resident_est is what its own post-allocation buffer
// report says actually landed on the GPU (see shared/types.ts's
// ProbeAttemptReport.gpu_layers_resident_est doc comment). Null when the
// worker predates the check, or nothing was computable for this rung (ngl<=0,
// or the load failed before tensor loading finished).
function residentCell(a: Pick<ProbeAttemptDto, "ngl" | "gpu_layers_resident_est" | "gpu_layers_resident_exact">): {
  text: string;
  warn: boolean;
  title?: string;
} {
  if (a.gpu_layers_resident_est == null) return { text: "—", warn: false };
  const prefix = a.gpu_layers_resident_exact === 1 ? "" : "~";
  const mismatched = a.ngl != null && a.gpu_layers_resident_est !== a.ngl;
  return {
    text: `${prefix}${a.gpu_layers_resident_est}`,
    warn: mismatched,
    title: mismatched
      ? `Claimed ${a.ngl} layers on GPU, but llama.cpp's own post-allocation buffer report shows ` +
        `${a.gpu_layers_resident_exact === 1 ? "exactly" : "an estimated"} ${a.gpu_layers_resident_est} ` +
        "actually landed there -- the rest fell back to CPU/system memory."
      : undefined,
  };
}

export interface ProbeAttemptsProps {
  testId: string;
  /** Re-fetched whenever the run's own status changes. */
  refreshKey?: unknown;
}

export function ProbeAttempts({ testId, refreshKey }: ProbeAttemptsProps) {
  const [attempts, setAttempts] = useState<ProbeAttemptDto[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getProbeAttempts(testId)
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
  }, [testId, refreshKey]);

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
              <th className="px-2 py-1.5 text-right">resident</th>
              <th className="px-2 py-1.5 text-right">VRAM needed</th>
              <th className="px-2 py-1.5 text-right">VRAM free</th>
              <th className="px-2 py-1.5 text-right">VRAM peak</th>
              <th className="px-2 py-1.5 text-right">shared</th>
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
              const resident = residentCell(a);
              return (
                <tr key={a.id} className="border-b border-border/40">
                  <td className="px-2 py-1.5 font-mono text-muted">{a.seq + 1}</td>
                  <td className="px-2 py-1.5 font-mono text-fg">{a.candidate_ctx.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{a.ngl ?? "—"}</td>
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${resident.warn ? "font-bold text-warning" : "text-muted"}`}
                    title={resident.title}
                  >
                    {resident.text}
                    {resident.warn ? " ⚠" : ""}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted" title={vramRatio ?? undefined}>
                    {mib(a.vram_needed_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_free_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_peak_mib)}</td>
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${a.vram_shared_peak_mib ? "font-bold text-warning" : "text-muted"}`}
                    title={
                      a.vram_shared_peak_mib != null
                        ? `${mib(a.vram_shared_peak_mib)} of this load's memory was system RAM the OS backed as GPU-accessible ` +
                          "memory (Windows WDDM \"Shared Usage\" / Linux amdgpu GTT), not real dedicated VRAM -- a direct " +
                          "measurement, not an inference from the needed-vs-peak gap."
                        : "No shared/system-RAM-backed GPU memory counter is available on this worker for this backend."
                    }
                  >
                    {mib(a.vram_shared_peak_mib)}
                  </td>
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
                      {a.ok && a.vram_discrepancy === 1 && (
                        <span
                          title={
                            a.error ??
                            `Observed VRAM peak came in far below what ${mib(a.vram_needed_mib)} of claimed offload should need -- ` +
                              "likely silently running from system RAM instead of erroring (seen on both NVIDIA/CUDA and AMD/Vulkan), not actual GPU offload."
                          }
                          className="rounded-full bg-warning-bg px-2 py-0.5 text-[10px] font-bold text-warning"
                        >
                          ⚠ possible VRAM fallback
                        </span>
                      )}
                      {a.reused_from_run_id && (
                        <Link
                          to={`/tests/${a.reused_from_run_id}`}
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
                <td colSpan={12} className="px-2 py-3 text-muted">
                  This probe recorded no loads. A probe run that never reached the machine leaves no rungs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        <b className="text-fg">Offload</b> is what was claimed; <b className="text-fg">resident</b> is how many of
        those layers llama.cpp's own post-allocation buffer report says actually landed on the GPU (a{" "}
        <span className="text-muted">~</span> prefix means it's a coarser estimate, not an exact per-layer count) —
        a mismatch is flagged the same way a VRAM-fallback row is.{" "}
        <b className="text-fg">Needed</b> is what the estimate predicted for that context and placement;{" "}
        <b className="text-fg">free</b> is what the machine actually had available just before the load;{" "}
        <b className="text-fg">peak</b> is what the load really used. A needed figure far above peak usually means
        the estimate is over-budgeting for this model — but when peak also stays far below free, it can instead mean
        the load silently ran (partly) from system RAM instead of true GPU memory, without llama.cpp reporting any
        error. A row flagged{" "}
        <b className="text-warning">⚠ possible VRAM fallback</b> was caught by that check; how it's handled — warn,
        retry once, or fail — is the worker's VRAM-discrepancy policy.{" "}
        <b className="text-fg">Shared</b> is that same spillover, but measured directly rather than inferred: the
        worker's own OS-level reading of how much of this process's memory was system RAM the driver backed as
        GPU-accessible memory instead of real dedicated VRAM. It's blank when no such counter exists for this
        worker's backend, which is different from a confirmed zero.
      </p>
    </div>
  );
}
