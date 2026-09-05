// BENCHMARKING_PLAN_V8.md N2 -- the ladder behind a verified ceiling.
//
// model_machine_limits stores the single context later runs consume; this
// panel shows what was actually loaded to get there. Every figure in the
// table is a direct measurement (free, peak, shared, resident) rather than a
// prediction -- a calculated pre-load estimate used to sit here as its own
// "VRAM needed"/"RAM needed" columns, at the same visual weight as the real
// measured ones, which read as fact to anyone skimming results in the app.
// That estimate still drives the ladder's own search and the "possible VRAM
// fallback" check (see vram_discrepancy below), just not as an on-screen
// peer of the numbers the worker actually measured.

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ProbeAttemptDto } from "../types";

function mib(value: number | null): string {
  return value == null ? "—" : `${Math.round(value).toLocaleString()} MiB`;
}

// How much of the free budget a load actually used, measured rather than
// predicted -- see the removal of the "VRAM needed" column below for why this
// reads peak, not the pre-load estimate: a calculated number sitting at the
// same visual weight as real measured columns (free, peak, shared) reads as
// a fact it isn't. The estimate still drives the ladder's own search and the
// "possible VRAM fallback" check below, just not this on-screen ratio.
// Paired against the TOTAL (whole-adapter/whole-system) peak, not the
// per-process one -- "free" was always measured whole-adapter/whole-system
// too (there's no such thing as "free" scoped to a process that doesn't
// exist yet), so that's the only pairing that's actually apples-to-apples.
function usedVsFree(peak: number | null, free: number | null): string | null {
  if (peak == null || free == null || free <= 0) return null;
  return `${Math.round((peak / free) * 100)}% of free`;
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

  // Derived from the rows already on screen -- no extra endpoint, and it can
  // only ever appear on a machine where a rung actually MEASURED the paging.
  // On a platform with no shared-memory counter (CUDA-on-Linux, Metal) every
  // vram_shared_peak_mib is null and this stays silent, which is correct:
  // there, an oversubscribed allocation fails outright instead of paging.
  const spilled = attempts.filter((a) => a.host_backed_method != null && a.vram_discrepancy === 1 && a.ngl != null);
  const lastClean = attempts
    .filter((a) => a.vram_shared_peak_mib != null && a.vram_discrepancy !== 1 && a.ngl != null)
    .reduce<ProbeAttemptDto | null>((best, a) => (best == null || a.ngl! > best.ngl! ? a : best), null);
  const worstSpill = spilled.reduce<ProbeAttemptDto | null>(
    (w, a) => (w == null || (a.vram_shared_peak_mib ?? 0) > (w.vram_shared_peak_mib ?? 0) ? a : w),
    null
  );
  const cliff = attempts
    .filter((a) => a.prefill_cliff === 1 && a.ngl != null)
    .reduce<ProbeAttemptDto | null>((best, a) => (best == null || a.ngl! < best.ngl! ? a : best), null);

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      {worstSpill && (
        <div className="mb-3 rounded-lg border border-warning/40 bg-warning-bg px-3 py-2 text-xs text-warning">
          <span className="font-bold">This GPU pages model weights to system RAM.</span>{" "}
          {lastClean?.ngl != null ? (
            <>
              Past <span className="font-mono font-bold">{lastClean.ngl}</span> layers, llama.cpp reports the layers on
              the GPU but the driver serves them from system RAM.{" "}
            </>
          ) : (
            <>llama.cpp reports layers on the GPU that the driver is serving from system RAM. </>
          )}
          Measured up to <span className="font-mono font-bold">{mib(worstSpill.vram_shared_peak_mib)}</span> in host
          memory on this probe. Layer counts above that boundary are not real GPU offload on this machine, so a
          maximum-offload result will not reflect its true GPU speed.
          {cliff?.ngl != null && (
            <>
              {" "}
              Prompt processing degrades earlier still, from{" "}
              <span className="font-mono font-bold">{cliff.ngl}</span> layers.
            </>
          )}
        </div>
      )}
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
            <tr className="border-b border-border/60 text-left text-[10px] uppercase tracking-wide text-muted">
              <th className="px-2 py-1.5" rowSpan={2}>#</th>
              <th className="px-2 py-1.5" rowSpan={2}>context</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>offload</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>resident</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>VRAM free</th>
              <th className="px-2 py-1.5 text-center" colSpan={2}>VRAM Peak</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>shared</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>RAM free</th>
              <th className="px-2 py-1.5 text-center" colSpan={2}>RAM Peak</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>gen tok/s</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2} title="Prompt-processing rate. Prefill has its own placement cliff, several layers BELOW the one where weights start spilling -- a rung can have the best gen tok/s while being several times worse to first token.">
                pp tok/s
              </th>
              <th className="px-2 py-1.5 text-right" rowSpan={2} title="Time to first token, measured from request send to the first streamed chunk.">
                TTFT
              </th>
              <th className="px-2 py-1.5 text-right" rowSpan={2} title="Layers' worth of system-RAM-backed GPU memory appearing per layer added, against a lower-offload load at the same context. Buffer overhead does not scale with layer count; spilled weights do, one for one -- so a value near 1 means the added layers are not on the GPU.">
                slope
              </th>
              <th className="px-2 py-1.5" rowSpan={2}>result</th>
            </tr>
            <tr className="border-b border-border text-left text-[10px] uppercase tracking-wide text-muted">
              <th className="px-2 py-1.5 text-right" title="Whole-adapter -- every process on this GPU combined, not just this load">
                total
              </th>
              <th className="px-2 py-1.5 text-right" title="This load's own process only">
                per process
              </th>
              <th className="px-2 py-1.5 text-right" title="Whole-system -- every process on this machine combined, not just this load">
                total
              </th>
              <th className="px-2 py-1.5 text-right" title="This load's own process only">
                per process
              </th>
            </tr>
          </thead>
          <tbody>
            {attempts.map((a) => {
              const vramRatio = usedVsFree(a.vram_peak_mib, a.vram_free_mib);
              const ramRatio = usedVsFree(a.ram_total_peak_mib, a.ram_free_mib);
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
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_free_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted" title={vramRatio ?? undefined}>
                    {mib(a.vram_peak_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_process_peak_mib)}</td>
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${a.vram_shared_peak_mib ? "font-bold text-warning" : "text-muted"}`}
                    title={
                      a.vram_shared_peak_mib != null
                        ? `${mib(a.vram_shared_peak_mib)} of this load's memory was system RAM the OS backed as GPU-accessible ` +
                          "memory (Windows WDDM \"Shared Usage\" / Linux amdgpu GTT), not real dedicated VRAM -- a direct " +
                          "measurement."
                        : "No shared/system-RAM-backed GPU memory counter is available on this worker for this backend."
                    }
                  >
                    {mib(a.vram_shared_peak_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.ram_free_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted" title={ramRatio ?? undefined}>
                    {mib(a.ram_total_peak_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.ram_peak_mib)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">
                    {a.gen_tps != null ? a.gen_tps.toFixed(1) : "—"}
                  </td>
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${a.prefill_cliff === 1 ? "font-bold text-warning" : "text-muted"}`}
                    title={
                      a.prefill_cliff === 1
                        ? "Prompt processing has collapsed at this placement -- the batch compute buffer is being served from system RAM. Generation can still look fine here; time to first token will not."
                        : undefined
                    }
                  >
                    {a.pp_tps != null ? a.pp_tps.toFixed(1) : "—"}
                    {a.prefill_cliff === 1 ? " ⚠" : ""}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">
                    {a.ttft_ms != null ? `${(a.ttft_ms / 1000).toFixed(2)}s` : "—"}
                  </td>
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${a.host_backed_slope != null && a.host_backed_slope > 0.5 ? "font-bold text-warning" : "text-muted"}`}
                    title={
                      a.host_backed_method === "ratio"
                        ? "No comparable lower-offload load at this context, so this rung was judged against its own predicted footprint instead of a slope."
                        : undefined
                    }
                  >
                    {a.host_backed_slope != null ? a.host_backed_slope.toFixed(2) : a.host_backed_method === "ratio" ? "n/a" : "—"}
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
                <td colSpan={13} className="px-2 py-3 text-muted">
                  This probe recorded no loads. A probe run that never reached the machine leaves no rungs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-muted">
        <b className="text-fg">Offload</b> is what was claimed; <b className="text-fg">resident</b> is the exact
        count llama.cpp's own per-layer "assigned to device" report says actually landed on the GPU — a mismatch
        against offload is flagged the same way a VRAM-fallback row is. It reads "—" rather than a guess when that
        report wasn't captured (the load failed before tensor loading finished, or an older build). Every other
        number in this row — <b className="text-fg">free</b>, <b className="text-fg">peak</b>,{" "}
        <b className="text-fg">shared</b> — is a direct measurement, not a prediction: free is what the machine
        actually had available just before the load. <b className="text-fg">Peak</b> is split into{" "}
        <b className="text-fg">total</b> (every process on the GPU/machine combined, not just this load — the same
        thing "free" measures, which is why the two are comparable) and{" "}
        <b className="text-fg">per process</b> (this load's own usage in isolation, when the backend could attribute
        it). They're genuinely different numbers, not two views of the same one: something else running on the same
        GPU or machine at the same time shows up in total but not per process, while per process can read "—" more
        often than total does — the tool/counter that attributes usage to this specific process can lag a fresh
        spawn, or never catch up at all on a very short load. A row flagged{" "}
        <b className="text-warning">⚠ possible VRAM fallback</b> means VRAM peak (total) came in far below what the
        ladder's own pre-load estimate expected this offload to need — a sign the load silently ran (partly) from
        system RAM instead of true GPU memory, without llama.cpp reporting any error; how it's handled — warn, retry
        once, or fail — is the worker's VRAM-discrepancy policy. <b className="text-fg">Shared</b> is that same
        spillover, but measured directly rather than inferred: the worker's own OS-level reading of how much of this
        process's memory was system RAM the driver backed as GPU-accessible memory instead of real dedicated VRAM.
        It's blank when no such counter exists for this worker's backend, which is different from a confirmed zero.
      </p>
    </div>
  );
}
