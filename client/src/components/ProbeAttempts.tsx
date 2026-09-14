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
import { PROBE_EXERCISED_TOKENS, PROBE_GEN_TOKENS, PROBE_PROMPT_TOKENS } from "../../../shared/probeLadder";
import { KV_HOST_BACKED_FAIL_FRAC } from "../../../shared/vramEstimate";

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

// Mirrors worker/src/runtimeBench.ts's failedForHostBackedLayers on the stored
// row: the only failure that leaves a rung with generation, no OOM, no adapter
// spill and a MEASURED discrepancy is the one where its layers were found in
// system RAM.
export function failedForHostBackedLayers(
  a: Pick<ProbeAttemptDto, "ok" | "oom" | "spill" | "gen_tps" | "vram_discrepancy" | "host_backed_method">
): boolean {
  return a.ok !== 1 && a.oom !== 1 && a.spill !== 1 && a.gen_tps != null && a.vram_discrepancy === 1 && a.host_backed_method != null;
}

// The context-axis counterpart: the rung generated, did not OOM or pass the
// adapter total, and most of what its larger context added went to system RAM
// (worker/src/runtimeBench.ts's probeSucceeded, KV_HOST_BACKED_FAIL_FRAC).
// Disjoint from the layer one -- a context-axis verdict never sets
// vram_discrepancy.
function failedForHostBackedCache(
  a: Pick<ProbeAttemptDto, "ok" | "oom" | "spill" | "gen_tps" | "kv_host_backed_frac">
): boolean {
  return (
    a.ok !== 1 &&
    a.oom !== 1 &&
    a.spill !== 1 &&
    a.gen_tps != null &&
    a.kv_host_backed_frac != null &&
    a.kv_host_backed_frac > KV_HOST_BACKED_FAIL_FRAC
  );
}

// Everything llama-server claimed on the GPU, however the driver split it
// between dedicated VRAM and shared system RAM. Measured as one sum per reading
// when the worker sent it; a rung stored before that falls back to adding the
// two separate peaks, marked "~" -- they can come from different moments of the
// load, so that sum can only read high, never low.
function claimedCell(
  a: Pick<ProbeAttemptDto, "vram_claimed_peak_mib" | "vram_process_peak_mib" | "vram_shared_peak_mib">
): { text: string; title: string; mib: number | null } {
  if (a.vram_claimed_peak_mib != null) {
    return {
      text: mib(a.vram_claimed_peak_mib),
      title: "llama-server's own dedicated VRAM plus its shared system RAM, summed within one reading -- everything it claimed on the GPU.",
      mib: a.vram_claimed_peak_mib,
    };
  }
  if (a.vram_process_peak_mib != null) {
    const sum = a.vram_process_peak_mib + (a.vram_shared_peak_mib ?? 0);
    return {
      text: `~${mib(sum)}`,
      title:
        "Recorded before claimed VRAM was measured directly: llama's VRAM peak plus its shared peak, which may come from different moments of the load, so this can read high.",
      mib: sum,
    };
  }
  return {
    text: "—",
    title:
      "No per-process VRAM reading on this worker (an older worker build, or a platform whose driver reports no per-process memory), so what llama-server claimed can't be measured.",
    mib: null,
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
  // The smallest context whose cache went mostly to system RAM -- the one
  // nearest the boundary the context search settled against.
  const kvFailed = attempts
    .filter(failedForHostBackedCache)
    .reduce<ProbeAttemptDto | null>((best, a) => (best == null || a.candidate_ctx < best.candidate_ctx ? a : best), null);
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
      {kvFailed && (
        <div className="mb-3 rounded-lg border border-warning/40 bg-warning-bg px-3 py-2 text-xs text-warning">
          <span className="font-bold">Past a point, the context's cache does not fit in VRAM.</span> At{" "}
          <span className="font-mono font-bold">{kvFailed.candidate_ctx.toLocaleString()}</span> tokens,{" "}
          <span className="font-mono font-bold">{Math.round((kvFailed.kv_host_backed_frac ?? 0) * 100)}%</span> of the
          memory that context added went to system RAM, so the load failed and the search tried smaller contexts. Each
          load here only runs a {PROBE_PROMPT_TOKENS}-token prompt and generates {PROBE_GEN_TOKENS} tokens, so that load
          can still look fast — a cache in system RAM only slows generation down once the context actually fills.
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
              <th
                className="px-2 py-1.5 text-right"
                rowSpan={2}
                title="Everything llama-server claimed on the GPU: its own dedicated VRAM plus its own shared system RAM, however the driver split the two."
              >
                VRAM claimed
              </th>
              <th className="px-2 py-1.5 text-center" colSpan={2}>VRAM Peak</th>
              <th className="px-2 py-1.5 text-center" colSpan={2}>Shared</th>
              <th className="px-2 py-1.5 text-right" rowSpan={2}>RAM free</th>
              <th className="px-2 py-1.5 text-center" colSpan={2}>RAM Peak</th>
              <th
                className="px-2 py-1.5 text-center"
                colSpan={3}
                title={`Every load runs the same ${PROBE_EXERCISED_TOKENS}-token workload whatever context it allocates, so these describe roughly ${PROBE_EXERCISED_TOKENS} tokens of context -- NOT the context in the row. A large context here is allocated and never read.`}
              >
                speed at ~{PROBE_EXERCISED_TOKENS} tok
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
              <th className="px-2 py-1.5 text-right" title="llama-server's own process only">
                llama
              </th>
              <th
                className="px-2 py-1.5 text-right"
                title="Whole-adapter -- every process's system-RAM-backed GPU memory combined, not just this load"
              >
                total
              </th>
              <th className="px-2 py-1.5 text-right" title="llama-server's own process only">
                llama
              </th>
              <th className="px-2 py-1.5 text-right" title="Whole-system -- every process on this machine combined, not just this load">
                total
              </th>
              <th className="px-2 py-1.5 text-right" title="llama-server's own process only">
                llama
              </th>
              <th className="px-2 py-1.5 text-right">gen tok/s</th>
              <th
                className="px-2 py-1.5 text-right"
                title="Prompt-processing rate. Prefill has its own placement cliff, several layers BELOW the one where weights start spilling -- a rung can have the best gen tok/s while being several times worse to first token."
              >
                pp tok/s
              </th>
              <th className="px-2 py-1.5 text-right" title="Time to first token, from request send to the first streamed chunk.">
                TTFT
              </th>
            </tr>
          </thead>
          <tbody>
            {attempts.map((a) => {
              const vramRatio = usedVsFree(a.vram_peak_mib, a.vram_free_mib);
              const ramRatio = usedVsFree(a.ram_total_peak_mib, a.ram_free_mib);
              const resident = residentCell(a);
              const claimed = claimedCell(a);
              // More claimed than was free before the load: the rest had to
              // land in system RAM, whatever the counters' timing.
              const claimedOverFree = claimed.mib != null && a.vram_free_mib != null && claimed.mib > a.vram_free_mib;
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
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${claimedOverFree ? "font-bold text-warning" : "text-muted"}`}
                    title={claimed.title}
                  >
                    {claimed.text}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted" title={vramRatio ?? undefined}>
                    {mib(a.vram_peak_mib)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-muted">{mib(a.vram_process_peak_mib)}</td>
                  <td
                    className="px-2 py-1.5 text-right font-mono text-muted"
                    title={
                      a.vram_shared_total_peak_mib != null
                        ? "Every process's system-RAM-backed GPU memory combined -- includes anything else running, not just this load."
                        : "No adapter-wide shared memory reading for this load (an older worker, or no such counter for this backend)."
                    }
                  >
                    {mib(a.vram_shared_total_peak_mib)}
                  </td>
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
                      a.kv_host_backed_frac != null
                        ? `Context axis: ${Math.round(a.kv_host_backed_frac * 100)}% of the memory this context added went to system RAM instead of VRAM. Above ${Math.round(KV_HOST_BACKED_FAIL_FRAC * 100)}% the load fails -- a cache in system RAM is slow once the context actually fills.`
                        : a.host_backed_method === "ratio"
                          ? "No comparable lower-offload load at this context, so this rung was judged against its own predicted footprint instead of a slope."
                          : undefined
                    }
                  >
                    {a.host_backed_slope != null ? a.host_backed_slope.toFixed(2) : a.host_backed_method === "ratio" ? "n/a" : "—"}
                    {a.kv_host_backed_frac != null && a.kv_host_backed_frac > KV_HOST_BACKED_FAIL_FRAC ? " kv" : ""}
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
                          {a.oom
                            ? "out of memory"
                            : a.spill
                              ? "spilled past VRAM"
                              : failedForHostBackedLayers(a)
                                ? "layers in system RAM"
                                : failedForHostBackedCache(a)
                                  ? "cache in system RAM"
                                  : "failed"}
                        </span>
                      )}
                      {/* === 1, not a bare `a.ok &&`: ok is a SQLite 0/1, and React
                          renders a literal 0 for `0 && ...` on every failed row. */}
                      {a.ok === 1 && a.vram_discrepancy === 1 && (
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
                <td colSpan={18} className="px-2 py-3 text-muted">
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
        report wasn't captured (nothing was claimed on the GPU, the load failed before tensor loading finished, or an older build). Every other
        number in this row — <b className="text-fg">free</b>, <b className="text-fg">claimed</b>,{" "}
        <b className="text-fg">peak</b>, <b className="text-fg">shared</b> — is a direct measurement, not a prediction:
        free is what the machine actually had available just before the load. <b className="text-fg">Claimed</b> is
        everything llama-server allocated on the GPU — its own dedicated VRAM plus its own shared memory, summed within
        one reading — however the driver split the two; once claimed exceeds VRAM free, the difference has to live in
        system RAM. A claimed value marked <b className="text-fg">~</b> comes from a load recorded before claimed was
        measured directly and adds the two separate peaks instead, so it can only read high.{" "}
        <b className="text-fg">Peak</b> and <b className="text-fg">Shared</b> are each split into{" "}
        <b className="text-fg">total</b> (every process on the GPU/machine combined, not just this load — the same
        thing "free" measures, which is why the two are comparable) and <b className="text-fg">llama</b> (llama-server's
        own usage in isolation, when the backend could attribute it). They're genuinely different numbers, not two
        views of the same one: something else running on the same GPU or machine at the same time shows up in total but
        not in llama, while llama can read "—" more often than total does — the tool/counter that attributes usage to
        this specific process can lag a fresh spawn, or never catch up at all on a very short load.{" "}
        <b className="text-fg">Shared</b> is the OS's own reading of system RAM the driver backed as GPU-accessible
        memory instead of real dedicated VRAM — the direct measurement of a silent spillover, not an inference about
        one. It's blank when no such counter exists for this worker's backend, which is different from a confirmed
        zero.
      </p>
      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        <b className="text-fg">Slope</b> is what decides a{" "}
        <b className="text-warning">⚠ possible VRAM fallback</b>: how much shared memory appears per unit of demand
        added, against an earlier load that held the other axis fixed. On the layer axis the unit is one layer's
        weights, so a value near 1 means the added layers went to system RAM rather than the GPU — buffer overhead
        doesn't scale with layer count, spilled weights do. A row reading <b className="text-fg">n/a</b> had no
        comparable earlier load yet and was judged against its own predicted footprint instead. A load whose layers
        are measurably in system RAM <b className="text-fg">fails</b> as <b className="text-warning">layers in system RAM</b>,
        so the search backs off to a layer count that really fits — when both memory counters agree, or at a small
        context where nothing but weights can account for the shared memory. Otherwise (the evidence is only an
        inference from the estimate, the counters disagree, or a large context could explain the reading) the load
        passes with the warning instead. A slope suffixed{" "}
        <b className="text-fg">kv</b> is the context axis, where what spilled is cache rather than weights: when more
        than {Math.round(KV_HOST_BACKED_FAIL_FRAC * 100)}% of the memory a larger context added went to system RAM, the
        load fails as <b className="text-warning">cache in system RAM</b> and the search tries a smaller context.{" "}
        <b className="text-fg">Speeds</b> come from the same fixed {PROBE_EXERCISED_TOKENS}-token workload on every
        row, so they say whether a configuration runs — not how fast it is at the context beside them. There is no
        minimum rate: a slow load is reported with its rate, and only one that generates nothing at all fails.
      </p>
    </div>
  );
}
