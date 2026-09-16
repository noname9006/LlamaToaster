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
// row. A row carrying host_backed_fail says outright which spill failed it. One
// from a worker predating that column falls back to the old derivation: the
// only failure that left it with generation, no OOM, no adapter spill and a
// MEASURED discrepancy was the one where its layers were found in system RAM.
export function failedForHostBackedLayers(
  a: Pick<
    ProbeAttemptDto,
    "ok" | "oom" | "spill" | "gen_tps" | "vram_discrepancy" | "host_backed_method" | "host_backed_fail"
  >
): boolean {
  if (a.host_backed_fail != null) return a.ok !== 1 && a.host_backed_fail === "layers";
  return a.ok !== 1 && a.oom !== 1 && a.spill !== 1 && a.gen_tps != null && a.vram_discrepancy === 1 && a.host_backed_method != null;
}

// The context-sized counterpart: what spilled was no more than the context's
// KV cache and compute buffer, so a smaller context can fix it. An older
// worker's row carries no host_backed_fail; its only other generating failure
// was the context-axis rule, the one that wrote kv_host_backed_frac.
function failedForHostBackedCache(
  a: Pick<
    ProbeAttemptDto,
    "ok" | "oom" | "spill" | "gen_tps" | "vram_discrepancy" | "host_backed_method" | "kv_host_backed_frac" | "host_backed_fail"
  >
): boolean {
  if (a.host_backed_fail != null) return a.ok !== 1 && a.host_backed_fail === "cache";
  return (
    a.ok !== 1 &&
    a.oom !== 1 &&
    a.spill !== 1 &&
    a.gen_tps != null &&
    a.kv_host_backed_frac != null &&
    !failedForHostBackedLayers(a)
  );
}

// The measured spill for one load (shared/gpuSpill.ts): how much of what
// llama.cpp put on the GPU this process's dedicated VRAM did not hold. "—" on a
// load that could not be measured, including every older worker's row.
function spillCell(
  a: Pick<
    ProbeAttemptDto,
    "gpu_buffers_mib" | "gpu_in_system_ram_mib" | "gpu_spill_jitter_mib" | "spill_ready_mib" | "spill_work_mib" | "load_kind"
  >
): {
  text: string;
  warn: boolean;
  title: string;
} {
  if (a.load_kind === "claim_stop") {
    return {
      text: "—",
      warn: false,
      title: "Not judged: the claim did not fit free VRAM, so the load stopped before any request.",
    };
  }
  if (a.gpu_in_system_ram_mib == null || a.gpu_buffers_mib == null) {
    return {
      text: "—",
      warn: false,
      title:
        "Not measured: this load had no llama.cpp buffer report or no per-process VRAM reading (an older worker, or a platform such as Metal that reports no per-process memory).",
    };
  }
  const jitter = a.gpu_spill_jitter_mib ?? 0;
  const inRam = a.gpu_in_system_ram_mib;
  const spilled = inRam > jitter;
  return {
    text: spilled ? mib(inRam) : "none",
    warn: spilled,
    title:
      `llama.cpp put ${mib(a.gpu_buffers_mib)} on the GPU; this process's dedicated VRAM held ` +
      `${mib(a.gpu_buffers_mib - inRam)} once the model loaded` +
      (jitter > 0 ? `, its readings moving by ${mib(jitter)}` : "") +
      ". " +
      (spilled ? `The other ${mib(inRam)} is being served from system RAM.` : "All of it is in VRAM.") +
      (a.spill_ready_mib != null || a.spill_work_mib != null
        ? ` By phase, from once-a-second readings: after load ${signedMib(a.spill_ready_mib)}, during prompt and generation ${signedMib(a.spill_work_mib)} -- the worse one decides.`
        : ""),
  };
}

function signedMib(value: number | null): string {
  return value == null ? "not read" : `${value > 0 ? "+" : ""}${Math.round(value).toLocaleString()} MiB`;
}

// How much a spilled row had in system RAM, measured where possible and the
// old shared reading otherwise -- for picking the worst one to quote.
function spilledMib(a: Pick<ProbeAttemptDto, "gpu_in_system_ram_mib" | "vram_shared_peak_mib">): number {
  return a.gpu_in_system_ram_mib ?? a.vram_shared_peak_mib ?? 0;
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

// The probe's second target on one load: llama.cpp's GPU claim against what
// --list-devices reported free before the probe's first load. "—" on a rung
// from a worker predating the reading, or one that printed no buffers.
function claimFitCell(
  a: Pick<ProbeAttemptDto, "gpu_buffers_mib" | "list_devices_free_mib" | "gen_tps" | "load_kind" | "claim_fits_free">
): {
  text: string;
  warn: boolean;
  title: string;
} {
  const free = a.list_devices_free_mib;
  if (free == null) {
    return { text: "—", warn: false, title: "No --list-devices reading for this probe (an older worker, or a build without --list-devices)." };
  }
  if (a.load_kind === "error") {
    return {
      text: "—",
      warn: false,
      title: "This load failed for a reason that is not memory (it never became ready, or crashed without running out of memory), so it says nothing about whether the claim fits.",
    };
  }
  if (a.gpu_buffers_mib == null) {
    return a.gen_tps == null
      ? { text: "no", warn: true, title: "The load never ran far enough to report its buffers, so it did not fit." }
      : { text: "—", warn: false, title: "This build printed no GPU buffer sizes, so the claim is unknown." };
  }
  const totalFits = a.gpu_buffers_mib < free;
  const diff = Math.abs(free - a.gpu_buffers_mib);
  const title = `llama.cpp claimed ${mib(a.gpu_buffers_mib)} on the GPU; --list-devices reported ${mib(free)} free before the probe's first load.`;
  // Judged per device by the worker: the total can have room while one GPU is full.
  if (a.claim_fits_free === 0 && totalFits) {
    return { text: "over on one GPU", warn: true, title: `${title} In total that fits, but at least one GPU's own claim did not fit its own free memory.` };
  }
  return {
    text: totalFits ? `${mib(diff)} under` : `${mib(diff)} over`,
    warn: !totalFits,
    title,
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
  // only ever appear on a machine where a load actually spilled.
  const layerSpills = attempts.filter((a) => failedForHostBackedLayers(a) && a.ngl != null);
  const lastClean = attempts
    .filter(
      (a) =>
        a.ok === 1 &&
        a.vram_discrepancy !== 1 &&
        a.ngl != null &&
        (a.gpu_in_system_ram_mib != null || a.vram_shared_peak_mib != null)
    )
    .reduce<ProbeAttemptDto | null>((best, a) => (best == null || a.ngl! > best.ngl! ? a : best), null);
  const worstSpill = layerSpills.reduce<ProbeAttemptDto | null>(
    (w, a) => (w == null || spilledMib(a) > spilledMib(w) ? a : w),
    null
  );
  // The smallest context whose buffers spilled -- the one nearest the boundary
  // the context search settled against.
  const cacheFailed = attempts
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
          Measured up to <span className="font-mono font-bold">{mib(spilledMib(worstSpill))}</span>{" "}
          {worstSpill.gpu_in_system_ram_mib != null ? "of llama.cpp's GPU memory in system RAM" : "in host memory"} on
          this probe. Layer counts above that boundary are not real GPU offload on this machine, so a maximum-offload
          result will not reflect its true GPU speed.
          {cliff?.ngl != null && (
            <>
              {" "}
              Prompt processing degrades earlier still, from{" "}
              <span className="font-mono font-bold">{cliff.ngl}</span> layers.
            </>
          )}
        </div>
      )}
      {cacheFailed && (
        <div className="mb-3 rounded-lg border border-warning/40 bg-warning-bg px-3 py-2 text-xs text-warning">
          <span className="font-bold">Past a point, the context's buffers do not fit in VRAM.</span> At{" "}
          <span className="font-mono font-bold">{cacheFailed.candidate_ctx.toLocaleString()}</span> tokens,{" "}
          {cacheFailed.gpu_in_system_ram_mib != null ? (
            <>
              <span className="font-mono font-bold">{mib(cacheFailed.gpu_in_system_ram_mib)}</span> of what llama.cpp put
              on the GPU was in system RAM, no more than that context's KV cache and compute buffer
            </>
          ) : (
            <>
              <span className="font-mono font-bold">{Math.round((cacheFailed.kv_host_backed_frac ?? 0) * 100)}%</span> of
              the memory that context added went to system RAM
            </>
          )}
          , so the load failed and the search tried smaller contexts. Each load here only runs a {PROBE_PROMPT_TOKENS}
          -token prompt and generates {PROBE_GEN_TOKENS} tokens, so that load can still look fast — memory in system RAM
          only slows generation down once the context actually fills.
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
              <th
                className="px-2 py-1.5 text-right"
                rowSpan={2}
                title="What llama.cpp reported putting on the GPU, minus what this process's dedicated VRAM actually held once the model loaded. Whatever is left over is being served from system RAM. Measured, not estimated: no limit is applied beyond the VRAM counter's own movement during the load."
              >
                in system RAM
              </th>
              <th
                className="px-2 py-1.5 text-right"
                rowSpan={2}
                title="llama.cpp's GPU claim against the free VRAM llama-server --list-devices reported before the probe's first load -- the probe's second target."
              >
                claim vs free
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
              const spill = spillCell(a);
              const claimFit = claimFitCell(a);
              // The second load of a point is its control (shared/probeLadder.ts).
              const isControl = attempts.some((b) => b.seq < a.seq && b.candidate_ctx === a.candidate_ctx && b.ngl === a.ngl);
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
                    className="px-2 py-1.5 text-right font-mono text-muted"
                    title={
                      a.vram_shared_peak_mib != null
                        ? `${mib(a.vram_shared_peak_mib)} of this load's memory was system RAM the OS backed as GPU-accessible ` +
                          "memory (Windows WDDM \"Shared Usage\" / Linux amdgpu GTT). Part of that is ordinary driver overhead, " +
                          "which is why a spill is judged in the \"in system RAM\" column instead."
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
                    className={`px-2 py-1.5 text-right font-mono ${spill.warn ? "font-bold text-warning" : "text-muted"}`}
                    title={spill.title}
                  >
                    {spill.text}
                  </td>
                  <td
                    className={`px-2 py-1.5 text-right font-mono ${claimFit.warn ? "font-bold text-warning" : "text-muted"}`}
                    title={claimFit.title}
                  >
                    {claimFit.text}
                  </td>
                  <td className="px-2 py-1.5 text-muted">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {a.load_kind === "error" ? (
                        <span
                          title={a.error ?? undefined}
                          className="rounded-full bg-surface px-2 py-0.5 text-[10px] font-bold text-muted"
                        >
                          failed — not a memory verdict
                        </span>
                      ) : a.load_kind === "claim_stop" ? (
                        <span
                          title={a.error ?? undefined}
                          className="rounded-full bg-surface px-2 py-0.5 text-[10px] font-bold text-muted"
                        >
                          claim over free — not generated
                        </span>
                      ) : a.ok ? (
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
                      {isControl && (
                        <span
                          title="The second load of this point: the control that confirms (or overturns) its no-spill verdict."
                          className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent"
                        >
                          control
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
        memory instead of real dedicated VRAM. It's blank when no such counter exists for this worker's backend, which
        is different from a confirmed zero.
      </p>
      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        <b className="text-fg">In system RAM</b> is measured, not estimated: llama.cpp logs exactly how much it put on
        the GPU — weights, KV cache, compute buffer — and the OS reports how much of this process actually sits in
        dedicated VRAM. Whatever VRAM does not hold is being served from system RAM, and the load{" "}
        <b className="text-fg">fails</b>; a clean load reads <b className="text-fg">none</b>. No limit is involved
        beyond the VRAM counter's own movement during the load. The failure names what spilled: more than the
        context's whole KV cache and compute buffer is <b className="text-warning">layers in system RAM</b>, and the
        search backs off to fewer layers; anything less is <b className="text-warning">cache in system RAM</b>, and the
        search tries a smaller context. <b className="text-warning">⚠ possible VRAM fallback</b> appears only where this
        could not be measured, as an inference from the estimate. <b className="text-fg">Speeds</b> come from the same
        fixed {PROBE_EXERCISED_TOKENS}-token workload on every row, so they say whether a configuration runs — not how
        fast it is at the context beside them. There is no minimum rate: a slow load is reported with its rate, and only
        one that generates nothing at all fails.
      </p>
    </div>
  );
}
