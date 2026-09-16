// Whether the GPU memory llama.cpp allocated actually sits in VRAM.
//
// Measured, never inferred, and never held against a tuned limit. Every load
// carries two independent accounts of the same memory:
//
//   llama.cpp's  after allocating, it logs exactly how many MiB it put in each
//                GPU buffer -- weights, KV cache, recurrent state, compute,
//                output (worker/src/bench.ts's parseGpuBufferReport)
//   the OS's     how much of this process's GPU memory is in dedicated VRAM
//                (Windows WDDM "Dedicated Usage", amdgpu fdinfo, nvidia-smi
//                per process)
//
// When VRAM holds less than llama.cpp put on the GPU, the difference is being
// served from system RAM: the driver backed it there instead of failing the
// allocation. Measured on a Radeon RX 6600 XT (Vulkan, b10956,
// Qwen3.6-35B-A3B-UD-IQ4_NL), in MiB:
//
//   load                  GPU buffers   in VRAM   difference
//   1024 tok,  0 layers           275       280          -5   clean
//   1024 tok,  4 layers          2002      2017         -16   clean
//   1024 tok,  8 layers          3635      2806        +828   layers spilled
//   1024 tok, 15 layers          6462      4623       +1839   layers spilled
//   262144 tok, 0 layers         1069       822        +247   compute buffer spilled
//   262144 tok, 4 layers         3320      3085        +234   compute buffer spilled
//
// A clean load reads at or just below zero: VRAM holds every buffer plus a
// sliver of driver overhead. The rest of the process's overhead lives in SHARED
// memory (~430-480MiB on that machine), which is why the shared counter on its
// own could never tell overhead from spill. The only tolerance applied is the
// dedicated counter's own movement across this load's readings, so a
// difference smaller than the counter can resolve is not called a spill.
//
// The driver does not wait for VRAM to run out: at 8 layers above, 828MiB was
// already in system RAM with half the card free. "Was the GPU full" therefore
// cannot be a precondition -- only the accounting can say.
//
// Not yet confirmed on Windows CUDA, where the CUDA context's own allocation
// may land in dedicated VRAM outside llama.cpp's buffers and pull clean
// readings further below zero -- which would hide a spill smaller than that.

/** What llama.cpp itself reported placing on GPU devices for one load. */
export interface GpuBufferReport {
  /** Every buffer on a GPU device: weights, KV cache, recurrent state, compute, output. */
  deviceMib: number;
  /** The part of deviceMib a smaller context shrinks: KV cache and compute buffer. */
  contextMib: number;
  /** deviceMib split by llama.cpp device name ("Vulkan0", "CUDA1"), so a claim
   * can be held against each device's own free memory. */
  byDeviceMib?: Record<string, number>;
}

export interface GpuSpillInput {
  buffers: GpuBufferReport | null;
  /** This process's dedicated VRAM: the highest reading taken after the model finished loading. */
  dedicatedMib: number | null;
  /** How far those readings moved between each other -- the counter's own resolution on this load. */
  dedicatedJitterMib: number | null;
  /** The load already runs at the smallest context the probe ever tries, so no
   * smaller context exists to fix a spill: whatever spilled is the layers'.
   * Measured: a rerun of 8 layers at 1,024 tokens put only 189MiB in system
   * RAM -- less than that context's own 216MiB of buffers -- and was otherwise
   * filed as a spill a smaller context could fix. */
  atSmallestContext?: boolean;
}

export interface GpuSpillVerdict {
  /** Both accounts existed. False with no per-process VRAM reading (Metal), no
   * buffer report from the build, or nothing placed on a GPU at all -- never a
   * silent "clean". */
  measured: boolean;
  /** GPU buffers minus dedicated VRAM. Negative when VRAM holds more than the
   * buffers, which is the normal clean reading. */
  inSystemRamMib: number | null;
  /** The tolerance applied: the dedicated counter's own movement on this load. */
  jitterMib: number | null;
  spilled: boolean;
  /** What a smaller context can do about it. "layers": more is in system RAM
   * than this context's entire KV cache and compute buffer, so the weights are
   * affected whatever the context -- or the load is already at the smallest
   * context, so there is no smaller one to try. "cache": no more than those
   * buffers, so a smaller context can bring it back. Null when nothing spilled. */
  cause: "layers" | "cache" | null;
}

export const UNMEASURED_GPU_SPILL: GpuSpillVerdict = {
  measured: false,
  inSystemRamMib: null,
  jitterMib: null,
  spilled: false,
  cause: null,
};

export function measureGpuSpill(input: GpuSpillInput): GpuSpillVerdict {
  const { buffers, dedicatedMib } = input;
  if (buffers == null || buffers.deviceMib <= 0 || dedicatedMib == null) return UNMEASURED_GPU_SPILL;
  const inSystemRamMib = buffers.deviceMib - dedicatedMib;
  const jitterMib = Math.max(0, input.dedicatedJitterMib ?? 0);
  const excessMib = inSystemRamMib - jitterMib;
  if (excessMib <= 0) return { measured: true, inSystemRamMib, jitterMib, spilled: false, cause: null };
  return {
    measured: true,
    inSystemRamMib,
    jitterMib,
    spilled: true,
    cause: input.atSmallestContext || excessMib > buffers.contextMib ? "layers" : "cache",
  };
}

// --- Spill by phase ----------------------------------------------------------
//
// One dedicated-VRAM figure per load (its peak once loaded) cannot show memory
// that the driver moves to system RAM only once it is used. So the context test
// reads this process's dedicated VRAM about once a second for the whole load
// and judges two phases separately:
//
//   ready -- a short hold after the server reports ready, before any request.
//            Judged on its LAST reading: a driver still paging buffers in right
//            after load would otherwise read as a spill.
//   work  -- the prompt and generation. Judged on the MEDIAN reading, so one
//            reading taken mid-reshuffle cannot decide it.
//
// Each phase's tolerance is how far its own readings moved. The load's verdict
// is its worse phase. A phase with no readings is left out; with neither, the
// load is unmeasured.

export interface MemoryReading {
  /** Epoch ms the reading was taken (the end of the counter's own sample). */
  atMs: number;
  dedicatedMib: number | null;
}

export interface PhasedSpillInput {
  buffers: GpuBufferReport | null;
  readings: readonly MemoryReading[];
  /** The ready hold: from the server reporting ready to the first request. */
  ready: { fromMs: number; toMs: number } | null;
  /** From the first request sent to the last token received. */
  work: { fromMs: number; toMs: number } | null;
  atSmallestContext?: boolean;
}

export interface PhasedSpillVerdict extends GpuSpillVerdict {
  readyPhase: GpuSpillVerdict;
  workPhase: GpuSpillVerdict;
}

/** How late after the work window a reading may land and still belong to it:
 * the Windows counter takes about a second to produce each sample. The ready
 * hold gets no such allowance -- a reading landing after the first request went
 * out partly covers the prompt, and the hold's whole point is that nothing has
 * touched the memory yet. */
const READING_LAG_MS = 1000;

export function measurePhasedGpuSpill(input: PhasedSpillInput): PhasedSpillVerdict {
  const inWindow = (w: { fromMs: number; toMs: number } | null, lagMs: number) =>
    w == null
      ? []
      : input.readings
          .filter((r) => r.dedicatedMib != null && r.atMs > w.fromMs && r.atMs <= w.toMs + lagMs)
          .map((r) => r.dedicatedMib as number);
  const spread = (values: number[]) => (values.length > 1 ? Math.max(...values) - Math.min(...values) : 0);
  const judge = (dedicatedMib: number | null, values: number[]) =>
    measureGpuSpill({
      buffers: input.buffers,
      dedicatedMib,
      dedicatedJitterMib: spread(values),
      atSmallestContext: input.atSmallestContext,
    });

  const readyValues = inWindow(input.ready, 0);
  const workValues = inWindow(input.work, READING_LAG_MS);
  const readyPhase = judge(readyValues.length > 0 ? readyValues[readyValues.length - 1] : null, readyValues);
  const workPhase = judge(workValues.length > 0 ? median(workValues) : null, workValues);

  const excess = (v: GpuSpillVerdict) => (v.measured ? (v.inSystemRamMib ?? 0) - (v.jitterMib ?? 0) : -Infinity);
  const measured = [readyPhase, workPhase].filter((v) => v.measured);
  if (measured.length === 0) return { ...UNMEASURED_GPU_SPILL, readyPhase, workPhase };
  const worst = measured.reduce((a, b) => (excess(b) > excess(a) ? b : a));
  return { ...worst, readyPhase, workPhase };
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
