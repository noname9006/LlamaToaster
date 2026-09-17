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
// That zero-based difference (measureGpuSpill) is no longer how the context test
// decides: it judges growth over an anchor load instead -- see "Spill as growth
// over an anchor" below for why zero is not a usable baseline. measureGpuSpill
// remains only as probeSucceeded's fallback for a caller that passes no verdict.
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

// --- Spill as growth over an anchor -----------------------------------------
//
// The context test's rule. The difference above assumes a clean load reads at
// or below zero, and on some builds it does not: on b11009 a 1,024-token load
// with nothing but the compute buffer on the GPU read +236MiB -- the whole
// buffer -- and failed every placement at that context. And the process's
// shared memory never reads zero: ~1GB of it appears with the first layer on
// the GPU and matches no llama.cpp buffer at all.
//
// So a load is judged against an ANCHOR, not against zero: the same model at 1
// layer and the smallest context, loaded first in the same probe (twice). What
// the anchor already holds -- driver overhead in shared memory, a compute
// buffer that never lands in VRAM -- cancels out, and only what changed since
// is judged. Two independent accounts of that change, per reading:
//
//   s = shared(n) - shared(A)                        shared memory grew
//   d = (claim(n) - claim(A)) - (ded(n) - ded(A))    claim grew, VRAM did not
//
// Real spill moves both by the same amount; overhead growing in shared moves s
// alone (dedicated still took the whole claim); a buffer moving back INTO VRAM
// drives d negative. Measured on the RX 6600 XT (b11009, Qwen3.8-27B, 1,024
// tokens, MiB): 2->4 layers s 0 / d -3; 4->9 s +145 / d +130; 9->20 s +716 /
// d +696; and at 262,144 tokens 5 layers s +20 / d -252 against the 1k anchor.
//
//   spill     = min(s, d)                (d alone where no shared counter exists)
//   tolerance = |s - d|                  how far the two accounts disagree
//             + range of s and d         how far this phase's readings moved
//             + anchor movement          how far the anchor's readings moved
//             + anchor noise             how far its second load differed
//   spilled   = spill > tolerance
//
// No constant anywhere: every term is a measurement of this probe.
//
// Phases: the ready hold (no request yet) is judged on its LAST reading, since
// a driver still paging buffers in right after load must not read as spill;
// the prompt and generation on the MEDIAN of paired readings, so a jump for a
// minority of the phase widens the tolerance instead of deciding it. The worse
// phase decides the load.

export interface MemoryReading {
  /** Epoch ms the reading was taken (the end of the counter's own sample). */
  atMs: number;
  dedicatedMib: number | null;
  /** This process's shared GPU memory from the SAME sample. Null or absent
   * where the platform has no such counter (nvidia-smi). */
  sharedMib?: number | null;
}

export interface PhaseWindow {
  fromMs: number;
  toMs: number;
}

/** One load's raw material for the rule. */
export interface MeasuredLoad {
  buffers: GpuBufferReport | null;
  readings: readonly MemoryReading[];
  /** The ready hold: from the server reporting ready to the first request. */
  ready: PhaseWindow | null;
  /** From the first request sent to the last token received. */
  work: PhaseWindow | null;
}

export interface AnchorPhase {
  dedicatedMib: number;
  /** Null where no shared counter exists. */
  sharedMib: number | null;
  /** How far the anchor's own readings moved in this phase. */
  movementMib: number;
  /** How far the anchor's second load differed from its first (0 until it runs). */
  noiseMib: number;
}

export interface SpillAnchor {
  claimMib: number;
  contextMib: number;
  ready: AnchorPhase | null;
  work: AnchorPhase | null;
}

/** How late after the work window a reading may land and still belong to it:
 * the Windows counter takes about a second to produce each sample. The ready
 * hold gets no such allowance -- a reading landing after the first request went
 * out partly covers the prompt, and the hold's whole point is that nothing has
 * touched the memory yet. */
const READING_LAG_MS = 1000;

type Phase = "ready" | "work";

function phaseReadings(load: MeasuredLoad, phase: Phase): MemoryReading[] {
  const w = phase === "ready" ? load.ready : load.work;
  const lag = phase === "ready" ? 0 : READING_LAG_MS;
  if (w == null) return [];
  return load.readings.filter((r) => r.dedicatedMib != null && r.atMs > w.fromMs && r.atMs <= w.toMs + lag);
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const range = (values: readonly number[]) => (values.length > 1 ? Math.max(...values) - Math.min(...values) : 0);

/** The value a phase is judged on: the hold's last reading, the work's median. */
function pick(values: readonly number[], phase: Phase): number {
  return phase === "ready" ? values[values.length - 1] : median(values);
}

/** Every reading has shared, or the phase is judged without it. */
function sharedSeries(readings: readonly MemoryReading[]): number[] | null {
  if (readings.length === 0 || readings.some((r) => r.sharedMib == null)) return null;
  return readings.map((r) => r.sharedMib as number);
}

function anchorPhase(load: MeasuredLoad, phase: Phase): AnchorPhase | null {
  const readings = phaseReadings(load, phase);
  if (readings.length === 0) return null;
  const dedicated = readings.map((r) => r.dedicatedMib as number);
  const shared = sharedSeries(readings);
  return {
    dedicatedMib: pick(dedicated, phase),
    sharedMib: shared ? pick(shared, phase) : null,
    movementMib: Math.max(range(dedicated), shared ? range(shared) : 0),
    noiseMib: 0,
  };
}

/** The anchor from its first load. Null when that load has no buffer report or
 * no reading in either phase -- every load of the probe is then unmeasured. */
export function buildSpillAnchor(first: MeasuredLoad): SpillAnchor | null {
  if (first.buffers == null || first.buffers.deviceMib <= 0) return null;
  const anchor: SpillAnchor = {
    claimMib: first.buffers.deviceMib,
    contextMib: first.buffers.contextMib,
    ready: anchorPhase(first, "ready"),
    work: anchorPhase(first, "work"),
  };
  return anchor.ready == null && anchor.work == null ? null : anchor;
}

/** The anchor with its second load folded in: per phase, the larger of that
 * load's |s| and |d| against the first, taken with no tolerance. A phase the
 * second load did not measure gets no noise term. */
export function withAnchorControl(anchor: SpillAnchor, second: MeasuredLoad): SpillAnchor {
  const bare: SpillAnchor = {
    ...anchor,
    ready: anchor.ready && { ...anchor.ready, movementMib: 0, noiseMib: 0 },
    work: anchor.work && { ...anchor.work, movementMib: 0, noiseMib: 0 },
  };
  const noiseOf = (phase: Phase): number => {
    const v = judgePhase(second, bare, phase, false);
    if (!v.measured) return 0;
    return Math.max(Math.abs(v.unlandedGrowthMib ?? 0), Math.abs(v.sharedGrowthMib ?? 0));
  };
  return {
    ...anchor,
    ready: anchor.ready && { ...anchor.ready, noiseMib: noiseOf("ready") },
    work: anchor.work && { ...anchor.work, noiseMib: noiseOf("work") },
  };
}

export interface GrowthSpillVerdict extends GpuSpillVerdict {
  /** s: shared memory grown since the anchor. Null where no shared counter exists. */
  sharedGrowthMib: number | null;
  /** d: claim grown since the anchor that dedicated VRAM did not take. */
  unlandedGrowthMib: number | null;
  /** How much of the claim's growth since the anchor is KV cache and compute
   * buffer -- what a smaller context can take back. */
  contextGrowthMib: number | null;
}

const UNMEASURED_GROWTH: GrowthSpillVerdict = {
  ...UNMEASURED_GPU_SPILL,
  sharedGrowthMib: null,
  unlandedGrowthMib: null,
  contextGrowthMib: null,
};

function judgePhase(load: MeasuredLoad, anchor: SpillAnchor, phase: Phase, atSmallestContext: boolean): GrowthSpillVerdict {
  const a = phase === "ready" ? anchor.ready : anchor.work;
  const readings = phaseReadings(load, phase);
  if (a == null || readings.length === 0 || load.buffers == null || load.buffers.deviceMib <= 0) return UNMEASURED_GROWTH;
  const claimGrowth = load.buffers.deviceMib - anchor.claimMib;
  // Paired per reading: s and d come from the same sample, never from medians of
  // two series that could describe different moments.
  const dSeries = readings.map((r) => claimGrowth - ((r.dedicatedMib as number) - a.dedicatedMib));
  const shared = a.sharedMib == null ? null : sharedSeries(readings);
  const sSeries = shared ? shared.map((v) => v - (a.sharedMib as number)) : null;
  const d = pick(dSeries, phase);
  const s = sSeries ? pick(sSeries, phase) : null;
  const spill = s == null ? d : Math.min(s, d);
  const tolerance =
    (s == null ? 0 : Math.abs(s - d)) + Math.max(range(dSeries), sSeries ? range(sSeries) : 0) + a.movementMib + a.noiseMib;
  const excess = spill - tolerance;
  const contextGrowthMib = load.buffers.contextMib - anchor.contextMib;
  const spilled = excess > 0;
  return {
    measured: true,
    inSystemRamMib: spill,
    jitterMib: tolerance,
    spilled,
    cause: !spilled ? null : atSmallestContext || excess > contextGrowthMib ? "layers" : "cache",
    sharedGrowthMib: s,
    unlandedGrowthMib: d,
    contextGrowthMib,
  };
}

export interface PhasedSpillVerdict extends GrowthSpillVerdict {
  readyPhase: GrowthSpillVerdict;
  workPhase: GrowthSpillVerdict;
}

/** One load against the anchor, both phases; the worse one decides. Unmeasured
 * without an anchor, never judged against zero instead. */
export function measureGrowthSpill(
  load: MeasuredLoad & { atSmallestContext?: boolean },
  anchor: SpillAnchor | null
): PhasedSpillVerdict {
  if (anchor == null) return { ...UNMEASURED_GROWTH, readyPhase: UNMEASURED_GROWTH, workPhase: UNMEASURED_GROWTH };
  const readyPhase = judgePhase(load, anchor, "ready", load.atSmallestContext === true);
  const workPhase = judgePhase(load, anchor, "work", load.atSmallestContext === true);
  const excess = (v: GpuSpillVerdict) => (v.measured ? (v.inSystemRamMib ?? 0) - (v.jitterMib ?? 0) : -Infinity);
  const measured = [readyPhase, workPhase].filter((v) => v.measured);
  if (measured.length === 0) return { ...UNMEASURED_GROWTH, readyPhase, workPhase };
  const worst = measured.reduce((x, y) => (excess(y) > excess(x) ? y : x));
  return { ...worst, readyPhase, workPhase };
}
