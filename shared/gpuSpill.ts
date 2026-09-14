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
