import si from "systeminformation";
import { readGpuMemory } from "./vram.js";
import type { Backend, GpuMemoryAccuracyLevel, GpuMemoryMeasurementSource } from "../../shared/types.js";

export interface SampleStats {
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  ram_avg_mib: number;
  vram_avg_mib: number | null;
  // Both peak and avg get the *same* accuracy/source -- they're two views
  // over the same sample stream (see MemorySampler's worst-tier-wins
  // tracking below), not independently measured.
  vram_peak_accuracy: GpuMemoryAccuracyLevel;
  vram_peak_source: GpuMemoryMeasurementSource | null;
  vram_avg_accuracy: GpuMemoryAccuracyLevel;
  vram_avg_source: GpuMemoryMeasurementSource | null;
}

export interface SampleCurrent {
  ram_mib: number;
  vram_mib: number | null;
}

export interface FreeMemoryBaseline {
  ram_free_before_mib: number;
  vram_free_before_mib: number | null;
  vram_free_before_accuracy: GpuMemoryAccuracyLevel;
  vram_free_before_source: GpuMemoryMeasurementSource | null;
  system_memory_total_mib: number | null;
  gpu_memory_total_mib: number | null;
  gpu_memory_total_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_total_source: GpuMemoryMeasurementSource | null;
}

const BYTES_PER_MIB = 1024 * 1024;

// One-time snapshot taken right before a llama-bench process spawns -- not
// part of MemorySampler's own interval loop since it's a single baseline
// reading, not something to keep resampling.
export async function captureFreeMemoryBaseline(backend: Backend): Promise<FreeMemoryBaseline> {
  let ramFreeMib = 0;
  let systemMemoryTotalMib: number | null = null;
  try {
    const mem = await si.mem();
    ramFreeMib = Math.round(mem.available / BYTES_PER_MIB);
    systemMemoryTotalMib = Math.round(mem.total / BYTES_PER_MIB);
  } catch {
    /* best-effort */
  }

  // No pid yet (the process hasn't spawned) -- readGpuMemory's "used" side
  // is therefore always its whole-adapter/non-process-specific tier here
  // (see vram.ts's readCudaGpuMemory, the only backend with a better tier at
  // all, which only attempts it when a pid is given). free = total - used
  // can't be more accurate than either input, and since "used" is never
  // better than "total" in this pid-less case for any backend implemented
  // here, "used"'s own accuracy/source already correctly describes the pair
  // -- no separate worst-of-two comparison needed.
  const gpuMemory = await readGpuMemory(backend, undefined);
  const vramFreeMib =
    gpuMemory.total.mib != null && gpuMemory.used.mib != null
      ? Math.max(0, gpuMemory.total.mib - gpuMemory.used.mib)
      : null;

  return {
    ram_free_before_mib: ramFreeMib,
    vram_free_before_mib: vramFreeMib,
    vram_free_before_accuracy: vramFreeMib != null ? gpuMemory.used.accuracy : "unavailable",
    vram_free_before_source: vramFreeMib != null ? gpuMemory.used.source : null,
    system_memory_total_mib: systemMemoryTotalMib,
    gpu_memory_total_mib: gpuMemory.total.mib,
    gpu_memory_total_accuracy: gpuMemory.total.accuracy,
    gpu_memory_total_source: gpuMemory.total.source,
  };
}

// How many RAM ticks to let pass between VRAM probes. Windows' generic path
// in vram.ts costs ~1-1.5s per call (confirmed live -- a real PDH subsystem
// cost, not just process-spawn overhead), which is too slow to run on every
// 2s RAM tick without risking sample() calls piling up on each other; the
// other platforms'/backends' probes are much cheaper but there's no reason
// to give them a different cadence just because they could afford one.
const VRAM_SAMPLE_EVERY_N_TICKS = 3;

// Lower = more trustworthy. Used to track the single worst tier seen across
// every VRAM sample in an item's run -- so a CUDA item where most samples
// hit the exact/process-specific path but a couple fell back to the
// whole-adapter reading (e.g. nvidia-smi's compute-apps list hadn't caught
// up yet) reports its aggregate honestly as "high", not an overclaimed
// "exact" from a partial hit rate.
const ACCURACY_RANK: Record<GpuMemoryAccuracyLevel, number> = {
  exact: 0,
  high: 1,
  estimated: 2,
  unavailable: 3,
};

export class MemorySampler {
  private timer: NodeJS.Timeout | null = null;
  private backend: Backend = "cpu";
  private ramPeakBytes = 0;
  private vramPeakBytes = 0;
  private ramCurrentBytes = 0;
  private vramCurrentBytes = 0;
  private ramSumBytes = 0;
  private ramSampleCount = 0;
  private vramSumBytes = 0;
  private vramSampleCount = 0;
  private vramMeasured = false;
  private vramTickCount = 0;
  private vramWorstAccuracy: GpuMemoryAccuracyLevel = "exact";
  private vramWorstSource: GpuMemoryMeasurementSource | null = null;
  private pid: number | undefined;

  start(pid: number | undefined, backend: Backend, intervalMs = 2000): void {
    this.pid = pid;
    this.backend = backend;
    this.ramPeakBytes = 0;
    this.vramPeakBytes = 0;
    this.ramCurrentBytes = 0;
    this.vramCurrentBytes = 0;
    this.ramSumBytes = 0;
    this.ramSampleCount = 0;
    this.vramSumBytes = 0;
    this.vramSampleCount = 0;
    this.vramMeasured = false;
    this.vramTickCount = 0;
    this.vramWorstAccuracy = "exact";
    this.vramWorstSource = null;
    this.sample();
    this.timer = setInterval(() => this.sample(), intervalMs);
  }

  stop(): SampleStats {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    return this.stats;
  }

  get stats(): SampleStats {
    const accuracy = this.vramMeasured ? this.vramWorstAccuracy : "unavailable";
    const source = this.vramMeasured ? this.vramWorstSource : null;
    return {
      ram_peak_mib: Math.round(this.ramPeakBytes / BYTES_PER_MIB),
      vram_peak_mib: this.vramMeasured ? Math.round(this.vramPeakBytes / BYTES_PER_MIB) : null,
      ram_avg_mib:
        this.ramSampleCount > 0 ? Math.round(this.ramSumBytes / this.ramSampleCount / BYTES_PER_MIB) : 0,
      vram_avg_mib:
        this.vramMeasured && this.vramSampleCount > 0
          ? Math.round(this.vramSumBytes / this.vramSampleCount / BYTES_PER_MIB)
          : null,
      vram_peak_accuracy: accuracy,
      vram_peak_source: source,
      vram_avg_accuracy: accuracy,
      vram_avg_source: source,
    };
  }

  // Latest sample rather than the running max/avg -- for live progress ticks
  // that want "what's it using right now".
  get current(): SampleCurrent {
    return {
      ram_mib: Math.round(this.ramCurrentBytes / BYTES_PER_MIB),
      vram_mib: this.vramMeasured ? Math.round(this.vramCurrentBytes / BYTES_PER_MIB) : null,
    };
  }

  private async sample(): Promise<void> {
    try {
      let ramBytes = 0;
      if (this.pid != null) {
        const procs = await si.processes();
        const proc = procs.list.find((p) => p.pid === this.pid);
        // systeminformation reports memRss in KiB (confirmed against its own
        // source: Windows divides WorkingSetSize bytes by 1024, the ps-based
        // paths read ps's native KB rss column directly).
        ramBytes = (proc?.memRss ?? 0) * 1024;
      } else {
        // No child pid known (shouldn't normally happen) -- fall back to
        // whole-machine active memory rather than reporting nothing.
        const mem = await si.mem();
        ramBytes = mem.active;
      }
      this.ramCurrentBytes = ramBytes;
      if (ramBytes > this.ramPeakBytes) this.ramPeakBytes = ramBytes;
      this.ramSumBytes += ramBytes;
      this.ramSampleCount++;

      const dueForVram = this.vramTickCount % VRAM_SAMPLE_EVERY_N_TICKS === 0;
      this.vramTickCount++;
      if (!dueForVram) return;

      try {
        const { used } = await readGpuMemory(this.backend, this.pid);
        // used.mib === null means "couldn't measure" (missing tool/driver/
        // permission, or the cpu backend's unconditional short-circuit); 0
        // is a legitimate reading and must still count.
        if (used.mib != null) {
          const vramBytes = used.mib * BYTES_PER_MIB;
          this.vramMeasured = true;
          this.vramCurrentBytes = vramBytes;
          if (vramBytes > this.vramPeakBytes) this.vramPeakBytes = vramBytes;
          this.vramSumBytes += vramBytes;
          this.vramSampleCount++;
          if (ACCURACY_RANK[used.accuracy] > ACCURACY_RANK[this.vramWorstAccuracy]) {
            this.vramWorstAccuracy = used.accuracy;
            this.vramWorstSource = used.source;
          }
        }
      } catch {
        /* VRAM visibility varies by OS/vendor/driver; best-effort */
      }
    } catch {
      /* sampler is non-fatal */
    }
  }
}
