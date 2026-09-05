import si from "systeminformation";
import { readGpuMemory } from "./vram.js";
import { readGpuSensors, SensorSampleBuffer } from "./sensors.js";
import type {
  Backend,
  GpuMemoryAccuracyLevel,
  GpuMemoryMeasurementSource,
  SensorMeasurementSource,
} from "../../shared/types.js";

export interface SampleStats {
  // The benchmark process's own RAM usage (systeminformation's per-pid RSS).
  ram_peak_mib: number;
  ram_avg_mib: number;
  // Whole-system RAM usage (si.mem().active -- every process combined),
  // sampled on the same interval.
  ram_total_peak_mib: number;
  ram_total_avg_mib: number;
  // Best-available VRAM usage stream (legacy meaning, kept for the existing
  // vram_avg/vram_peak columns, CSV names, and the VRAM-discrepancy
  // heuristic): per-process when the backend could read it that tick,
  // whole-adapter otherwise.
  vram_peak_mib: number | null;
  vram_avg_mib: number | null;
  // Whole-adapter VRAM usage (every process on the GPU combined) -- one
  // accuracy/source pair shared by avg+peak, since they're two views over
  // the same sample stream (worst-tier-wins tracking below).
  vram_total_used_peak_mib: number | null;
  vram_total_used_avg_mib: number | null;
  vram_total_used_accuracy: GpuMemoryAccuracyLevel;
  vram_total_used_source: GpuMemoryMeasurementSource | null;
  // The benchmark process's own VRAM usage (only when the backend exposes a
  // per-process reading -- nvidia-smi compute-apps, Windows GPU Process
  // Memory, rocm-smi --showpids / amdgpu fdinfo). Same shared
  // accuracy/source pair convention as the whole-adapter stream above.
  vram_process_peak_mib: number | null;
  vram_process_avg_mib: number | null;
  vram_process_accuracy: GpuMemoryAccuracyLevel;
  vram_process_source: GpuMemoryMeasurementSource | null;
  // The SAME process's system-RAM-backed GPU allocation -- vram.ts's
  // GpuMemoryReading.processShared (Windows WDDM "Shared Usage", or Linux
  // amdgpu's GTT domain). Null wherever the platform/backend has no such
  // counter at all (see that field's own doc comment for which those are),
  // never a false "0 spilled".
  vram_process_shared_peak_mib: number | null;
  vram_process_shared_avg_mib: number | null;
  vram_process_shared_accuracy: GpuMemoryAccuracyLevel;
  vram_process_shared_source: GpuMemoryMeasurementSource | null;
  vram_peak_accuracy: GpuMemoryAccuracyLevel;
  vram_peak_source: GpuMemoryMeasurementSource | null;
  vram_avg_accuracy: GpuMemoryAccuracyLevel;
  vram_avg_source: GpuMemoryMeasurementSource | null;
  // BENCHMARKING_PLAN_V8.md M6 -- adapter clock/temperature telemetry
  // piggybacked on the VRAM tick (every third 2s sampler tick, ~6s). The
  // sample SERIES is part of the spec, not a nice-to-have: aggregates alone
  // cannot see "sagged between halves". NULL on every sensorless backend.
  gpu_temp_c_max: number | null;
  gpu_clock_mhz_min: number | null;
  gpu_clock_samples: number[] | null;
  sensor_source: SensorMeasurementSource | null;
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
  private pid: number | undefined;
  private vramTickCount = 0;

  // The benchmark process's own RAM (per-pid RSS) -- the legacy stream.
  private ramPeakBytes = 0;
  private ramCurrentBytes = 0;
  private ramSumBytes = 0;
  private ramSampleCount = 0;
  // Whole-system RAM (si.mem().active -- every process combined).
  private ramTotalPeakBytes = 0;
  private ramTotalSumBytes = 0;
  private ramTotalSampleCount = 0;
  // Best-available VRAM (process when the backend could read it this tick,
  // else whole adapter) -- the legacy stream, keeps vram_peak_mib/vram_avg_mib
  // semantics stable for every existing consumer (CSV names, the
  // VRAM-discrepancy heuristic in worker/src/index.ts).
  private vramPeakBytes = 0;
  private vramCurrentBytes = 0;
  private vramSumBytes = 0;
  private vramSampleCount = 0;
  private vramMeasured = false;
  private vramWorstAccuracy: GpuMemoryAccuracyLevel = "exact";
  private vramWorstSource: GpuMemoryMeasurementSource | null = null;
  // Whole-adapter VRAM usage (every process on the GPU combined).
  private vramTotalPeakBytes = 0;
  private vramTotalSumBytes = 0;
  private vramTotalSampleCount = 0;
  private vramTotalMeasured = false;
  private vramTotalWorstAccuracy: GpuMemoryAccuracyLevel = "exact";
  private vramTotalWorstSource: GpuMemoryMeasurementSource | null = null;
  // The benchmark process's own VRAM usage.
  private vramProcessPeakBytes = 0;
  private vramProcessSumBytes = 0;
  private vramProcessSampleCount = 0;
  private vramProcessMeasured = false;
  private vramProcessWorstAccuracy: GpuMemoryAccuracyLevel = "exact";
  private vramProcessWorstSource: GpuMemoryMeasurementSource | null = null;
  // The same process's system-RAM-backed GPU allocation (WDDM Shared Usage /
  // Linux GTT) -- same tracking shape as vramProcess* above.
  private vramProcessSharedPeakBytes = 0;
  private vramProcessSharedSumBytes = 0;
  private vramProcessSharedSampleCount = 0;
  private vramProcessSharedMeasured = false;
  private vramProcessSharedWorstAccuracy: GpuMemoryAccuracyLevel = "exact";
  private vramProcessSharedWorstSource: GpuMemoryMeasurementSource | null = null;
  // M6 -- clock/temp samples, on the same tick as VRAM.
  private sensors = new SensorSampleBuffer();

  start(pid: number | undefined, backend: Backend, intervalMs = 2000): void {
    this.pid = pid;
    this.backend = backend;
    this.vramTickCount = 0;
    this.ramPeakBytes = 0;
    this.ramCurrentBytes = 0;
    this.ramSumBytes = 0;
    this.ramSampleCount = 0;
    this.ramTotalPeakBytes = 0;
    this.ramTotalSumBytes = 0;
    this.ramTotalSampleCount = 0;
    this.vramPeakBytes = 0;
    this.vramCurrentBytes = 0;
    this.vramSumBytes = 0;
    this.vramSampleCount = 0;
    this.vramMeasured = false;
    this.vramWorstAccuracy = "exact";
    this.vramWorstSource = null;
    this.vramTotalPeakBytes = 0;
    this.vramTotalSumBytes = 0;
    this.vramTotalSampleCount = 0;
    this.vramTotalMeasured = false;
    this.vramTotalWorstAccuracy = "exact";
    this.vramTotalWorstSource = null;
    this.vramProcessPeakBytes = 0;
    this.vramProcessSumBytes = 0;
    this.vramProcessSampleCount = 0;
    this.vramProcessMeasured = false;
    this.vramProcessWorstAccuracy = "exact";
    this.vramProcessWorstSource = null;
    this.vramProcessSharedPeakBytes = 0;
    this.vramProcessSharedSumBytes = 0;
    this.vramProcessSharedSampleCount = 0;
    this.vramProcessSharedMeasured = false;
    this.vramProcessSharedWorstAccuracy = "exact";
    this.vramProcessSharedWorstSource = null;
    this.sensors.reset();
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

  // M6's detection window is the TIMED work, not the spawn: server items
  // bracket sampling from the first timed request onward, since model-load
  // idle clocks inside the sampled span would otherwise mask or fabricate
  // sag. llama-bench items never call this -- their repeats are
  // binary-internal, so the whole item is the stated approximation.
  openSensorWindow(): void {
    this.sensors.openTimedWindow();
  }

  get stats(): SampleStats {
    // repeats only gates the throttle FLAG (derived at item end by the
    // caller, which knows the repeat count) -- the aggregates and the series
    // this getter returns are repeat-independent.
    const sensorReport = this.sensors.report(0);
    const accuracy = this.vramMeasured ? this.vramWorstAccuracy : "unavailable";
    const source = this.vramMeasured ? this.vramWorstSource : null;
    const totalAccuracy = this.vramTotalMeasured ? this.vramTotalWorstAccuracy : "unavailable";
    const totalSource = this.vramTotalMeasured ? this.vramTotalWorstSource : null;
    const processAccuracy = this.vramProcessMeasured ? this.vramProcessWorstAccuracy : "unavailable";
    const processSource = this.vramProcessMeasured ? this.vramProcessWorstSource : null;
    const processSharedAccuracy = this.vramProcessSharedMeasured ? this.vramProcessSharedWorstAccuracy : "unavailable";
    const processSharedSource = this.vramProcessSharedMeasured ? this.vramProcessSharedWorstSource : null;
    return {
      ram_peak_mib: Math.round(this.ramPeakBytes / BYTES_PER_MIB),
      vram_peak_mib: this.vramMeasured ? Math.round(this.vramPeakBytes / BYTES_PER_MIB) : null,
      ram_avg_mib:
        this.ramSampleCount > 0 ? Math.round(this.ramSumBytes / this.ramSampleCount / BYTES_PER_MIB) : 0,
      vram_avg_mib:
        this.vramMeasured && this.vramSampleCount > 0
          ? Math.round(this.vramSumBytes / this.vramSampleCount / BYTES_PER_MIB)
          : null,
      ram_total_peak_mib:
        this.ramTotalSampleCount > 0 ? Math.round(this.ramTotalPeakBytes / BYTES_PER_MIB) : 0,
      ram_total_avg_mib:
        this.ramTotalSampleCount > 0 ? Math.round(this.ramTotalSumBytes / this.ramTotalSampleCount / BYTES_PER_MIB) : 0,
      vram_total_used_peak_mib: this.vramTotalMeasured ? Math.round(this.vramTotalPeakBytes / BYTES_PER_MIB) : null,
      vram_total_used_avg_mib:
        this.vramTotalMeasured && this.vramTotalSampleCount > 0
          ? Math.round(this.vramTotalSumBytes / this.vramTotalSampleCount / BYTES_PER_MIB)
          : null,
      vram_total_used_accuracy: totalAccuracy,
      vram_total_used_source: totalSource,
      vram_process_peak_mib: this.vramProcessMeasured ? Math.round(this.vramProcessPeakBytes / BYTES_PER_MIB) : null,
      vram_process_avg_mib:
        this.vramProcessMeasured && this.vramProcessSampleCount > 0
          ? Math.round(this.vramProcessSumBytes / this.vramProcessSampleCount / BYTES_PER_MIB)
          : null,
      vram_process_accuracy: processAccuracy,
      vram_process_source: processSource,
      vram_process_shared_peak_mib: this.vramProcessSharedMeasured
        ? Math.round(this.vramProcessSharedPeakBytes / BYTES_PER_MIB)
        : null,
      vram_process_shared_avg_mib:
        this.vramProcessSharedMeasured && this.vramProcessSharedSampleCount > 0
          ? Math.round(this.vramProcessSharedSumBytes / this.vramProcessSharedSampleCount / BYTES_PER_MIB)
          : null,
      vram_process_shared_accuracy: processSharedAccuracy,
      vram_process_shared_source: processSharedSource,
      vram_peak_accuracy: accuracy,
      vram_peak_source: source,
      vram_avg_accuracy: accuracy,
      vram_avg_source: source,
      gpu_temp_c_max: sensorReport.gpu_temp_c_max,
      gpu_clock_mhz_min: sensorReport.gpu_clock_mhz_min,
      gpu_clock_samples: sensorReport.gpu_clock_samples,
      sensor_source: this.sensors.source,
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

      // Whole-system RAM used (every process combined) -- one extra cheap
      // si.mem() call per tick, independent of whether a child pid exists.
      const mem = await si.mem();
      if (typeof mem.active === "number" && Number.isFinite(mem.active)) {
        if (mem.active > this.ramTotalPeakBytes) this.ramTotalPeakBytes = mem.active;
        this.ramTotalSumBytes += mem.active;
        this.ramTotalSampleCount++;
      }

      const dueForVram = this.vramTickCount % VRAM_SAMPLE_EVERY_N_TICKS === 0;
      this.vramTickCount++;
      if (!dueForVram) return;

      try {
        const { used, process: processUsed, processShared } = await readGpuMemory(this.backend, this.pid);
        // used.mib === null means "couldn't measure" (missing tool/driver/
        // permission, or the cpu backend's unconditional short-circuit); 0
        // is a legitimate reading and must still count.
        if (used.mib != null) {
          const vramBytes = used.mib * BYTES_PER_MIB;
          // Legacy best-available stream: the process's own reading when the
          // backend produced one this tick, else the whole adapter.
          this.vramMeasured = true;
          this.vramCurrentBytes = vramBytes;
          if (vramBytes > this.vramPeakBytes) this.vramPeakBytes = vramBytes;
          this.vramSumBytes += vramBytes;
          this.vramSampleCount++;
          if (ACCURACY_RANK[used.accuracy] > ACCURACY_RANK[this.vramWorstAccuracy]) {
            this.vramWorstAccuracy = used.accuracy;
            this.vramWorstSource = used.source;
          }
          // Whole-adapter stream (always separate from the process stream --
          // they're different numbers and the report shows both).
          this.vramTotalMeasured = true;
          if (vramBytes > this.vramTotalPeakBytes) this.vramTotalPeakBytes = vramBytes;
          this.vramTotalSumBytes += vramBytes;
          this.vramTotalSampleCount++;
          if (ACCURACY_RANK[used.accuracy] > ACCURACY_RANK[this.vramTotalWorstAccuracy]) {
            this.vramTotalWorstAccuracy = used.accuracy;
            this.vramTotalWorstSource = used.source;
          }
        }
        // processUsed is absent (not null) when this backend/platform has no
        // per-process reading at all -- e.g. Metal -- or the driver hadn't
        // caught up to this process yet that tick. Either way the whole
        // process stream just doesn't grow this tick.
        if (processUsed?.mib != null) {
          const processBytes = processUsed.mib * BYTES_PER_MIB;
          this.vramProcessMeasured = true;
          if (processBytes > this.vramProcessPeakBytes) this.vramProcessPeakBytes = processBytes;
          this.vramProcessSumBytes += processBytes;
          this.vramProcessSampleCount++;
          if (ACCURACY_RANK[processUsed.accuracy] > ACCURACY_RANK[this.vramProcessWorstAccuracy]) {
            this.vramProcessWorstAccuracy = processUsed.accuracy;
            this.vramProcessWorstSource = processUsed.source;
          }
        }
        // processShared is absent (not null) wherever the platform/backend
        // has no such counter at all -- see GpuMemoryReading.processShared's
        // own doc comment for which those are.
        if (processShared?.mib != null) {
          const sharedBytes = processShared.mib * BYTES_PER_MIB;
          this.vramProcessSharedMeasured = true;
          if (sharedBytes > this.vramProcessSharedPeakBytes) this.vramProcessSharedPeakBytes = sharedBytes;
          this.vramProcessSharedSumBytes += sharedBytes;
          this.vramProcessSharedSampleCount++;
          if (ACCURACY_RANK[processShared.accuracy] > ACCURACY_RANK[this.vramProcessSharedWorstAccuracy]) {
            this.vramProcessSharedWorstAccuracy = processShared.accuracy;
            this.vramProcessSharedWorstSource = processShared.source;
          }
        }
      } catch {
        /* VRAM visibility varies by OS/vendor/driver; best-effort */
      }

      // M6 -- same cadence, same best-effort posture: a sensorless platform
      // simply contributes nothing and its columns stay NULL.
      try {
        this.sensors.add(await readGpuSensors(this.backend));
      } catch {
        /* no cross-platform sensor source; declared unavailable, never fatal */
      }
    } catch {
      /* sampler is non-fatal */
    }
  }
}
