import si from "systeminformation";

export interface SampleStats {
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  ram_avg_mib: number;
  vram_avg_mib: number | null;
}

export interface SampleCurrent {
  ram_mib: number;
  vram_mib: number | null;
}

export interface FreeMemoryBaseline {
  ram_free_before_mib: number;
  vram_free_before_mib: number | null;
}

const BYTES_PER_MIB = 1024 * 1024;

// One-time snapshot taken right before a llama-bench process spawns -- not
// part of MemorySampler's own interval loop since it's a single baseline
// reading, not something to keep resampling.
export async function captureFreeMemoryBaseline(): Promise<FreeMemoryBaseline> {
  let ramFreeMib = 0;
  try {
    const mem = await si.mem();
    ramFreeMib = Math.round(mem.available / BYTES_PER_MIB);
  } catch {
    /* best-effort */
  }

  let vramFreeMib: number | null = null;
  try {
    const graphics = await si.graphics();
    let totalMib = 0;
    let usedMib = 0;
    let measured = false;
    for (const controller of graphics.controllers) {
      if (controller.memoryTotal == null || controller.memoryUsed == null) continue;
      totalMib += controller.memoryTotal;
      usedMib += controller.memoryUsed;
      measured = true;
    }
    if (measured) vramFreeMib = Math.max(0, Math.round(totalMib - usedMib));
  } catch {
    /* VRAM visibility on Vulkan/AMD is best-effort; ignore */
  }

  return { ram_free_before_mib: ramFreeMib, vram_free_before_mib: vramFreeMib };
}

export class MemorySampler {
  private timer: NodeJS.Timeout | null = null;
  private ramPeakBytes = 0;
  private vramPeakBytes = 0;
  private ramCurrentBytes = 0;
  private vramCurrentBytes = 0;
  private ramSumBytes = 0;
  private ramSampleCount = 0;
  private vramSumBytes = 0;
  private vramSampleCount = 0;
  private vramMeasured = false;
  private pid: number | undefined;

  start(pid: number | undefined, intervalMs = 2000): void {
    this.pid = pid;
    this.ramPeakBytes = 0;
    this.vramPeakBytes = 0;
    this.ramCurrentBytes = 0;
    this.vramCurrentBytes = 0;
    this.ramSumBytes = 0;
    this.ramSampleCount = 0;
    this.vramSumBytes = 0;
    this.vramSampleCount = 0;
    this.vramMeasured = false;
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
    return {
      ram_peak_mib: Math.round(this.ramPeakBytes / BYTES_PER_MIB),
      vram_peak_mib: this.vramMeasured ? Math.round(this.vramPeakBytes / BYTES_PER_MIB) : null,
      ram_avg_mib:
        this.ramSampleCount > 0 ? Math.round(this.ramSumBytes / this.ramSampleCount / BYTES_PER_MIB) : 0,
      vram_avg_mib:
        this.vramMeasured && this.vramSampleCount > 0
          ? Math.round(this.vramSumBytes / this.vramSampleCount / BYTES_PER_MIB)
          : null,
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

      try {
        const graphics = await si.graphics();
        let vramBytes = 0;
        for (const controller of graphics.controllers) {
          if (controller.memoryUsed == null) continue;
          // systeminformation reports memoryUsed in MiB.
          const used = controller.memoryUsed * BYTES_PER_MIB;
          if (used > vramBytes) vramBytes = used;
        }
        if (vramBytes > 0) {
          this.vramMeasured = true;
          this.vramCurrentBytes = vramBytes;
          if (vramBytes > this.vramPeakBytes) this.vramPeakBytes = vramBytes;
          this.vramSumBytes += vramBytes;
          this.vramSampleCount++;
        }
      } catch {
        /* VRAM visibility on Vulkan/AMD is best-effort; ignore */
      }
    } catch {
      /* sampler is non-fatal */
    }
  }
}
