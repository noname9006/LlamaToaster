// The Serve page's memory bars: each of VRAM and RAM split into this app, other
// apps and free, with llama.cpp's own account of what the app's share holds.
//
// Measured with Qwen 1B fully offloaded at ctx 8192: adapter used 1901, before load
// 791, process dedicated 1110; logged buffers 863 + 192 + 43 = 1098. So process
// dedicated is the app's share, and the logged buffers explain it.
//
// A GPU read is slow (first ~4s, warm ~1s on Windows) and the GPU counter query
// disturbs inference timing, so callers only do this while someone is watching.

import { readFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { platform } from "node:os";
import si from "systeminformation";
import type { Backend } from "../../../shared/types.js";
import { isSharedMemoryGpu } from "../../../shared/types.js";
import type { DaemonMemorySnapshot } from "../../../shared/daemonTypes.js";
import { measureGpuSpill, UNMEASURED_GPU_SPILL, type GpuSpillInput, type GpuSpillVerdict } from "../../../shared/gpuSpill.js";
import { parseGpuBufferReport } from "../bench.js";
import { readGpuMemory } from "../vram.js";
import { runPowerShell } from "../psSession.js";
import type { LoadReport } from "./logParse.js";

const execFileAsync = promisify(execFile);
const BYTES_PER_MIB = 1024 * 1024;

/** This process's resident memory, in MiB. Null when the OS will not say. */
export async function readProcessRssMib(pid: number): Promise<number | null> {
  try {
    const os = platform();
    if (os === "linux") {
      const status = await readFile(`/proc/${pid}/status`, "utf8");
      const m = /^VmRSS:\s+(\d+)\s+kB/m.exec(status);
      return m ? Math.round(Number(m[1]) / 1024) : null;
    }
    if (os === "win32") {
      const script = `(Get-Process -Id ${pid}).WorkingSet64`;
      const out = await runPowerShell(script, 5_000, () =>
        execFileAsync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], { timeout: 5_000, windowsHide: true })
      );
      const n = Number(out.trim());
      return Number.isFinite(n) && n > 0 ? Math.round(n / BYTES_PER_MIB) : null;
    }
    const { stdout } = await execFileAsync("ps", ["-o", "rss=", "-p", String(pid)], { timeout: 5_000 });
    const kb = Number(stdout.trim());
    return Number.isFinite(kb) && kb > 0 ? Math.round(kb / 1024) : null;
  } catch {
    return null;
  }
}

export interface MemoryInputs {
  at: number;
  unified: boolean;
  gpuName: string | null;
  gpuCount?: number;
  ramTotalMib: number | null;
  ramAvailableMib: number | null;
  /** llama-server's resident memory, or null with nothing running or no reading. */
  childRssMib: number | null;
  childRunning: boolean;
  workerRssMib: number;
  gpuTotalMib: number | null;
  gpuUsedMib: number | null;
  gpuProcessMib: number | null;
  report: LoadReport | null;
  spill: { measured: boolean; inSystemRamMib: number | null; spilled: boolean };
}

function nonNegative(n: number): number {
  return Math.max(0, Math.round(n));
}

/** Pure, so the arithmetic and the "no reading" cases are testable anywhere. */
export function buildMemorySnapshot(i: MemoryInputs): DaemonMemorySnapshot {
  const ramUsed = i.ramTotalMib != null && i.ramAvailableMib != null ? i.ramTotalMib - i.ramAvailableMib : null;
  // With llama-server running but unreadable, the app's share is unknown: the
  // worker's own RSS alone would read as "the app is small".
  const ramApp = i.childRunning && i.childRssMib == null ? null : i.workerRssMib + (i.childRssMib ?? 0);
  const ram = {
    total: i.ramTotalMib,
    app: ramApp,
    other: ramUsed != null && ramApp != null ? nonNegative(ramUsed - ramApp) : null,
    free: i.ramAvailableMib,
  };

  let vram: DaemonMemorySnapshot["vram"] = null;
  if (!i.unified && i.gpuTotalMib != null) {
    // Nothing running means nothing of ours on the GPU, which is a real zero.
    const app = i.childRunning ? i.gpuProcessMib : 0;
    vram = {
      total: i.gpuTotalMib,
      app,
      other: i.gpuUsedMib != null && app != null ? nonNegative(i.gpuUsedMib - app) : null,
      free: i.gpuUsedMib != null ? nonNegative(i.gpuTotalMib - i.gpuUsedMib) : null,
    };
  }

  return {
    at: i.at,
    unified: i.unified,
    gpu_name: i.gpuName,
    gpu_count: i.gpuCount ?? (i.gpuName ? 1 : 0),
    vram,
    ram,
    breakdown: {
      weights_gpu: i.report?.weightsGpuMib ?? null,
      kv_gpu: i.report?.kvGpuMib ?? null,
      compute_gpu: i.report?.computeGpuMib ?? null,
      weights_host: i.report?.weightsHostMib ?? null,
      compute_host: i.report?.computeHostMib ?? null,
      prompt_cache: i.report?.promptCacheMib ?? null,
      worker: i.workerRssMib,
    },
    spill: { measured: i.spill.measured, in_system_ram_mib: i.spill.inSystemRamMib, spilled: i.spill.spilled },
  };
}

export interface HardwareLike {
  gpu?: { model?: string; vendor?: string; vram_dynamic?: boolean }[];
  unified_memory?: boolean;
}

/** Which GPU this machine serves on. An integrated GPU listed first, which is common,
 * must not make a machine with a discrete card read as unified memory. */
export function pickGpu(hardware: HardwareLike, backend: Backend): { name: string | null; count: number; unified: boolean } {
  const gpus = hardware.gpu ?? [];
  const discrete = gpus.find((g) => !isSharedMemoryGpu(g, hardware.unified_memory));
  const unified = hardware.unified_memory === true || (gpus.length > 0 && !discrete && backend !== "cpu");
  return { name: (discrete ?? gpus[0])?.model ?? null, count: gpus.length, unified };
}

// Readings of this process's dedicated VRAM kept for the spill check.
const MAX_DEDICATED_READINGS = 8;
// The counter is read in whole MiB, so it cannot tell a smaller gap from rounding.
const READING_RESOLUTION_MIB = 1;

/** The tolerance is the counter's own movement over these readings, as in a context
 * test. With fewer than two there is no movement to judge by, so nothing is claimed. */
export function spillFromReadings(buffers: GpuSpillInput["buffers"], readings: number[]): GpuSpillVerdict {
  if (readings.length < 2) return UNMEASURED_GPU_SPILL;
  const highest = Math.max(...readings);
  const movement = highest - Math.min(...readings);
  return measureGpuSpill({ buffers, dedicatedMib: highest, dedicatedJitterMib: movement + READING_RESOLUTION_MIB });
}

export async function readDaemonMemory(input: {
  backend: Backend;
  hardware: HardwareLike;
  pid: number | null;
  report: LoadReport | null;
  /** Earlier readings of this load, which this appends to. Absent: no spill check,
   * because the model is not settled (it is loading, waking or asleep). */
  dedicatedReadings?: number[];
}): Promise<DaemonMemorySnapshot> {
  const { name: gpuName, count: gpuCount, unified } = pickGpu(input.hardware, input.backend);
  let ramTotalMib: number | null = null;
  let ramAvailableMib: number | null = null;
  try {
    const mem = await si.mem();
    ramTotalMib = Math.round(mem.total / BYTES_PER_MIB);
    ramAvailableMib = Math.round(mem.available / BYTES_PER_MIB);
  } catch {
    // Reads as "no figure" on the page.
  }
  const childRssMib = input.pid != null ? await readProcessRssMib(input.pid) : null;
  const reading = !unified && input.backend !== "cpu" ? await readGpuMemory(input.backend, input.pid ?? undefined) : null;
  const buffers = input.report ? parseGpuBufferReport(input.report.bufferLines.join("\n")) : null;
  // Same rule as a context test: with mlock, CPU weights sit in pinned host
  // buffers that WDDM counts as shared. Spill here only compares dedicated VRAM
  // against device buffers, which do not include those.
  const readings = input.dedicatedReadings;
  const dedicated = reading?.process?.mib;
  if (readings && dedicated != null) {
    readings.push(dedicated);
    if (readings.length > MAX_DEDICATED_READINGS) readings.shift();
  }
  const spill = readings ? spillFromReadings(buffers, readings) : UNMEASURED_GPU_SPILL;
  return buildMemorySnapshot({
    at: Date.now(),
    unified,
    gpuName,
    gpuCount,
    ramTotalMib,
    ramAvailableMib,
    childRssMib,
    childRunning: input.pid != null,
    workerRssMib: Math.round(process.memoryUsage().rss / BYTES_PER_MIB),
    gpuTotalMib: reading?.total.mib ?? null,
    gpuUsedMib: reading?.used.mib ?? null,
    gpuProcessMib: reading?.process?.mib ?? null,
    report: input.report,
    spill,
  });
}
