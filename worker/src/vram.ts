import { platform as osPlatform } from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, readdir } from "node:fs/promises";
import si from "systeminformation";
import type { Backend, GpuMemoryAccuracyLevel, GpuMemoryMeasurementSource } from "../../shared/types.js";

// Live VRAM usage has no single cross-platform source, and no single
// accuracy tier either -- systeminformation's own `memoryTotal`/`memoryUsed`
// fields (what sampler.ts used to rely on exclusively) are simply absent for
// some real vendor/OS combos -- confirmed live against a Radeon RX 6600 XT on
// Windows, where si.graphics() returns `vram: 8176` (a static total) but
// `memoryTotal`/`memoryUsed` both undefined, so nothing was ever reported.
// readGpuMemory below dispatches per declared backend (not just per OS) so
// each gets the collection strategy -- and the honest accuracy/provenance
// label -- it actually deserves. See shared/types.ts's
// GpuMemoryAccuracyLevel/GpuMemoryMeasurementSource for what the labels mean.

const execFileAsync = promisify(execFile);
// 5000 previously -- raised for headroom around Windows' Get-Counter path
// (readWindowsUsedMib below), whose cold-start cost (spawning powershell.exe
// + the "GPU Adapter Memory" performance-counter provider's own first-use
// init) was confirmed live to occasionally exceed 5s on a freshly started
// worker process, silently reporting vram_free_before as "unavailable" for
// that one call. worker/src/index.ts now also fires a warm-up call at
// startup so this mostly shouldn't matter in practice, but a wider margin
// here is a cheap second line of defense (every other exec call using this
// constant normally completes in well under 1s either way, so raising it
// doesn't meaningfully slow down a genuine "tool isn't installed" failure).
const EXEC_TIMEOUT_MS = 10_000;
const BYTES_PER_MIB = 1024 * 1024;

export interface GpuMemoryValue {
  mib: number | null;
  accuracy: GpuMemoryAccuracyLevel;
  source: GpuMemoryMeasurementSource | null;
}

export interface GpuMemoryReading {
  total: GpuMemoryValue;
  used: GpuMemoryValue;
}

function unavailable(): GpuMemoryValue {
  return { mib: null, accuracy: "unavailable", source: null };
}

function reading(
  mib: number | null,
  accuracy: GpuMemoryAccuracyLevel,
  source: GpuMemoryMeasurementSource
): GpuMemoryValue {
  return mib == null ? unavailable() : { mib, accuracy, source };
}

const UNAVAILABLE_READING: GpuMemoryReading = { total: unavailable(), used: unavailable() };

// The one entry point every caller (worker/src/sampler.ts) uses. `pid` is
// the spawned llama-bench/llama-server child's pid when known (the interval
// sampler has one; the pre-spawn baseline capture never does, since the
// process doesn't exist yet -- which is exactly why free_start can never be
// "exact"/process_gpu_usage for any backend, only model_avg/model_peak can).
export async function readGpuMemory(backend: Backend, pid: number | undefined): Promise<GpuMemoryReading> {
  try {
    // No subprocess call at all on the cpu backend -- both correctness (the
    // spec requires null here) and a real perf win, skipping the ~1-1.5s
    // Windows PDH cost (see readGenericGpuMemory below) on every CPU item.
    if (backend === "cpu") return UNAVAILABLE_READING;
    if (backend === "cuda") return await readCudaGpuMemory(pid);
    if (backend === "rocm") return await readRocmGpuMemory();
    if (backend === "metal" || osPlatform() === "darwin") return await readDarwinGpuMemory();
    return await readGenericGpuMemory();
  } catch {
    return UNAVAILABLE_READING;
  }
}

interface VramSample {
  totalMib: number | null;
  usedMib: number | null;
}

const NULL_SAMPLE: VramSample = { totalMib: null, usedMib: null };

// ---------------------------------------------------------------------------
// Generic OS-level path (vulkan, and any other/unrecognized backend string --
// sycl, opencl-adreno, ...): the "GPU Adapter Memory" performance counter
// category on Windows is vendor-agnostic (published by any WDDM driver --
// NVIDIA, AMD, Intel all expose it), which is why this is used instead of a
// vendor-specific tool even on an NVIDIA box running the vulkan backend.
// Confirmed live: `Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage'`
// returns real, per-adapter byte counts on this machine's Radeon RX 6600 XT
// with no elevation required, taking ~1-1.5s per call (a real PDH subsystem
// cost, not just process-spawn overhead -- reproduced identically from a
// warm, already-running PowerShell host). Labeled "high"/driver_reported_memory
// rather than the spec's own "estimated"/memory_budget_estimate Vulkan
// example: that example assumes a real VK_EXT_memory_budget Vulkan API call,
// which this environment can't implement (no native-addon compilation
// available) and which was never actually built -- what runs here is a
// genuine driver counter, a real reading, not a budget heuristic.
async function readGenericGpuMemory(): Promise<GpuMemoryReading> {
  const plat = osPlatform();
  let sample: VramSample = NULL_SAMPLE;
  if (plat === "win32") sample = await readWindowsVram();
  else if (plat === "linux") sample = await readLinuxVram();
  return {
    total: reading(sample.totalMib, "high", "driver_reported_memory"),
    used: reading(sample.usedMib, "high", "driver_reported_memory"),
  };
}

// No "Dedicated Limit" counter instance exists on this system to read total
// capacity the same way (confirmed live -- only Shared Usage/Dedicated
// Usage/Total Committed are published here, and that counter set is known to
// vary across Windows builds/drivers), so total still comes from
// systeminformation's static `vram`/`memoryTotal` fields, which -- unlike
// the live usage fields -- were confirmed working for this card. Cached
// after the first successful read since total capacity can't change during
// a worker process's lifetime.
let cachedWindowsTotalMib: number | null | undefined;

async function readWindowsVram(): Promise<VramSample> {
  const [totalMib, usedMib] = await Promise.all([readWindowsTotalMib(), readWindowsUsedMib()]);
  return { totalMib, usedMib };
}

async function readWindowsTotalMib(): Promise<number | null> {
  if (cachedWindowsTotalMib !== undefined) return cachedWindowsTotalMib;
  try {
    const graphics = await si.graphics();
    let totalMib = 0;
    for (const c of graphics.controllers) {
      const t = c.memoryTotal ?? c.vram;
      if (typeof t === "number" && t > 0) totalMib += t;
    }
    cachedWindowsTotalMib = totalMib > 0 ? Math.round(totalMib) : null;
  } catch {
    cachedWindowsTotalMib = null;
  }
  return cachedWindowsTotalMib;
}

async function readWindowsUsedMib(): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples | " +
          "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum",
      ],
      { timeout: EXEC_TIMEOUT_MS, windowsHide: true }
    );
    const bytes = Number(stdout.trim());
    return Number.isFinite(bytes) ? Math.round(bytes / BYTES_PER_MIB) : null;
  } catch {
    // No counter provider, no GPU, or the call timed out -- best-effort.
    return null;
  }
}

// Linux (generic/vulkan path): NVIDIA first (nvidia-smi ships with the
// driver package and is the standard, always-available source), falling
// back to amdgpu's own sysfs accounting -- exposed by the stock open-source
// kernel driver with no ROCm/rocm-smi install required. Neither of these
// has been run live (this session only has Windows hardware available);
// both are standard, documented interfaces widely relied on by other GPU
// monitoring tools (nvidia-smi's CSV output; amdgpu's mem_info_vram_* attrs
// are documented at kernel.org/doc/html/latest/gpu/amdgpu/thermal.html), but
// should be verified against a real Linux worker before being trusted the
// way the Windows path above has been.
async function readLinuxVram(): Promise<VramSample> {
  const nvidia = await readNvidiaSmiVram();
  if (nvidia) return nvidia;
  const amd = await readAmdgpuSysfsVram();
  if (amd) return amd;
  return NULL_SAMPLE;
}

async function readNvidiaSmiVram(): Promise<VramSample | null> {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      ["--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
      { timeout: EXEC_TIMEOUT_MS }
    );
    // One line per GPU -- only the first is read, matching this app's
    // existing single-GPU-per-worker assumption elsewhere.
    const line = stdout.trim().split("\n")[0] ?? "";
    const [totalStr, usedStr] = line.split(",").map((s) => s.trim());
    const totalMib = Number(totalStr);
    const usedMib = Number(usedStr);
    if (!Number.isFinite(totalMib) || !Number.isFinite(usedMib)) return null;
    return { totalMib, usedMib };
  } catch {
    // nvidia-smi missing (no NVIDIA GPU/driver on this box) is the expected
    // case on an AMD/Intel worker -- not an error, just try the next vendor.
    return null;
  }
}

const DRM_CARD_DIR = "/sys/class/drm";
const CARD_DIR_RE = /^card\d+$/;

async function readAmdgpuSysfsVram(): Promise<VramSample | null> {
  try {
    const entries = await readdir(DRM_CARD_DIR);
    for (const entry of entries) {
      if (!CARD_DIR_RE.test(entry)) continue; // skip cardN-<connector> subdirs
      const base = `${DRM_CARD_DIR}/${entry}/device`;
      try {
        const [totalRaw, usedRaw] = await Promise.all([
          readFile(`${base}/mem_info_vram_total`, "utf8"),
          readFile(`${base}/mem_info_vram_used`, "utf8"),
        ]);
        const totalBytes = Number(totalRaw.trim());
        const usedBytes = Number(usedRaw.trim());
        if (Number.isFinite(totalBytes) && Number.isFinite(usedBytes)) {
          return {
            totalMib: Math.round(totalBytes / BYTES_PER_MIB),
            usedMib: Math.round(usedBytes / BYTES_PER_MIB),
          };
        }
      } catch {
        // This card has no amdgpu vram_info files (e.g. an Intel iGPU
        // sharing the same /sys/class/drm listing) -- try the next entry.
        continue;
      }
    }
    return null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// CUDA: nvidia-smi --query-compute-apps is NVML's own per-process memory
// accounting exposed as a stable, long-documented CLI -- the one backend
// here that can honestly claim "exact"/process_gpu_usage, preferred over
// the whole-adapter reading per the spec's own "prefer process-specific"
// requirement. Not run live this session -- no NVIDIA hardware available --
// verify against a real CUDA worker before fully trusting the PID-matching
// behavior in practice.
async function readCudaGpuMemory(pid: number | undefined): Promise<GpuMemoryReading> {
  const wholeAdapter = await readNvidiaSmiWholeAdapter();
  const totalReading = reading(wholeAdapter?.totalMib ?? null, "exact", "driver_reported_memory");
  if (pid != null) {
    const processUsedMib = await readNvidiaSmiProcessUsed(pid);
    if (processUsedMib != null) {
      return { total: totalReading, used: reading(processUsedMib, "exact", "process_gpu_usage") };
    }
  }
  // No pid yet (pre-spawn baseline), or this process hasn't shown up in
  // nvidia-smi's own compute-apps list yet -- fall back to whole-adapter
  // usage: a real driver reading, just not process-isolated.
  return { total: totalReading, used: reading(wholeAdapter?.usedMib ?? null, "high", "driver_reported_memory") };
}

async function readNvidiaSmiWholeAdapter(): Promise<VramSample | null> {
  return readNvidiaSmiVram();
}

// One line per (pid, used_memory) pair currently allocating GPU compute
// memory, e.g. "12345, 2048" -- matched against the spawned llama-bench/
// llama-server child's own pid. Empty output (not an error) when nothing is
// currently using the GPU for compute, or when this specific process hasn't
// been picked up by nvidia-smi's own polling yet.
async function readNvidiaSmiProcessUsed(pid: number): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
      { timeout: EXEC_TIMEOUT_MS }
    );
    for (const line of stdout.trim().split("\n")) {
      if (!line.trim()) continue;
      const [pidStr, usedStr] = line.split(",").map((s) => s.trim());
      if (Number(pidStr) === pid) {
        const usedMib = Number(usedStr);
        return Number.isFinite(usedMib) ? usedMib : null;
      }
    }
    return null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// ROCm: deliberately no process-specific reading attempted (see the plan's
// "Deviations from the literal spec" -- unlike NVML's --query-compute-apps,
// there's no verified-reliable per-process equivalent in classic rocm-smi,
// and guessing at one risks mislabeling accuracy, which "never fabricate"
// forbids just as much as fabricating a number would). Falls back to the
// same amdgpu sysfs files the generic Linux path already reads when
// rocm-smi itself isn't installed -- same underlying kernel data either way,
// ROCm and Vulkan both sit on top of the same amdgpu driver on Linux. Not
// run live this session -- no ROCm install available -- verify against a
// real ROCm worker before fully trusting, especially the exact rocm-smi
// JSON key names below.
async function readRocmGpuMemory(): Promise<GpuMemoryReading> {
  const sample = (await readRocmSmiVram()) ?? (await readAmdgpuSysfsVram());
  return {
    total: reading(sample?.totalMib ?? null, "exact", "driver_reported_memory"),
    used: reading(sample?.usedMib ?? null, "high", "driver_reported_memory"),
  };
}

interface RocmSmiVramEntry {
  "VRAM Total Memory (B)"?: string;
  "VRAM Total Used Memory (B)"?: string;
}

async function readRocmSmiVram(): Promise<VramSample | null> {
  try {
    const { stdout } = await execFileAsync("rocm-smi", ["--showmeminfo", "vram", "--json"], {
      timeout: EXEC_TIMEOUT_MS,
    });
    const parsed = JSON.parse(stdout) as Record<string, RocmSmiVramEntry>;
    // Keyed by card id (e.g. "card0") -- only the first is read, matching
    // this app's existing single-GPU-per-worker assumption elsewhere.
    const first = Object.values(parsed)[0];
    if (!first) return null;
    const totalBytes = Number(first["VRAM Total Memory (B)"]);
    const usedBytes = Number(first["VRAM Total Used Memory (B)"]);
    if (!Number.isFinite(totalBytes) || !Number.isFinite(usedBytes)) return null;
    return {
      totalMib: Math.round(totalBytes / BYTES_PER_MIB),
      usedMib: Math.round(usedBytes / BYTES_PER_MIB),
    };
  } catch {
    // rocm-smi missing (not installed, or this is actually a plain Vulkan
    // box rather than a ROCm one) -- not an error, fall back to sysfs.
    return null;
  }
}

// ---------------------------------------------------------------------------
// Metal / Apple Silicon: no discrete VRAM at all -- the GPU draws from
// unified system memory up to a dynamic ceiling (Metal's
// recommendedMaxWorkingSetSize), so unlike every other backend here "total"
// is defined as total *system* memory (sysctl hw.memsize), per the spec's
// own explicit "use total unified memory" instruction for this backend --
// not the GPU's own allocation ceiling, which isn't obtainable from a shell
// command at all. "In use system memory" (bytes) inside ioreg's
// IOAccelerator "PerformanceStatistics" dict is the same figure community
// tools (asitop, stats.app) read for GPU memory usage. Not run live this
// session -- no macOS hardware available -- verify against a real Metal
// worker before fully trusting.
const IOREG_IN_USE_RE = /"In use system memory"\s*=\s*(\d+)/;

async function readDarwinGpuMemory(): Promise<GpuMemoryReading> {
  const [totalBytes, usedBytes] = await Promise.all([readDarwinTotalMemoryBytes(), readDarwinGpuUsedBytes()]);
  return {
    total: reading(
      totalBytes != null ? Math.round(totalBytes / BYTES_PER_MIB) : null,
      "estimated",
      "unified_memory_estimate"
    ),
    used: reading(
      usedBytes != null ? Math.round(usedBytes / BYTES_PER_MIB) : null,
      "estimated",
      "unified_memory_estimate"
    ),
  };
}

async function readDarwinTotalMemoryBytes(): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync("sysctl", ["-n", "hw.memsize"], { timeout: EXEC_TIMEOUT_MS });
    const bytes = Number(stdout.trim());
    return Number.isFinite(bytes) ? bytes : null;
  } catch {
    return null;
  }
}

async function readDarwinGpuUsedBytes(): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync("ioreg", ["-r", "-d", "1", "-c", "IOAccelerator"], {
      timeout: EXEC_TIMEOUT_MS,
    });
    const match = IOREG_IN_USE_RE.exec(stdout);
    if (!match) return null;
    const bytes = Number(match[1]);
    return Number.isFinite(bytes) ? bytes : null;
  } catch {
    return null;
  }
}
