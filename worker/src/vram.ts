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
  // Capacity (whole adapter).
  total: GpuMemoryValue;
  // Whole-adapter usage -- every process on the GPU combined (the desktop,
  // other apps, this benchmark), never process-isolated.
  used: GpuMemoryValue;
  // This worker's own benchmark child process's usage -- only present on
  // backends/platforms with a per-process reading (nvidia-smi compute-apps,
  // Windows' WDDM "GPU Process Memory" counter, rocm-smi --showpids /
  // amdgpu fdinfo), and only once a pid exists AND the driver has caught up
  // to that process's allocations. Absent = "no per-process reading
  // available" (which is different from a measured 0, and is why this is an
  // optional field rather than a nullable one: a fallback that silently
  // reports the whole adapter as "the process's usage" is exactly the
  // mislabeling the accuracy/source fields exist to prevent).
  process?: GpuMemoryValue;
  // The same process's SYSTEM-RAM-backed GPU allocation -- Windows' WDDM
  // "GPU Process Memory \ Shared Usage" counter (a distinct PDH category from
  // Dedicated Usage above, confirmed live on this machine's Radeon RX 6600 XT
  // -- see the comment on readWindowsUsedMib), or Linux amdgpu's GTT domain
  // via /proc/<pid>/fdinfo's drm-memory-gtt entries -- the same DRM
  // accounting `process` above reads for the VRAM domain. This is the
  // measured answer to "how much did the OS silently spill into system RAM
  // instead of erroring" -- shared/vramEstimate.ts's isVramDiscrepancy only
  // ever INFERRED that from a needed-vs-peak gap; a `processShared` reading
  // sees it directly. Absent (not a measured 0) wherever no such counter/file
  // exists at all: CUDA-on-Linux has no OS-level silent-paging mechanism to
  // read here (an oversubscribed cudaMalloc there fails outright rather than
  // falling back), and Metal has no discrete/shared split to begin with.
  processShared?: GpuMemoryValue;
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
// "exact"/process_gpu_usage for any backend, only the per-process reading
// below can). Every backend returns `used` (whole adapter) unconditionally
// when measurable; `process` is only ever present when a pid was given AND
// the driver exposes per-process accounting for it.
export async function readGpuMemory(backend: Backend, pid: number | undefined): Promise<GpuMemoryReading> {
  try {
    // No subprocess call at all on the cpu backend -- both correctness (the
    // spec requires null here) and a real perf win, skipping the ~1-1.5s
    // Windows PDH cost (see readGenericGpuMemory below) on every CPU item.
    if (backend === "cpu") return UNAVAILABLE_READING;
    if (backend === "cuda") return await readCudaGpuMemory(pid);
    // ROCm collection (rocm-smi CLI, amdgpu sysfs) is Linux-only -- neither
    // exists on a Windows box running a win-rocm/win-hip llama.cpp build
    // (the HIP SDK/TheRock tarball ships no rocm-smi; /sys/class/drm is a
    // Linux path), so every probe silently returned "unavailable" there and
    // the UI showed n/a for all VRAM columns. Route win32 through the same
    // vendor-agnostic WDDM "GPU Adapter Memory" counter the vulkan/generic
    // path uses -- confirmed live against a Radeon RX 6600 XT on Windows --
    // rather than letting the ROCm-only probes dead-end.
    if (backend === "rocm") {
      return osPlatform() === "win32" ? await readGenericGpuMemory(pid) : await readRocmGpuMemory(pid);
    }
    if (backend === "metal" || osPlatform() === "darwin") return await readDarwinGpuMemory();
    return await readGenericGpuMemory(pid);
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
async function readGenericGpuMemory(pid: number | undefined): Promise<GpuMemoryReading> {
  const plat = osPlatform();
  let sample: VramSample = NULL_SAMPLE;
  let processReading: GpuMemoryValue | undefined;
  let processSharedReading: GpuMemoryValue | undefined;
  if (plat === "win32") {
    const w = await readWindowsVram(pid);
    sample = w.sample;
    if (w.processMib != null) processReading = reading(w.processMib, "exact", "process_gpu_usage");
    if (w.processSharedMib != null) processSharedReading = reading(w.processSharedMib, "exact", "process_gpu_usage");
  } else if (plat === "linux") {
    sample = await readLinuxVram();
    if (pid != null) {
      processReading = (await readLinuxProcessVram(pid)) ?? undefined;
      processSharedReading = (await readAmdgpuFdinfoGtt(pid)) ?? undefined;
    }
  }
  return {
    total: reading(sample.totalMib, "high", "driver_reported_memory"),
    used: reading(sample.usedMib, "high", "driver_reported_memory"),
    ...(processReading ? { process: processReading } : {}),
    ...(processSharedReading ? { processShared: processSharedReading } : {}),
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

async function readWindowsVram(
  pid: number | undefined
): Promise<{ sample: VramSample; processMib: number | null; processSharedMib: number | null }> {
  const [totalMib, usedAndProcess] = await Promise.all([readWindowsTotalMib(), readWindowsUsedMib(pid)]);
  return {
    sample: { totalMib, usedMib: usedAndProcess.usedMib },
    processMib: usedAndProcess.processMib,
    processSharedMib: usedAndProcess.processSharedMib,
  };
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

function parseMiB(raw: string | undefined): number | null {
  const bytes = Number(raw?.trim());
  return Number.isFinite(bytes) && bytes > 0 ? Math.round(bytes / BYTES_PER_MIB) : null;
}

// One powershell.exe spawn for ALL THREE readings (each Get-Counter call pays
// the same ~1-1.5s PDH cold-start cost, so separate spawns would multiply it):
//   1. the whole-adapter "GPU Adapter Memory" Dedicated Usage sum (used),
//   2. the per-process "GPU Process Memory" Dedicated Usage for the spawned
//      child's pid -- the same VidMm data Task Manager's per-process "GPU
//      memory" column reads,
//   3. that SAME process's "GPU Process Memory" Shared Usage -- a separate
//      PDH counter (confirmed live on this machine's Radeon RX 6600 XT: only
//      Shared Usage/Dedicated Usage/Total Committed are published under
//      "GPU Process Memory", see readWindowsTotalMib's own comment) tracking
//      system RAM WDDM is backing as GPU-accessible memory for this process
//      -- exactly the bytes an oversubscribed allocation silently spills
//      into instead of erroring (this file's own top comment; NVIDIA's
//      Control Panel names the mechanism "CUDA - Sysmem Fallback Policy",
//      but the WDDM counter itself is vendor-agnostic, so it applies
//      identically to an AMD/Vulkan or an NVIDIA/CUDA process's pid).
// The counter instances are named `pid_<pid>_luid_0x..._phys_<n>` (one per
// memory segment), so the wildcard matches and Measure-Object sums them.
// Caveats: an instance only exists while the process actually holds that
// kind of GPU allocation (a freshly-spawned llama-bench may not have one yet
// -- fine, the sampler just doesn't grow that stream this tick), and
// "Dedicated Usage" is VidMm's *committed* accounting (what the driver
// charged the process), not hardware residency -- same semantics as
// nvidia-smi's per-process used_memory.
async function readWindowsUsedMib(
  pid: number | undefined
): Promise<{ usedMib: number | null; processMib: number | null; processSharedMib: number | null }> {
  try {
    const lines = [
      `$u = (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum`,
      pid != null
        ? `$p = (Get-Counter '\\GPU Process Memory(pid_${pid}*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum`
        : null,
      pid != null
        ? `$s = (Get-Counter '\\GPU Process Memory(pid_${pid}*)\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum`
        : null,
      `Write-Output $u`,
      pid != null ? `Write-Output $p` : null,
      pid != null ? `Write-Output $s` : null,
    ].filter(Boolean);
    const { stdout } = await execFileAsync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", lines.join("; ")], {
      timeout: EXEC_TIMEOUT_MS,
      windowsHide: true,
    });
    // Three lines, one per Write-Output when a pid was given (one otherwise)
    // -- a line is blank when its counter instance doesn't exist yet.
    const [usedRaw, processRaw, sharedRaw] = stdout.trim().split("\n");
    return {
      usedMib: parseMiB(usedRaw),
      processMib: pid != null ? parseMiB(processRaw) : null,
      processSharedMib: pid != null ? parseMiB(sharedRaw) : null,
    };
  } catch {
    // No counter provider, no GPU, or the call timed out -- best-effort.
    return { usedMib: null, processMib: null, processSharedMib: null };
  }
}

// Windows-only, per-process Shared Usage in isolation -- for a caller
// (readCudaGpuMemory) that already has its own total/used/process readings
// from a vendor-specific source (nvidia-smi) and just needs this ONE
// vendor-agnostic WDDM figure added on top, without re-querying "GPU Adapter
// Memory" Dedicated Usage it has no use for. See readWindowsUsedMib's own
// comment for what this counter means and why it's the right one to read.
async function readWindowsProcessSharedMib(pid: number): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        `(Get-Counter '\\GPU Process Memory(pid_${pid}*)\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum`,
      ],
      { timeout: EXEC_TIMEOUT_MS, windowsHide: true }
    );
    return parseMiB(stdout);
  } catch {
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
      // windowsHide: keep this child off the worker's visible console -- same
      // reasoning as binary-probe.ts's helpText (console children inherit the
      // parent console when not hidden, letting them reconfigure it).
      { timeout: EXEC_TIMEOUT_MS, windowsHide: true }
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

// Per-process VRAM on Linux, vendor-agnostic (used by the generic/vulkan
// path, where the box can be NVIDIA or AMD): nvidia-smi's own per-process
// compute-apps accounting first (identical mechanism to the cuda backend),
// then amdgpu's /proc/<pid>/fdinfo drm-memory-vram entries (the standard DRM
// fdinfo interface -- what amdgpu_top's per-process mode reads; works with
// just the stock amdgpu kernel driver, no ROCm install needed). Null when
// neither vendor's source knows the process.
async function readLinuxProcessVram(pid: number): Promise<GpuMemoryValue | null> {
  const nvidiaMib = await readNvidiaSmiProcessUsed(pid);
  if (nvidiaMib != null) return reading(nvidiaMib, "exact", "process_gpu_usage");
  const amd = await readAmdgpuFdinfoVram(pid);
  if (amd) return amd;
  return null;
}

const DRM_CARD_DIR = "/sys/class/drm";
const CARD_DIR_RE = /^card\d+$/;

// Per-process GPU memory via amdgpu's fdinfo accounting, generalized over
// which DRM domain to sum: every fd open on a DRM render node gets one
// `drm-memory-<domain>: <bytes>` line per domain in /proc/<pid>/fdinfo,
// summed across fds -- same data amdgpu_top's per-process mode reads. Works
// on a stock kernel/amdgpu driver with no ROCm install at all (matching the
// sysfs fallback above for the whole-adapter side). Returns null when the
// process has no GPU fds, none of the entries are readable, or this domain
// never appears (a process that never touched the GPU, or never touched THIS
// domain, reports nothing, not 0).
async function readAmdgpuFdinfoDomain(pid: number, domainKey: string): Promise<GpuMemoryValue | null> {
  try {
    const fdinfoDir = `/proc/${pid}/fdinfo`;
    const entries = await readdir(fdinfoDir);
    let totalBytes = 0;
    for (const entry of entries) {
      let content: string;
      try {
        content = await readFile(`${fdinfoDir}/${entry}`, "utf8");
      } catch {
        // fd closed between readdir and read -- skip it.
        continue;
      }
      for (const line of content.split("\n")) {
        if (!line.startsWith(domainKey)) continue;
        const bytes = Number(line.slice(line.indexOf(":") + 1).trim());
        if (Number.isFinite(bytes)) totalBytes += bytes;
      }
    }
    return totalBytes > 0 ? reading(Math.round(totalBytes / BYTES_PER_MIB), "exact", "process_gpu_usage") : null;
  } catch {
    return null;
  }
}

async function readAmdgpuFdinfoVram(pid: number): Promise<GpuMemoryValue | null> {
  return readAmdgpuFdinfoDomain(pid, "drm-memory-vram:");
}

// GTT (Graphics Translation Table) -- amdgpu's own domain for system-RAM
// pages the GPU can address directly, the Linux/AMD analogue of Windows'
// WDDM "Shared Usage": a buffer the scheduler couldn't place in VRAM lands
// here instead of failing, over the same DRM fdinfo interface readAmdgpuFdinfoVram
// already reads for the VRAM domain (see GpuMemoryReading.processShared's own
// doc comment for the fuller picture). Not run live this session -- no Linux
// AMD hardware available -- verify against a real box before fully trusting,
// same posture as every other not-yet-run-live path in this file.
async function readAmdgpuFdinfoGtt(pid: number): Promise<GpuMemoryValue | null> {
  return readAmdgpuFdinfoDomain(pid, "drm-memory-gtt:");
}

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
// behavior in practice. `used` stays the whole-adapter reading (reported
// separately from `process` below -- the two are different numbers, and
// collapsing them into one hybrid field is what made the old "vram_peak"
// mean different things on different backends).
async function readCudaGpuMemory(pid: number | undefined): Promise<GpuMemoryReading> {
  const wholeAdapter = await readNvidiaSmiWholeAdapter();
  const totalReading = reading(wholeAdapter?.totalMib ?? null, "exact", "driver_reported_memory");
  const usedReading = reading(wholeAdapter?.usedMib ?? null, "high", "driver_reported_memory");
  let processReading: GpuMemoryValue | undefined;
  let processSharedReading: GpuMemoryValue | undefined;
  if (pid != null) {
    const processUsedMib = await readNvidiaSmiProcessUsed(pid);
    // null = this process hasn't shown up in nvidia-smi's own compute-apps
    // list yet (its polling lags a fresh spawn) -- no process reading this
    // tick, not a measured 0.
    if (processUsedMib != null) processReading = reading(processUsedMib, "exact", "process_gpu_usage");
    // NVML/nvidia-smi has no counterpart to WDDM's "Shared Usage" -- the
    // sysmem-fallback bytes an oversubscribed CUDA allocation silently
    // spills to (NVIDIA Control Panel's own "CUDA - Sysmem Fallback Policy")
    // are invisible to nvidia-smi entirely. Windows' own WDDM performance
    // counter is vendor-agnostic though (any driver model, any API) -- an
    // NVIDIA process's pid publishes the identical "GPU Process Memory"
    // category a Vulkan/AMD process's does, so reading it here catches
    // exactly what nvidia-smi can't, on Windows only (Linux CUDA has no
    // equivalent silent-paging mechanism to read -- see
    // GpuMemoryReading.processShared's own doc comment).
    if (osPlatform() === "win32") {
      const sharedMib = await readWindowsProcessSharedMib(pid);
      if (sharedMib != null) processSharedReading = reading(sharedMib, "exact", "process_gpu_usage");
    }
  }
  return {
    total: totalReading,
    used: usedReading,
    ...(processReading ? { process: processReading } : {}),
    ...(processSharedReading ? { processShared: processSharedReading } : {}),
  };
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

export interface NvidiaDriverInfo {
  driverVersion: string;
  // The maximum CUDA version this DRIVER supports -- not the CUDA runtime
  // version the llama.cpp binary was built against (that side shows up in
  // the process's own stderr, see bench.ts's extractCudaDiagnosticLines).
  // Juxtaposing both is how a too-old-driver-for-this-build mismatch gets
  // ruled in/out from the log alone.
  cudaVersion: string;
}

// Plain `nvidia-smi` (no args) banner, e.g.
//   | NVIDIA-SMI 566.36       Driver Version: 566.36       CUDA Version: 12.7 |
// parsed for exactly those two figures. Best-effort diagnostics for the
// VRAM-discrepancy postmortem -- null whenever nvidia-smi is missing/fails
// (non-NVIDIA worker) or the banner shape ever changes.
export async function readNvidiaDriverInfo(): Promise<NvidiaDriverInfo | null> {
  try {
    const { stdout } = await execFileAsync("nvidia-smi", [], { timeout: EXEC_TIMEOUT_MS });
    const m = /Driver Version:\s*(\S+)\s+CUDA Version:\s*(\S+)/.exec(stdout);
    return m ? { driverVersion: m[1], cudaVersion: m[2] } : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// ROCm (Linux): whole-adapter side reads rocm-smi's meminfo (or the same
// amdgpu sysfs files the generic Linux path already reads when rocm-smi
// isn't installed -- same underlying kernel data either way, ROCm and Vulkan
// both sit on top of the same amdgpu driver on Linux). Per-process side uses
// rocm-smi --showpids (KFD's per-PID VRAM accounting, reliable since ROCm
// 6.1/6.3 -- earlier releases reported UNKNOWN on some cards), falling back
// to /proc/<pid>/fdinfo drm-memory-vram when rocm-smi itself is missing.
// Neither path has been run live this session -- no ROCm install available --
// verify against a real ROCm worker before fully trusting, especially the
// exact rocm-smi JSON/key/column names below.
async function readRocmGpuMemory(pid: number | undefined): Promise<GpuMemoryReading> {
  const sample = (await readRocmSmiVram()) ?? (await readAmdgpuSysfsVram());
  let processReading: GpuMemoryValue | undefined;
  let processSharedReading: GpuMemoryValue | undefined;
  if (pid != null) {
    processReading = (await readRocmSmiProcessVram(pid)) ?? (await readAmdgpuFdinfoVram(pid)) ?? undefined;
    // rocm-smi --showpids has no GTT column -- fdinfo is the only source for
    // this domain regardless of whether rocm-smi itself is installed.
    processSharedReading = (await readAmdgpuFdinfoGtt(pid)) ?? undefined;
  }
  return {
    total: reading(sample?.totalMib ?? null, "exact", "driver_reported_memory"),
    used: reading(sample?.usedMib ?? null, "high", "driver_reported_memory"),
    ...(processReading ? { process: processReading } : {}),
    ...(processSharedReading ? { processShared: processSharedReading } : {}),
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

// Per-process VRAM from `rocm-smi --showpids` -- KFD's per-PID accounting:
//   PID PROCESS NAME GPU(s) VRAM USED SDMA USED CU OCCUPANCY
//   132174 llama-server 2 31923675136 0 0
// VRAM USED is bytes (0 when the process isn't actually holding VRAM right
// now) and may read UNKNOWN on old ROCm releases / unsupported archs (a
// real "no data" answer, never fabricated). Columns are whitespace-split and
// read from the END (last three tokens are VRAM USED / SDMA USED / CU
// OCCUPANCY) so a process name containing spaces can't shift them.
async function readRocmSmiProcessVram(pid: number): Promise<GpuMemoryValue | null> {
  try {
    const { stdout } = await execFileAsync("rocm-smi", ["--showpids"], { timeout: EXEC_TIMEOUT_MS });
    let bestMib: number | null = null;
    for (const rawLine of stdout.split("\n")) {
      const tokens = rawLine.trim().split(/\s+/);
      if (tokens.length < 4) continue;
      if (tokens[0] === "PID" || !/^\d+$/.test(tokens[0])) continue; // header / prose
      if (Number(tokens[0]) !== pid) continue;
      const vramUsed = tokens[tokens.length - 3];
      if (vramUsed === "UNKNOWN" || vramUsed === "N/A") continue;
      const bytes = Number(vramUsed);
      if (!Number.isFinite(bytes)) continue;
      const mib = Math.round(bytes / BYTES_PER_MIB);
      if (bestMib == null || mib > bestMib) bestMib = mib;
    }
    return bestMib != null ? reading(bestMib, "exact", "process_gpu_usage") : null;
  } catch {
    // rocm-smi missing -- fall back to amdgpu fdinfo (see readRocmGpuMemory).
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
