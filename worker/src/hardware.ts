import si from "systeminformation";
import { platform as osPlatform, arch as osArch } from "node:os";
import { detectBackend, isSharedMemoryGpu } from "../../shared/types.js";
import type { ListedDevice } from "./binary-probe.js";
import { readNvidiaDriverInfo } from "./vram.js";
import { detectCpuTopology } from "./cpuTopology.js";

// Re-exported for this file's existing callers (worker/src/index.ts imports
// both detectHardware and detectBackend from here) -- the implementation
// itself now lives in shared/types.ts so client/src/pages/NewRun.tsx can run
// the identical per-GPU logic, see that file's own doc comment for why.
export { detectBackend };

export interface HardwareInfo {
  platform: string;
  arch: string;
  cpu: { manufacturer: string; brand: string; flags: string[]; cores: number };
  // vram_mb is si.graphics().controllers[].vram (MB), null when the driver
  // doesn't report it at all (confirmed happens on some setups) -- optional
  // (rather than always-present-but-nullable) so a worker process still
  // running old code (hasn't been restarted since this field was added)
  // keeps reporting a valid snapshot, same reasoning as mem_total_bytes
  // below. vram_dynamic is true only for shared/unified memory (typically
  // an iGPU, e.g. Intel UHD 620) where vram_mb is an estimate/allocation
  // rather than a fixed dedicated pool, so callers should label it as such
  // rather than presenting it with the same confidence as a discrete GPU's
  // real VRAM size. It is NOT a verbatim copy of systeminformation's
  // vramDynamic: that library's Windows derivation (VideoMemoryType === '2')
  // flags *dedicated* VRAM as dynamic, so detectHardware below resets the
  // flag to false whenever an actual dedicated pool (vram_mb > 0) exists.
  // vram_usable_mb / vram_listed_total_mb: see shared/types.ts's mirrored copy.
  gpu: {
    vendor: string;
    model: string;
    vram_mb?: number | null;
    vram_dynamic?: boolean;
    vram_usable_mb?: number | null;
    vram_listed_total_mb?: number | null;
  }[];
  // OS name, RAM type and unified-memory flag -- see shared/types.ts.
  os?: { family: string; name: string } | null;
  mem_type?: string | null;
  unified_memory?: boolean;
  // Total system RAM in bytes (on Apple Silicon, total unified memory) --
  // see si.mem().total in detectHardware below. Kept in sync with the
  // mirrored copy of this interface in shared/types.ts.
  mem_total_bytes?: number;
  // Best-effort nvidia-smi probe (driver version + the max CUDA toolkit it
  // runs) -- present only when an NVIDIA GPU was detected and nvidia-smi
  // answered. This is what lets the server pick between llama.cpp's
  // cuda-12.x/cuda-13.x build variants for this box; see shared/types.ts's
  // HardwareInfo.nvidia_driver for why cuda_version (not driverVersion) is
  // the field that matters there. Kept optional for old-worker tolerance.
  nvidia_driver?: { version: string; cuda_version: string } | null;
}

// Diagnostic only -- nothing reads `flags` to pick between build variants.
// Checked live on a Windows box: si.cpu().flags comes back non-empty but
// missing modern ISA bits a Ryzen 5600X definitely has (no avx/avx2/sse4),
// so it's populated but not trustworthy enough to gate a decision on,
// especially on Windows. Doesn't matter in practice either way -- current
// llama.cpp releases ship exactly one generic CPU build per platform+arch,
// no AVX-tiered variants to choose between (see github-releases.ts).
export async function detectHardware(): Promise<HardwareInfo> {
  const [cpu, graphics, mem, osInfo, memLayout] = await Promise.all([
    si.cpu(),
    si.graphics(),
    si.mem(),
    // Cosmetic only (the OS / RAM-type labels) -- must never fail detection.
    bestEffort(() => si.osInfo()),
    bestEffort(() => si.memLayout()),
  ]);
  const gpu = graphics.controllers.filter((c) => c.vendor).map((c) => {
    // A concrete dedicated VRAM budget (>0) is the ground truth that the
    // card is NOT shared/unified memory. systeminformation's Windows
    // vramDynamic derivation (graphics.js:1227, v5.32.0) treats
    // Win32_VideoController.VideoMemoryType === '2' as dynamic, but '2' is
    // the enum value for *dedicated* VRAM -- so it flags exactly the cards
    // with the most on-die memory as "shared". Correct the verdict whenever
    // a real dedicated pool was reported, keeping the tag only for adapters
    // that genuinely have no dedicated VRAM of their own (macOS unified
    // memory, Windows iGPUs that report no pool at all).
    const vram_mb = typeof c.vram === "number" && c.vram > 0 ? c.vram : null;
    const vram_dynamic = (c.vramDynamic ?? false) && vram_mb == null;
    return {
      vendor: c.vendor ?? "",
      model: c.model ?? "",
      vram_mb,
      vram_dynamic,
    };
  });
  // Apple Silicon: CPU and GPU draw from one pool, so there is no separate VRAM.
  const unifiedMemory = osPlatform() === "darwin" && osArch() === "arm64";
  // Only boxes that actually have an NVIDIA GPU pay the ~200-500ms nvidia-smi
  // round trip (once per process -- detectHardware runs once at startup).
  // Null on failure (nvidia-smi missing/not on PATH, driver too old to
  // report a CUDA version, ...): the server then treats CUDA compatibility
  // as unknown and just orders variants conservatively instead.
  const hasNvidia = gpu.some((g) => /nvidia/i.test(g.vendor));
  const nvidiaDriverInfo = hasNvidia ? await readNvidiaDriverInfo().catch(() => null) : null;
  // Per-core topology for the optimization flow's thread defaults -- best
  // effort, reuses the si.cpu() reading above on Windows (see cpuTopology.ts).
  const cpuTopology = await detectCpuTopology(process.platform, cpu).catch(() => null);
  return {
    platform: osPlatform(),
    arch: osArch(),
    cpu: {
      manufacturer: cpu.manufacturer ?? "",
      brand: cpu.brand ?? "",
      flags:
        typeof cpu.flags === "string" && cpu.flags.length
          ? cpu.flags.split(/\s+/).filter(Boolean)
          : [],
      // Logical processor count (si.cpu().cores falls back to os.cpus().length) --
      // the real ceiling for llama-bench's -t, including SMT/hyperthreads.
      cores: typeof cpu.cores === "number" && cpu.cores > 0 ? cpu.cores : 0,
      ...(typeof cpu.physicalCores === "number" && cpu.physicalCores > 0 ? { physical_cores: cpu.physicalCores } : {}),
    },
    ...(cpuTopology ? { cpu_topology: cpuTopology } : {}),
    gpu,
    os: describeOs(osPlatform(), osInfo),
    mem_type: describeMemType(memLayout),
    unified_memory: unifiedMemory,
    mem_total_bytes: typeof mem.total === "number" && mem.total > 0 ? mem.total : undefined,
    nvidia_driver: nvidiaDriverInfo
      ? { version: nvidiaDriverInfo.driverVersion, cuda_version: nvidiaDriverInfo.cudaVersion }
      : null,
  };
}

async function bestEffort<T>(read: () => Promise<T>): Promise<T | null> {
  try {
    return await read();
  } catch {
    return null;
  }
}

const OS_FAMILY_BY_PLATFORM: Record<string, string> = { win32: "Windows", darwin: "macOS", linux: "Linux" };

/** "Windows 10 Pro N (10.0.19045)", "macOS 14.5 (Sonoma)", "Ubuntu 22.04". */
export function describeOs(
  platform: string,
  info: { distro?: string; release?: string; codename?: string } | null
): { family: string; name: string } {
  const family = OS_FAMILY_BY_PLATFORM[platform] ?? platform;
  // si prefixes Windows editions with the vendor ("Microsoft Windows 10 Pro N").
  const distro = (info?.distro ?? "").replace(/^Microsoft\s+/i, "").trim();
  const release = (info?.release ?? "").trim();
  if (!distro) return { family, name: release ? `${family} ${release}` : family };
  let name = distro;
  if (release && !distro.includes(release)) {
    // Windows: the release is the NT build ("10.0.19045") -- parenthesised,
    // since the edition already says which Windows it is. Elsewhere it is the
    // version itself ("22.04", "14.5").
    name = platform === "win32" ? `${distro} (${release})` : `${distro} ${release}`;
  }
  const codename = (info?.codename ?? "").trim();
  if (platform === "darwin" && codename && !name.toLowerCase().includes(codename.toLowerCase())) {
    name = `${name} (${codename.replace(/^macOS\s+/i, "")})`;
  }
  return { family, name };
}

/** "DDR4-3200" from si.memLayout()'s first module that names a type. Null when none does. */
export function describeMemType(layout: { type?: string; clockSpeed?: number | null }[] | null): string | null {
  const module = layout?.find((m) => typeof m.type === "string" && /ddr|lpddr|hbm/i.test(m.type));
  if (!module?.type) return null;
  const type = module.type.trim().toUpperCase();
  const speed = typeof module.clockSpeed === "number" && module.clockSpeed > 0 ? module.clockSpeed : null;
  return speed ? `${type}-${speed}` : type;
}

// Words that name a vendor/product line rather than a specific card -- dropped
// before comparing si's adapter name with llama.cpp's device description.
const GPU_NAME_NOISE = new Set([
  "nvidia", "geforce", "amd", "radeon", "intel", "advanced", "micro", "devices", "inc", "corporation",
  "graphics", "gpu", "series", "laptop", "r", "tm", "apple",
]);

function gpuNameTokens(name: string): string[] {
  return name
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t && !GPU_NAME_NOISE.has(t));
}

function gpuNamesMatch(model: string, description: string): boolean {
  const a = gpuNameTokens(model);
  const b = gpuNameTokens(description);
  if (a.length === 0 || b.length === 0) return false;
  const [small, large] = a.length <= b.length ? [a, b] : [b, a];
  return small.every((t) => large.includes(t));
}

/**
 * Attaches llama.cpp's `--list-devices` figures to each GPU: "max usable VRAM"
 * (the device's reported free memory) and the device's listed total. Each
 * listed device is claimed by at most one GPU, in order, so two identical
 * cards map to two devices. Shared-memory GPUs (iGPUs, unified memory) get
 * nothing -- their "VRAM" is system RAM, already shown as RAM. A GPU no device
 * matches is left as it was. Returns a copy; `hw` is not mutated.
 */
export function mergeListedDevices(hw: HardwareInfo, devices: readonly ListedDevice[]): HardwareInfo {
  const claimed = new Set<number>();
  const listed = devices.filter((d) => d.totalMib > 0);
  const gpu = hw.gpu.map((g) => {
    if (isSharedMemoryGpu(g, hw.unified_memory)) return g;
    let idx = listed.findIndex((d, i) => !claimed.has(i) && gpuNamesMatch(g.model, d.description));
    // A lone discrete GPU with a lone listed device is the same card even when
    // the two names share no token (driver-specific naming).
    if (idx < 0 && hw.gpu.filter((x) => !isSharedMemoryGpu(x, hw.unified_memory)).length === 1) {
      idx = listed.findIndex((_, i) => !claimed.has(i));
      if (idx >= 0 && listed.length !== 1) idx = -1;
    }
    if (idx < 0) return g;
    claimed.add(idx);
    return { ...g, vram_usable_mb: listed[idx].freeMib, vram_listed_total_mb: listed[idx].totalMib };
  });
  return { ...hw, gpu };
}
