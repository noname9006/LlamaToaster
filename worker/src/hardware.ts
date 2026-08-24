import si from "systeminformation";
import { platform as osPlatform, arch as osArch } from "node:os";
import { detectBackend } from "../../shared/types.js";
import { readNvidiaDriverInfo } from "./vram.js";

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
  // doesn't report it at all (confirmed happens on some Linux/Vulkan-only
  // setups) -- optional (rather than always-present-but-nullable) so a
  // worker process still running old code (hasn't been restarted since this
  // field was added) keeps reporting a valid snapshot, same reasoning as
  // mem_total_bytes below. vram_dynamic mirrors that same entry's
  // vramDynamic -- true for shared/unified memory (typically an iGPU, e.g.
  // Intel UHD 620) where vram_mb is an estimate/allocation rather than a
  // fixed dedicated pool, so callers displaying it should label it as such
  // rather than presenting it with the same confidence as a discrete GPU's
  // real VRAM size.
  gpu: { vendor: string; model: string; vram_mb?: number | null; vram_dynamic?: boolean }[];
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
  const [cpu, graphics, mem] = await Promise.all([si.cpu(), si.graphics(), si.mem()]);
  const gpu = graphics.controllers
    .filter((c) => c.vendor)
    .map((c) => ({
      vendor: c.vendor ?? "",
      model: c.model ?? "",
      vram_mb: typeof c.vram === "number" && c.vram > 0 ? c.vram : null,
      vram_dynamic: c.vramDynamic ?? false,
    }));
  // Only boxes that actually have an NVIDIA GPU pay the ~200-500ms nvidia-smi
  // round trip (once per process -- detectHardware runs once at startup).
  // Null on failure (nvidia-smi missing/not on PATH, driver too old to
  // report a CUDA version, ...): the server then treats CUDA compatibility
  // as unknown and just orders variants conservatively instead.
  const hasNvidia = gpu.some((g) => /nvidia/i.test(g.vendor));
  const nvidiaDriverInfo = hasNvidia ? await readNvidiaDriverInfo().catch(() => null) : null;
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
    },
    gpu,
    mem_total_bytes: typeof mem.total === "number" && mem.total > 0 ? mem.total : undefined,
    nvidia_driver: nvidiaDriverInfo
      ? { version: nvidiaDriverInfo.driverVersion, cuda_version: nvidiaDriverInfo.cudaVersion }
      : null,
  };
}
