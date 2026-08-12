import si from "systeminformation";
import { platform as osPlatform, arch as osArch } from "node:os";
import type { Backend } from "../../shared/types.js";

export interface HardwareInfo {
  platform: string;
  arch: string;
  cpu: { manufacturer: string; brand: string; flags: string[]; cores: number };
  gpu: { vendor: string; model: string }[];
}

// Manually-maintained snapshot of AMD GPU model tokens ROCm officially
// supports, checked live against AMD's own docs this session (Aug 2026):
// rocm.docs.amd.com's Windows "system requirements" page (HIP SDK 7.2) and
// Linux "compatibility matrix" (ROCm 7.0) -- these two lists actually differ
// (ROCm-on-Windows supports a narrower set than Linux), but this app doesn't
// know a worker's exact driver/ROCm version, so the two are merged into one
// permissive union rather than trying to be precise per-platform. This WILL
// go stale as AMD updates support -- re-check those two pages if a "should
// this default to rocm" question comes up for a model not listed here.
const ROCM_SUPPORTED_AMD_GPU_MARKERS = [
  // RDNA3 / RDNA4 consumer (Radeon RX)
  "7600", "7650 gre", "7700 xt", "7800 xt", "7900 xt", "7900 xtx",
  "9060", "9070",
  // Radeon PRO / workstation
  "w6800", "w7700", "w7800", "w7900", "v620", "v710", "r9700", "r9600",
  // Instinct (data center)
  "mi100", "mi200", "mi300", "mi325x", "mi350x", "mi355x",
  // Ryzen AI Max APUs
  "ryzen ai max",
];

function isRocmSupportedAmdGpu(model: string): boolean {
  const lower = model.toLowerCase();
  return ROCM_SUPPORTED_AMD_GPU_MARKERS.some((marker) => lower.includes(marker));
}

// Best-effort default when a worker's config.json doesn't pin a `backend`.
// Only a guess -- an explicit value in config.json always overrides this,
// see worker/src/index.ts's startup logic. Picks among KNOWN_BACKENDS
// (shared/types.ts); a user who wants something this heuristic doesn't know
// about (or gets wrong for their exact hardware) can always set `backend`
// explicitly instead -- see server/src/github-releases.ts's generic asset
// matching, which accepts any backend string, not just these.
//
// macOS is a hard special case, confirmed against a real llama.cpp release's
// asset list (ggml-org/llama.cpp): it ships exactly "macos-arm64"/"macos-x64"
// with no vulkan/cuda/rocm variant whatsoever -- Metal is just baked into
// that one build (macOS only has the one GPU API to begin with, unlike
// Windows/Linux which support several competing ones). Reporting "vulkan"
// there (as an earlier version of this function did, based only on "a GPU
// exists") pointed a Mac at zero installable assets. So macOS always
// resolves to `cpu` regardless of GPU vendor -- the bucket its one real
// build actually falls into.
//
// win32/linux: NVIDIA gets `cuda` (broad CUDA compatibility across
// virtually every NVIDIA GPU, unlike AMD's much narrower ROCm support below
// -- note current ubuntu releases ship no cuda asset at all, so this guess
// can come up empty on a Linux+NVIDIA worker specifically until upstream
// adds one back). AMD gets `rocm` only when the specific detected model is
// one ROCm actually supports (see isRocmSupportedAmdGpu above); otherwise
// `vulkan`, same as any other detected GPU (Intel, unrecognized AMD, ...) --
// the broadest cross-vendor option that reliably works. No GPU at all falls
// back to `cpu`.
export function detectBackend(platform: string, gpu: HardwareInfo["gpu"]): Backend {
  if (platform === "darwin") return "cpu";
  const text = gpu.map((g) => `${g.vendor} ${g.model}`.toLowerCase()).join(" ");
  if (text.includes("nvidia")) return "cuda";
  const amdGpu = gpu.find((g) => /amd|advanced micro devices/i.test(g.vendor));
  if (amdGpu && isRocmSupportedAmdGpu(amdGpu.model)) return "rocm";
  if (gpu.length > 0) return "vulkan";
  return "cpu";
}

// Diagnostic only -- nothing reads `flags` to pick between build variants.
// Checked live on a Windows box: si.cpu().flags comes back non-empty but
// missing modern ISA bits a Ryzen 5600X definitely has (no avx/avx2/sse4),
// so it's populated but not trustworthy enough to gate a decision on,
// especially on Windows. Doesn't matter in practice either way -- current
// llama.cpp releases ship exactly one generic CPU build per platform+arch,
// no AVX-tiered variants to choose between (see github-releases.ts).
export async function detectHardware(): Promise<HardwareInfo> {
  const [cpu, graphics] = await Promise.all([si.cpu(), si.graphics()]);
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
    gpu: graphics.controllers
      .filter((c) => c.vendor)
      .map((c) => ({ vendor: c.vendor ?? "", model: c.model ?? "" })),
  };
}
