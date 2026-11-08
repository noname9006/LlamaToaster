import type { HelperGpu } from "./helper";
import { isFastUnified } from "../../shared/fitGrade";
// What a browser can learn about the machine's memory.
//
// RAM: chrome.system.memory gives the real total and available figures.
// GPU: WebGL's unmasked renderer string names the card; WebGPU says the
// vendor and whether a real adapter exists. No browser API reports VRAM size
// or how much of it is free, so the total comes from a table of known cards
// (the user can override it) and "available" is total minus what the user
// says the desktop and other apps hold.

import type { GpuInfo, Settings } from "./settings";

const MIB_PER_GB = 1024;

// Most specific first: "4070 Ti Super" must match before "4070 Ti" and "4070".
const VRAM_TABLE: [RegExp, number][] = [
  // NVIDIA laptop parts (differ from their desktop namesakes)
  [/RTX\s*5090.*Laptop/i, 24], [/RTX\s*5080.*Laptop/i, 16], [/RTX\s*5070\s*Ti.*Laptop/i, 12], [/RTX\s*5070.*Laptop/i, 8],
  [/RTX\s*4090.*Laptop/i, 16], [/RTX\s*4080.*Laptop/i, 12], [/RTX\s*4070.*Laptop/i, 8], [/RTX\s*4060.*Laptop/i, 8], [/RTX\s*4050.*Laptop/i, 6],
  [/RTX\s*3080\s*Ti.*Laptop/i, 16], [/RTX\s*3080.*Laptop/i, 8], [/RTX\s*3070.*Laptop/i, 8], [/RTX\s*3060.*Laptop/i, 6], [/RTX\s*3050.*Laptop/i, 4],
  // NVIDIA desktop
  [/RTX\s*PRO\s*6000/i, 96], [/RTX\s*6000\s*Ada/i, 48], [/RTX\s*A6000/i, 48], [/RTX\s*A5000/i, 24], [/RTX\s*A4000/i, 16],
  [/RTX\s*5090/i, 32], [/RTX\s*5080/i, 16], [/RTX\s*5070\s*Ti/i, 16], [/RTX\s*5070/i, 12], [/RTX\s*5060\s*Ti/i, 16], [/RTX\s*5060/i, 8], [/RTX\s*5050/i, 8],
  [/RTX\s*4090/i, 24], [/RTX\s*4080/i, 16], [/RTX\s*4070\s*Ti\s*Super/i, 16], [/RTX\s*4070\s*Ti/i, 12], [/RTX\s*4070/i, 12],
  [/RTX\s*4060\s*Ti/i, 8], [/RTX\s*4060/i, 8],
  [/RTX\s*3090/i, 24], [/RTX\s*3080\s*Ti/i, 12], [/RTX\s*3080/i, 10], [/RTX\s*3070/i, 8], [/RTX\s*3060\s*Ti/i, 8], [/RTX\s*3060/i, 12], [/RTX\s*3050/i, 8],
  [/TITAN\s*RTX/i, 24], [/RTX\s*2080\s*Ti/i, 11], [/RTX\s*2080/i, 8], [/RTX\s*2070/i, 8], [/RTX\s*2060\s*Super/i, 8], [/RTX\s*2060/i, 6],
  [/GTX\s*1660/i, 6], [/GTX\s*1650/i, 4], [/GTX\s*1080\s*Ti/i, 11], [/GTX\s*1080/i, 8], [/GTX\s*1070/i, 8], [/GTX\s*1060/i, 6], [/GTX\s*1050\s*Ti/i, 4],
  [/Tesla\s*P40/i, 24], [/Tesla\s*V100/i, 16], [/\bA100\b/i, 80], [/\bH100\b/i, 80], [/\bL40S?\b/i, 48], [/\bL4\b/i, 24], [/\bT4\b/i, 16],
  // AMD
  [/Radeon\s*AI\s*PRO\s*R9700/i, 32], [/RX\s*9070\s*XT/i, 16], [/RX\s*9070/i, 16], [/RX\s*9060\s*XT/i, 16],
  [/RX\s*7900\s*XTX/i, 24], [/RX\s*7900\s*XT/i, 20], [/RX\s*7900\s*GRE/i, 16], [/RX\s*7800\s*XT/i, 16], [/RX\s*7700\s*XT/i, 12], [/RX\s*7600\s*XT/i, 16], [/RX\s*7600/i, 8],
  [/RX\s*6950\s*XT/i, 16], [/RX\s*6900\s*XT/i, 16], [/RX\s*6800/i, 16], [/RX\s*6750/i, 12], [/RX\s*6700\s*XT/i, 12], [/RX\s*6700/i, 10],
  [/RX\s*6650\s*XT/i, 8], [/RX\s*6600/i, 8], [/RX\s*6500/i, 4], [/RX\s*6400/i, 4],
  [/RX\s*5700/i, 8], [/RX\s*5600/i, 6], [/RX\s*5500/i, 8], [/Radeon\s*VII/i, 16], [/Vega\s*64/i, 8], [/Vega\s*56/i, 8],
  [/Radeon\s*PRO\s*W7900/i, 48], [/Radeon\s*PRO\s*W7800/i, 32], [/Instinct\s*MI300X/i, 192], [/Instinct\s*MI250/i, 128],
  // Intel Arc (discrete)
  [/Arc.*B580/i, 12], [/Arc.*B570/i, 10], [/Arc.*A770/i, 16], [/Arc.*A750/i, 8], [/Arc.*A580/i, 8], [/Arc.*A380/i, 6], [/Arc.*A310/i, 4],
];

// GPUs that draw from system RAM rather than their own memory.
const UNIFIED_PATTERNS = [
  /Apple/i,
  /Intel.*(UHD|Iris|HD Graphics)/i,
  /Intel.*Arc.*(1[34]0V|Graphics)\b/i,
  /Radeon\s*\(TM\)\s*Graphics/i,
  /Radeon\s*Graphics/i,
  /Radeon.*\b(6[0-9]0M|7[0-9]0M|8[0-9]0M|8[0-9]0S|Vega\s*\d+\b(?!\s*(56|64)))/i,
  /Adreno/i,
  /Mali/i,
];

export function lookupVramMib(name: string | null): number | null {
  if (!name) return null;
  for (const [re, gb] of VRAM_TABLE) if (re.test(name)) return gb * MIB_PER_GB;
  return null;
}

export function looksUnified(name: string | null, vendor: string | null): boolean {
  if (vendor && /apple/i.test(vendor)) return true;
  if (!name) return false;
  if (lookupVramMib(name) != null) return false;
  return UNIFIED_PATTERNS.some((re) => re.test(name));
}

/** "ANGLE (AMD, AMD Radeon RX 6600 XT (0x000073FF) Direct3D11 vs_5_0 ps_5_0, D3D11)" -> "AMD Radeon RX 6600 XT". */
export function cleanRendererName(raw: string): string {
  let s = raw.trim();
  const angle = /^ANGLE \((.*)\)$/.exec(s);
  if (angle) {
    const parts = angle[1].split(", ");
    s = parts.length >= 2 ? parts[1] : parts[0];
  }
  return s
    .replace(/\s*\(0x[0-9a-f]+\)/gi, "")
    .replace(/\s*(Direct3D|D3D)\d*.*$/i, "")
    .replace(/\s*(vs|ps)_\d_\d.*$/i, "")
    .replace(/\s*OpenGL.*$/i, "")
    .replace(/\s*Metal.*$/i, "")
    .replace(/\/PCIe\/SSE2/i, "")
    .trim();
}

function webglRenderer(): string | null {
  try {
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : document.createElement("canvas");
    const gl = (canvas as HTMLCanvasElement).getContext("webgl") as WebGLRenderingContext | null;
    if (!gl) return null;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const raw = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return typeof raw === "string" && raw ? raw : null;
  } catch {
    return null;
  }
}

/** Detects the GPU (call from a page: popup or content script). */
export async function detectGpu(): Promise<GpuInfo> {
  const renderer = webglRenderer();
  let vendor: string | null = null;
  let architecture: string | null = null;
  let description: string | null = null;
  let noGpu = false;
  try {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    const adapter = gpu ? await gpu.requestAdapter(/Windows/.test(navigator.userAgent) ? undefined : { powerPreference: "high-performance" }) : null;
    if (!adapter) noGpu = true;
    else {
      const info = adapter.info;
      vendor = info?.vendor || null;
      architecture = info?.architecture || null;
      description = info?.description || null;
      if ((info as GPUAdapterInfo & { isFallbackAdapter?: boolean })?.isFallbackAdapter) noGpu = true;
    }
  } catch {
    // No WebGPU: fall back on WebGL alone.
  }
  const name = renderer ? cleanRendererName(renderer) : description;
  if (name && /SwiftShader|llvmpipe|Software|Basic Render/i.test(name)) noGpu = true;
  if (!renderer && noGpu === false && !vendor) noGpu = true;
  return {
    name,
    vendor,
    architecture,
    tableVramMib: lookupVramMib(name),
    unified: looksUnified(name, vendor),
    noGpu,
    detectedAt: Date.now(),
  };
}

export interface RamReading {
  totalMib: number;
  availableMib: number;
  /** The native helper's GPU reading, when the helper is installed. */
  helper?: HelperGpu | null;
  /** Resident RAM of the llama.cpp processes, from the helper. */
  llamaRamMib?: number;
}

export async function readRam(): Promise<RamReading | null> {
  try {
    const m = await chrome.system.memory.getInfo();
    return { totalMib: m.capacity / 1048576, availableMib: m.availableCapacity / 1048576 };
  } catch {
    return null;
  }
}

export interface CpuReading {
  model: string;
  threads: number;
}

export async function readCpu(): Promise<CpuReading | null> {
  try {
    const c = await chrome.system.cpu.getInfo();
    return { model: c.modelName.trim(), threads: c.numOfProcessors };
  } catch {
    return null;
  }
}

/** The memory picture the fit check runs against. */
export interface MemoryPicture {
  kind: "vram" | "unified" | "cpu";
  /** Device memory fit may fill, MiB (0 for CPU only). */
  deviceTotalMib: number;
  deviceFreeMib: number;
  ramTotalMib: number | null;
  ramAvailableMib: number | null;
  /** The VRAM total was not found and not entered. */
  vramUnknown: boolean;
  /** Totals and free memory come from the native helper, not an estimate. */
  measured: boolean;
  /** Unified memory fast enough to grade like VRAM (Apple silicon, Strix
   * Halo, GB10); an integrated GPU on ordinary DDR grades like the CPU. */
  fastUnified?: boolean;
}

export function memoryPicture(gpu: GpuInfo | null, ram: RamReading | null, s: Settings, helper: HelperGpu | null = null): MemoryPicture {
  const ramTotal = ram?.totalMib ?? null;
  const ramAvail = ram?.availableMib ?? null;
  if (helper && !s.cpuOnly && s.unifiedOverride == null) {
    const base = { ramTotalMib: ramTotal, ramAvailableMib: ramAvail, vramUnknown: false, measured: true };
    if (helper.integrated || helper.vendor === "Apple") {
      // Integrated GPUs run from the shared pool the OS gives them
      // (Windows: half of RAM; macOS: reported as all RAM, Metal uses ~3/4).
      const apple = helper.vendor === "Apple";
      const pool = helper.sharedTotalMib ?? ramTotal ?? 0;
      const total = apple ? pool * 0.75 : pool;
      const used = helper.sharedUsedMib ?? 0;
      const free = Math.max(0, Math.min(total, pool - used, ramAvail ?? Infinity));
      return { kind: "unified", deviceTotalMib: total, deviceFreeMib: free, ...base, fastUnified: apple || isFastUnified([helper.name]) };
    }
    const total = s.vramOverrideMib ?? helper.dedicatedTotalMib;
    const used = helper.dedicatedUsedMib ?? s.vramReservedMib;
    return { kind: "vram", deviceTotalMib: total, deviceFreeMib: Math.max(0, total - used), ...base };
  }
  return estimatedPicture(gpu, ram, s);
}

function estimatedPicture(gpu: GpuInfo | null, ram: RamReading | null, s: Settings): MemoryPicture {
  const ramTotal = ram?.totalMib ?? null;
  const ramAvail = ram?.availableMib ?? null;
  const noGpu = s.cpuOnly || !gpu || gpu.noGpu;
  if (noGpu) return { kind: "cpu", deviceTotalMib: 0, deviceFreeMib: 0, ramTotalMib: ramTotal, ramAvailableMib: ramAvail, vramUnknown: false, measured: false };
  const unified = s.unifiedOverride ?? gpu.unified;
  if (unified) {
    // Metal's working-set budget is about 3/4 of RAM; an integrated GPU
    // elsewhere can use what the OS has available.
    const apple = /apple/i.test(gpu.vendor ?? "") || /apple/i.test(gpu.name ?? "");
    const total = ramTotal != null ? (apple ? ramTotal * 0.75 : ramTotal) : 0;
    const free = ramAvail != null ? Math.min(total, ramAvail) : total;
    return {
      kind: "unified",
      deviceTotalMib: total,
      deviceFreeMib: free,
      ramTotalMib: ramTotal,
      ramAvailableMib: ramAvail,
      vramUnknown: false,
      measured: false,
      fastUnified: apple || isFastUnified([gpu.name, gpu.vendor]),
    };
  }
  const total = s.vramOverrideMib ?? gpu.tableVramMib;
  if (total == null) return { kind: "vram", deviceTotalMib: 0, deviceFreeMib: 0, ramTotalMib: ramTotal, ramAvailableMib: ramAvail, vramUnknown: true, measured: false };
  return {
    kind: "vram",
    deviceTotalMib: total,
    deviceFreeMib: Math.max(0, total - s.vramReservedMib),
    ramTotalMib: ramTotal,
    ramAvailableMib: ramAvail,
    vramUnknown: false,
    measured: false,
  };
}
