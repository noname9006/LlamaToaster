// What the user picks (context, cache type, margin, memory overrides) and what
// was detected/measured, kept in chrome.storage so the popup and every
// Hugging Face tab see the same values and update together.

export interface GpuInfo {
  /** Best name found: WebGL's unmasked renderer, else WebGPU's description. */
  name: string | null;
  vendor: string | null;
  architecture: string | null;
  /** Total VRAM from the lookup table, MiB; null when the GPU is not in it. */
  tableVramMib: number | null;
  /** The GPU shares system memory (Apple silicon, integrated graphics). */
  unified: boolean;
  /** WebGPU is unavailable or only a software adapter answered. */
  noGpu: boolean;
  detectedAt: number;
}

export interface BandwidthResult {
  measuredAt: number;
  /** How READ was measured: "wasm-simd" (memory-bound) or "js" (a lower bound: the loop, not the memory, is the limit). Absent on results saved by older versions, which were all "js". */
  ram: { logicalCores: number; readKernel?: "wasm-simd" | "js"; points: { threads: number; readGBs: number; writeGBs: number; copyGBs: number }[] };
  vram: { deviceName: string; bufferMib: number; readGBs: number; writeGBs: number; copyGBs: number } | null;
  vramUnavailableReason?: string;
}

/** How much memory the fit keeps free: a fixed amount (any value, set on the
 * popup's lever), or what is in use when the check starts (captured once, so
 * the marks don't move as usage moves). "custom" is an older saved value: it
 * reads as "preset" with the typed amount. */
export type MarginMode = "preset" | "current";

export interface Settings {
  /** "auto": every GGUF page is checked as it opens. "manual": only when the
   * toaster on the page is clicked. */
  mode: "manual" | "auto";
  ctx: number;
  marginMode: MarginMode;
  /** The fixed amount kept free on the GPU (marginMode "preset"), MiB. */
  marginMib: number;
  /** Kept free in system RAM: the same two modes as the GPU's. */
  ramMarginMode: MarginMode;
  /** The fixed amount kept free in RAM (ramMarginMode "preset"), MiB. */
  ramMarginMib: number;
  /** User-entered total VRAM, MiB; null = use the table. */
  vramOverrideMib: number | null;
  /** VRAM already taken by the desktop and other apps, MiB. A browser cannot
   * read this, so it is the user's estimate. */
  vramReservedMib: number;
  /** null = auto (detected). */
  unifiedOverride: boolean | null;
  /** Run as CPU-only regardless of the GPU found. */
  cpuOnly: boolean;
  /** The toolbar button opens the side panel instead of the popup. */
  sidePanel: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  mode: "manual",
  ctx: 4096,
  marginMode: "preset",
  marginMib: 1024,
  // "In use now" for RAM is what the check did before RAM had a setting:
  // the model gets what the OS reports as available.
  ramMarginMode: "current",
  ramMarginMib: 4096,
  vramOverrideMib: null,
  vramReservedMib: 512,
  unifiedOverride: null,
  cpuOnly: false,
  sidePanel: false,
};

export * from "./ctxSlider";

// Settings live in storage.local: storage.sync allows ~120 writes a minute,
// and dragging the context slider writes on every tick, so sync ran out of
// quota and the pages silently stopped following the slider. Old sync values
// are read once as a fallback.
export async function getSettings(): Promise<Settings> {
  const { settings } = await chrome.storage.local.get("settings");
  let saved = settings as Partial<Settings> | undefined;
  if (!saved) {
    try {
      saved = (await chrome.storage.sync.get("settings")).settings as Partial<Settings> | undefined;
    } catch {
      // sync unavailable: defaults
    }
  }
  const out = { ...DEFAULT_SETTINGS, ...(saved ?? {}) } as Settings & { customMarginMib?: number };
  // The typed-amount mode became the lever's free value.
  if ((out.marginMode as string) === "custom") {
    out.marginMode = "preset";
    if (typeof out.customMarginMib === "number") out.marginMib = out.customMarginMib;
  }
  delete out.customMarginMib;
  return out;
}

// Writes run one after another, so a slow early write can never land after
// (and overwrite) a later one.
let writeChain: Promise<unknown> = Promise.resolve();
export function setSettings(patch: Partial<Settings>): Promise<Settings> {
  const run = writeChain.then(async () => {
    const next = { ...(await getSettings()), ...patch };
    await chrome.storage.local.set({ settings: next });
    return next;
  });
  writeChain = run.catch(() => undefined);
  return run;
}

export async function getGpu(): Promise<GpuInfo | null> {
  const { gpu } = await chrome.storage.local.get("gpu");
  return (gpu as GpuInfo | undefined) ?? null;
}

export async function getBandwidth(): Promise<BandwidthResult | null> {
  const { bandwidth } = await chrome.storage.local.get("bandwidth");
  return (bandwidth as BandwidthResult | undefined) ?? null;
}
