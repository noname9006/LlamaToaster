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

/** How much VRAM the fit keeps free: a fixed amount, what is in use when the
 * check starts (captured once, so the marks don't move as usage moves), or
 * an amount the user typed. */
export type MarginMode = "preset" | "current" | "custom";

export interface Settings {
  /** "auto": every GGUF page is checked as it opens. "manual": only when the
   * toaster on the page is clicked. */
  mode: "manual" | "auto";
  ctx: number;
  marginMode: MarginMode;
  /** The preset amount (marginMode "preset"), MiB. */
  marginMib: number;
  /** The typed amount (marginMode "custom"), MiB. */
  customMarginMib: number;
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
  customMarginMib: 2048,
  vramOverrideMib: null,
  vramReservedMib: 512,
  unifiedOverride: null,
  cpuOnly: false,
  sidePanel: false,
};

/** Labelled anchor points (tokens); the slider itself is continuous between them. */
export const CTX_STOPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 10485760];
export const CTX_MIN = CTX_STOPS[0];
export const CTX_MAX = CTX_STOPS[CTX_STOPS.length - 1];
/** 1k..1M takes 100 positions per doubling. The 1M..10M stretch is rare (Llama 4
 * Scout, Mistral Nemo's header) and gets one short segment at the end. */
const PER_DOUBLING = 100;
const MAIN_END = (CTX_STOPS.length - 2) * PER_DOUBLING; // 1M
const TAIL = 70;
export const CTX_SLIDER_MAX = MAIN_END + TAIL;
const MAIN_MAX = CTX_STOPS[CTX_STOPS.length - 2];
/** Within this many positions of an anchor the thumb snaps to it (the anchors are "magnetic"). */
const SNAP_RADIUS = 12;
const TAIL_SNAP = 6;

/** Where each labelled anchor sits on the slider. */
export function stopPositions(): number[] {
  return CTX_STOPS.map((_, i) => (i < CTX_STOPS.length - 1 ? i * PER_DOUBLING : CTX_SLIDER_MAX));
}

export function ctxFromSlider(pos: number): number {
  if (pos > MAIN_END) {
    if (CTX_SLIDER_MAX - pos <= TAIL_SNAP) return CTX_MAX;
    if (pos - MAIN_END <= TAIL_SNAP) return MAIN_MAX;
    const raw = MAIN_MAX * (CTX_MAX / MAIN_MAX) ** ((pos - MAIN_END) / TAIL);
    return Math.min(CTX_MAX, Math.round(raw / 131072) * 131072);
  }
  const nearest = Math.round(pos / PER_DOUBLING);
  if (Math.abs(pos - nearest * PER_DOUBLING) <= SNAP_RADIUS) return CTX_STOPS[nearest];
  const raw = CTX_MIN * 2 ** (pos / PER_DOUBLING);
  const step = raw < 8192 ? 256 : raw < 65536 ? 1024 : 8192;
  return Math.min(MAIN_MAX, Math.max(CTX_MIN, Math.round(raw / step) * step));
}

export function sliderFromCtx(ctx: number): number {
  const c = Math.min(CTX_MAX, Math.max(CTX_MIN, ctx));
  if (c > MAIN_MAX) return Math.round(MAIN_END + (Math.log(c / MAIN_MAX) / Math.log(CTX_MAX / MAIN_MAX)) * TAIL);
  return Math.round(Math.log2(c / CTX_MIN) * PER_DOUBLING);
}

export function ctxLabel(ctx: number): string {
  if (ctx >= 1048576) return `${+(ctx / 1048576).toFixed(2)}M`;
  return `${+(ctx / 1024).toFixed(2)}k`;
}

/** Tick labels placed where the slider thumb actually sits (22px thumb inside a
 * 1px track border: its centre travels from 12px to W-12). Up to maxPos only
 * (a slider capped at a model's maximum shows the ticks it can reach). */
export function ticksHtml(maxPos = CTX_SLIDER_MAX): string {
  const pos = stopPositions();
  return CTX_STOPS.map((c, i) => [c, pos[i]] as const)
    .filter(([, p]) => p <= maxPos)
    // The last two anchors sit close together: the 10M label hangs right of its tick.
    .map(([c, p]) => `<span class="${c === CTX_MAX ? "tail" : ""}" style="left: calc(12px + (100% - 24px) * ${p / maxPos})">${ctxLabel(c)}</span>`)
    .join("");
}

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
  return { ...DEFAULT_SETTINGS, ...(saved ?? {}) };
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
