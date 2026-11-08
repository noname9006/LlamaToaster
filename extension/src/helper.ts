// The worker's VRAM host (worker/chrome-host/, registered by the worker
// installer and updater). Chrome exposes no VRAM figures to extensions, so
// this native-messaging host reads them:
// DXGI + the GPU Adapter Memory counters on Windows (any vendor), DRM sysfs
// and nvidia-smi on Linux, sysctl/vm_stat on macOS.

export const HELPER_NAME = "com.llamatoaster.vram";

export interface HelperGpu {
  name: string;
  vendor: string;
  vendorId: number;
  deviceId: number;
  /** A carve-out of <= 512 MiB: the GPU really runs from shared RAM. */
  integrated: boolean;
  dedicatedTotalMib: number;
  sharedTotalMib: number | null;
  dedicatedUsedMib: number | null;
  sharedUsedMib: number | null;
  /** What this machine's llama.cpp processes (LlamaToaster's own) hold on this GPU. */
  llamaDedicatedMib?: number;
  llamaSharedMib?: number;
  /** The apps holding the most dedicated VRAM on this GPU (only when asked for apps; null where the driver names no processes). */
  apps?: HelperApp[] | null;
}

/** One app (all its processes together) and the memory it holds. */
export interface HelperApp {
  name: string;
  mib: number;
}

/** The installed RAM, from SMBIOS (Windows: Win32_PhysicalMemory; macOS:
 * system_profiler; Linux: dmidecode, which usually needs root). */
export interface HelperRam {
  /** "DDR4", "DDR5", "LPDDR5", ... null when the firmware doesn't say. */
  type: string | null;
  /** The speed the memory runs at, MT/s (the slowest module's). */
  mts: number | null;
  /** The modules' rated speed, MT/s, when it differs (XMP/EXPO off, mixed kits). */
  ratedMts: number | null;
  modules: number;
}

export interface HelperReading {
  os: string;
  gpus: HelperGpu[];
  /** Only when asked for ({ ram: true }) and the helper is new enough. */
  ram: HelperRam | null;
  /** Resident RAM of the llama.cpp processes. */
  llamaRamMib: number;
  /** The apps holding the most RAM (only when asked for apps and the helper is new enough). */
  topRam: HelperApp[] | null;
  readAt: number;
}

export type HelperResult = { ok: true; reading: HelperReading } | { ok: false; missing: boolean; error: string };

/** Ask the helper for a fresh reading. Only works from extension pages and the service worker. */
export function readHelper(opts: { ram?: boolean; apps?: boolean } = {}): Promise<HelperResult> {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendNativeMessage(HELPER_NAME, { cmd: "read", ram: !!opts.ram, apps: !!opts.apps }, (resp) => {
        const err = chrome.runtime.lastError;
        if (err || !resp) {
          const msg = err?.message ?? "no reply";
          resolve({ ok: false, missing: /not found|forbidden|not allowed/i.test(msg), error: msg });
        } else if (!resp.ok) {
          resolve({ ok: false, missing: false, error: String(resp.error ?? "helper error") });
        } else {
          resolve({ ok: true, reading: { os: resp.os, gpus: resp.gpus ?? [], ram: resp.ram ?? null, llamaRamMib: resp.llamaRamMib ?? 0, topRam: Array.isArray(resp.topRam) ? resp.topRam : null, readAt: Date.now() } });
        }
      });
    } catch (e) {
      resolve({ ok: false, missing: true, error: e instanceof Error ? e.message : String(e) });
    }
  });
}

/** The GPU a model would run on: the discrete card with the most VRAM, else the integrated one. */
export function primaryGpu(r: HelperReading | null): HelperGpu | null {
  if (!r || r.gpus.length === 0) return null;
  const discrete = r.gpus.filter((g) => !g.integrated).sort((a, b) => b.dedicatedTotalMib - a.dedicatedTotalMib);
  return discrete[0] ?? r.gpus[0];
}

/** "DDR4-3600", "LPDDR5-6400", "DDR5-4800 (rated 6000)"; null when nothing is known. */
export function ramLabel(r: HelperRam | null | undefined): string | null {
  if (!r || (!r.type && !r.mts)) return null;
  const base = [r.type, r.mts].filter(Boolean).join("-");
  return r.ratedMts && r.mts && r.ratedMts > r.mts ? `${base} (rated ${r.ratedMts})` : base;
}
