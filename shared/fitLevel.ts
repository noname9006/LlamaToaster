// "Will it fit?" on the Models page (docs/plans/MODEL_FIT_CHECK_PLAN.md): one of
// three levels from a llama-fit-params answer and what the machine has available.
//
// Fit only sizes the GPU: it never checks RAM against what is available, and its
// budget is the driver's own figure. The RAM side is judged here, against the
// OS's reading -- llama.cpp's --list-devices has no row for the CPU.
//
// Computed on read rather than stored, so a change to this rule applies to
// answers already on file.

import type { FitPoint } from "./fitParams.js";

export type FitLevel = "fits" | "partial" | "no_fit";

/** The mark each level is drawn with, on the Models page and in the Chrome extension. */
export const FIT_MARKS: Record<FitLevel, string> = { fits: "✅✅", partial: "✅", no_fit: "❗" };

export interface FitMemory {
  /** What --list-devices reports for each GPU; its "free" is really "available". */
  devices: { name: string; description: string; total_mib: number; free_mib: number }[];
  ram_available_mib: number | null;
  ram_total_mib: number | null;
  /** The GPU draws from system memory (Apple silicon, integrated GPUs). */
  unified: boolean;
}

export function fitLevel(point: FitPoint, memory: FitMemory): FitLevel | null {
  if (point.verdict === "error") return null;
  const host = point.host_used_mib ?? 0;
  const device = point.dev_used_mib ?? 0;
  const ram = memory.ram_available_mib;

  // No GPU: all of it is one pool, so there is no partial offload to speak of.
  if (memory.devices.length === 0) {
    if (ram != null) return host <= ram ? "fits" : "no_fit";
    return point.verdict === "doesnt_fit" ? "no_fit" : "fits";
  }

  if (memory.unified) {
    // macOS under-reports free RAM, and Metal's budget is the better figure
    // there; elsewhere the OS's is. Either way one pool serves both sides.
    const deviceFree = memory.devices.reduce((sum, d) => sum + d.free_mib, 0);
    if (device + host > Math.max(deviceFree, ram ?? 0)) return "no_fit";
    return point.verdict === "full" ? "fits" : "partial";
  }

  if (ram != null && host > ram) return "no_fit";
  if (point.verdict === "full") return "fits";
  // No layer fit within the margin, but what stays on the device still fits: it
  // runs on the CPU. Over what the device has, even with no layers on it (the
  // compute buffer alone can be GBs), it spills or fails: that is not enough memory.
  if (point.verdict === "doesnt_fit") return (point.dev_free_mib ?? 0) < 0 ? "no_fit" : "partial";
  return "partial";
}

/** Same file, same answer, whichever list it was asked from. Null for a model
 * with neither a Hugging Face source nor an id. */
export function fitTargetKey(m: { id?: string; hf_repo?: string | null; hf_file?: string | null }): string | null {
  if (m.hf_repo && m.hf_file) return `hf:${m.hf_repo}/${m.hf_file}`;
  return m.id ? `model:${m.id}` : null;
}
