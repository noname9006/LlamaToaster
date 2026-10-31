// The words of "Will it fit?" (docs/plans/MODEL_FIT_CHECK_PLAN.md), apart from
// the components so they can be tested and read in one place.

import type { FitLevel } from "../../../../shared/fitLevel.js";
import type { ModelFitResult } from "../../types";
import { formatBytes } from "../../utils";

export const FIT_MARKS: Record<FitLevel, string> = { fits: "✅✅", partial: "✅", no_fit: "❗" };

export const ACCURACY_NOTE =
  "This measurement is quite accurate, but in certain circumstances the real experience may differ, depending on context size, cache quantization and other details.";

// A header is about this much on the models measured; only the first shard of a
// split model carries the tokenizer.
const FIRST_FILE_MIB = 15;
const LATER_SHARD_MIB = 1;

/** Roughly what checking one file downloads. Shown as an estimate. */
export function headerDownloadMib(shards: number): number {
  return FIRST_FILE_MIB + Math.max(0, shards - 1) * LATER_SHARD_MIB;
}

const gb = (mib: number) => `${(mib / 1024).toFixed(1)} GB`;

function memoryKind(r: ModelFitResult): "vram" | "unified" | "ram" {
  if (r.memory.devices.length === 0) return "ram";
  return r.memory.unified ? "unified" : "vram";
}

/** One line: what the mark means for this answer. */
export function fitHeadline(r: ModelFitResult): string {
  if (r.level === "fits") {
    const kind = memoryKind(r);
    return kind === "unified" ? "Fits in unified memory" : kind === "ram" ? "Fits in RAM" : "Fits in VRAM";
  }
  if (r.level === "no_fit") return "Not enough memory: will not run, or will run with degraded speed";
  if (r.level === "partial") {
    const p = r.point;
    if (p?.layers_gpu === 0) return "Fits with partial offload: no layers fit on the GPU, it would run on the CPU";
    if (p && p.layers_total != null && p.layers_gpu != null && p.layers_gpu < p.layers_total) {
      return `Fits with partial offload: ${p.layers_gpu} of ${p.layers_total} layers on the GPU`;
    }
    if (p && p.overflow_layers > 0) return `Fits with partial offload: experts of ${p.overflow_layers} layers in RAM`;
    return "Fits with partial offload";
  }
  return "The check could not finish";
}

/** Supporting figures for the details view, one per line. */
export function fitDetails(r: ModelFitResult): string[] {
  const lines: string[] = [];
  const p = r.point;
  if (p) {
    const available = r.memory.devices.reduce((sum, d) => sum + d.free_mib, 0);
    if (r.memory.devices.length > 0) {
      lines.push(`${r.memory.unified ? "Unified memory" : "VRAM"}: needs ${gb(p.dev_used_mib ?? 0)} of ${gb(available)} available, keeping ${gb(r.margin_mib)} free`);
    }
    if (r.memory.ram_available_mib != null) lines.push(`RAM: needs ${gb(p.host_used_mib ?? 0)} of ${gb(r.memory.ram_available_mib)} available`);
    else lines.push(`RAM: needs ${gb(p.host_used_mib ?? 0)}`);
    lines.push(`Context: ${p.ctx.toLocaleString("en-US")} tokens`);
  }
  lines.push(`llama.cpp ${r.llama_cpp_build} (${r.backend})`);
  if (r.source === "header") lines.push(`Sized from the model's header: ${r.fetched_bytes != null ? formatBytes(r.fetched_bytes) : "a few MB"} downloaded, then deleted`);
  else if (r.source === "local") lines.push("Sized from the file already on this machine");
  return lines;
}
