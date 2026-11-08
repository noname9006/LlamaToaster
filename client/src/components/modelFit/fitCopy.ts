// The words of "Will it fit?" (docs/plans/MODEL_FIT_CHECK_PLAN.md), apart from
// the components so they can be tested and read in one place.

import { FIT_MARKS } from "../../../../shared/fitLevel.js";
import { ctxShort, fitGrade, fitGradeLabel, fitScale, type FitGrade, type FitScale } from "../../../../shared/fitGrade.js";
import type { ModelFitResult } from "../../types";
import { formatBytes } from "../../utils";

export { FIT_MARKS };

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

/** The bread grade of an answer (shared/fitGrade.ts), null when the check failed. */
export function fitBreadGrade(r: ModelFitResult): { scale: FitScale; grade: FitGrade } | null {
  const scale = fitScale(r.memory);
  const grade = fitGrade({ level: r.level, point: r.point, scale, fitsAtTrained: r.ceiling?.fits_at_trained ?? false });
  return grade == null ? null : { scale, grade };
}

/** The bread mark's tooltip: what the grade means, then the figure behind it. */
export function fitBreadHeadline(r: ModelFitResult): string {
  const g = fitBreadGrade(r);
  if (!g) return fitHeadline(r);
  const detail = fitBreadDetail(r, g.scale, g.grade);
  // Partial offload names the exact layer count instead of the grade's band.
  if (g.scale === "gpu" && (g.grade === 1 || g.grade === 2) && detail) return `Partial offload: ${detail}`;
  const label = fitGradeLabel(g.scale, g.grade, "shorter");
  return detail ? `${label} (${detail})` : label;
}

function fitBreadDetail(r: ModelFitResult, scale: FitScale, grade: FitGrade): string | null {
  const p = r.point;
  const c = r.ceiling;
  if (grade === 0) return null;
  if (scale === "ram") return c ? `max ${ctxShort(c.trained_ctx)}` : null;
  if (grade === 5) return c ? `${ctxShort(c.trained_ctx)}` : null;
  if (grade === 4) {
    if (c?.max_full_ctx) return `up to ${ctxShort(c.max_full_ctx)} of ${ctxShort(c.trained_ctx)}`;
    if (c) return `under 4k of ${ctxShort(c.trained_ctx)}`;
    return null;
  }
  if (!p || p.layers_total == null || p.layers_gpu == null) return null;
  if (grade === 3) return `experts of ${p.overflow_layers} layers in RAM`;
  return `${p.layers_gpu} of ${p.layers_total} layers on the GPU${p.overflow_layers > 0 ? `, ${expertsNote(p.overflow_layers, p.layers_gpu)}` : ""}`;
}

// The output layer has no experts, so every GPU block's experts are in RAM
// once the overflow covers layers_gpu - 1.
function expertsNote(overflow: number, layersGpu: number): string {
  return overflow >= layersGpu - 1 ? "all experts in RAM" : `experts of ${overflow} layers in RAM`;
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
  const c = r.ceiling;
  if (c) {
    const trained = `${c.trained_ctx.toLocaleString("en-US")} tokens`;
    if (c.fits_at_trained) lines.push(`Maximum context: ${trained}, fits as well`);
    else if (c.max_full_ctx) lines.push(`Maximum context: ${trained}; every layer stays on the GPU up to ${c.max_full_ctx.toLocaleString("en-US")}`);
    else lines.push(`Maximum context: ${trained}, does not fit as well`);
  }
  lines.push(`llama.cpp ${r.llama_cpp_build} (${r.backend})`);
  if (r.source === "header") lines.push(`Sized from the model's header: ${r.fetched_bytes != null ? formatBytes(r.fetched_bytes) : "a few MB"} downloaded, then deleted`);
  else if (r.source === "local") lines.push("Sized from the file already on this machine");
  return lines;
}
