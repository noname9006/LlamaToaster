// Bread grades for "Will it fit?" (the Models page and the Chrome extension):
// a finer reading of shared/fitLevel.ts's three levels, drawn as toast
// (shared/breadMarks.ts).
//
// Two scales, picked by the machine:
//   gpu -- a GPU with its own VRAM, or fast unified memory (Apple silicon,
//          Strix Halo, GB10): 5 all layers at the maximum context, 4 all
//          layers at a shorter context, 3 all layers with MoE experts in RAM,
//          2 partial offload with at least half the layers on the GPU,
//          1 under half, 0 not enough memory.
//   ram -- CPU only, or an integrated GPU on ordinary DDR (the GPU barely
//          helps decoding there, so placement is not worth grading):
//          2 fits at the maximum context, 1 fits at a shorter one, 0 not
//          enough memory.

import type { FitLevel, FitMemory } from "./fitLevel.js";

export type FitScale = "gpu" | "ram";
export type FitGrade = 0 | 1 | 2 | 3 | 4 | 5;

/** What a second look at the model's maximum context found. */
export interface FitCeiling {
  /** The model's maximum context: the header's <arch>.context_length. Not
   * necessarily what it was trained on -- RoPE-scaled models (YaRN, Llama 4's
   * 10M) advertise far more than their training length. */
  trained_ctx: number;
  /** Fits as fully at the maximum context as at the context asked. */
  fits_at_trained: boolean;
  /** gpu scale: the longest context that keeps every layer on the GPU when
   * the maximum one does not (fit shrinks the context before it moves
   * layers). Null when unknown or under fit's 4096 floor. */
  max_full_ctx: number | null;
}

// Unified memory fast enough to grade like VRAM. Matched against the GPU's
// name or description (llama.cpp's --list-devices on the worker, WebGL's
// renderer string in the extension).
const FAST_UNIFIED = [
  /\bapple\b/i,
  /\bM[1-9]\s*(pro|max|ultra)?\b/i,
  /\bMTL\d*\b/, // Metal device names
  /radeon\S*\s*80[4-6]0S/i, // Strix Halo (Ryzen AI Max) iGPUs
  /ryzen ai max/i,
  /\bGB10\b|dgx spark/i,
];

export function isFastUnified(names: (string | null | undefined)[]): boolean {
  return names.some((n) => !!n && FAST_UNIFIED.some((re) => re.test(n)));
}

export function fitScale(memory: Pick<FitMemory, "devices" | "unified">): FitScale {
  if (memory.devices.length === 0) return "ram";
  if (memory.unified && !isFastUnified(memory.devices.flatMap((d) => [d.name, d.description]))) return "ram";
  return "gpu";
}

export interface FitGradeInput {
  level: FitLevel | null;
  /** Placement at the context asked (null when the check failed). */
  point: { layers_gpu: number | null; layers_total: number | null; overflow_layers: number } | null;
  scale: FitScale;
  /** Also fits at the model's maximum context (false when unknown). */
  fitsAtTrained: boolean;
}

export function fitGrade(i: FitGradeInput): FitGrade | null {
  if (i.level == null) return null;
  if (i.level === "no_fit") return 0;
  if (i.scale === "ram") return i.fitsAtTrained ? 2 : 1;
  if (i.level === "fits") return i.fitsAtTrained ? 5 : 4;
  const gpu = i.point?.layers_gpu ?? 0;
  const total = i.point?.layers_total ?? 0;
  if (total > 0 && gpu >= total && (i.point?.overflow_layers ?? 0) > 0) return 3;
  return total > 0 && gpu * 2 >= total ? 2 : 1;
}

/** Where the grade is shown: the Models page sizes at a fixed small context,
 * the extension at the context the user picked. */
export type GradeContextWording = "shorter" | "yours";

export function fitGradeLabel(scale: FitScale, grade: FitGrade, wording: GradeContextWording): string {
  const shorter = wording === "shorter" ? "at a shorter context than the maximum" : "at your context";
  if (scale === "ram") {
    if (grade === 2) return "Fits in RAM at the maximum context";
    if (grade === 1) return `Fits in RAM ${shorter}`;
    return "Not enough memory";
  }
  switch (grade) {
    case 5:
      return "All layers on the GPU at the maximum context";
    case 4:
      return `All layers on the GPU ${shorter}`;
    case 3:
      return "All layers on the GPU, experts in RAM";
    case 2:
      return "Partial offload: half the layers or more on the GPU";
    case 1:
      return "Partial offload: under half the layers on the GPU";
    default:
      return "Not enough memory";
  }
}

/** Every grade of a scale, best first, for a legend. */
/** A legend row lists every grade, so it says which ones only some models can get. */
export function fitGradeLegendLabel(scale: FitScale, grade: FitGrade, wording: GradeContextWording): string {
  const label = fitGradeLabel(scale, grade, wording);
  return scale === "gpu" && grade === 3 ? `${label} (MoE models only)` : label;
}

export function fitGrades(scale: FitScale): FitGrade[] {
  return scale === "ram" ? [2, 1, 0] : [5, 4, 3, 2, 1, 0];
}

/** "48k" style context, for tooltips. */
export function ctxShort(ctx: number): string {
  if (ctx >= 1048576) return `${+(ctx / 1048576).toFixed(2)}M`;
  return `${+(ctx / 1024).toFixed(1)}k`;
}
