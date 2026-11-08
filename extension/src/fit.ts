// One quant's answer at the chosen context: the app's header-only stand-in for
// llama-fit-params (shared/fitSimulate.ts) for the placement, the app's own
// three-level rule (shared/fitLevel.ts), and the bread grade drawn from it
// (shared/fitGrade.ts, shared/breadMarks.ts) -- the Models page's marks.

import { simulateFit, type FitSimPoint } from "../../shared/fitSimulate";
import { fitLevel, type FitLevel, type FitMemory } from "../../shared/fitLevel";
import { KV_TYPE_UNITS, type FitPoint } from "../../shared/fitParams";
import { FIT_MARKS } from "../../client/src/components/modelFit/fitCopy";
import { ctxShort, fitGrade, fitGradeLabel, type FitGrade, type FitScale } from "../../shared/fitGrade";
import type { ModelFacts } from "./hfHeader";
import type { MemoryPicture } from "./hardware";
import type { Settings } from "./settings";
import { ctxLabel } from "./settings";

export { FIT_MARKS };

export interface FitAnswer {
  level: FitLevel | null;
  /** Bread grade (shared/fitGrade.ts); null when the check could not finish. */
  grade: FitGrade | null;
  scale: FitScale;
  mark: string;
  headline: string;
  details: string[];
  /** The context actually checked (capped at the model's maximum context). */
  ctx: number;
  ctxCapped: boolean;
  sim: FitSimPoint | null;
}

const gb = (mib: number) => `${(mib / 1024).toFixed(1)} GB`;

function bytesPerElement(type: string): number {
  return (KV_TYPE_UNITS[type] ?? 64) / 32;
}

/** The device memory the fit may fill and how much of it stays free. */
export interface DeviceBudget {
  freeMib: number;
  marginMib: number;
  /** Why that much stays free, for the tooltip ("" for a plain amount). */
  marginNote: string;
  /** What freeMib is, for the tooltip: the device's whole memory or what it has available. */
  freeLabel: "total" | "available";
}

/** VRAM in use right now (total minus free): what "keep free: in use now" captures when a check starts. */
export function deviceInUseMib(mem: MemoryPicture): number {
  return Math.max(0, mem.deviceTotalMib - mem.deviceFreeMib);
}

/**
 * "In use now" keeps free what was in use when the check started and fits
 * against the whole device, so the marks stay put as other apps' usage moves
 * (or as llama.cpp itself loads). The other modes keep a fixed amount free out
 * of the device's whole memory (8 GB with 1 GB kept free: the model gets 7 GB),
 * so "in use now" is just the same rule with the amount other apps hold.
 *
 * Where the browser can't see what other apps hold (no worker), or the memory
 * is shared with RAM, the total isn't the right base: the user's own "other
 * apps use" figure is already taken off deviceFreeMib, and it stays.
 */
export function deviceBudget(mem: MemoryPicture, s: Settings, inUseAtStartMib: number | null): DeviceBudget {
  if (mem.kind === "cpu") return { freeMib: 0, marginMib: 0, marginNote: "", freeLabel: "available" };
  if (s.marginMode === "current") {
    return { freeMib: mem.deviceTotalMib, marginMib: inUseAtStartMib ?? deviceInUseMib(mem), marginNote: " (in use when the check started)", freeLabel: "total" };
  }
  const marginMib = s.marginMode === "custom" ? s.customMarginMib : s.marginMib;
  if (mem.kind === "vram" && mem.measured) return { freeMib: mem.deviceTotalMib, marginMib, marginNote: "", freeLabel: "total" };
  return { freeMib: mem.deviceFreeMib, marginMib, marginNote: "", freeLabel: "available" };
}

/** Which bread scale this machine grades on. */
export function memoryScale(mem: MemoryPicture): FitScale {
  if (mem.kind === "cpu") return "ram";
  if (mem.kind === "unified" && !mem.fastUnified) return "ram";
  return "gpu";
}

export function evaluateFit(facts: ModelFacts, mem: MemoryPicture, s: Settings, budget: DeviceBudget = deviceBudget(mem, s, null)): FitAnswer {
  const ctx = facts.trainedCtx && s.ctx > facts.trainedCtx ? facts.trainedCtx : s.ctx;
  const ctxCapped = ctx !== s.ctx;
  const hasDevice = mem.kind !== "cpu";
  const freeMib = hasDevice ? budget.freeMib : 0;
  const marginMib = hasDevice ? budget.marginMib : 0;
  const scale = memoryScale(mem);
  const memory: FitMemory = {
    devices: hasDevice ? [{ name: "gpu", description: "", total_mib: mem.deviceTotalMib, free_mib: freeMib }] : [],
    ram_available_mib: mem.ramAvailableMib,
    ram_total_mib: mem.ramTotalMib,
    unified: mem.kind === "unified",
  };

  const at = (c: number) => {
    const sim = simulateFit({
      breakdown: facts.breakdown,
      layout: facts.layout,
      nLayer: facts.nLayer,
      ctx: c,
      freeMib,
      marginMib,
      bytesK: bytesPerElement("f16"),
      bytesV: bytesPerElement("f16"),
    });
    if (!sim) return null;
    // fit says "doesn't fit" when even no layers on the GPU leave too little
    // device memory (the compute buffer alone can be GBs).
    const deviceShort = hasDevice && sim.layers_gpu === 0 && sim.dev_used_mib > freeMib - marginMib;
    const point = {
      verdict: deviceShort ? "doesnt_fit" : sim.verdict,
      dev_used_mib: sim.dev_used_mib,
      host_used_mib: sim.host_used_mib,
      dev_free_mib: freeMib - sim.dev_used_mib,
    } as FitPoint;
    return { sim, level: fitLevel(point, memory) };
  };

  const first = at(ctx);
  if (!first) return { level: null, grade: null, scale, mark: "⚠️", headline: "The check could not finish", details: [], ctx, ctxCapped, sim: null };
  const { sim, level } = first;

  // Grades 5 and 2 need the maximum context to fit as fully as this one.
  const full = (l: FitLevel | null) => (scale === "ram" ? l != null && l !== "no_fit" : l === "fits");
  let fitsAtTrained = false;
  if (full(level)) {
    if (!facts.trainedCtx || facts.trainedCtx <= ctx) fitsAtTrained = !!facts.trainedCtx;
    else fitsAtTrained = full(at(facts.trainedCtx)?.level ?? null);
  }
  const grade = fitGrade({ level, point: sim, scale, fitsAtTrained });

  const details: string[] = [];
  if (hasDevice) {
    details.push(`${mem.kind === "unified" ? "Unified memory" : "VRAM"}: needs ${gb(sim.dev_used_mib)} of ${gb(freeMib)} ${budget.freeLabel}, keeping ${gb(marginMib)} free${budget.marginNote}`);
  }
  if (mem.ramAvailableMib != null) details.push(`RAM: needs ${gb(sim.host_used_mib)} of ${gb(mem.ramAvailableMib)} available`);
  else details.push(`RAM: needs ${gb(sim.host_used_mib)}`);
  if (hasDevice) {
    details.push(`Layers on the GPU: ${sim.layers_gpu} of ${sim.layers_total}${sim.overflow_layers > 0 ? `, experts of ${sim.overflow_layers} layers in RAM` : ""}`);
  }
  details.push(`KV cache: ${gb(sim.kv_mib)} (f16) · compute buffer ${gb(sim.compute_mib)}`);
  details.push(`Context: ${ctx.toLocaleString("en-US")} tokens${ctxCapped ? ` (the model's maximum; you picked ${ctxLabel(s.ctx)})` : ""}`);
  details.push(`File: ${gb(facts.totalBytes / 1048576)}${facts.shards.length > 1 ? ` in ${facts.shards.length} parts` : ""}, ${facts.nLayer} layers`);

  return {
    level,
    grade,
    scale,
    mark: level ? FIT_MARKS[level] : "⚠️",
    headline: grade != null ? breadHeadline(scale, grade, sim, ctx, facts.trainedCtx) : headline(level, sim, mem),
    details,
    ctx,
    ctxCapped,
    sim,
  };
}

function breadHeadline(scale: FitScale, grade: FitGrade, p: FitSimPoint, ctx: number, trained: number | null | undefined): string {
  // Partial offload names the exact layer count instead of the grade's band.
  if (scale === "gpu" && (grade === 1 || grade === 2)) return `Partial offload: ${p.layers_gpu} of ${p.layers_total} layers on the GPU${p.overflow_layers > 0 ? `, ${expertsNote(p.overflow_layers, p.layers_gpu)}` : ""}`;
  const label = fitGradeLabel(scale, grade, "yours");
  let detail: string | null = null;
  if (grade === 0) detail = null;
  else if (scale === "ram" || grade === 4 || grade === 5) detail = grade === 5 || (scale === "ram" && grade === 2) ? (trained ? ctxShort(trained) : null) : `${ctxShort(ctx)}${trained ? ` of ${ctxShort(trained)}` : ""}`;
  else if (grade === 3) detail = `experts of ${p.overflow_layers} layers in RAM`;
  else detail = `${p.layers_gpu} of ${p.layers_total} layers on the GPU`;
  return detail ? `${label} (${detail})` : label;
}

// The output layer has no experts, so every GPU block's experts are in RAM
// once the overflow covers layers_gpu - 1.
function expertsNote(overflow: number, layersGpu: number): string {
  return overflow >= layersGpu - 1 ? "all experts in RAM" : `experts of ${overflow} layers in RAM`;
}

// fitCopy.ts's fitHeadline, for a simulated point.
function headline(level: FitLevel | null, p: FitSimPoint, mem: MemoryPicture): string {
  if (level === "fits") return mem.kind === "unified" ? "Fits in unified memory" : mem.kind === "cpu" ? "Fits in RAM" : "Fits in VRAM";
  if (level === "no_fit") return "Not enough memory: will not run, or will run with degraded speed";
  if (level === "partial") {
    if (p.layers_gpu === 0) return "Fits with partial offload: no layers fit on the GPU, it would run on the CPU";
    if (p.layers_gpu < p.layers_total) return `Fits with partial offload: ${p.layers_gpu} of ${p.layers_total} layers on the GPU`;
    if (p.overflow_layers > 0) return `Fits with partial offload: experts of ${p.overflow_layers} layers in RAM`;
    return "Fits with partial offload";
  }
  return "The check could not finish";
}
