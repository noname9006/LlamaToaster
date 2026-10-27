import type { HardwareInfo, Model, Worker } from "../../types";
import { backendVisibleGpus } from "../../types";
import {
  estimateKvCacheMib,
  estimateKvLayoutMib,
  estimateRuntimeExtrasMib,
  placeWeightPools,
  RAM_ESTIMATE_FIXED_OVERHEAD_MIB,
  VRAM_ESTIMATE_FIXED_OVERHEAD_MIB,
} from "../../vramEstimate";
import type { LtIconName } from "../ltIcons";
import { fmtGb } from "./ui";
import { ctxLabel } from "../../../../shared/optimizeFlow.js";
import { simulateFit } from "../../../../shared/fitSimulate.js";
import { DEFAULT_FIT_MARGIN_MIB } from "../../../../shared/fitParams.js";

const BYTES_PER_MIB = 1048576;

// What kind of memory a machine has, and the quick pre-run verdict for a
// model on it -- shared by the New test model list and the Models fit table.

export interface MachineKind {
  hasGpu: boolean;
  unified: boolean;
  gpus: HardwareInfo["gpu"];
}

export function machineKind(w: Worker): MachineKind {
  const gpus = w.hardware ? (w.backend ? backendVisibleGpus(w.hardware.gpu, w.backend) : w.hardware.gpu) : [];
  const hasGpu = w.backend !== "cpu" && gpus.length > 0;
  const unified = w.backend === "metal" || w.hardware?.unified_memory === true || gpus.some((g) => g.vram_dynamic === true);
  return { hasGpu, unified, gpus };
}

export function stateLabel(w: Worker): string {
  if (w.status === "offline") return "Offline";
  if (w.status === "busy") return "Testing";
  return "Online · idle";
}

export interface Verdict {
  icon: LtIconName;
  text: string;
  tone: string;
}

// One fit answer for a model at one context: llama-fit-params' own (a fit
// check on the worker) or the header-only simulation of it.
export interface FitAnswer {
  layers_gpu: number;
  layers_total: number;
  overflow_layers: number;
  dev_used_mib: number;
  host_used_mib: number;
  source: "fit" | "estimate";
}

// The header-only stand-in for llama-fit-params (shared/fitSimulate.ts), at
// fit's default margin so it reads like a fit check. Null when the row lacks
// the header facts it needs.
export function estimateFitAnswer(model: Model, freeMib: number, ctx: number): FitAnswer | null {
  const n = model.metadata.n_layer;
  const breakdown = model.metadata.tensor_layer_bytes ?? null;
  const layout = model.metadata.kv_layout ?? null;
  if (typeof n !== "number" || n <= 0 || !breakdown || !layout) return null;
  const p = simulateFit({ breakdown, layout, nLayer: n, ctx, freeMib, marginMib: DEFAULT_FIT_MARGIN_MIB });
  return p ? { ...p, source: "estimate" } : null;
}

function fitVerdict(a: FitAnswer, freeMib: number, ramMib: number | null, ctx: number): Verdict {
  const tag = a.source === "fit" ? " · llama.cpp fit" : " (estimate)";
  const ramPart = ramMib != null ? ` · RAM ${fmtGb(a.host_used_mib)} / ${fmtGb(ramMib, 0)}` : "";
  if (ramMib != null && a.host_used_mib > ramMib * 0.9) {
    return { icon: "oom", text: `Too big for RAM · ${fmtGb(a.host_used_mib)} / ${fmtGb(ramMib, 0)} at ${ctxLabel(ctx)}${tag}`, tone: "text-danger" };
  }
  const vram = `VRAM ${fmtGb(a.dev_used_mib)} / ${fmtGb(freeMib)}`;
  if (a.layers_gpu >= a.layers_total && a.overflow_layers === 0) {
    return { icon: "ok", text: `Fits · ${vram}${ramPart}${tag}`, tone: "text-fg" };
  }
  if (a.layers_gpu >= a.layers_total) {
    return { icon: "spill", text: `Fits with ${a.overflow_layers} layers' experts in RAM · ${vram}${ramPart}${tag}`, tone: "text-warning" };
  }
  if (a.layers_gpu === 0) {
    return { icon: "spill", text: `No layers fit in VRAM at ${ctxLabel(ctx)} · runs on the CPU${ramPart}${tag}`, tone: "text-danger" };
  }
  return {
    icon: "spill",
    text: `${a.layers_gpu} of ${a.layers_total} layers on the GPU at ${ctxLabel(ctx)}${ramPart}${tag}`,
    tone: "text-warning",
  };
}

export function modelVerdict(model: Model, w: Worker | undefined, ctx: number, fit?: FitAnswer | null): Verdict | null {
  if (!w) return null;
  const n = model.metadata.n_layer;
  if (typeof n !== "number" || n <= 0) return { icon: "chip", text: "Layer count unknown", tone: "text-muted" };
  {
    const { hasGpu, unified } = machineKind(w);
    const free = w.vram?.vram_free_before_mib ?? (w.hardware?.gpu[0]?.vram_usable_mb ?? w.hardware?.gpu[0]?.vram_mb ?? null);
    const ramMib = w.hardware?.mem_total_bytes ? w.hardware.mem_total_bytes / BYTES_PER_MIB : null;
    if (hasGpu && !unified && free != null) {
      const answer = fit ?? estimateFitAnswer(model, free, ctx);
      if (answer) return fitVerdict(answer, free, ramMib, ctx);
    }
  }
  // Weights split the way llama.cpp places them at full offload: the
  // transformer blocks (+ output) in VRAM, the token embedding and misc
  // tensors in system RAM. No breakdown (older rows): the whole file in VRAM.
  const breakdown = model.metadata.tensor_layer_bytes ?? null;
  let gpuWeights: number;
  let cpuWeights: number;
  if (breakdown) {
    const placed = placeWeightPools(breakdown, n, n + 1);
    gpuWeights = placed.gpuBytes / BYTES_PER_MIB;
    cpuWeights = placed.cpuBytes / BYTES_PER_MIB;
  } else if (model.size_bytes > 0) {
    gpuWeights = model.size_bytes / BYTES_PER_MIB;
    cpuWeights = 0;
  } else {
    return null;
  }
  // Header-read KV layout first: counts only the layers that really cache
  // tokens (hybrid recurrent layers and SWA windows sized as llama.cpp does).
  // Older catalog rows without one fall back to the all-layers formula.
  const layout = model.metadata.kv_layout;
  const kv = layout
    ? estimateKvLayoutMib(layout, { tokens: ctx, cacheTypeK: "f16", cacheTypeV: "f16" })
    : typeof model.metadata.n_head_kv === "number"
      ? estimateKvCacheMib({
          nLayer: n,
          nHeadKv: model.metadata.n_head_kv,
          headDimK: model.metadata.head_dim_k ?? undefined,
          headDimV: model.metadata.head_dim_v ?? undefined,
          nEmbd: model.metadata.n_embd ?? undefined,
          nHead: model.metadata.n_head ?? undefined,
          cacheTypeK: "f16",
          cacheTypeV: "f16",
          tokens: ctx,
        })
      : null;
  // Recurrent state + compute buffer from the header; a flat allowance when
  // the row predates those fields.
  const runtime = layout
    ? estimateRuntimeExtrasMib(layout) + (layout.vocab ? 0 : VRAM_ESTIMATE_FIXED_OVERHEAD_MIB)
    : VRAM_ESTIMATE_FIXED_OVERHEAD_MIB;
  const deviceNeed = gpuWeights + (kv ?? 0) + runtime;
  const { hasGpu, unified } = machineKind(w);
  const ramMib = w.hardware?.mem_total_bytes ? w.hardware.mem_total_bytes / BYTES_PER_MIB : null;
  if (!hasGpu || unified) {
    if (ramMib == null) return null;
    const need = deviceNeed + cpuWeights;
    return need <= ramMib * 0.9
      ? { icon: "chip", text: `${unified ? "Unified" : "CPU only"} · ${fmtGb(need)} / ${fmtGb(ramMib, 0)}`, tone: "text-fg-2" }
      : { icon: "oom", text: "Too big for RAM", tone: "text-danger" };
  }
  const free = w.vram?.vram_free_before_mib ?? (w.hardware?.gpu[0]?.vram_usable_mb ?? w.hardware?.gpu[0]?.vram_mb ?? null);
  if (free == null) return null;
  // Two pools: the GPU holds blocks + KV + compute, system RAM holds what
  // llama.cpp keeps CPU-side (+ the runtime's own allowance).
  const vramBudget = free * 0.95;
  const ramBudget = ramMib != null ? ramMib * 0.9 : null;
  const ramNeed = cpuWeights + RAM_ESTIMATE_FIXED_OVERHEAD_MIB;
  const ramPart = ramMib != null ? ` · RAM ${fmtGb(ramNeed)} / ${fmtGb(ramMib, 0)}` : "";
  if (deviceNeed <= vramBudget) {
    if (ramBudget != null && ramNeed > ramBudget) {
      return { icon: "oom", text: `Too big for RAM · ${fmtGb(ramNeed)} / ${fmtGb(ramMib!, 0)}`, tone: "text-danger" };
    }
    return { icon: "ok", text: `Fits · VRAM ${fmtGb(deviceNeed)} / ${fmtGb(free)}${ramPart}`, tone: "text-fg" };
  }
  // Rough on purpose (f16 KV, default ubatch; rows without kv_layout
  // overstate hybrid/SWA KV); the fit map replaces it with llama.cpp's own
  // sizing.
  const over = deviceNeed - vramBudget;
  const overText = `~${fmtGb(over)} over VRAM at ${ctxLabel(ctx)} (estimate)`;
  if (ramBudget == null) return { icon: "spill", text: overText, tone: "text-danger" };
  // The overflow has to live in RAM too (partial offload).
  if (ramNeed + over > ramBudget) {
    return { icon: "oom", text: `${overText} · no room in RAM (${fmtGb(ramNeed + over)} / ${fmtGb(ramMib!, 0)})`, tone: "text-danger" };
  }
  return { icon: "spill", text: `${overText} · fits with offload to RAM (${fmtGb(ramNeed + over)} / ${fmtGb(ramMib!, 0)})`, tone: "text-warning" };
}
