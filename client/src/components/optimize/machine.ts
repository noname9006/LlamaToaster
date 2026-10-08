import type { HardwareInfo, Model, Worker } from "../../types";
import { backendVisibleGpus } from "../../types";
import { estimateKvCacheMib, estimateVramNeededMib } from "../../vramEstimate";
import type { LtIconName } from "../ltIcons";
import { fmtGb } from "./ui";
import { ctxLabel } from "../../../../shared/optimizeFlow.js";

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

export function modelVerdict(model: Model, w: Worker | undefined, ctx: number): Verdict | null {
  if (!w) return null;
  const n = model.metadata.n_layer;
  if (typeof n !== "number" || n <= 0) return { icon: "chip", text: "Layer count unknown", tone: "text-muted" };
  const weights = estimateVramNeededMib({
    modelSizeBytes: model.size_bytes,
    totalModelLayers: n + 1,
    requestedNgl: n + 1,
    tensorBreakdown: model.metadata.tensor_layer_bytes ?? null,
  });
  const kv =
    typeof model.metadata.n_head_kv === "number"
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
  if (weights == null) return null;
  const need = weights + (kv ?? 0);
  const { hasGpu, unified } = machineKind(w);
  const ramMib = w.hardware?.mem_total_bytes ? w.hardware.mem_total_bytes / 1048576 : null;
  if (!hasGpu || unified) {
    if (ramMib == null) return null;
    return need <= ramMib * 0.9
      ? { icon: "chip", text: `${unified ? "Unified" : "CPU only"} · ${fmtGb(need)} / ${fmtGb(ramMib, 0)}`, tone: "text-fg-2" }
      : { icon: "oom", text: "Too big for RAM", tone: "text-danger" };
  }
  const free = w.vram?.vram_free_before_mib ?? (w.hardware?.gpu[0]?.vram_usable_mb ?? w.hardware?.gpu[0]?.vram_mb ?? null);
  if (free == null) return null;
  if (need <= free * 0.95) return { icon: "ok", text: `Fits · ${fmtGb(need)} / ${fmtGb(free)}`, tone: "text-fg" };
  // Rough on purpose (hybrid attention models get their KV overstated); the
  // fit map replaces it with llama.cpp's own sizing.
  return { icon: "spill", text: `~${fmtGb(need - free * 0.95)} over VRAM at ${ctxLabel(ctx)} (estimate)`, tone: "text-danger" };
}

