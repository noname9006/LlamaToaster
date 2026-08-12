import { api } from "./client";
import type { Model } from "../types";
import { formatFlashAttn } from "../utils";

// Every source here already has a route the rest of the app uses
// (Dashboard/Workers read hardware via getWorkerLlamaCpp, Models reads
// listModels, RunDetail's CSV export link uses exportUrl) -- no new server
// endpoint needed just to hand the same data to the chat panel as text.
const MAX_RESULT_ROWS = 150;

// Mirrors the shape server/src/routes/results.ts's /api/results/export
// actually returns (a `SELECT r.* ...` over the `results` table, see
// db/schema.sql) -- that route's own ResultExportRow type is server-local
// and not exported, so this is a client-side re-declaration of the same
// runtime shape rather than an import.
interface ExportRow {
  worker_name: string;
  model_filename: string;
  test_type: string;
  n_prompt: number;
  n_gen: number;
  n_threads: number;
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  avg_tps: number;
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  created_at: number;
}

async function hardwareSummary(): Promise<string> {
  const workers = await api.listWorkers();
  if (workers.length === 0) return "(no workers configured)";
  const lines = await Promise.all(
    workers.map(async (w) => {
      try {
        const info = await api.getWorkerLlamaCpp(w.name);
        const hw = info.hardware;
        const cpu = hw ? hw.cpu.brand || hw.cpu.manufacturer || "unknown CPU" : "unknown CPU";
        const gpu = hw && hw.gpu.length > 0 ? hw.gpu.map((g) => g.model).join(", ") : "no discrete GPU reported";
        const state = info.busy ? "busy" : "idle";
        return `- ${w.name} — ${info.backend} backend, ${info.platform}/${info.arch}, CPU: ${cpu}, GPU: ${gpu}, build ${info.active_tag ?? w.llama_cpp_build} (${state})`;
      } catch {
        return `- ${w.name} — unreachable right now (hardware unknown).`;
      }
    })
  );
  return lines.join("\n");
}

function modelsSummary(models: Model[]): string {
  if (models.length === 0) return "(none registered)";
  return models
    .map((m) => {
      const family = typeof m.metadata.arch === "string" ? m.metadata.arch : "unknown arch";
      const quant = typeof m.metadata.quant === "string" ? `, ${m.metadata.quant}` : "";
      const repo = m.hf_repo ? ` — ${m.hf_repo}` : "";
      return `- ${m.filename} (${family}${quant})${repo}`;
    })
    .join("\n");
}

async function resultsSummary(): Promise<string> {
  try {
    const res = await fetch(api.exportUrl("json"));
    if (!res.ok) return "(couldn't load benchmark results)";
    const rows = (await res.json()) as ExportRow[];
    if (rows.length === 0) return "(no benchmark results yet)";
    const sorted = [...rows].sort((a, b) => b.created_at - a.created_at);
    const capped = sorted.slice(0, MAX_RESULT_ROWS);
    const header =
      "worker | model | test | n_prompt | n_gen | threads | ngl | batch | ubatch | ctk | ctv | fa | avg_tps | ram_peak_mib | vram_peak_mib";
    const lines = capped.map(
      (r) =>
        `${r.worker_name} | ${r.model_filename} | ${r.test_type} | ${r.n_prompt} | ${r.n_gen} | ${r.n_threads} | ` +
        `${r.n_gpu_layers} | ${r.batch_size} | ${r.ubatch_size} | ${r.cache_type_k} | ${r.cache_type_v} | ${formatFlashAttn(r.flash_attn)} | ` +
        `${r.avg_tps.toFixed(2)} | ${r.ram_peak_mib} | ${r.vram_peak_mib ?? "n/a"}`
    );
    const scope = rows.length > capped.length ? `most recent ${capped.length} of ${rows.length} total` : `${rows.length} total`;
    return `(${scope})\n${header}\n${lines.join("\n")}`;
  } catch {
    return "(couldn't load benchmark results)";
  }
}

// Snapshot, not a live subscription -- the caller decides when to refresh
// (opening the panel, or an explicit "Refresh context" click) so token usage
// per message stays predictable instead of re-fetching everything on every
// keystroke.
export async function buildContextSnapshot(): Promise<string> {
  const [hardware, models] = await Promise.all([hardwareSummary(), api.listModels()]);
  const results = await resultsSummary();
  return [
    "## Hardware",
    hardware,
    "",
    `## Registered models (n=${models.length})`,
    modelsSummary(models),
    "",
    "## Recent benchmark results",
    results,
  ].join("\n");
}
