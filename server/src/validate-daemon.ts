// Worker-reported Serve state is trusted no further than any other worker report:
// numbers are checked, strings capped, and a malformed usage row is dropped rather
// than poisoning a chart. See validate-worker-state.ts for the same posture.

import { validateLoadSettings } from "../../shared/daemonSettings.js";
import type { DaemonActualLoad, DaemonInterface, DaemonMemorySide, DaemonMemorySnapshot, DaemonPhase, DaemonStatus } from "../../shared/daemonTypes.js";
import type { DaemonEndpoint, DaemonUsageRow } from "../../shared/daemonUsage.js";

const PHASES: ReadonlySet<DaemonPhase> = new Set(["idle", "loading", "ready", "sleeping", "stopping", "suspended", "failed", "crashed"]);
const ENDPOINTS: ReadonlySet<DaemonEndpoint> = new Set(["chat", "completions", "responses", "messages", "embeddings", "rerank"]);
const MAX_USAGE_ROWS = 500;
const MAX_TOKENS = 1e9;
const MAX_MS = 7 * 86_400_000;
const MAX_MIB = 1e8;
const SKEW_MS = 5 * 60_000;

function str(v: unknown, max: number): string | null {
  return typeof v === "string" ? v.slice(0, max) : null;
}

function bounded(v: unknown, max: number): number | null {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= max ? v : null;
}

function record(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function memorySide(v: unknown): DaemonMemorySide | null {
  const m = record(v);
  if (!m) return null;
  return { total: bounded(m.total, MAX_MIB), app: bounded(m.app, MAX_MIB), other: bounded(m.other, MAX_MIB), free: bounded(m.free, MAX_MIB) };
}

function memory(v: unknown): DaemonMemorySnapshot | null {
  const m = record(v);
  const ram = memorySide(m?.ram);
  if (!m || !ram) return null;
  const b = record(m.breakdown) ?? {};
  const spill = record(m.spill) ?? {};
  return {
    at: bounded(m.at, Number.MAX_SAFE_INTEGER) ?? Date.now(),
    unified: m.unified === true,
    gpu_name: str(m.gpu_name, 128),
    gpu_count: bounded(m.gpu_count, 256) ?? 0,
    vram: memorySide(m.vram),
    ram,
    breakdown: {
      weights_gpu: bounded(b.weights_gpu, MAX_MIB),
      kv_gpu: bounded(b.kv_gpu, MAX_MIB),
      compute_gpu: bounded(b.compute_gpu, MAX_MIB),
      weights_host: bounded(b.weights_host, MAX_MIB),
      compute_host: bounded(b.compute_host, MAX_MIB),
      prompt_cache: bounded(b.prompt_cache, MAX_MIB),
      worker: bounded(b.worker, MAX_MIB),
    },
    spill: {
      measured: spill.measured === true,
      // Negative is the normal clean reading (VRAM holds more than the buffers).
      in_system_ram_mib: typeof spill.in_system_ram_mib === "number" && Number.isFinite(spill.in_system_ram_mib) ? spill.in_system_ram_mib : null,
      spilled: spill.spilled === true,
    },
  };
}

function actual(v: unknown): DaemonActualLoad | null {
  const a = record(v);
  if (!a) return null;
  return {
    gpu_layers: bounded(a.gpu_layers, 100_000),
    total_layers: bounded(a.total_layers, 100_000),
    n_ctx: bounded(a.n_ctx, MAX_TOKENS),
    n_slots: bounded(a.n_slots, 100_000),
  };
}

function interfaces(v: unknown): DaemonInterface[] {
  if (!Array.isArray(v)) return [];
  const out: DaemonInterface[] = [];
  for (const entry of v.slice(0, 16)) {
    const e = record(entry);
    const address = str(e?.address, 64);
    if (address) out.push({ name: str(e?.name, 64) ?? "", address });
  }
  return out;
}

export function parseDaemonStatus(raw: unknown): DaemonStatus | null {
  const s = record(raw);
  if (!s || !PHASES.has(s.phase as DaemonPhase)) return null;
  const gateway = record(s.gateway) ?? {};
  const settings = s.settings != null ? validateLoadSettings(s.settings) : null;
  const defaults: Record<string, number> = {};
  for (const [key, value] of Object.entries(record(s.generation_defaults) ?? {}).slice(0, 64)) {
    if (typeof value === "number" && Number.isFinite(value)) defaults[key.slice(0, 64)] = value;
  }
  return {
    phase: s.phase as DaemonPhase,
    message: str(s.message, 2000),
    model_id: str(s.model_id, 128),
    model_name: str(s.model_name, 512),
    alias: str(s.alias, 128),
    build_tag: str(s.build_tag, 128),
    load_hash: str(s.load_hash, 64),
    settings: settings && settings.errors.length === 0 ? settings.values : null,
    mmproj_filename: str(s.mmproj_filename, 512),
    draft_model_id: str(s.draft_model_id, 128),
    actual: actual(s.actual),
    loaded_at: bounded(s.loaded_at, Number.MAX_SAFE_INTEGER),
    in_flight: bounded(s.in_flight, 1e6) ?? 0,
    requests_total: bounded(s.requests_total, Number.MAX_SAFE_INTEGER) ?? 0,
    generation_defaults: Object.keys(defaults).length > 0 ? defaults : null,
    gateway: {
      listening: gateway.listening === true,
      host: str(gateway.host, 64) ?? "",
      port: bounded(gateway.port, 65535) ?? 0,
      error: str(gateway.error, 500),
    },
    interfaces: interfaces(s.interfaces),
    active_build_tag: str(s.active_build_tag, 128),
    supported_settings: Array.isArray(s.supported_settings) ? s.supported_settings.filter((k): k is string => typeof k === "string").slice(0, 256).map((k) => k.slice(0, 64)) : [],
    restarts: bounded(s.restarts, 1e6) ?? 0,
    usage_dropped: bounded(s.usage_dropped, Number.MAX_SAFE_INTEGER) ?? 0,
    autostart_model_id: str(s.autostart_model_id, 128),
    memory: memory(s.memory),
  };
}

function usageRow(raw: unknown): DaemonUsageRow | null {
  const r = record(raw);
  if (!r) return null;
  const id = str(r.id, 64);
  const modelId = str(r.model_id, 128);
  const loadHash = str(r.load_hash, 64);
  const at = bounded(r.at, Date.now() + SKEW_MS);
  const status = bounded(r.status, 999);
  const totalMs = bounded(r.total_ms, MAX_MS);
  const concurrency = bounded(r.concurrency, 100_000);
  if (!id || !modelId || !loadHash || at == null || status == null || totalMs == null || concurrency == null) return null;
  if (!ENDPOINTS.has(r.endpoint as DaemonEndpoint)) return null;
  return {
    id,
    at,
    model_id: modelId,
    load_hash: loadHash,
    endpoint: r.endpoint as DaemonEndpoint,
    stream: r.stream === true,
    status,
    cancelled: r.cancelled === true,
    processed_tokens: bounded(r.processed_tokens, MAX_TOKENS),
    cached_tokens: bounded(r.cached_tokens, MAX_TOKENS),
    generated_tokens: bounded(r.generated_tokens, MAX_TOKENS),
    prompt_ms: bounded(r.prompt_ms, MAX_MS),
    predicted_ms: bounded(r.predicted_ms, MAX_MS),
    total_ms: totalMs,
    concurrency,
    watched: r.watched === true,
  };
}

/** Every row with an id is acknowledged, valid or not, so a bad row is not resent forever. */
export function parseDaemonUsage(raw: unknown): { rows: DaemonUsageRow[]; acked: string[] } {
  if (!Array.isArray(raw)) return { rows: [], acked: [] };
  const rows: DaemonUsageRow[] = [];
  const acked: string[] = [];
  for (const entry of raw.slice(0, MAX_USAGE_ROWS)) {
    const id = str(record(entry)?.id, 64);
    if (id) acked.push(id);
    const row = usageRow(entry);
    if (row) rows.push(row);
  }
  return { rows, acked };
}
