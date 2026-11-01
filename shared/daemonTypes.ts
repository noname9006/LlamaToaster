// The protocol between the Serve page, the server and the worker's model daemon.
// See docs/plans/MODEL_DAEMON_PLAN.md.

import type { GatewaySettings, SamplingSettings, SettingValues } from "./daemonSettings.js";
import type { DaemonUsageRow } from "./daemonUsage.js";

export type DaemonPhase =
  | "idle"
  // A daemon_load job is starting llama-server.
  | "loading"
  | "ready"
  // llama-server freed the model after --sleep-idle-seconds; the next request reloads it.
  | "sleeping"
  // Unloading: running requests get time to finish first.
  | "stopping"
  // Unloaded so a test can run; comes back when the queue drains.
  | "suspended"
  | "failed"
  // Died after it was ready. Restarted up to a bound, then stays here.
  | "crashed";

/** A serve request: which model, and how to load it. */
export interface DaemonLoadSpec {
  model_id: string;
  model_name: string;
  /** Registered model the worker resolves to a path, same as a test does. */
  model: import("./types.js").Model;
  alias: string;
  settings: SettingValues;
  sampling: SamplingSettings;
  /** model_dir-relative, checked against what the worker has on disk. */
  mmproj_filename?: string | null;
  draft_model?: import("./types.js").Model | null;
}

export interface DaemonLoadJobPayload {
  spec: DaemonLoadSpec;
  config: DaemonConfig;
  /** A job older than this is dropped, so a machine reconnecting after hours
   * does not suddenly claim its GPU. */
  requested_at: number;
}

/** Everything the gateway needs. Persisted on the worker, so agents keep working
 * while the server is unreachable. */
export interface DaemonConfig {
  version: number;
  gateway: GatewaySettings;
  /** SHA-256 hex only: neither end stores a key. */
  key_hashes: string[];
  /** Live sampling for the model below, applied on the next request. */
  sampling_model_id: string | null;
  sampling: SamplingSettings | null;
}

export interface DaemonActualLoad {
  gpu_layers: number | null;
  total_layers: number | null;
  n_ctx: number | null;
  n_slots: number | null;
}

/** One side of the memory display, in MiB. A null means no reading exists, which
 * the page says rather than showing a guess. */
export interface DaemonMemorySide {
  total: number | null;
  /** This app: llama-server, plus the worker on the RAM side. */
  app: number | null;
  /** Everything else using it. */
  other: number | null;
  free: number | null;
}

export interface DaemonMemorySnapshot {
  at: number;
  /** Same physical memory for CPU and GPU: `ram` is the whole picture. */
  unified: boolean;
  gpu_name: string | null;
  /** GPUs on the machine. With several, the figures may combine them: on Windows the
   * total adds up every adapter, so the page says so rather than implying one card. */
  gpu_count: number;
  vram: DaemonMemorySide | null;
  ram: DaemonMemorySide;
  /** What the app's share is made of, from llama.cpp's own report. */
  breakdown: {
    weights_gpu: number | null;
    kv_gpu: number | null;
    compute_gpu: number | null;
    weights_host: number | null;
    compute_host: number | null;
    prompt_cache: number | null;
    worker: number | null;
  };
  /** GPU buffers that are not in dedicated VRAM, from shared/gpuSpill.ts. */
  spill: { measured: boolean; in_system_ram_mib: number | null; spilled: boolean };
}

export interface DaemonInterface {
  name: string;
  address: string;
}

export interface DaemonStatus {
  phase: DaemonPhase;
  /** Why it is failed, crashed or suspended, in words for the page. */
  message: string | null;
  model_id: string | null;
  model_name: string | null;
  alias: string | null;
  build_tag: string | null;
  load_hash: string | null;
  settings: SettingValues | null;
  /** The rest of what a reload could change, so the page can tell. */
  mmproj_filename: string | null;
  draft_model_id: string | null;
  actual: DaemonActualLoad | null;
  loaded_at: number | null;
  in_flight: number;
  requests_total: number;
  /** Build defaults from /props, for placeholders. */
  generation_defaults: Record<string, number> | null;
  gateway: { listening: boolean; host: string; port: number; error: string | null };
  interfaces: DaemonInterface[];
  /** Active build's supported load settings, parsed from its --help. */
  active_build_tag: string | null;
  supported_settings: string[];
  restarts: number;
  /** Usage rows lost to a long outage. */
  usage_dropped: number;
  autostart_model_id: string | null;
  memory: DaemonMemorySnapshot | null;
}

export interface DaemonLoadRecord {
  load_hash: string;
  model_id: string;
  build_tag: string | null;
  settings: SettingValues;
  actual: DaemonActualLoad | null;
}

/** POST /api/worker/daemon/telemetry */
export interface DaemonTelemetryInput {
  machine_id: string;
  hostname?: string;
  config_version: number;
  /** Whose sampling the stored config carries. Two configs of one version can differ:
   * sampling follows the loaded model. */
  config_model_id: string | null;
  status: DaemonStatus;
  usage: DaemonUsageRow[];
}

export interface DaemonTelemetryResponse {
  /** Someone has the page open: read memory, tick faster. */
  watched: boolean;
  /** Present when the worker's config_version is behind. */
  config?: DaemonConfig;
  /** Usage rows stored or already known, so the worker can drop them. */
  acked: string[];
}

export interface DaemonApiKeyInfo {
  id: string;
  name: string;
  prefix: string;
  created_at: number;
  revoked_at: number | null;
}

/** GET /api/workers/:id/daemon */
export interface DaemonView {
  status: DaemonStatus | null;
  /** When the worker last reported, so the page can say how old this is. */
  reported_at: number | null;
  capable: boolean;
  gateway: GatewaySettings;
  keys: DaemonApiKeyInfo[];
  pending_job: { id: string; type: "daemon_load" | "daemon_unload"; status: string } | null;
  last_error: string | null;
}

/** GET /api/workers/:id/daemon/models/:modelId/settings */
export interface DaemonModelSettings {
  settings: SettingValues;
  sampling: SamplingSettings;
  alias: string | null;
  mmproj_filename: string | null;
  draft_model_id: string | null;
}

/** GET /api/workers/:id/daemon/usage */
export interface DaemonUsageView {
  loads: (DaemonLoadRecord & { first_seen: number; requests: number })[];
  profile: import("./daemonUsage.js").UsageProfile;
  curve: import("./daemonUsage.js").SpeedCurvePoint[];
  /** Newest first, capped. */
  points: { at: number; ctx: number | null; pp: number | null; tg: number | null }[];
  /** Rows left out of the chart because other requests were running. */
  concurrent_excluded: number;
  window_days: number;
}
