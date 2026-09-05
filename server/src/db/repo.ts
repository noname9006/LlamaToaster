import { v4 as uuid } from "uuid";
import { createHash } from "node:crypto";
import { getDb } from "./migrate.js";
import { deriveWorkerStatus, LEASE_MS } from "../liveness.js";
import { generateSessionId, generateRefreshToken, hashToken } from "../session.js";
// Only the pure filename parser is used from here -- adminRepo.stats()'s
// quant card needs it and it must match what the client's model picker shows
// (client/src/modelGrouping.ts's extractQuant mirrors this same pattern).
import { parseQuant } from "../hf.js";
import type {
  Model,
  ModelMetadata,
  RegisterModelInput,
  Test,
  TestConfig,
  TestStatus,
  ResultRow,
  IngestResultInput,
  TestItem,
  TestItemTickInput,
  TestItemTerminalInput,
  GpuMemoryAccuracyLevel,
  Worker,
  WorkerStatePush,
  ActiveJobReport,
  QueueJob,
  AuthUser,
  SessionInfo,
  AdminStats,
  AdminTestSummary,
  AdminTestFilters,
  AdminUserSummary,
  CommunityAggregateRow,
  CommunityAggregateFilters,
  CommunityFacets,
  PossibleDuplicateWorker,
  HardwareInfo,
  Backend,
  TestType,
  AppSettings,
  VramDiscrepancyPolicy,
  TestKind,
  ProbeResultInput,
  ProbeAttemptReport,
  QualityResultInput,
} from "../../../shared/types.js";
import {
  isVramDiscrepancyPolicy,
  isValidProbeMaxLoads,
  parseCaveatFlags,
  CHAIN_WALL_CLOCK_MS,
} from "../../../shared/types.js";
import type { SweepItem } from "../../../shared/sweep.js";
import type { TensorLayerBreakdown } from "../../../shared/vramEstimate.js";
import { configHash, type ConfigHashInput } from "../../../shared/configHash.js";
import { engineFromItem, specTypeFor, type EngineKind } from "../../../shared/engineSpec.js";

interface ModelRow {
  id: string;
  filename: string;
  size_bytes: number;
  source: string;
  hf_repo: string | null;
  hf_file: string | null;
  metadata: string;
  created_at: number;
}

interface TestRow {
  id: string;
  root_run_id?: string | null;
  kind?: string | null;
  comparison_id?: string | null;
  // Multi-user Stage 4/5: present on every real row (COLUMN_MIGRATIONS,
  // §4.1) but deliberately absent from the public Test type mapTest produces
  // (same "DB-internal only" posture as models.created_by) -- declared here
  // only because adminRepo.listTests (§5.1) is the one place that reads it
  // back out, for cross-tenant display.
  user_id?: string | null;
  worker_name: string;
  worker_id: string | null;
  llama_cpp_build: string;
  llama_cpp_backend: string;
  backend_device_name: string | null;
  model_id: string;
  config: string;
  status: string;
  error: string | null;
  started_at: number;
  completed_at: number | null;
  model_filename?: string | null;
  items_total?: number;
  items_done?: number;
  items_failed?: number;
  items_cancelled?: number;
}

interface TestItemRow {
  id: string;
  run_id: string;
  idx: number;
  n_prompt: number;
  n_gen: number;
  n_depth: number | null;
  concurrency: number | null;
  n_threads: number;
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  n_gpu_layers_draft: number;
  n_cpu_moe: number;
  status: string;
  detail: string | null;
  ram_mib: number | null;
  vram_mib: number | null;
  ram_peak_mib: number | null;
  vram_peak_mib: number | null;
  ram_avg_mib: number | null;
  vram_avg_mib: number | null;
  ram_free_before_mib: number | null;
  vram_free_before_mib: number | null;
  live_tps: number | null;
  error: string | null;
  started_at: number | null;
  completed_at: number | null;
}

interface ResultRowRaw {
  id: string;
  run_id: string;
  idx: number | null;
  model_id: string;
  // Denormalized from the parent run via getResultsForTest's JOIN, not a
  // physical column on `results` itself -- see ResultRow's own doc comment.
  backend_type: string;
  backend_device_name: string | null;
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
  mtp: string;
  n_gpu_layers_draft: number;
  n_cpu_moe: number;
  avg_tps: number;
  stddev_tps: number;
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  ram_avg_mib: number;
  vram_avg_mib: number | null;
  ram_free_before_mib: number | null;
  vram_free_before_mib: number | null;
  // Raw storage is nullable regardless of shared/types.ts's ResultRow
  // typing accuracy fields as non-nullable strings: NULL here means this row
  // predates the migration that added these columns, not "unavailable" as a
  // deliberate measurement -- mapResult below coalesces that distinction
  // away for anything reading through the public type.
  system_memory_total_mib: number | null;
  gpu_memory_total_mib: number | null;
  gpu_memory_total_accuracy: string | null;
  gpu_memory_total_source: string | null;
  gpu_memory_free_start_accuracy: string | null;
  gpu_memory_free_start_source: string | null;
  gpu_memory_model_avg_accuracy: string | null;
  gpu_memory_model_avg_source: string | null;
  gpu_memory_model_peak_accuracy: string | null;
  gpu_memory_model_peak_source: string | null;
  // Whole-adapter/per-process VRAM usage + whole-system RAM usage avg/peak --
  // see shared/types.ts's ResultRow doc comments. Same raw-nullable storage
  // and pre-migration coalescing rules as the gpu_memory_* fields above.
  gpu_memory_used_avg_mib: number | null;
  gpu_memory_used_peak_mib: number | null;
  gpu_memory_used_avg_accuracy: string | null;
  gpu_memory_used_avg_source: string | null;
  gpu_memory_used_peak_accuracy: string | null;
  gpu_memory_used_peak_source: string | null;
  gpu_memory_process_avg_mib: number | null;
  gpu_memory_process_peak_mib: number | null;
  gpu_memory_process_avg_accuracy: string | null;
  gpu_memory_process_avg_source: string | null;
  gpu_memory_process_peak_accuracy: string | null;
  gpu_memory_process_peak_source: string | null;
  ram_total_used_avg_mib: number | null;
  ram_total_used_peak_mib: number | null;
  gpu_layers_loaded: number | null;
  total_model_layers: number | null;
  gpu_layers_loaded_draft: number | null;
  total_model_layers_draft: number | null;
  // See shared/types.ts's ResultRow.gpu_layers_resident_est -- worker-derived
  // residency estimate, written on every row where it was computable.
  gpu_layers_resident_est: number | null;
  // See shared/types.ts's ResultRow.gpu_layers_resident_est_draft -- the
  // MTP/draft companion's own residency estimate; NULL on every non-MTP row.
  gpu_layers_resident_est_draft: number | null;
  sample_count: number | null;
  suspect_count: number | null;
  suspect_samples: string | null;
  repeat_samples: string | null;
  spec_drafted: number | null;
  spec_accepted: number | null;
  // BENCHMARKING_PLAN_V8.md §0 columns -- NULL on every row predating them.
  method_version: number | null;
  n_depth: number | null;
  config_hash: string | null;
  prompt_offset: number | null;
  spec_type: string | null;
  spec_n_max: number | null;
  spec_n_min: number | null;
  speedup: number | null;
  speedup_status: string | null;
  caveat_flags: string | null;
  concurrency: number | null;
  gpu_temp_c_max: number | null;
  gpu_clock_mhz_min: number | null;
  gpu_clock_samples: string | null;
  cpu_isa: string | null;
  imported_bundle_id: string | null;
  import_opt_in: number | null;
  ttft_ms_p50: number | null;
  ttft_ms_p95: number | null;
  ttft_n: number | null;
  e2e_ms_mean: number | null;
  worker_id: string | null;
  llama_cpp_build: string | null;
  engine: string | null;
  raw_json_path: string | null;
  created_at: number;
}

interface WorkerRow {
  id: string;
  machine_id: string;
  display_name: string;
  backend: string | null;
  platform: string | null;
  arch: string | null;
  hostname: string | null;
  hardware_json: string | null;
  installed_builds_json: string | null;
  model_files_json: string | null;
  // See WorkerVramInfo's doc comment (shared/types.ts) -- last-reported
  // free-VRAM reading, updated on every idle heartbeat. Null until the
  // worker's first idle heartbeat.
  vram_json: string | null;
  last_heartbeat_at: number | null;
  active_job_id: string | null;
  capabilities_json: string | null;
  sensors_json: string | null;
  cpu_isa: string | null;
  pause_requested: number;
  created_at: number;
  updated_at: number;
  // Only present when the query joins worker_jobs on active_job_id (see
  // WORKER_SELECT_SQL below) -- null whenever active_job_id is null, or the
  // active job is a type with no run_id (install_build/download_model/delete_*).
  active_job_run_id?: string | null;
  active_job_progress_json?: string | null;
  // Multi-user Stage 3 (MULTIUSER_PLAN.md §3.3) -- device-flow enrolment.
  // Present on every row (SELECT workers.* below) but deliberately NEVER
  // copied into the public Worker shape by mapWorker: enrolment_code_hash is
  // a lookup secret like a session token hash, and user_code/expiry/approval
  // timing have no reason to leak through the general GET /api/workers
  // listing an already-approved worker's owner sees.
  user_id: string | null;
  enrolment_code_hash: string | null;
  user_code: string | null;
  enrolment_expires_at: number | null;
  approved_at: number | null;
}

// Server-internal shape for the device-flow routes (routes/device.ts) --
// deliberately narrower than the public Worker type, which never exposes
// enrolment_code_hash/user_code/user_id/approved_at at all (see WorkerRow's
// own doc comment above).
export interface WorkerEnrolment {
  id: string;
  userId: string | null;
  displayName: string;
  hostname: string | null;
  platform: string | null;
  arch: string | null;
  gpuModel: string | null; // first reported GPU, for the §3.1 confirm card
  enrolmentExpiresAt: number | null;
  approvedAt: number | null;
}

// The richer shape actually stored in workers.model_files_json (see
// WorkerStatePush.model_files) -- ModelDirFile plus the optional GGUF
// metadata a worker may have read off the file header. Local to this file;
// the public Worker.modelFiles type stays the narrower ModelDirFile[].
interface ModelDirFileMeta {
  path: string;
  size_bytes: number;
  n_layer?: number | null;
  mtp_layers?: number | null;
  // Present in stored model_files_json now that the heartbeat validator
  // persists it -- see validate-worker-state.ts's parseModelFiles.
  state?: string;
  // Real per-tensor weight-byte breakdown -- see ModelMetadata.tensor_layer_bytes
  // (shared/types.ts). Read by findModelFileMeta below for the same
  // "backfill a hand-dropped file's missing catalog metadata" story as
  // n_layer/mtp_layers.
  tensor_layer_bytes?: TensorLayerBreakdown | null;
}

interface WorkerJobRow {
  id: string;
  worker_id: string;
  job_type: string;
  payload_json: string;
  status: string;
  attempts: number;
  lease_expires_at: number | null;
  cancel_requested: number;
  discard_requested: number;
  progress_json: string | null;
  run_id: string | null;
  created_at: number;
  claimed_at: number | null;
  completed_at: number | null;
  error: string | null;
}

interface UserRow {
  id: string;
  display_name: string | null;
  avatar_url: string | null;
  share_benchmarks: number;
  created_at: number;
  updated_at: number;
  last_login_at: number | null;
}

// Server-internal -- isSuperadmin is NOT a column (MULTIUSER_PLAN.md §5.1),
// so a row alone can never populate it; every caller that needs a real
// AuthUser gets this back from getUser with isSuperadmin hardcoded false and
// is expected to override it (see server/src/auth-middleware.ts), exactly
// like the plan's own `{ ...user, isSuperadmin }` spread.
export type UserRecord = AuthUser & { shareBenchmarks: boolean; createdAt: number; updatedAt: number; lastLoginAt: number | null };

interface IdentityRow {
  id: string;
  user_id: string;
  provider: string;
  provider_user_id: string;
  provider_login: string | null;
  created_at: number;
}

export interface Identity {
  id: string;
  userId: string;
  provider: string;
  providerUserId: string;
  providerLogin: string | null;
  createdAt: number;
}

interface SessionRow {
  id: string;
  user_id: string;
  token_hash: string;
  refresh_hash: string | null;
  prev_refresh_hash: string | null;
  expires_at: number;
  is_worker: number;
  worker_id: string | null;
  label: string | null;
  last_seen_at: number | null;
  created_at: number;
}

// Server-internal session shape (camelCase) -- the wire-facing SessionInfo
// (shared/types.ts) is a narrower, non-sensitive subset of this returned
// only by GET /api/sessions.
export interface SessionRecord {
  id: string;
  userId: string;
  expiresAt: number;
  isWorker: boolean;
  workerId: string | null;
  label: string | null;
  lastSeenAt: number | null;
  createdAt: number;
}

function mapModel(row: ModelRow): Model {
  return {
    id: row.id,
    filename: row.filename,
    size_bytes: row.size_bytes,
    source: row.source as Model["source"],
    hf_repo: row.hf_repo ?? undefined,
    hf_file: row.hf_file ?? undefined,
    metadata: JSON.parse(row.metadata || "{}"),
    created_at: row.created_at,
  };
}

function mapTest(row: TestRow): Test {
  return {
    id: row.id,
    root_run_id: row.root_run_id ?? row.id,
    kind: (row.kind as TestKind | null) ?? null,
    comparison_id: row.comparison_id ?? null,
    worker_name: row.worker_name,
    worker_id: row.worker_id ?? undefined,
    llama_cpp_build: row.llama_cpp_build,
    llama_cpp_backend: row.llama_cpp_backend as Test["llama_cpp_backend"],
    backend_device_name: row.backend_device_name ?? undefined,
    model_id: row.model_id,
    model_filename: row.model_filename ?? undefined,
    config: JSON.parse(row.config),
    status: row.status as TestStatus,
    error: row.error ?? undefined,
    started_at: row.started_at,
    completed_at: row.completed_at ?? undefined,
    items_total: row.items_total ?? 0,
    items_done: row.items_done ?? 0,
    items_failed: row.items_failed ?? 0,
    items_cancelled: row.items_cancelled ?? 0,
  };
}

function mapTestItem(row: TestItemRow): TestItem {
  return {
    id: row.id,
    run_id: row.run_id,
    idx: row.idx,
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_depth: row.n_depth ?? 0,
    concurrency: row.concurrency ?? 1,
    n_threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    mtp: row.mtp,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    n_cpu_moe: row.n_cpu_moe,
    status: row.status as TestItem["status"],
    detail: row.detail ?? undefined,
    ram_mib: row.ram_mib,
    vram_mib: row.vram_mib,
    ram_peak_mib: row.ram_peak_mib,
    vram_peak_mib: row.vram_peak_mib,
    ram_avg_mib: row.ram_avg_mib,
    vram_avg_mib: row.vram_avg_mib,
    ram_free_before_mib: row.ram_free_before_mib,
    vram_free_before_mib: row.vram_free_before_mib,
    live_tps: row.live_tps,
    error: row.error ?? undefined,
    started_at: row.started_at ?? undefined,
    completed_at: row.completed_at ?? undefined,
  };
}

function mapUser(row: UserRow): UserRecord {
  return {
    id: row.id,
    displayName: row.display_name,
    avatarUrl: row.avatar_url,
    isSuperadmin: false, // always overridden by the caller -- see UserRecord's doc comment
    shareBenchmarks: row.share_benchmarks === 1,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    lastLoginAt: row.last_login_at,
  };
}

function mapIdentity(row: IdentityRow): Identity {
  return {
    id: row.id,
    userId: row.user_id,
    provider: row.provider,
    providerUserId: row.provider_user_id,
    providerLogin: row.provider_login,
    createdAt: row.created_at,
  };
}

function mapSession(row: SessionRow): SessionRecord {
  return {
    id: row.id,
    userId: row.user_id,
    expiresAt: row.expires_at,
    isWorker: row.is_worker === 1,
    workerId: row.worker_id,
    label: row.label,
    lastSeenAt: row.last_seen_at,
    createdAt: row.created_at,
  };
}

// Tolerates a corrupt/unexpected JSON blob by falling back to the given
// default rather than throwing -- a worker-supplied state push is validated
// before it's ever written (see server/src/validate-worker-state.ts), but a
// read-side parse failure (a hand-edited DB row, a future format change)
// should degrade to "nothing reported yet," not take the whole route down.
function safeParseJson<T>(json: string | null, fallback: T): T {
  if (!json) return fallback;
  try {
    return JSON.parse(json) as T;
  } catch {
    return fallback;
  }
}

// Used by workerRepo.findPossibleDuplicate below to corroborate (or, in the
// future, stand in for) a hostname match with an actual hardware fingerprint
// -- hostname alone is a weak signal (a generic default like
// "DESKTOP-XXXXXXX" collides across genuinely different machines). Compares
// only the fields that are stable across reboots/reinstalls of the SAME
// physical box: CPU brand+cores, the set of GPU models (not vram/driver
// fields, which can read back slightly differently run to run), platform+
// arch, and total RAM within a small tolerance (firmware/OS-reserved amounts
// can differ by a sliver from boot to boot). Deliberately conservative: two
// genuinely different but identically-specced headless/no-GPU machines (e.g.
// two identical cloud VMs) could false-positive here -- acceptable since this
// only ever feeds a human-confirmed suggestion, never an automatic merge.
const RAM_TOLERANCE_FRACTION = 0.03;

function hardwareLooksTheSame(
  a: HardwareInfo | null,
  b: HardwareInfo | null,
  platform: string | null,
  arch: string | null
): boolean {
  if (!a || !b) return false;
  if (a.platform !== platform || a.arch !== arch || b.platform !== platform || b.arch !== arch) return false;
  if (a.cpu.brand.trim().toLowerCase() !== b.cpu.brand.trim().toLowerCase()) return false;
  if (a.cpu.cores !== b.cpu.cores) return false;
  const gpuSet = (h: HardwareInfo) => new Set(h.gpu.map((g) => g.model.trim().toLowerCase()));
  const ga = gpuSet(a);
  const gb = gpuSet(b);
  if (ga.size !== gb.size || [...ga].some((m) => !gb.has(m))) return false;
  if (a.mem_total_bytes != null && b.mem_total_bytes != null) {
    const diff = Math.abs(a.mem_total_bytes - b.mem_total_bytes);
    if (diff > a.mem_total_bytes * RAM_TOLERANCE_FRACTION) return false;
  }
  return true;
}

// Unlike the single active_job_id FK on `workers` (JOINed in via
// WORKER_SELECT_SQL above), several download_model jobs can be genuinely
// 'claimed' for the same worker at once (worker/src/index.ts's
// downloadJobPool), so this is a second, one-to-many query rather than
// another JOIN column. Small worker/job counts in practice make the N+1
// query pattern (once per mapWorker call) a non-issue here.
function getActiveDownloadReports(workerId: string): ActiveJobReport[] {
  const rows = getDb()
    .prepare(
      `SELECT progress_json FROM worker_jobs
       WHERE worker_id = ? AND status = 'claimed' AND job_type = 'download_model'
       ORDER BY created_at ASC`
    )
    .all(workerId) as { progress_json: string | null }[];
  return rows
    .map((r) => (r.progress_json ? safeParseJson<ActiveJobReport | null>(r.progress_json, null) : null))
    .filter((r): r is ActiveJobReport => r !== null);
}

function mapWorker(row: WorkerRow): Worker {
  const activeDownloads = getActiveDownloadReports(row.id);
  return {
    id: row.id,
    machineId: row.machine_id,
    displayName: row.display_name,
    hostname: row.hostname,
    backend: row.backend,
    platform: row.platform,
    arch: row.arch,
    hardware: safeParseJson(row.hardware_json, null),
    installedBuilds: safeParseJson(row.installed_builds_json, []),
    modelFiles: safeParseJson(row.model_files_json, []),
    capabilities: safeParseJson<string[]>(row.capabilities_json, ["benchmark"]),
    sensors: safeParseJson<Worker["sensors"]>(row.sensors_json, null),
    cpuIsa: row.cpu_isa,
    vram: safeParseJson(row.vram_json, null),
    status: deriveWorkerStatus(row),
    lastHeartbeatAt: row.last_heartbeat_at,
    activeJobId: row.active_job_id,
    activeTestId: row.active_job_run_id ?? undefined,
    activeJobProgress: row.active_job_progress_json
      ? safeParseJson<ActiveJobReport | null>(row.active_job_progress_json, null) ?? undefined
      : undefined,
    activeDownloads: activeDownloads.length > 0 ? activeDownloads : undefined,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function mapWorkerEnrolment(row: WorkerRow): WorkerEnrolment {
  const hardware = safeParseJson<{ gpu?: { model?: string }[] } | null>(row.hardware_json, null);
  return {
    id: row.id,
    userId: row.user_id,
    displayName: row.display_name,
    hostname: row.hostname,
    platform: row.platform,
    arch: row.arch,
    gpuModel: hardware?.gpu?.[0]?.model ?? null,
    enrolmentExpiresAt: row.enrolment_expires_at,
    approvedAt: row.approved_at,
  };
}

// LEFT JOIN so a worker with no active job (the common case) still returns
// its row -- active_job_run_id/active_job_progress_json are simply null
// then. Only a 'benchmark' job carries a run_id (MULTIUSER_PLAN.md §1.12's
// enqueueJob call), so install_build/download_model/etc leave
// active_job_run_id null even while genuinely active -- the client shows
// activeJobProgress's phase/bytes for those instead of a run link.
const WORKER_SELECT_SQL = `
  SELECT workers.*, wj.run_id AS active_job_run_id, wj.progress_json AS active_job_progress_json
  FROM workers
  LEFT JOIN worker_jobs wj ON wj.id = workers.active_job_id
`;

function safeParseNumberArray(json: string): number[] | undefined {
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) && parsed.every((v) => typeof v === "number") ? parsed : undefined;
  } catch {
    return undefined;
  }
}

// A raw DB NULL here means this row predates the migration that added these
// columns -- reads the same as a deliberate "unavailable" measurement to
// anything consuming the public ResultRow type, per shared/types.ts's
// GpuMemoryAccuracyLevel doc comment (accuracy is always a concrete level,
// never absent).
function coalesceAccuracy(raw: string | null): GpuMemoryAccuracyLevel {
  return (raw as GpuMemoryAccuracyLevel | null) ?? "unavailable";
}

function mapResult(row: ResultRowRaw): ResultRow {
  return {
    id: row.id,
    run_id: row.run_id,
    // -1 only for pre-run_items-era historical rows that backfillResultIdx
    // (migrate.ts) couldn't recover -- harmless, it just never matches a
    // real item.idx in the client's join.
    idx: row.idx ?? -1,
    model_id: row.model_id,
    test_type: row.test_type as ResultRow["test_type"],
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    mtp: row.mtp,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    n_cpu_moe: row.n_cpu_moe,
    avg_tps: row.avg_tps,
    stddev_tps: row.stddev_tps,
    ram_peak_mib: row.ram_peak_mib,
    vram_peak_mib: row.vram_peak_mib,
    ram_avg_mib: row.ram_avg_mib,
    vram_avg_mib: row.vram_avg_mib,
    ram_free_before_mib: row.ram_free_before_mib,
    vram_free_before_mib: row.vram_free_before_mib,
    backend_type: row.backend_type,
    backend_device_name: row.backend_device_name,
    system_memory_total_mb: row.system_memory_total_mib,
    gpu_memory_total_mb: row.gpu_memory_total_mib,
    gpu_memory_total_accuracy: coalesceAccuracy(row.gpu_memory_total_accuracy),
    gpu_memory_total_source: row.gpu_memory_total_source as ResultRow["gpu_memory_total_source"],
    gpu_memory_free_start_accuracy: coalesceAccuracy(row.gpu_memory_free_start_accuracy),
    gpu_memory_free_start_source: row.gpu_memory_free_start_source as ResultRow["gpu_memory_free_start_source"],
    gpu_memory_model_avg_accuracy: coalesceAccuracy(row.gpu_memory_model_avg_accuracy),
    gpu_memory_model_avg_source: row.gpu_memory_model_avg_source as ResultRow["gpu_memory_model_avg_source"],
    gpu_memory_model_peak_accuracy: coalesceAccuracy(row.gpu_memory_model_peak_accuracy),
    gpu_memory_model_peak_source: row.gpu_memory_model_peak_source as ResultRow["gpu_memory_model_peak_source"],
    // Whole-adapter/per-process/whole-RAM streams are optional on the public
    // type (version-skew tolerance for old workers) -- mapped with ?? instead
    // of coalesceAccuracy so a pre-migration row reads as plain "absent"
    // (undefined), while a current row's null keeps its "unavailable"
    // accuracy pair as-is.
    gpu_memory_used_avg_mib: row.gpu_memory_used_avg_mib ?? undefined,
    gpu_memory_used_peak_mib: row.gpu_memory_used_peak_mib ?? undefined,
    gpu_memory_used_avg_accuracy: coalesceAccuracy(row.gpu_memory_used_avg_accuracy),
    gpu_memory_used_avg_source: row.gpu_memory_used_avg_source as ResultRow["gpu_memory_used_avg_source"],
    gpu_memory_used_peak_accuracy: coalesceAccuracy(row.gpu_memory_used_peak_accuracy),
    gpu_memory_used_peak_source: row.gpu_memory_used_peak_source as ResultRow["gpu_memory_used_peak_source"],
    gpu_memory_process_avg_mib: row.gpu_memory_process_avg_mib ?? undefined,
    gpu_memory_process_peak_mib: row.gpu_memory_process_peak_mib ?? undefined,
    gpu_memory_process_avg_accuracy: coalesceAccuracy(row.gpu_memory_process_avg_accuracy),
    gpu_memory_process_avg_source: row.gpu_memory_process_avg_source as ResultRow["gpu_memory_process_avg_source"],
    gpu_memory_process_peak_accuracy: coalesceAccuracy(row.gpu_memory_process_peak_accuracy),
    gpu_memory_process_peak_source: row.gpu_memory_process_peak_source as ResultRow["gpu_memory_process_peak_source"],
    ram_total_used_avg_mib: row.ram_total_used_avg_mib ?? undefined,
    ram_total_used_peak_mib: row.ram_total_used_peak_mib ?? undefined,
    gpu_layers_loaded: row.gpu_layers_loaded,
    total_model_layers: row.total_model_layers,
    gpu_layers_loaded_draft: row.gpu_layers_loaded_draft ?? undefined,
    total_model_layers_draft: row.total_model_layers_draft ?? undefined,
    gpu_layers_resident_est: row.gpu_layers_resident_est ?? undefined,
    gpu_layers_resident_est_draft: row.gpu_layers_resident_est_draft ?? undefined,
    sample_count: row.sample_count ?? undefined,
    suspect_count: row.suspect_count ?? undefined,
    // Stored as JSON text (SQLite has no array column type) -- tolerate a
    // corrupt/unexpected value by dropping it rather than throwing, since
    // this is diagnostic-only data, not load-bearing for the run itself.
    suspect_samples: row.suspect_samples ? safeParseNumberArray(row.suspect_samples) : undefined,
    repeat_samples: row.repeat_samples ? safeParseNumberArray(row.repeat_samples) : undefined,
    spec_drafted: row.spec_drafted ?? undefined,
    spec_accepted: row.spec_accepted ?? undefined,
    method_version: row.method_version ?? null,
    n_depth: row.n_depth ?? 0,
    config_hash: row.config_hash,
    prompt_offset: row.prompt_offset,
    spec_type: row.spec_type,
    spec_n_max: row.spec_n_max,
    spec_n_min: row.spec_n_min,
    speedup: row.speedup,
    speedup_status: (row.speedup_status as ResultRow["speedup_status"]) ?? null,
    caveat_flags: parseCaveatFlags(row.caveat_flags),
    concurrency: row.concurrency,
    gpu_temp_c_max: row.gpu_temp_c_max,
    gpu_clock_mhz_min: row.gpu_clock_mhz_min,
    gpu_clock_samples: row.gpu_clock_samples ? safeParseNumberArray(row.gpu_clock_samples) ?? null : null,
    cpu_isa: row.cpu_isa,
    imported_bundle_id: row.imported_bundle_id,
    import_opt_in: row.import_opt_in === 1,
    ttft_ms_p50: row.ttft_ms_p50,
    ttft_ms_p95: row.ttft_ms_p95,
    ttft_n: row.ttft_n,
    e2e_ms_mean: row.e2e_ms_mean,
    worker_id: row.worker_id,
    llama_cpp_build: row.llama_cpp_build,
    engine: (row.engine as EngineKind | null) ?? null,
    raw_json_path: row.raw_json_path ?? undefined,
    created_at: row.created_at,
  };
}

// The VPS has no filesystem access to the actual model file (it lives on the
// worker's machine), so it can't compute a real sha256 of the bytes. When the
// caller omits an id, derive a stable one from the identifying fields it did
// supply, so re-registering the same model doesn't create a duplicate row
// with a fresh random id each time.
function deriveModelId(input: RegisterModelInput): string {
  const key =
    input.source === "huggingface"
      ? `hf:${input.hf_repo ?? ""}/${input.hf_file ?? ""}`
      : `local:${input.filename ?? ""}:${input.size_bytes ?? 0}`;
  return createHash("sha256").update(key).digest("hex");
}

// appSettingsRepo.get()'s default when no admin override has ever been
// stored -- see that repo method's own comment for why this is a literal
// rather than an import from shared/probeLadder.ts's own PROBE_MAX_LOADS.
const DEFAULT_PROBE_MAX_LOADS = 24;

// The k-anonymity floor communityRepo's aggregates are subject to: the
// minimum number of DISTINCT opted-in contributors a group must combine
// before it is returned at all. See communityRepo's own comment for why the
// default is 1 (no floor beyond consent) rather than BENCHMARKING_PLAN_V8.md
// §0.9's original 5. Read per call rather than at module load so a test (or
// an operator restarting with a different value) gets the current setting;
// the result is validated to a positive integer here because it is
// interpolated into SQL rather than bound as a parameter -- a HAVING
// threshold cannot be a bound parameter in the prepared statements below
// without reshuffling every existing positional argument.
export const DEFAULT_COMMUNITY_MIN_CONTRIBUTORS = 1;
export function communityMinContributors(): number {
  const raw = process.env.COMMUNITY_MIN_CONTRIBUTORS;
  if (raw == null || raw.trim() === "") return DEFAULT_COMMUNITY_MIN_CONTRIBUTORS;
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 1) return DEFAULT_COMMUNITY_MIN_CONTRIBUTORS;
  return n;
}

export const repo = {
  listModels(): Model[] {
    const rows = getDb()
      .prepare("SELECT * FROM models ORDER BY created_at DESC")
      .all() as ModelRow[];
    return rows.map(mapModel);
  },

  getModel(id: string): Model | undefined {
    const row = getDb().prepare("SELECT * FROM models WHERE id = ?").get(id) as
      | ModelRow
      | undefined;
    return row ? mapModel(row) : undefined;
  },

  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): created_by is set ONLY on
  // the first insert (never touched by the ON CONFLICT branch, regardless of
  // who's calling) -- re-registering an existing model never lets a caller
  // "adopt" one they didn't originally create. The conflict WHERE clause is
  // the actual ownership enforcement: an unclaimed model (created_by NULL --
  // single-tenant mode, or a legacy row) stays freely updatable by anyone;
  // once a model has a real creator, only that same creator's calls may
  // overwrite filename/hf_repo/hf_file/metadata -- everyone else's identical
  // upsert-shaped call just silently no-ops instead of erroring (updateModelMetadata
  // below remains the always-open path for filling in a missing field, e.g.
  // backfilling n_layer, which isn't an identity field). `IS` rather than
  // `=` throughout so two NULLs (single-tenant mode on both sides) compare
  // equal -- plain `=` never does for NULL in SQL.
  registerModel(input: RegisterModelInput): Model {
    const id = input.id ?? deriveModelId(input);
    const now = Date.now();
    const metadata = JSON.stringify(input.metadata ?? {});
    getDb()
      .prepare(
        `INSERT INTO models (id, filename, size_bytes, source, hf_repo, hf_file, metadata, created_by, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           filename = excluded.filename,
           size_bytes = excluded.size_bytes,
           source = excluded.source,
           hf_repo = excluded.hf_repo,
           hf_file = excluded.hf_file,
           metadata = excluded.metadata
         WHERE models.created_by IS NULL OR models.created_by IS excluded.created_by`
      )
      .run(
        id,
        input.filename ?? "",
        input.size_bytes ?? 0,
        input.source,
        input.hf_repo ?? null,
        input.hf_file ?? null,
        metadata,
        input.created_by ?? null,
        now
      );
    return this.getModel(id)!;
  },

  // Merges into a model's existing metadata rather than replacing it
  // wholesale like registerModel's upsert does -- used to backfill a single
  // derived field (e.g. n_layer) onto a model that predates it without
  // disturbing anything else already recorded (arch, quant, ...).
  updateModelMetadata(id: string, patch: Partial<ModelMetadata>): Model | undefined {
    const existing = this.getModel(id);
    if (!existing) return undefined;
    const metadata = JSON.stringify({ ...existing.metadata, ...patch });
    getDb().prepare("UPDATE models SET metadata = ? WHERE id = ?").run(metadata, id);
    return this.getModel(id);
  },

  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): internal-only read of a
  // model's creator, for DELETE /api/models/:id's ownership check --
  // deliberately not part of the public Model type (nothing in the UI needs
  // it yet). undefined means the model doesn't exist at all, distinct from
  // null (exists, but unclaimed -- single-tenant mode or a legacy row).
  getModelCreatedBy(id: string): string | null | undefined {
    const row = getDb().prepare(`SELECT created_by FROM models WHERE id = ?`).get(id) as
      | { created_by: string | null }
      | undefined;
    return row?.created_by;
  },

  // results.model_id is a foreign key (foreign_keys=ON, see migrate.ts) with
  // no ON DELETE clause, so SQLite rejects deleting a model any existing
  // result row still references -- check first so the route can return a
  // clean 409 instead of a raw SQLite constraint error.
  countResultsForModel(modelId: string): number {
    const row = getDb().prepare("SELECT COUNT(*) as n FROM results WHERE model_id = ?").get(modelId) as {
      n: number;
    };
    return row.n;
  },

  // A model deleted while a run against it is still in flight would let the
  // worker's eventual per-item terminal report (POST /api/runs/:id/items/:idx,
  // see routes/runs.ts) hit the same FK constraint above once it tries to
  // insert a `results` row for a model that's already gone -- that insert
  // would fail, the worker's safeItemTerminal would retry and give up, and
  // that item (and potentially the whole run) would be stuck "running"
  // forever with no result and no clear error. Block deletion up front
  // instead.
  countRunningTestsForModel(modelId: string): number {
    const row = getDb()
      .prepare("SELECT COUNT(*) as n FROM runs WHERE model_id = ? AND status = 'running'")
      .get(modelId) as { n: number };
    return row.n;
  },

  deleteModel(id: string): Model | undefined {
    const model = this.getModel(id);
    if (!model) return undefined;
    getDb().prepare("DELETE FROM models WHERE id = ?").run(id);
    return model;
  },

  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3): userId undefined means
  // "single-tenant mode" -- either AUTH_ENABLED is off (authMiddleware never
  // runs, so no route ever has a real req.user to pass) or the caller is
  // system code with no request/user context at all (the reaper, a worker's
  // own heartbeat-driven reconciliation). Both cases correctly mean "operate
  // on every row, not just one user's" -- the `? IS NULL OR ...` clause below
  // is what makes that a single query instead of two near-duplicate ones.
  // Once a real userId is passed, this is the actual enforcement: ownership
  // is checked in SQL, not layered on afterward.
  listTests(userId: string | undefined): Test[] {
    const rows = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'done') AS items_done,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status IN ('failed','failed_oom')) AS items_failed,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'cancelled') AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         WHERE (? IS NULL OR runs.user_id = ?)
         ORDER BY runs.started_at DESC`
      )
      .all(userId ?? null, userId ?? null) as TestRow[];
    return rows.map(mapTest);
  },

  getTest(userId: string | undefined, id: string): Test | undefined {
    const row = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'done') AS items_done,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status IN ('failed','failed_oom')) AS items_failed,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'cancelled') AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         WHERE runs.id = ? AND (? IS NULL OR runs.user_id = ?)`
      )
      .get(id, userId ?? null, userId ?? null) as TestRow | undefined;
    return row ? mapTest(row) : undefined;
  },

  getTestWithResults(userId: string | undefined, id: string): { run: Test; results: ResultRow[]; items: TestItem[] } | undefined {
    const run = this.getTest(userId, id);
    if (!run) return undefined;
    const results = this.getResultsForTest(id);
    const items = this.getTestItems(id);
    return { run, results, items };
  },

  // N7 -- write an imported bundle's rows. They land on a synthetic run
  // badged `imported`, and every row carries its bundle id so it can be shown
  // as imported everywhere and kept out of local profile scoring unless the
  // importer opted in for that bundle.
  importBundleRows(
    userId: string | undefined,
    input: {
      bundleId: string;
      optIn: boolean;
      run: {
        id: string;
        kind: string | null;
        model_filename: string;
        model_quant: string | null;
        model_sha256?: string | null;
        llama_cpp_build: string | null;
        config: Record<string, unknown>;
      };
      hardwareClass: { gpu_name: string | null; backend: string | null; vram_class_mib: number | null };
      rows: ImportedResultRow[];
    }
  ): { runId: string } {
    const database = getDb();
    const now = Date.now();
    const runId = uuid();
    // Never overwrite a local model record from an import: link to it when the
    // content hash already exists here, register a minimal stand-in otherwise.
    const modelId = input.run.model_sha256 ?? `imported:${input.bundleId}`;
    if (!this.getModel(modelId)) {
      this.registerModel({
        id: modelId,
        filename: input.run.model_filename,
        size_bytes: 0,
        source: "local",
        metadata: input.run.model_quant ? { quant: input.run.model_quant } : {},
      });
    }

    const insertResult = database.prepare(
      `INSERT INTO results
         (id, run_id, user_id, idx, model_id, test_type, n_prompt, n_gen, n_depth, n_threads, n_gpu_layers,
          batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft, n_cpu_moe,
          avg_tps, stddev_tps, ram_peak_mib, vram_peak_mib, ram_avg_mib, vram_avg_mib,
          gpu_memory_total_accuracy, gpu_memory_free_start_accuracy,
          gpu_memory_model_avg_accuracy, gpu_memory_model_peak_accuracy,
          sample_count, suspect_count, method_version, config_hash, caveat_flags, concurrency,
          gpu_temp_c_max, gpu_clock_mhz_min, ttft_ms_p50, ttft_ms_p95, ttft_n, e2e_ms_mean,
          llama_cpp_build, engine, imported_bundle_id, import_opt_in, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );

    const tx = database.transaction(() => {
      database
        .prepare(
          `INSERT INTO runs (id, root_run_id, kind, user_id, worker_name, worker_id, llama_cpp_build,
                             llama_cpp_backend, backend_device_name, model_id, config, status, error,
                             started_at, completed_at)
           VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 'done', ?, ?, ?)`
        )
        .run(
          runId,
          runId,
          input.run.kind,
          userId ?? null,
          // A point-in-time label, not a machine identity: an imported bundle
          // carries an anonymized hardware CLASS, never a hostname.
          input.hardwareClass.gpu_name ?? "imported",
          input.run.llama_cpp_build,
          input.hardwareClass.backend,
          input.hardwareClass.gpu_name,
          modelId,
          JSON.stringify({ ...input.run.config, imported_bundle_id: input.bundleId }),
          `imported from bundle ${input.bundleId}`,
          now,
          now
        );

      input.rows.forEach((row, idx) => {
        insertResult.run(
          uuid(),
          runId,
          userId ?? null,
          idx,
          modelId,
          row.test_type,
          row.n_prompt,
          row.n_gen,
          row.n_depth,
          row.n_threads,
          row.n_gpu_layers,
          row.batch_size,
          row.ubatch_size,
          row.cache_type_k,
          row.cache_type_v,
          row.flash_attn,
          row.mtp,
          row.n_gpu_layers_draft,
          row.n_cpu_moe,
          row.avg_tps,
          row.stddev_tps,
          row.ram_peak_mib,
          row.vram_peak_mib,
          row.ram_peak_mib,
          row.vram_peak_mib,
          "unavailable",
          "unavailable",
          "unavailable",
          "unavailable",
          row.sample_count,
          row.suspect_count,
          row.method_version,
          // Stored verbatim: it was recomputed from the canonical form at
          // export time and re-verified on the way in.
          row.config_hash,
          row.caveat_flags.length > 0 ? JSON.stringify(row.caveat_flags) : null,
          row.concurrency,
          row.gpu_temp_c_max,
          row.gpu_clock_mhz_min,
          row.ttft_ms_p50,
          row.ttft_ms_p95,
          row.ttft_n,
          row.e2e_ms_mean,
          input.run.llama_cpp_build,
          row.engine,
          input.bundleId,
          input.optIn ? 1 : 0,
          row.created_at
        );
      });
    });
    tx();
    return { runId };
  },

  // N1 curve read path -- a curve is a deterministic GROUPING over `results`,
  // not a table: rows sharing (model_id, worker_id, llama_cpp_build, engine,
  // method_version), ordered by effective context. idx_results_curve covers
  // exactly those five keys, so this stays an index range scan.
  // Serves the CALLER's own rows; other tenants' contributions only ever
  // surface through the consent-gated communityRepo aggregates, never here.
  listCurveRows(
    userId: string | undefined,
    modelId: string,
    filters: { workerId?: string; build?: string; engine?: string; methodVersion?: number | null }
  ): ResultRow[] {
    const rows = getDb()
      .prepare(
        `SELECT results.*, runs.llama_cpp_backend AS backend_type,
                runs.backend_device_name AS backend_device_name
         FROM results
         JOIN runs ON runs.id = results.run_id
         WHERE results.model_id = ?
           AND (? IS NULL OR results.user_id = ?)
           AND (? IS NULL OR results.worker_id = ?)
           AND (? IS NULL OR results.llama_cpp_build = ?)
           AND (? IS NULL OR results.engine = ?)
           AND (? IS NULL OR results.method_version = ?)
         ORDER BY results.created_at ASC, results.id ASC`
      )
      .all(
        modelId,
        userId ?? null,
        userId ?? null,
        filters.workerId ?? null,
        filters.workerId ?? null,
        filters.build ?? null,
        filters.build ?? null,
        filters.engine ?? null,
        filters.engine ?? null,
        filters.methodVersion ?? null,
        filters.methodVersion ?? null
      ) as (ResultRowRaw & { backend_type: Backend; backend_device_name: string | null })[];
    return rows.map((row) => ({
      ...mapResult(row),
      backend_type: row.backend_type,
      backend_device_name: row.backend_device_name,
    }));
  },

  // Every run under one chain root, whatever its kind -- the chain-scoped
  // read §0.5 makes an indexed equality predicate (idx_runs_root).
  listTestsUnderRoot(userId: string | undefined, rootRunId: string): Test[] {
    const rows = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                0 AS items_total, 0 AS items_done, 0 AS items_failed, 0 AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         WHERE COALESCE(runs.root_run_id, runs.id) = ?
           AND (? IS NULL OR runs.user_id = ?)
         ORDER BY runs.started_at ASC`
      )
      .all(rootRunId, userId ?? null, userId ?? null) as TestRow[];
    return rows.map(mapTest);
  },

  // §0.3 scoring universe for one chain: every run under the root whose kind
  // is 'sweep' (or NULL -- a standalone legacy run is its own sweep, so runs
  // that predate the kind column keep producing profiles), with N3's
  // comparison members excluded: trimmed comparison grids must not compete
  // with Test A's full-grid profiles.
  listChainScoringTests(userId: string | undefined, rootRunId: string): Test[] {
    const rows = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                0 AS items_total, 0 AS items_done, 0 AS items_failed, 0 AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         WHERE COALESCE(runs.root_run_id, runs.id) = ?
           AND (? IS NULL OR runs.user_id = ?)
           AND (runs.kind IS NULL OR runs.kind = 'sweep')
           AND runs.comparison_id IS NULL
         ORDER BY runs.started_at ASC`
      )
      .all(rootRunId, userId ?? null, userId ?? null) as TestRow[];
    return rows.map(mapTest);
  },

  // §0.5 duplicate-trigger guard -- a caller's NON-terminal run matching
  // (user_id, model_id, worker_id) with the same non-NULL kind blocks a new
  // trigger; roots only (root_run_id IS NULL OR root_run_id = id); NULL-kind
  // matches only NULL-kind. The 409 body names the active root and kind.
  findBlockingTest(
    userId: string | undefined,
    modelId: string,
    workerId: string,
    kind: string | null
  ): { id: string; kind: string | null } | undefined {
    return getDb()
      .prepare(
        `SELECT id, kind FROM runs
         WHERE (? IS NULL OR user_id = ?)
           AND model_id = ? AND worker_id = ?
           AND status IN ('running','scheduled')
           AND (root_run_id IS NULL OR root_run_id = id)
           AND ((kind IS NULL AND ? IS NULL) OR kind = ?)
         ORDER BY started_at DESC LIMIT 1`
      )
      .get(userId ?? null, userId ?? null, modelId, workerId, kind ?? null, kind ?? null) as
      | { id: string; kind: string | null }
      | undefined;
  },

  // §0.5 chain quotas -- ≤ 3 active roots per user; probes are exempt from
  // the active-roots quota (minutes-long by construction); a comparison
  // group counts once, not once per member.
  countActiveRoots(userId: string | undefined): number {
    const row = getDb()
      .prepare(
        `SELECT COUNT(DISTINCT COALESCE(comparison_id, id)) AS n FROM runs
         WHERE (? IS NULL OR user_id = ?)
           AND status IN ('running','scheduled')
           AND (root_run_id IS NULL OR root_run_id = id)
           AND (kind IS NULL OR kind <> 'probe')`
      )
      .get(userId ?? null, userId ?? null) as { n: number };
    return row.n;
  },

  // §0.5 -- chain depth of a prospective child whose parent resolves to this
  // root. Depth counts tuning→refine→sweep links; standalone roots are depth 0.
  getChainDepth(rootRunId: string): number {
    const rows = getDb()
      .prepare(
        `SELECT id, config FROM runs WHERE root_run_id = ? AND (root_run_id IS NOT NULL)`
      )
      .all(rootRunId) as { id: string; config: string }[];
    let maxDepth = 1;
    for (const row of rows) {
      try {
        const cfg = JSON.parse(row.config) as { chain_depth?: number };
        if (typeof cfg.chain_depth === "number") maxDepth = Math.max(maxDepth, cfg.chain_depth);
      } catch {
        /* legacy config without a depth marker contributes depth 1 */
      }
    }
    return maxDepth;
  },

  getTestOwnerId(runId: string): string | null | undefined {
    const row = getDb().prepare(`SELECT user_id FROM runs WHERE id = ?`).get(runId) as
      | { user_id: string | null }
      | undefined;
    return row?.user_id;
  },

  // N3 -- every member of one comparison group, oldest first (the first is
  // the fairness reference every later member is checked against).
  listComparisonMembers(comparisonId: string, userId?: string | undefined): Test[] {
    const rows = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'done') AS items_done,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status IN ('failed','failed_oom')) AS items_failed,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'cancelled') AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         WHERE runs.comparison_id = ? AND (? IS NULL OR runs.user_id = ?)
         ORDER BY runs.started_at ASC`
      )
      .all(comparisonId, userId ?? null, userId ?? null) as TestRow[];
    return rows.map(mapTest);
  },

  listComparisonMemberRepeats(comparisonId: string): number[] {
    const rows = getDb()
      .prepare(`SELECT config FROM runs WHERE comparison_id = ?`)
      .all(comparisonId) as { config: string }[];
    const repeats: number[] = [];
    for (const row of rows) {
      try {
        const cfg = JSON.parse(row.config) as TestConfig;
        if (typeof cfg.sweep?.repeats === "number") repeats.push(cfg.sweep.repeats);
      } catch {
        /* unparseable legacy config -- skip */
      }
    }
    return repeats;
  },

  countComparisonMembers(comparisonId: string): Promise<number> | number {
    return (
      getDb().prepare(`SELECT COUNT(*) AS n FROM runs WHERE comparison_id = ?`).get(comparisonId) as { n: number }
    ).n;
  },

  hasActiveComparisonMember(comparisonId: string): boolean {
    const row = getDb()
      .prepare(
        `SELECT COUNT(*) AS n FROM runs WHERE comparison_id = ? AND status IN ('running','scheduled')`
      )
      .get(comparisonId) as { n: number };
    return row.n > 0;
  },

  // §0.5 -- 48h wall clock enforced continuously against RUNNING roots.
  // Returns the ids that were just force-finalized (the caller cancels their
  // pending jobs). Called from the trigger path and the maintenance reaper.
  cancelExpiredRoots(): string[] {
    const db = getDb();
    const cutoff = Date.now() - CHAIN_WALL_CLOCK_MS;
    const expired = db
      .prepare(
        `SELECT DISTINCT COALESCE(root_run_id, id) AS root FROM runs
         WHERE status = 'running' AND started_at < ?`
      )
      .all(cutoff) as { root: string }[];
    const cancelledIds: string[] = [];
    for (const { root } of expired) {
      const stillRunning = db
        .prepare(`SELECT id FROM runs WHERE COALESCE(root_run_id, id) = ? AND status = 'running'`)
        .all(root);
      if (stillRunning.length === 0) continue;
      for (const r of stillRunning) {
        this.reconcileStaleTest(undefined, (r as { id: string }).id, "chain exceeded its 48 hour wall clock");
        cancelledIds.push((r as { id: string }).id);
      }
    }
    return cancelledIds;
  },

  // backend_type/backend_device_name are denormalized from the parent run
  // here, not physical columns on `results` itself -- see ResultRow's own
  // doc comment for why (backend can't vary between one run's own items,
  // so repeating an identical string on every row would be pure noise).
  // Mirrors worker_name's own existing JOIN in the CSV/MD export
  // (server/src/routes/results.ts).
  getResultsForTest(runId: string): ResultRow[] {
    const rows = getDb()
      .prepare(
        `SELECT r.*, runs.llama_cpp_backend AS backend_type, runs.backend_device_name AS backend_device_name
         FROM results r
         JOIN runs ON runs.id = r.run_id
         WHERE r.run_id = ?
         ORDER BY r.created_at ASC`
      )
      .all(runId) as ResultRowRaw[];
    return rows.map(mapResult);
  },

  // §0.12 / N6 -- the >1/3 thermally_throttled ratio, computed the same way
  // for both the on-demand /sustained read and the once-per-completed-run
  // observability event: denominator is every non-cancelled item, skipped
  // included; numerator is items with >=1 result flagged thermally_throttled.
  thermallyFlaggedRatio(runId: string): { ratio: number; flagged: number; denominator: number } {
    const items = this.getTestItems(runId);
    const results = this.getResultsForTest(runId);
    const flaggedIdx = new Set(
      results.filter((r) => (r.caveat_flags ?? []).includes("thermally_throttled")).map((r) => r.idx)
    );
    const denominator = items.filter((i) => i.status !== "cancelled").length;
    const ratio = denominator > 0 ? flaggedIdx.size / denominator : 0;
    return { ratio, flagged: flaggedIdx.size, denominator };
  },

  // M6 -- "the sample series is this plan's first bulky column (~2 KB per
  // item). It may be pruned 30 days after ingest -- detection already ran at
  // ingest time, and gpu_clock_mhz_min, gpu_temp_c_max and the caveat flag
  // persist indefinitely, which is everything later readers (scoring, N6
  // policy) consume." Nulls only the sample series; every other thermal
  // column and caveat_flags are untouched. Called from reaper.ts's
  // runMaintenanceSweep, same interval as the other retention prunes.
  pruneOldGpuClockSamples(days: number): number {
    const cutoff = Date.now() - days * 24 * 3600 * 1000;
    return getDb()
      .prepare(`UPDATE results SET gpu_clock_samples = NULL WHERE gpu_clock_samples IS NOT NULL AND created_at < ?`)
      .run(cutoff).changes;
  },

  createTest(userId: string | undefined, run: Test): void {
    getDb()
      .prepare(
        `INSERT INTO runs (id, root_run_id, kind, comparison_id, user_id, worker_name, worker_id, llama_cpp_build, llama_cpp_backend, backend_device_name, model_id, config, status, error, started_at, completed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        run.id,
        run.root_run_id ?? run.id,
        run.kind ?? null,
        run.comparison_id ?? null,
        userId ?? null,
        run.worker_name,
        run.worker_id ?? null,
        run.llama_cpp_build,
        run.llama_cpp_backend,
        run.backend_device_name ?? null,
        run.model_id,
        JSON.stringify(run.config),
        run.status,
        run.error ?? null,
        run.started_at,
        run.completed_at ?? null
      );
  },

  // Pre-creates the full planned item list at trigger time (before the
  // worker is even contacted) so the UI can show "N total tests" and a
  // queued list immediately. idx must match what the worker will report
  // against -- both sides compute it by calling shared/sweep.ts's
  // expandSweep on the exact same sweep JSON, so there's no separate
  // registration round trip.
  createTestItems(userId: string | undefined, runId: string, items: SweepItem[]): void {
    const database = getDb();
    const insert = database.prepare(
      `INSERT INTO run_items
         (id, run_id, user_id, idx, n_prompt, n_gen, n_depth, concurrency, n_threads, n_gpu_layers,
          batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft, n_cpu_moe, status)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')`
    );
    const tx = database.transaction((rows: SweepItem[]) => {
      for (const item of rows) {
        insert.run(
          uuid(),
          runId,
          userId ?? null,
          item.idx,
          item.n_prompt,
          item.n_gen,
          item.n_depth ?? 0,
          item.concurrency ?? 1,
          item.threads,
          item.n_gpu_layers,
          item.batch_size,
          item.ubatch_size,
          item.cache_type_k,
          item.cache_type_v,
          item.flash_attn,
          item.mtp,
          item.n_gpu_layers_draft,
          item.n_cpu_moe
        );
      }
    });
    tx(items);
  },

  getTestItems(runId: string): TestItem[] {
    const rows = getDb()
      .prepare("SELECT * FROM run_items WHERE run_id = ? ORDER BY idx ASC")
      .all(runId) as TestItemRow[];
    return rows.map(mapTestItem);
  },

  // Best-effort progress tick (loading/processing/generating/benchmarking) --
  // called far more often than the terminal update below, so this is a
  // plain UPDATE with no completion bookkeeping. ram_free_before_mib/
  // vram_free_before_mib are set once via COALESCE-on-existing-value (same
  // pattern as started_at) since they're only ever sent on an item's very
  // first tick.
  updateTestItemTick(runId: string, idx: number, patch: TestItemTickInput): void {
    getDb()
      .prepare(
        `UPDATE run_items SET
           status = ?,
           detail = COALESCE(?, detail),
           ram_mib = ?,
           vram_mib = ?,
           live_tps = COALESCE(?, live_tps),
           ram_free_before_mib = COALESCE(ram_free_before_mib, ?),
           vram_free_before_mib = COALESCE(vram_free_before_mib, ?),
           started_at = COALESCE(started_at, ?)
         WHERE run_id = ? AND idx = ?`
      )
      .run(
        patch.status,
        patch.detail ?? null,
        patch.ram_mib ?? null,
        patch.vram_mib ?? null,
        patch.live_tps ?? null,
        patch.ram_free_before_mib ?? null,
        patch.vram_free_before_mib ?? null,
        Date.now(),
        runId,
        idx
      );
  },

  // Records one item's final outcome, writes a `results` row for a
  // successful item, and -- once every item for the run is terminal --
  // finalizes the run itself (done/partial/failed). Returns undefined if the
  // run doesn't exist so the route can 404 instead of silently no-op'ing.
  recordTestItemTerminal(runId: string, idx: number, input: TestItemTerminalInput): Test | undefined {
    const database = getDb();
    // undefined userId -- this is a worker-authenticated write, not a
    // browser request; the route verifies worker.id === run.worker_id itself
    // before ever calling this (MULTIUSER_PLAN.md §4.3's own distinction:
    // worker-reported writes are authorized by worker identity, not user id).
    const run = this.getTest(undefined, runId);
    if (!run) return undefined;
    // The public Test type deliberately doesn't expose user_id (kept
    // DB-internal, same as models.created_by) -- read it directly so every
    // `results` row inserted below can be denormalized-stamped with the same
    // owner as its parent run (§4.3: results.user_id, unlike run_items',
    // isn't set at creation time -- results rows don't exist until an item
    // goes terminal, here).
    const runUserId = (
      database.prepare(`SELECT user_id FROM runs WHERE id = ?`).get(runId) as { user_id: string | null } | undefined
    )?.user_id ?? null;

    const tx = database.transaction(() => {
      const now = Date.now();
      database
        .prepare(
          `UPDATE run_items SET
             status = ?, error = ?, completed_at = ?,
             ram_peak_mib = ?, vram_peak_mib = ?,
             ram_avg_mib = ?, vram_avg_mib = ?
           WHERE run_id = ? AND idx = ?`
        )
        .run(
          input.status,
          input.error ?? null,
          now,
          input.ram_peak_mib ?? null,
          input.vram_peak_mib ?? null,
          input.ram_avg_mib ?? null,
          input.vram_avg_mib ?? null,
          runId,
          idx
        );

      if (input.status === "done" && input.results) {
        const insertResult = database.prepare(
          `INSERT INTO results
             (id, run_id, user_id, idx, model_id, test_type, n_prompt, n_gen, n_depth, n_threads, n_gpu_layers,
              batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft, n_cpu_moe,
              avg_tps, stddev_tps, ram_peak_mib, vram_peak_mib,
              ram_avg_mib, vram_avg_mib, ram_free_before_mib, vram_free_before_mib,
              system_memory_total_mib, gpu_memory_total_mib,
              gpu_memory_total_accuracy, gpu_memory_total_source,
              gpu_memory_free_start_accuracy, gpu_memory_free_start_source,
              gpu_memory_model_avg_accuracy, gpu_memory_model_avg_source,
              gpu_memory_model_peak_accuracy, gpu_memory_model_peak_source,
              gpu_memory_used_avg_mib, gpu_memory_used_peak_mib,
              gpu_memory_used_avg_accuracy, gpu_memory_used_avg_source,
              gpu_memory_used_peak_accuracy, gpu_memory_used_peak_source,
              gpu_memory_process_avg_mib, gpu_memory_process_peak_mib,
              gpu_memory_process_avg_accuracy, gpu_memory_process_avg_source,
              gpu_memory_process_peak_accuracy, gpu_memory_process_peak_source,
              ram_total_used_avg_mib, ram_total_used_peak_mib,
              gpu_layers_loaded, total_model_layers, gpu_layers_loaded_draft, total_model_layers_draft,
              gpu_layers_resident_est, gpu_layers_resident_est_draft,
              sample_count, suspect_count, suspect_samples, repeat_samples, spec_drafted, spec_accepted,
              method_version, prompt_offset, spec_type, spec_n_max, spec_n_min, config_hash,
              caveat_flags, concurrency, gpu_temp_c_max, gpu_clock_mhz_min, gpu_clock_samples,
              cpu_isa, ttft_ms_p50, ttft_ms_p95, ttft_n, e2e_ms_mean,
              worker_id, llama_cpp_build, engine, raw_json_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        );
        // Up to two rows for one idx (a pp row and a tg row from the same
        // benchmark process) -- distinguished by test_type, see
        // worker/src/index.ts's runSweepItem/runSweepItemViaServer.
        for (const resultInput of input.results) {
          const row = buildResultRow(runId, run.model_id, idx, resultInput);
          // §0.2/§0.8 -- one engine derivation feeds both the stored column
          // and the config hash, so twins can never disagree on engine.
          const engine = deriveEngineForResult(row, run);
          // §0.8 -- server and worker agree via the one shared function; the
          // server (re)computes here so a version-skewed worker that never
          // heard of config_hash still gets joinable rows.
          const hash = configHash(configHashInputFromResult(row, resultInput, engine));
          insertResult.run(
            row.id,
            row.run_id,
            runUserId,
            row.idx,
            row.model_id,
            row.test_type,
            row.n_prompt,
            row.n_gen,
            row.n_depth ?? 0,
            row.n_threads,
            row.n_gpu_layers,
            row.batch_size,
            row.ubatch_size,
            row.cache_type_k,
            row.cache_type_v,
            row.flash_attn,
            row.mtp,
            row.n_gpu_layers_draft,
            row.n_cpu_moe,
            row.avg_tps,
            row.stddev_tps,
            row.ram_peak_mib,
            row.vram_peak_mib,
            row.ram_avg_mib,
            row.vram_avg_mib,
            row.ram_free_before_mib,
            row.vram_free_before_mib,
            row.system_memory_total_mb,
            row.gpu_memory_total_mb,
            row.gpu_memory_total_accuracy,
            row.gpu_memory_total_source,
            row.gpu_memory_free_start_accuracy,
            row.gpu_memory_free_start_source,
            row.gpu_memory_model_avg_accuracy,
            row.gpu_memory_model_avg_source,
            row.gpu_memory_model_peak_accuracy,
            row.gpu_memory_model_peak_source,
            row.gpu_memory_used_avg_mib,
            row.gpu_memory_used_peak_mib,
            row.gpu_memory_used_avg_accuracy,
            row.gpu_memory_used_avg_source,
            row.gpu_memory_used_peak_accuracy,
            row.gpu_memory_used_peak_source,
            row.gpu_memory_process_avg_mib,
            row.gpu_memory_process_peak_mib,
            row.gpu_memory_process_avg_accuracy,
            row.gpu_memory_process_avg_source,
            row.gpu_memory_process_peak_accuracy,
            row.gpu_memory_process_peak_source,
            row.ram_total_used_avg_mib,
            row.ram_total_used_peak_mib,
            row.gpu_layers_loaded,
            row.total_model_layers,
            row.gpu_layers_loaded_draft ?? null,
            row.total_model_layers_draft ?? null,
            row.gpu_layers_resident_est ?? null,
            row.gpu_layers_resident_est_draft ?? null,
            row.sample_count ?? null,
            row.suspect_count ?? null,
            row.suspect_samples ? JSON.stringify(row.suspect_samples) : null,
            row.repeat_samples ? JSON.stringify(row.repeat_samples) : null,
            row.spec_drafted ?? null,
            row.spec_accepted ?? null,
            row.method_version ?? null,
            row.prompt_offset ?? null,
            row.spec_type ?? null,
            row.spec_n_max ?? null,
            row.spec_n_min ?? null,
            hash,
            row.caveat_flags && row.caveat_flags.length > 0 ? JSON.stringify(row.caveat_flags) : null,
            row.concurrency ?? null,
            row.gpu_temp_c_max ?? null,
            row.gpu_clock_mhz_min ?? null,
            row.gpu_clock_samples ? JSON.stringify(row.gpu_clock_samples) : null,
            row.cpu_isa ?? null,
            row.ttft_ms_p50 ?? null,
            row.ttft_ms_p95 ?? null,
            row.ttft_n ?? null,
            row.e2e_ms_mean ?? null,
            run.worker_id ?? null,
            run.llama_cpp_build ?? null,
            engine,
            null,
            row.created_at
          );
          markSpeedupStatuses(database, row.id, hash, engine, resultInput.wall_clock_fallback === true);
        }
      }

      // Tier-2 backend_device_name upgrade (see shared/types.ts's
      // Test.backend_device_name) -- a plain UPDATE, not COALESCE: the Tier-1
      // fallback is always written first at trigger time (createTest,
      // routes/tests.ts's resolveBuildForRun), strictly before any item can
      // go terminal, so this write is always the intended upgrade, never a
      // downgrade.
      if (input.backend_device_name) {
        database.prepare(`UPDATE runs SET backend_device_name = ? WHERE id = ?`).run(input.backend_device_name, runId);
      }

      if (this.countUnfinishedItems(runId) === 0) {
        this.finalizeTest(runId, now);
      }
    });
    tx();

    return this.getTest(undefined, runId);
  },

  countUnfinishedItems(runId: string): number {
    const row = getDb()
      .prepare(
        `SELECT COUNT(*) as n FROM run_items
         WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','failed_unsupported','cancelled','skipped')`
      )
      .get(runId) as { n: number };
    return row.n;
  },

  // Derives the run's overall status from its items' final statuses. Only
  // meaningful once countUnfinishedItems(runId) === 0 -- callers are
  // responsible for that check (recordTestItemTerminal does it after every
  // terminal write; failAllTestItems calls it directly since it just made
  // that true itself).
  //
  // cancelReason distinguishes *why* items ended up 'cancelled': omitted
  // (the default) means a genuine per-item user stop (worker/src/index.ts's
  // stopRequested loop, reported one item at a time through the normal
  // recordTestItemTerminal path) -- "stopped by user" is accurate there.
  // reconcileStaleTest passes an explicit reason instead, since its
  // 'cancelled' items were never actually confirmed stopped by anyone --
  // the DB just lost track of a run the worker itself had already moved on
  // from (its own terminal report may have been rejected or never arrived).
  // Labelling that "stopped by user" too would be a lie the user has no way
  // to tell apart from a real stop.
  finalizeTest(runId: string, completedAt: number = Date.now(), cancelReason?: string): void {
    const counts = getDb()
      .prepare(
        `SELECT
           SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as done,
           SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped,
           SUM(CASE WHEN status = 'failed_oom' THEN 1 ELSE 0 END) as oom,
           SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
           COUNT(*) as total
         FROM run_items WHERE run_id = ?`
      )
      .get(runId) as { done: number; skipped: number; oom: number; cancelled: number; total: number };

    let status: TestStatus;
    let error: string | null = null;
    if (counts.cancelled > 0) {
      // A genuine user stop always reads "cancelled" regardless of how much
      // finished first -- that's the outcome the user asked for. An
      // unconfirmed/reconciled stop (cancelReason set) is different: if some
      // combinations *did* complete and are saved, "partial" reflects that
      // there's real, usable data here rather than implying the whole run
      // is worthless.
      status = cancelReason && counts.done > 0 ? "partial" : "cancelled";
      const reason = cancelReason ?? "stopped by user";
      error = `${reason} after ${counts.done} of ${counts.total} test${counts.total === 1 ? "" : "s"} completed`;
    } else if (counts.total === 0 || (counts.done === 0 && counts.skipped === 0)) {
      status = "failed";
    } else if (counts.done + counts.skipped === counts.total) {
      status = "done";
      if (counts.skipped > 0) {
        error = `${counts.skipped} of ${counts.total} test${counts.total === 1 ? "" : "s"} skipped`;
      }
    } else {
      status = "partial";
    }

    // Guards on counts.cancelled (not status) -- a reconciled run with some
    // completions now reads status "partial" too (see above), but its error
    // was already set from cancelReason and must not be overwritten with a
    // "tests failed" message; nothing here was a bench failure.
    if (counts.cancelled === 0) {
      const failedCount = counts.total - counts.done - counts.skipped;
      if (failedCount > 0) {
        error = `${failedCount} of ${counts.total} test${counts.total === 1 ? "" : "s"} failed`;
        if (counts.oom > 0) error += ` (${counts.oom} OOM)`;
        if (counts.skipped > 0) error += `, ${counts.skipped} skipped`;
      }
    }

    getDb()
      .prepare(`UPDATE runs SET status = ?, error = ?, completed_at = ? WHERE id = ?`)
      .run(status, error, completedAt, runId);
  },

  // Used when the worker rejects a trigger or is unreachable -- the run's
  // items already exist (created up front, see routes/tests.ts) but nothing
  // ever ran, so mark all of them failed rather than leaving a permanently
  // "queued" list for a run that never started.
  // userId is an ownership check, not a value written anywhere here -- a run
  // whose items are all still pre-existing rows already carries its own
  // user_id from createTestItems; this just refuses to act on a run the
  // caller doesn't own (undefined = system context, no check -- see this
  // section's own header comment above listTests).
  failAllTestItems(userId: string | undefined, runId: string, error: string): void {
    if (!this.getTest(userId, runId)) return;
    const database = getDb();
    const now = Date.now();
    const tx = database.transaction(() => {
      database
        .prepare(
          `UPDATE run_items SET status = 'failed', error = ?, completed_at = ?
           WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','failed_unsupported','cancelled','skipped')`
        )
        .run(error, now, runId);
      this.finalizeTest(runId, now);
    });
    tx();
  },

  // Marks a run's still-unfinished items 'cancelled' (not 'failed' --
  // nothing here indicates an actual bench failure, just that the worker
  // isn't tracking this run anymore) and finalizes it. `note` is both the
  // per-item error text and finalizeTest's cancelReason, so the run-level
  // message stays honest about this being an unconfirmed/lost run, not a
  // genuine user stop (see finalizeTest's own doc comment).
  // Called from system contexts with no user in scope (the reaper, and the
  // browser-facing /stop route) for a run this server genuinely lost track
  // of or the user asked to stop -- NOT for a worker's own job-completion
  // failure report, which has a real reported reason and belongs on
  // reportJobFailure below instead (see that function's own doc comment for
  // why conflating the two mislabels a genuine error as "cancelled").
  reconcileStaleTest(userId: string | undefined, runId: string, note: string): Test | undefined {
    if (!this.getTest(userId, runId)) return undefined;
    const database = getDb();
    const now = Date.now();
    const tx = database.transaction(() => {
      database
        .prepare(
          `UPDATE run_items SET status = 'cancelled', error = ?, completed_at = ?
           WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','failed_unsupported','cancelled','skipped')`
        )
        .run(note, now, runId);
      this.finalizeTest(runId, now, note);
    });
    tx();
    return this.getTest(userId, runId);
  },

  // A worker's own job-completion report of `ok: false` -- unlike
  // reconcileStaleTest above, this is a CONFIRMED failure with a real reason
  // the worker gave us (e.g. the terminal-result POST itself came back 500,
  // or install_build genuinely failed), not a case of the server losing
  // track of a run. Marks unfinished items 'failed' (never 'cancelled' --
  // "cancelled" reads as "stopped by user, or lost with nothing completed"
  // everywhere else in this app, per Tests.tsx's own column description,
  // which is actively misleading for a run that failed with a real,
  // reported error) so finalizeTest's ordinary done/failed accounting lands
  // on "failed" (nothing completed) or "partial" (something did) instead.
  // finalizeTest's own error message is then a generic "N of M tests failed"
  // summary; overwritten here with the worker's actual reported text so the
  // run-level line stays specific instead of vague.
  reportJobFailure(userId: string | undefined, runId: string, note: string): Test | undefined {
    if (!this.getTest(userId, runId)) return undefined;
    const database = getDb();
    const now = Date.now();
    const tx = database.transaction(() => {
      database
        .prepare(
          `UPDATE run_items SET status = 'failed', error = ?, completed_at = ?
           WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','failed_unsupported','cancelled','skipped')`
        )
        .run(note, now, runId);
      this.finalizeTest(runId, now);
      database.prepare(`UPDATE runs SET error = ? WHERE id = ?`).run(note, runId);
    });
    tx();
    return this.getTest(userId, runId);
  },

  // --- Multi-user Stage 1: workers (MULTIUSER_PLAN.md §1.2/§1.5/§1.6) ---
  workerRepo: {
    getWorker(id: string): Worker | undefined {
      const row = getDb().prepare(`${WORKER_SELECT_SQL} WHERE workers.id = ?`).get(id) as WorkerRow | undefined;
      return row ? mapWorker(row) : undefined;
    },

    // A worker's own persisted config.json (worker/src/index.ts) -- not
    // anything Tailscale/network-derived, unlike the old config.ts. Used by
    // routes/queue.ts to resolve a re-enrolling or re-connecting worker's
    // existing row without creating a duplicate.
    getByMachineId(machineId: string): Worker | undefined {
      const row = getDb().prepare(`${WORKER_SELECT_SQL} WHERE workers.machine_id = ?`).get(machineId) as
        | WorkerRow
        | undefined;
      return row ? mapWorker(row) : undefined;
    },

    listWorkers(): Worker[] {
      const rows = getDb().prepare(`${WORKER_SELECT_SQL} ORDER BY workers.created_at ASC`).all() as WorkerRow[];
      return rows.map(mapWorker);
    },

    // Permanently forgets a machine (routes/workers.ts's DELETE
    // /api/workers/:id -- the ownership + not-busy checks live there, this
    // is just the row removal). Deliberately NOT the same posture as
    // pruneExpiredEnrolments' "never delete a row with history" -- this IS
    // the user asking to delete it. Safe regardless: runs.worker_id is
    // `ON DELETE SET NULL` and runs.worker_name is a point-in-time snapshot
    // (COLUMN_MIGRATIONS in migrate.ts), so every past run stays fully
    // readable, just no longer linked to a live worker row. worker_jobs rows
    // (`ON DELETE CASCADE`) go with it -- correct, since the caller already
    // refused to delete a busy machine, so there's nothing queued/in-flight
    // to lose. If this machine reconnects later, its machine_id is gone from
    // this table too, so the server treats it as a brand-new enrolment.
    deleteWorker(id: string): void {
      getDb().prepare(`DELETE FROM workers WHERE id = ?`).run(id);
    },

    // Cosmetic, user-controlled rename -- the DB column's own comment
    // ("user-editable; defaults to the reported hostname") anticipated this,
    // and reissueEnrolment above already takes care not to clobber it across
    // a re-enrolment; this was simply the write path that never got built.
    // Ownership is checked by the route (assertOwnsWorker), same as every
    // other per-worker action -- this just does the update.
    renameWorker(id: string, displayName: string): Worker | undefined {
      getDb().prepare(`UPDATE workers SET display_name = ?, updated_at = ? WHERE id = ?`).run(displayName, Date.now(), id);
      return this.getWorker(id);
    },

    // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.6) -- the caller's own
    // machines only, for the AI context snapshot (routes/ai.ts's
    // hardwareSummary). A separate function rather than listWorkers(userId?)
    // so the common global-listing call site (the Workers page's own
    // GET /api/workers, still unscoped -- a user manages only their own
    // machines in practice via ownership checks elsewhere, but nothing here
    // needs a second, filtered listing of everyone else's) isn't forced to
    // pass an always-undefined argument.
    listWorkersForUser(userId: string): Worker[] {
      const rows = getDb()
        .prepare(`${WORKER_SELECT_SQL} WHERE workers.user_id = ? ORDER BY workers.created_at ASC`)
        .all(userId) as WorkerRow[];
      return rows.map(mapWorker);
    },

    // Searches every worker's cached model_files_json (from its last
    // heartbeat, see WorkerStatePush.model_files) for a matching filename --
    // replaces the old live "ask every worker" fan-out (server/src/routes/
    // models.ts's findGgufInfoFromWorkers) now that there's no outbound HTTP
    // to workers at all. The stored JSON is the richer WorkerStatePush shape
    // (path + size_bytes + n_layer/mtp_layers), even though the public
    // Worker.modelFiles type only promises plain ModelDirFile -- this is the
    // one place that reads the extra fields back out, so the cast stays
    // contained here rather than leaking into route code.
    // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.5): scoped to the caller's own
    // machines once authenticated -- every machine in single-tenant mode
    // (userId undefined), unchanged from before this scoping existed.
    findModelFileMeta(
      userId: string | undefined,
      filename: string
    ):
      | { n_layer: number | null; mtp_layers: number | null; tensor_layer_bytes: TensorLayerBreakdown | null }
      | undefined {
      const rows = getDb()
        .prepare(`SELECT model_files_json FROM workers WHERE (? IS NULL OR user_id = ?)`)
        .all(userId ?? null, userId ?? null) as { model_files_json: string | null }[];
      for (const row of rows) {
        if (!row.model_files_json) continue;
        const files = safeParseJson<(ModelDirFileMeta)[]>(row.model_files_json, []);
        // state "missing" entries are kept by the worker's cache for
        // re-download bookkeeping after the file was deleted -- their stale
        // n_layer/mtp_layers must not satisfy a lookup for a live file.
        const match = files.find((f) => f.path === filename && f.state !== "missing");
        if (
          match &&
          (typeof match.n_layer === "number" || typeof match.mtp_layers === "number" || match.tensor_layer_bytes)
        ) {
          return {
            n_layer: match.n_layer ?? null,
            mtp_layers: match.mtp_layers ?? null,
            tensor_layer_bytes: match.tensor_layer_bytes ?? null,
          };
        }
      }
      return undefined;
    },

    // Self-announce: a worker identifies itself purely by the machine_id it
    // persists locally -- the FIRST heartbeat/queue-poll from a machine_id
    // the server has never seen creates its `workers` row (MULTIUSER_PLAN.md
    // §1.2). Stage 3 replaces this shared-secret-plus-self-announce identity
    // with real enrolment; this function's `getByMachineId` reuse is exactly
    // what lets that later transition happen with no migration of existing
    // rows (§3.4).
    getOrCreateByMachineId(machineId: string, hostname: string): Worker {
      const existing = this.getByMachineId(machineId);
      if (existing) return existing;
      const now = Date.now();
      const id = uuid();
      getDb()
        .prepare(
          `INSERT INTO workers (id, machine_id, display_name, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)`
        )
        .run(id, machineId, hostname, now, now);
      return this.getWorker(id)!;
    },

    // Called on every heartbeat/queue poll (MULTIUSER_PLAN.md §1.5).
    // activeJobId is null while idle -- the caller (routes/queue.ts) derives
    // it from the ActiveJobReport it received, if any, not from anything
    // stored here previously.
    recordHeartbeat(workerId: string, state: WorkerStatePush, activeJobId: string | null): void {
      getDb()
        .prepare(
          `UPDATE workers SET
             backend = ?, platform = ?, arch = ?, hostname = ?,
             hardware_json = ?, installed_builds_json = ?, model_files_json = ?,
             vram_json = COALESCE(?, vram_json),
             capabilities_json = ?, sensors_json = COALESCE(?, sensors_json), cpu_isa = COALESCE(?, cpu_isa),
             last_heartbeat_at = ?, active_job_id = ?, updated_at = ?
           WHERE id = ?`
        )
        .run(
          state.backend,
          state.hardware.platform,
          state.hardware.arch,
          state.hostname,
          JSON.stringify(state.hardware),
          JSON.stringify(state.installed_builds),
          JSON.stringify(state.model_files),
          state.vram !== undefined ? JSON.stringify(state.vram) : null,
          JSON.stringify(state.capabilities ?? []),
          state.sensors !== undefined ? JSON.stringify(state.sensors) : null,
          state.cpu_isa ?? null,
          Date.now(),
          activeJobId,
          Date.now(),
          workerId
        );
    },

    // MULTIUSER_PLAN.md §1.14: POST /api/runs/:id/pause|resume flips this
    // per-WORKER (not per-job) flag; the heartbeat handler reads it on every
    // poll to build HeartbeatResponse.control.pause. Persists until an
    // explicit resume -- unlike a job's one-shot cancel_requested, pause is
    // a standing state, matching the worker's own existing
    // pauseRequested/resume semantics (worker/src/index.ts).
    setPauseRequested(workerId: string, paused: boolean): void {
      getDb().prepare(`UPDATE workers SET pause_requested = ? WHERE id = ?`).run(paused ? 1 : 0, workerId);
    },

    getPauseRequested(workerId: string): boolean {
      const row = getDb().prepare(`SELECT pause_requested FROM workers WHERE id = ?`).get(workerId) as
        | { pause_requested: number }
        | undefined;
      return row?.pause_requested === 1;
    },

    // --- Multi-user Stage 3: device-flow enrolment (MULTIUSER_PLAN.md §3.3/§3.4) ---

    getEnrolmentById(id: string): WorkerEnrolment | undefined {
      const row = getDb().prepare(`SELECT * FROM workers WHERE id = ?`).get(id) as WorkerRow | undefined;
      return row ? mapWorkerEnrolment(row) : undefined;
    },

    // The human-facing lookup -- used by both GET /api/device/status (any
    // state) and POST /api/device/approve (must still be pending). Not
    // scoped by state itself; callers check approvedAt/enrolmentExpiresAt.
    getByUserCode(userCode: string): WorkerEnrolment | undefined {
      const row = getDb().prepare(`SELECT * FROM workers WHERE user_code = ?`).get(userCode) as WorkerRow | undefined;
      return row ? mapWorkerEnrolment(row) : undefined;
    },

    // The worker-held lookup -- POST /api/device/token polls by this, never
    // by user_code (that's the human's code, not the machine's).
    getByEnrolmentCodeHash(hash: string): WorkerEnrolment | undefined {
      const row = getDb().prepare(`SELECT * FROM workers WHERE enrolment_code_hash = ?`).get(hash) as
        | WorkerRow
        | undefined;
      return row ? mapWorkerEnrolment(row) : undefined;
    },

    // First enrolment of a brand-new machine (§3.1 step 2) -- display_name
    // defaults to the reported hostname (§3.3's own naming rule: no
    // generated names, no identity-derived names). user_id/approved_at stay
    // NULL until a human approves it (§3.4).
    createPending(opts: {
      machineId: string;
      hostname: string;
      platform: string;
      arch: string;
      deviceCode: string;
      userCode: string;
      expiresAt: number;
      // Not yet known from a heartbeat (there's been none) -- carried
      // straight from the worker's own POST /api/device/start body instead,
      // so findPossibleDuplicate below has something to compare against
      // before this machine is even approved. Optional purely for a
      // not-yet-updated worker binary; a normal fresh install always sends it.
      hardware?: HardwareInfo;
    }): WorkerEnrolment {
      const now = Date.now();
      const id = uuid();
      getDb()
        .prepare(
          `INSERT INTO workers
             (id, machine_id, display_name, hostname, platform, arch, hardware_json,
              enrolment_code_hash, user_code, enrolment_expires_at, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          id,
          opts.machineId,
          opts.hostname,
          opts.hostname,
          opts.platform,
          opts.arch,
          opts.hardware ? JSON.stringify(opts.hardware) : null,
          hashToken(opts.deviceCode),
          opts.userCode,
          opts.expiresAt,
          now,
          now
        );
      return this.getEnrolmentById(id)!;
    },

    // Re-enrolment of an ALREADY-KNOWN machine (§3.4) -- an owned machine
    // reconnecting (expired session, revoked from Settings, rebuilt disk) or
    // a still-pending one retrying after an abandoned attempt. Deliberately
    // does NOT touch display_name (preserves a user's rename across
    // reconnects) or user_id/approved_at (preserves existing ownership
    // exactly as-is; only a FIRST enrolment ever sets those, via approve()
    // below) -- only the self-reported hostname/platform/arch and the fresh
    // enrolment code/expiry are refreshed.
    reissueEnrolment(
      workerId: string,
      opts: {
        hostname: string;
        platform: string;
        arch: string;
        deviceCode: string;
        userCode: string;
        expiresAt: number;
        hardware?: HardwareInfo;
      }
    ): WorkerEnrolment {
      getDb()
        .prepare(
          `UPDATE workers SET
             hostname = ?, platform = ?, arch = ?, hardware_json = COALESCE(?, hardware_json),
             enrolment_code_hash = ?, user_code = ?, enrolment_expires_at = ?, updated_at = ?
           WHERE id = ?`
        )
        .run(
          opts.hostname,
          opts.platform,
          opts.arch,
          opts.hardware ? JSON.stringify(opts.hardware) : null,
          hashToken(opts.deviceCode),
          opts.userCode,
          opts.expiresAt,
          Date.now(),
          workerId
        );
      return this.getEnrolmentById(workerId)!;
    },

    // Bug fix: worker/config.json's machine_id (worker/src/index.ts) is
    // generated once with randomUUID() and lives ONLY in that file -- there's
    // no other on-disk or OS-level trace of it. So a user who deletes a
    // worker's install folder and re-runs setup gets a fresh machine_id;
    // getByMachineId above always misses, and createPending would otherwise
    // silently insert a second, indistinguishable `workers` row (same
    // hostname/platform/arch/owner) with no history and no link to the old
    // one. Matches on EITHER self-reported hostname OR a hardware fingerprint
    // (hardwareLooksTheSame above) against the user's OWN already-approved
    // machines (never another user's, and never a still-pending one) --
    // hostname alone is a weak signal (a generic default like
    // "DESKTOP-XXXXXXX" collides across genuinely different machines), so
    // hardware corroborates or, when the hostname was actually changed,
    // stands in for it. Takes the CANDIDATE's own worker id (rather than a
    // bare hostname) so it can pull hostname/platform/arch/hardware_json for
    // that row itself -- both call sites (GET /api/device/status, POST
    // /api/device/approve) already have this id at hand. A soft signal, not
    // a uniqueness constraint: used to ask before creating a duplicate (or to
    // offer merging into the match, see mergeEnrolment below), never to block
    // outright.
    findPossibleDuplicate(userId: string, candidateWorkerId: string): PossibleDuplicateWorker | undefined {
      const candidate = getDb()
        .prepare(`SELECT hostname, platform, arch, hardware_json FROM workers WHERE id = ?`)
        .get(candidateWorkerId) as
        | { hostname: string | null; platform: string | null; arch: string | null; hardware_json: string | null }
        | undefined;
      if (!candidate) return undefined;
      const candidateHardware = safeParseJson<HardwareInfo | null>(candidate.hardware_json, null);
      const rows = getDb()
        .prepare(
          `SELECT id, display_name, last_heartbeat_at, hostname, hardware_json FROM workers
           WHERE user_id = ? AND approved_at IS NOT NULL AND id != ?
           ORDER BY last_heartbeat_at DESC`
        )
        .all(userId, candidateWorkerId) as {
        id: string;
        display_name: string;
        last_heartbeat_at: number | null;
        hostname: string | null;
        hardware_json: string | null;
      }[];
      for (const row of rows) {
        const hostnameMatch = candidate.hostname != null && row.hostname === candidate.hostname;
        const hardwareMatch = hardwareLooksTheSame(
          candidateHardware,
          safeParseJson<HardwareInfo | null>(row.hardware_json, null),
          candidate.platform,
          candidate.arch
        );
        if (hostnameMatch || hardwareMatch) {
          return { id: row.id, displayName: row.display_name, lastHeartbeatAt: row.last_heartbeat_at, hostnameMatch, hardwareMatch };
        }
      }
      return undefined;
    },

    // Re-points an EXISTING approved worker row onto a freshly-enrolling
    // machine's identity, instead of approving the pending row as a second,
    // disconnected one -- the actual fix for findPossibleDuplicate's warning
    // above, called once a human confirms "yes, this is the same machine"
    // (routes/device.ts's POST /api/device/approve, `merge_into`).
    // Deliberately preserves the target's id/display_name/user_id/
    // approved_at/created_at (that continuity -- same worker id, same
    // history, same human-assigned name -- is the whole point) and instead
    // transplants the PENDING row's machine_id/hostname/platform/arch/
    // hardware_json plus its enrolment_code_hash/user_code/
    // enrolment_expires_at, so the connecting worker's next POST
    // /api/device/token poll (which looks itself up by that hash) finds the
    // target row -- already approved, so it succeeds immediately. The pending
    // row is deleted FIRST, in the same transaction: machine_id is UNIQUE NOT
    // NULL, so its value has to be freed before the target row can take it.
    mergeEnrolment(pendingWorkerId: string, targetWorkerId: string): Worker {
      const database = getDb();
      const tx = database.transaction(() => {
        const pending = database
          .prepare(
            `SELECT machine_id, hostname, platform, arch, hardware_json,
                    enrolment_code_hash, user_code, enrolment_expires_at
             FROM workers WHERE id = ?`
          )
          .get(pendingWorkerId) as {
          machine_id: string;
          hostname: string | null;
          platform: string | null;
          arch: string | null;
          hardware_json: string | null;
          enrolment_code_hash: string | null;
          user_code: string | null;
          enrolment_expires_at: number | null;
        };
        database.prepare(`DELETE FROM workers WHERE id = ?`).run(pendingWorkerId);
        database
          .prepare(
            `UPDATE workers SET
               machine_id = ?, hostname = ?, platform = ?, arch = ?, hardware_json = ?,
               enrolment_code_hash = ?, user_code = ?, enrolment_expires_at = ?, updated_at = ?
             WHERE id = ?`
          )
          .run(
            pending.machine_id,
            pending.hostname,
            pending.platform,
            pending.arch,
            pending.hardware_json,
            pending.enrolment_code_hash,
            pending.user_code,
            pending.enrolment_expires_at,
            Date.now(),
            targetWorkerId
          );
      });
      tx();
      return this.getWorker(targetWorkerId)!;
    },

    // The ONLY place workers.user_id is ever set (§3.1 step 5) -- the
    // approval click IS the identity binding, not a rubber stamp on a
    // decision made earlier (§3.1's own point). Deliberately leaves
    // enrolment_code_hash/user_code in place -- the worker's own
    // /api/device/token poll still needs to find this row by hash to redeem
    // its session; clearEnrolmentCode below is the actual one-shot
    // consumption point, called once that redemption succeeds, not here.
    approve(workerId: string, userId: string): void {
      getDb()
        .prepare(`UPDATE workers SET user_id = ?, approved_at = ?, updated_at = ? WHERE id = ?`)
        .run(userId, Date.now(), Date.now(), workerId);
    },

    // Called once POST /api/device/token successfully issues a session for
    // this worker -- the true single-use consumption point (see approve's
    // own comment for why it isn't done there instead). Clears BOTH codes:
    // enrolment_code_hash so the device_code can never be redeemed a second
    // time (replay protection), and user_code so it stops occupying the
    // partial-unique "no two pending enrolments collide" index and can't be
    // guessed against a now-meaningless target.
    clearEnrolmentCode(workerId: string): void {
      getDb()
        .prepare(
          `UPDATE workers SET enrolment_code_hash = NULL, user_code = NULL, enrolment_expires_at = NULL, updated_at = ? WHERE id = ?`
        )
        .run(Date.now(), workerId);
    },

    // §1.7's own maintenance-sweep spec: "expired-enrolment pruning". Clears
    // the code fields only -- deliberately NEVER deletes the row itself,
    // even though a genuinely-abandoned first-time enrolment has no history
    // to lose. A worker's own machine_id is meant to be stable, and
    // /api/device/start's getByMachineId reuse (§3.4) means a Stage 1
    // legacy machine UPGRADING to device-flow enrolment reuses its existing
    // (historied) row via reissueEnrolment -- if that particular attempt
    // then expires unapproved, deleting the row would destroy a real
    // machine's identity/history, not just tidy up a throwaway one. Clearing
    // just the code fields is safe either way: it frees the
    // idx_workers_user_code_pending unique-index slot and retires a
    // now-dead code (defense in depth -- /api/device/approve's own expiry
    // check already rejects it) without ever risking data loss.
    pruneExpiredEnrolments(): number {
      return getDb()
        .prepare(
          `UPDATE workers SET user_code = NULL, enrolment_code_hash = NULL, enrolment_expires_at = NULL
           WHERE approved_at IS NULL AND enrolment_expires_at IS NOT NULL AND enrolment_expires_at < ?`
        )
        .run(Date.now()).changes;
    },
  },

  // --- Multi-user Stage 1: persistent pull queue (MULTIUSER_PLAN.md §1.2/§1.4/§1.5/§1.7) ---
  queueRepo: {
    // Atomic under Node's single-threaded/synchronous-transaction model:
    // better-sqlite3's `transaction()` runs its callback with no `await`
    // inside, so nothing else (including another concurrent
    // POST /api/worker/queue call) can interleave between the SELECT and the
    // UPDATE below -- two concurrent callers for the same worker can never
    // both claim the same job (§1.17's exit criterion, tested in
    // queue.test.ts).
    //
    // Skips (never claims) a job that was cancelled while still pending --
    // e.g. the user stopped a 'scheduled' run before the worker ever picked
    // it up. attempts is deliberately NOT incremented here -- it only counts
    // lease-expiry requeues (see returnToPending), not claims.
    //
    // Flips the parent run to 'running' in the SAME transaction as the claim
    // (MULTIUSER_PLAN.md §1.13) so the two can never disagree.
    claimNextJob(workerId: string): QueueJob | undefined {
      const database = getDb();
      const tx = database.transaction((): QueueJob | undefined => {
        const job = database
          .prepare(
            `SELECT * FROM worker_jobs
             WHERE worker_id = ? AND status = 'pending' AND cancel_requested = 0
             ORDER BY created_at ASC, rowid ASC LIMIT 1`
          )
          .get(workerId) as WorkerJobRow | undefined;
        if (!job) return undefined;

        const now = Date.now();
        database
          .prepare(`UPDATE worker_jobs SET status = 'claimed', lease_expires_at = ?, claimed_at = ? WHERE id = ?`)
          .run(now + LEASE_MS, now, job.id);

        // Set immediately, in the same transaction as the claim -- NOT left
        // to wait for the worker's first heartbeat (which reports active_job
        // and would otherwise be the only thing that ever writes this).
        // Without this, Worker.status/activeTestId/activeJobProgress would
        // all read stale ("idle", no run) for up to one heartbeat interval
        // right after a claim.
        database.prepare(`UPDATE workers SET active_job_id = ? WHERE id = ?`).run(job.id, workerId);

        // A 'benchmark', 'run_probe' or 'measure_quality' claim means the run
        // itself started executing -- those are the three job types that
        // actually carry out a run's own work (N2/N4 ride their own job type
        // instead of 'benchmark', see routes/tests.ts's trigger route). An
        // 'install_build' job can share the same run_id (so the reaper can
        // reconcile the run if the install permanently fails, see
        // markJobFailed below) but claiming it must NOT flip the run to
        // 'running' early. Without run_probe/measure_quality here, a probe or
        // quality run sat at 'scheduled' for its ENTIRE execution (the Tests
        // page mislabeled it, TestDetail hid its elapsed-time/Stop controls)
        // and only ever moved once its terminal result landed.
        if (job.run_id && (job.job_type === "benchmark" || job.job_type === "run_probe" || job.job_type === "measure_quality")) {
          database
            .prepare(
              `UPDATE runs SET status = 'running', started_at = ?
               WHERE id = ? AND status NOT IN ('done','partial','failed','cancelled')`
            )
            .run(now, job.run_id);
        }

        return { job_id: job.id, type: job.job_type, payload: JSON.parse(job.payload_json) } as QueueJob;
      });
      return tx();
    },

    // Called on every heartbeat while a job is active (MULTIUSER_PLAN.md
    // §1.5): extends the lease and records live progress. Returns undefined
    // when the server no longer considers this job live for this worker
    // (wrong worker, already terminal, or -- rare -- claimed by a different
    // attempt after a reaped lease) so the caller tells the worker to stop
    // rather than let two executions of the same job both report results.
    extendLeaseAndGetFlags(
      jobId: string,
      workerId: string,
      report: ActiveJobReport
    ): { cancel_requested: boolean; discard_requested: boolean } | undefined {
      const database = getDb();
      const job = database.prepare(`SELECT * FROM worker_jobs WHERE id = ?`).get(jobId) as
        | WorkerJobRow
        | undefined;
      if (!job || job.worker_id !== workerId || job.status !== "claimed") return undefined;
      database
        .prepare(`UPDATE worker_jobs SET lease_expires_at = ?, progress_json = ? WHERE id = ?`)
        .run(Date.now() + LEASE_MS, JSON.stringify(report), jobId);
      return { cancel_requested: job.cancel_requested === 1, discard_requested: job.discard_requested === 1 };
    },

    enqueueJob(
      workerId: string,
      job: { type: string; payload: unknown; runId?: string }
    ): string {
      const id = uuid();
      getDb()
        .prepare(
          `INSERT INTO worker_jobs (id, worker_id, job_type, payload_json, status, run_id, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?, ?)`
        )
        .run(id, workerId, job.type, JSON.stringify(job.payload), job.runId ?? null, Date.now());
      return id;
    },

    // Every non-terminal (pending/claimed) job for a run -- a run can have up
    // to two (install_build + benchmark, both stamped with the same run_id),
    // so stop/pause and the reaper need to account for either one, not just
    // "the most recent." MULTIUSER_PLAN.md §1.14.
    getNonTerminalJobsForTest(runId: string): { id: string; workerId: string; status: string; jobType: string }[] {
      const rows = getDb()
        .prepare(
          `SELECT id, worker_id, status, job_type FROM worker_jobs
           WHERE run_id = ? AND status IN ('pending','claimed')`
        )
        .all(runId) as { id: string; worker_id: string; status: string; job_type: string }[];
      return rows.map((r) => ({ id: r.id, workerId: r.worker_id, status: r.status, jobType: r.job_type }));
    },

    // Marks a still-pending (never claimed) job cancelled outright -- the
    // worker never saw it, so there's nothing to signal. Returns false if the
    // job wasn't pending (caller should fall back to requestCancel instead).
    cancelPendingJob(jobId: string): boolean {
      const result = getDb()
        .prepare(`UPDATE worker_jobs SET status = 'cancelled', completed_at = ? WHERE id = ? AND status = 'pending'`)
        .run(Date.now(), jobId);
      return result.changes > 0;
    },

    // Cancels every still-pending job for a run outright in one statement --
    // used both by a user Stop that arrives before the worker ever claimed
    // anything, and by the reaper cleaning up a benchmark job that would
    // otherwise be orphaned after its sibling install_build permanently
    // fails (see index.ts's reapExpiredLeases).
    cancelPendingJobsForTest(runId: string): void {
      getDb()
        .prepare(`UPDATE worker_jobs SET status = 'cancelled', completed_at = ? WHERE run_id = ? AND status = 'pending'`)
        .run(Date.now(), runId);
    },

    // Flags an already-claimed job for cancellation -- delivered to the
    // worker on its next heartbeat (MULTIUSER_PLAN.md §1.5/§1.14), not
    // immediate. Also set on a still-pending job as a defensive no-op (in
    // case it's claimed a moment later, claimNextJob's WHERE clause above
    // will skip it rather than hand it out).
    requestCancel(jobId: string): void {
      getDb().prepare(`UPDATE worker_jobs SET cancel_requested = 1 WHERE id = ?`).run(jobId);
    },

    // Cancel Download -- same delivery as requestCancel above (the worker
    // sees this job's id in the next heartbeat's cancel_job_ids either way),
    // but also flags discard_requested so the worker deletes the .part file
    // instead of keeping it for resume (see routes/queue.ts's heartbeat
    // handler and worker/src/index.ts's discard handling). Only meaningful
    // for a download_model job -- the route calling this already checked
    // that (server/src/routes/workers.ts).
    requestDiscard(jobId: string): void {
      getDb().prepare(`UPDATE worker_jobs SET cancel_requested = 1, discard_requested = 1 WHERE id = ?`).run(jobId);
    },

    // Test-scoped version of requestCancel -- flags every currently-claimed
    // job for a run (in practice at most one: a run's install_build and
    // benchmark jobs are never claimed simultaneously, since the worker
    // executes its queue strictly one job at a time).
    requestCancelForTest(runId: string): void {
      getDb().prepare(`UPDATE worker_jobs SET cancel_requested = 1 WHERE run_id = ? AND status = 'claimed'`).run(runId);
    },

    listExpiredLeases(
      now: number
    ): { id: string; worker_id: string; run_id: string | null; attempts: number; cancel_requested: number }[] {
      return getDb()
        .prepare(
          `SELECT id, worker_id, run_id, attempts, cancel_requested FROM worker_jobs
           WHERE status = 'claimed' AND lease_expires_at < ?`
        )
        .all(now) as {
        id: string;
        worker_id: string;
        run_id: string | null;
        attempts: number;
        cancel_requested: number;
      }[];
    },

    // A job only a lease-expiry has already returned to 'pending' once
    // (attempts >= 1) is only ever reclaimable by its OWN worker_id -- jobs
    // aren't stealable across workers. If that worker never comes back
    // (a genuine crash, not a brief pm2/systemd restart blip), the job would
    // otherwise sit 'pending' forever: listExpiredLeases only ever looks at
    // 'claimed' rows, so a job that's already back in 'pending' never gets a
    // second look. Swept on the same reap cycle as listExpiredLeases, gated
    // on the worker's own last_heartbeat_at rather than a fixed delay so it
    // tracks the real liveness definition (see server/src/liveness.ts).
    // attempts >= 1 (not 0) deliberately excludes a job still awaiting its
    // very first claim -- that path is instead guarded at trigger time
    // (server/src/routes/runs.ts rejects triggering against an
    // already-offline worker), so a fresh job's worker going offline before
    // ever claiming anything is not this function's concern.
    listStuckPendingJobsForOfflineWorkers(
      offlineCutoff: number
    ): { id: string; worker_id: string; run_id: string | null; attempts: number; cancel_requested: number }[] {
      return getDb()
        .prepare(
          `SELECT wj.id, wj.worker_id, wj.run_id, wj.attempts, wj.cancel_requested
           FROM worker_jobs wj
           JOIN workers w ON w.id = wj.worker_id
           WHERE wj.status = 'pending' AND wj.attempts >= 1
             AND (w.last_heartbeat_at IS NULL OR w.last_heartbeat_at < ?)`
        )
        .all(offlineCutoff) as {
        id: string;
        worker_id: string;
        run_id: string | null;
        attempts: number;
        cancel_requested: number;
      }[];
    },

    // Looked up by the job-completion endpoint (routes/queue.ts) to verify
    // the reporting worker actually owns this job before trusting its
    // done/failed report -- one worker must never be able to complete or
    // fail another's job.
    getJob(
      jobId: string
    ): { id: string; workerId: string; runId: string | null; status: string; jobType: string } | undefined {
      const row = getDb()
        .prepare(`SELECT id, worker_id, run_id, status, job_type FROM worker_jobs WHERE id = ?`)
        .get(jobId) as
        | { id: string; worker_id: string; run_id: string | null; status: string; job_type: string }
        | undefined;
      if (!row) return undefined;
      return { id: row.id, workerId: row.worker_id, runId: row.run_id, status: row.status, jobType: row.job_type };
    },

    // Clears workers.active_job_id whenever it still points at THIS job --
    // the WHERE guard (rather than an unconditional clear by workerId) means
    // this can never clobber a different, newer active_job_id the same
    // worker has already moved on to by the time this runs.
    markJobFailed(jobId: string, error: string): void {
      const database = getDb();
      database
        .prepare(`UPDATE worker_jobs SET status = 'failed', error = ?, completed_at = ? WHERE id = ?`)
        .run(error, Date.now(), jobId);
      database.prepare(`UPDATE workers SET active_job_id = NULL WHERE active_job_id = ?`).run(jobId);
    },

    markJobCompleted(jobId: string): void {
      const database = getDb();
      database
        .prepare(`UPDATE worker_jobs SET status = 'completed', completed_at = ? WHERE id = ?`)
        .run(Date.now(), jobId);
      database.prepare(`UPDATE workers SET active_job_id = NULL WHERE active_job_id = ?`).run(jobId);
    },

    // Returns an expired-lease job to the queue for re-claim. attempts is
    // incremented HERE (and only here) -- it counts lease-expiry requeues,
    // not claims, so the reaper's MAX_ATTEMPTS check means "this job's lease
    // expired without completing N times," not "N total claims."
    returnToPending(jobId: string): void {
      const database = getDb();
      database
        .prepare(
          `UPDATE worker_jobs SET status = 'pending', attempts = attempts + 1, lease_expires_at = NULL, claimed_at = NULL
           WHERE id = ?`
        )
        .run(jobId);
      database.prepare(`UPDATE workers SET active_job_id = NULL WHERE active_job_id = ?`).run(jobId);
    },

    // §1.7's own maintenance-sweep spec: "terminal-job pruning
    // (pruneCompletedOlderThan, 7 days)". A terminal job row (completed/
    // failed/cancelled) has no further use once its outcome has had time to
    // be seen -- unlike a pending/claimed row, nothing reads it again after
    // that. Called from reaper.ts's runMaintenanceSweep, same interval as
    // the job-lease reaper.
    pruneCompletedOlderThan(days: number): number {
      const cutoff = Date.now() - days * 24 * 3600 * 1000;
      return getDb()
        .prepare(`DELETE FROM worker_jobs WHERE status IN ('completed', 'failed', 'cancelled') AND completed_at < ?`)
        .run(cutoff).changes;
    },
  },

  // --- Multi-user Stage 2: users/identities (MULTIUSER_PLAN.md §2.1/§2.4) ---
  userRepo: {
    getUser(userId: string): UserRecord | undefined {
      const row = getDb().prepare(`SELECT * FROM users WHERE id = ?`).get(userId) as UserRow | undefined;
      return row ? mapUser(row) : undefined;
    },

    getIdentities(userId: string): Identity[] {
      const rows = getDb()
        .prepare(`SELECT * FROM identities WHERE user_id = ? ORDER BY created_at ASC`)
        .all(userId) as IdentityRow[];
      return rows.map(mapIdentity);
    },

    // Looked up before either upsertByIdentity or linkIdentity act -- lets
    // the CALLER (the OAuth callback route) decide what an existing link
    // means in context (a normal login vs. "this identity is already
    // somebody else's account, reject the link" -- MULTIUSER_PLAN.md §2.4's
    // own warning about naive auto-merge) rather than baking that policy
    // into the data layer.
    findByIdentity(provider: string, providerUserId: string): UserRecord | undefined {
      const row = getDb()
        .prepare(
          `SELECT users.* FROM users
           JOIN identities ON identities.user_id = users.id
           WHERE identities.provider = ? AND identities.provider_user_id = ?`
        )
        .get(provider, providerUserId) as UserRow | undefined;
      return row ? mapUser(row) : undefined;
    },

    // Normal login/sign-up, one and the same (§2.1) -- a provider_user_id
    // never seen before creates a fresh users+identities row pair in one
    // transaction; a recognized one just refreshes provider_login (mutable,
    // e.g. a GitHub username change) and last_login_at.
    upsertByIdentity(
      provider: string,
      profile: { providerUserId: string; login: string; avatarUrl: string | null }
    ): UserRecord {
      const database = getDb();
      const now = Date.now();
      const existing = database
        .prepare(`SELECT user_id FROM identities WHERE provider = ? AND provider_user_id = ?`)
        .get(provider, profile.providerUserId) as { user_id: string } | undefined;

      if (existing) {
        database
          .prepare(`UPDATE identities SET provider_login = ? WHERE provider = ? AND provider_user_id = ?`)
          .run(profile.login, provider, profile.providerUserId);
        database.prepare(`UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?`).run(now, now, existing.user_id);
        return this.getUser(existing.user_id)!;
      }

      const userId = uuid();
      const tx = database.transaction(() => {
        database
          .prepare(
            `INSERT INTO users (id, display_name, avatar_url, created_at, updated_at, last_login_at)
             VALUES (?, ?, ?, ?, ?, ?)`
          )
          .run(userId, profile.login, profile.avatarUrl, now, now, now);
        database
          .prepare(
            `INSERT INTO identities (id, user_id, provider, provider_user_id, provider_login, created_at)
             VALUES (?, ?, ?, ?, ?, ?)`
          )
          .run(uuid(), userId, provider, profile.providerUserId, profile.login, now);
      });
      tx();
      return this.getUser(userId)!;
    },

    // Settings -> "Connect another account" (§2.4). Assumes the caller has
    // already used findByIdentity to confirm this identity isn't linked to a
    // DIFFERENT user -- attaches it to userId (or, if it's already linked to
    // userId itself, just refreshes provider_login).
    linkIdentity(
      userId: string,
      provider: string,
      profile: { providerUserId: string; login: string; avatarUrl: string | null }
    ): UserRecord {
      const database = getDb();
      const now = Date.now();
      const existing = database
        .prepare(`SELECT user_id FROM identities WHERE provider = ? AND provider_user_id = ?`)
        .get(provider, profile.providerUserId) as { user_id: string } | undefined;

      if (existing) {
        database
          .prepare(`UPDATE identities SET provider_login = ? WHERE provider = ? AND provider_user_id = ?`)
          .run(profile.login, provider, profile.providerUserId);
      } else {
        database
          .prepare(
            `INSERT INTO identities (id, user_id, provider, provider_user_id, provider_login, created_at)
             VALUES (?, ?, ?, ?, ?, ?)`
          )
          .run(uuid(), userId, provider, profile.providerUserId, profile.login, now);
      }
      return this.getUser(userId)!;
    },

    // Settings' own toggle (§5.4) -- the one thing on the users row a
    // caller can write about themselves. Default is 1 (opt-in by default,
    // disclosed in Settings' own copy), so this only ever needs to flip it.
    setShareBenchmarks(userId: string, value: boolean): UserRecord {
      getDb().prepare(`UPDATE users SET share_benchmarks = ?, updated_at = ? WHERE id = ?`).run(value ? 1 : 0, Date.now(), userId);
      return this.getUser(userId)!;
    },

    // Security checklist's own "Account deletion ships in v1" item. A plain
    // `DELETE FROM users` would fail outright once PRAGMA foreign_keys=ON
    // meets `models.created_by` -- that column deliberately has NO ON DELETE
    // clause (see migrate.ts's own comment: a deleted user's created_by
    // attribution on a model nobody else has touched should block deletion
    // rather than silently orphan/cascade a shared catalog row). This makes
    // that explicit: reassign the caller's own models to ownerless (global,
    // same as any model with no known creator) FIRST, in the same
    // transaction, then delete the user -- which cascades sessions,
    // identities, workers, runs, results, and run_items automatically (all
    // declared ON DELETE CASCADE, §4.1). Community aggregates need no
    // separate cleanup: they're computed live from runs/results, which are
    // already gone by the time any aggregate query runs again.
    deleteAccount(userId: string): void {
      const database = getDb();
      const tx = database.transaction(() => {
        database.prepare(`UPDATE models SET created_by = NULL WHERE created_by = ?`).run(userId);
        database.prepare(`DELETE FROM users WHERE id = ?`).run(userId);
      });
      tx();
    },
  },

  // --- Multi-user Stage 2: sessions (MULTIUSER_PLAN.md §2.2/§2.5) ---
  sessionRepo: {
    // Returns the token/refresh ONCE -- the only place either value ever
    // exists outside the client; only their hashes are persisted.
    create(
      userId: string,
      opts: { isWorker?: boolean; workerId?: string; label?: string } = {}
    ): { token: string; refresh: string | null; expiresAt: number } {
      const token = generateSessionId();
      const refresh = opts.isWorker ? generateRefreshToken() : null;
      const now = Date.now();
      const expiresAt = now + (opts.isWorker ? 90 : 30) * 24 * 3600 * 1000;
      getDb()
        .prepare(
          `INSERT INTO sessions
             (id, user_id, token_hash, refresh_hash, expires_at, is_worker, worker_id, label, last_seen_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          uuid(),
          userId,
          hashToken(token),
          refresh ? hashToken(refresh) : null,
          expiresAt,
          opts.isWorker ? 1 : 0,
          opts.workerId ?? null,
          opts.label ?? null,
          now,
          now
        );
      return { token, refresh, expiresAt };
    },

    getByTokenHash(hash: string): SessionRecord | undefined {
      const row = getDb().prepare(`SELECT * FROM sessions WHERE token_hash = ?`).get(hash) as SessionRow | undefined;
      return row ? mapSession(row) : undefined;
    },

    getByRefreshHash(hash: string): SessionRecord | undefined {
      const row = getDb().prepare(`SELECT * FROM sessions WHERE refresh_hash = ?`).get(hash) as SessionRow | undefined;
      return row ? mapSession(row) : undefined;
    },

    // Replay detection (§2.5): a presented refresh token matching a PREVIOUS
    // (already-rotated-away) one means it was captured and reused.
    getByPrevRefreshHash(hash: string): SessionRecord | undefined {
      const row = getDb().prepare(`SELECT * FROM sessions WHERE prev_refresh_hash = ?`).get(hash) as
        | SessionRow
        | undefined;
      return row ? mapSession(row) : undefined;
    },

    // sessions.id is STABLE across rotation -- callers that stored it
    // earlier (e.g. workers.session_id, Stage 3) are never orphaned by a
    // later refresh, unlike rotating the primary key itself would.
    rotate(sessionId: string): { token: string; refresh: string; expiresAt: number } {
      const database = getDb();
      const current = database.prepare(`SELECT refresh_hash FROM sessions WHERE id = ?`).get(sessionId) as
        | { refresh_hash: string | null }
        | undefined;
      const token = generateSessionId();
      const refresh = generateRefreshToken();
      const now = Date.now();
      const expiresAt = now + 90 * 24 * 3600 * 1000;
      database
        .prepare(
          `UPDATE sessions SET token_hash = ?, refresh_hash = ?, prev_refresh_hash = ?, expires_at = ? WHERE id = ?`
        )
        .run(hashToken(token), hashToken(refresh), current?.refresh_hash ?? null, expiresAt, sessionId);
      return { token, refresh, expiresAt };
    },

    revokeById(id: string): void {
      getDb().prepare(`DELETE FROM sessions WHERE id = ?`).run(id);
    },

    // Settings -> "Sign out everywhere else" (§2.5) -- keeps the caller's own
    // current session alive.
    revokeAllExcept(userId: string, exceptSessionId: string): void {
      getDb().prepare(`DELETE FROM sessions WHERE user_id = ? AND id != ?`).run(userId, exceptSessionId);
    },

    // Multi-user Stage 3 (MULTIUSER_PLAN.md §3.4): called on re-enrolment
    // (POST /api/device/start against an already-known machine_id) so a
    // fresh enrolment never leaves an old worker session live alongside the
    // new one -- closes the exact hole v3.3 had (its own review finding D6).
    revokeWorkerSessions(workerId: string): void {
      getDb().prepare(`DELETE FROM sessions WHERE worker_id = ? AND is_worker = 1`).run(workerId);
    },

    listForUser(userId: string): SessionRecord[] {
      const rows = getDb()
        .prepare(`SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC`)
        .all(userId) as SessionRow[];
      return rows.map(mapSession);
    },

    // Reusable session (§2.8): returns the most recent non-expired,
    // non-worker session for a user if one exists, so the OAuth callback can
    // hand back the SAME cookie on every login instead of minting a new
    // session row per page refresh. Workers are excluded -- a worker session
    // is a different identity space (it authenticates via its own refresh
    // token, never via this cookie) and must never be reused as a browser
    // session. Returns undefined when there's nothing to reuse (first login
    // ever, every session expired, all sessions are worker sessions).
    getActiveBrowserSession(userId: string): SessionRecord | undefined {
      const row = getDb()
        .prepare(
          `SELECT * FROM sessions
            WHERE user_id = ? AND is_worker = 0 AND expires_at > ?
            ORDER BY last_seen_at DESC, created_at DESC
            LIMIT 1`
        )
        .get(userId, Date.now()) as SessionRow | undefined;
      return row ? mapSession(row) : undefined;
    },

    // Issues a fresh access token for an existing session row, keeping the
    // same id/created_at/refresh_hash (§2.8). The raw token is only ever
    // returned here and at create() time -- the DB stores only its hash, so
    // this is the only way to get a usable token for a row that already
    // exists. Used by the OAuth callback to reuse a session across logins
    // instead of minting a new row each time.
    regenerateToken(sessionId: string): { token: string; expiresAt: number } {
      const token = generateSessionId();
      const now = Date.now();
      const expiresAt = now + 30 * 24 * 3600 * 1000;
      getDb()
        .prepare(
          `UPDATE sessions SET token_hash = ?, expires_at = ?, last_seen_at = ? WHERE id = ?`
        )
        .run(hashToken(token), expiresAt, now, sessionId);
      return { token, expiresAt };
    },

    // Sliding expiry (§2.5): extends expires_at to now + the session's own
    // lifetime, throttled to at most one write per hour per session so an
    // active user's every request doesn't turn into a write.
    touch(session: SessionRecord): void {
      const now = Date.now();
      if (session.lastSeenAt && now - session.lastSeenAt < 3600_000) return;
      const expiresAt = now + (session.isWorker ? 90 : 30) * 24 * 3600 * 1000;
      getDb().prepare(`UPDATE sessions SET last_seen_at = ?, expires_at = ? WHERE id = ?`).run(now, expiresAt, session.id);
    },

    // §1.7's own maintenance-sweep spec: "expired-session pruning -- which
    // v3.3 never had, so sessions would grow forever despite having an
    // expires_at index." Called from reaper.ts's runMaintenanceSweep, same
    // interval as the job-lease reaper. Returns the deleted count purely for
    // logging -- callers don't otherwise need it.
    pruneExpired(): number {
      return getDb().prepare(`DELETE FROM sessions WHERE expires_at < ?`).run(Date.now()).changes;
    },
  },

  // --- Multi-user Stage 2: /api/ai/chat budget (MULTIUSER_PLAN.md §2.6) ---
  aiUsageRepo: {
    // Upsert-and-increment the (user, day, hour) bucket -- called once per
    // request that actually reaches the provider (routes/ai.ts checks the
    // budget BEFORE calling this, so a request rejected for being over
    // budget is never counted against itself).
    recordUsage(userId: string, day: string, hour: number): void {
      getDb()
        .prepare(
          `INSERT INTO ai_usage (user_id, day, hour, count) VALUES (?, ?, ?, 1)
           ON CONFLICT (user_id, day, hour) DO UPDATE SET count = count + 1`
        )
        .run(userId, day, hour);
    },

    getHourCount(userId: string, day: string, hour: number): number {
      const row = getDb().prepare(`SELECT count FROM ai_usage WHERE user_id = ? AND day = ? AND hour = ?`).get(
        userId,
        day,
        hour
      ) as { count: number } | undefined;
      return row?.count ?? 0;
    },

    getUserDayCount(userId: string, day: string): number {
      const row = getDb().prepare(`SELECT SUM(count) as total FROM ai_usage WHERE user_id = ? AND day = ?`).get(
        userId,
        day
      ) as { total: number | null } | undefined;
      return row?.total ?? 0;
    },

    // No user_id filter -- the global circuit breaker (AI_LIMITS.globalPerDay)
    // protects the operator's provider budget regardless of who's calling.
    getGlobalDayCount(day: string): number {
      const row = getDb().prepare(`SELECT SUM(count) as total FROM ai_usage WHERE day = ?`).get(day) as
        | { total: number | null }
        | undefined;
      return row?.total ?? 0;
    },
  },

  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.2): everything created before
  // AUTH_ENABLED existed (or while it was off) has user_id/user_id-on-workers
  // NULL -- without this, the operator's entire pre-auth benchmark history
  // becomes invisible in the normal UI the moment they log in as a NEW user,
  // reachable only through a future admin view. One-shot, guarded by the meta
  // table (not a boot-time migration -- the claiming user's row has to exist
  // first, see routes/auth.ts's OAuth callback, which calls this right after
  // a superadmin-listed identity's login). Returns whether this call actually
  // performed the claim (false on every call after the first, or if some
  // other login already claimed it first) so the caller can log accordingly.
  claimLegacyHistory(userId: string): boolean {
    const database = getDb();
    let claimed = false;
    const tx = database.transaction(() => {
      const already = database.prepare(`SELECT 1 FROM meta WHERE key = 'legacy_history_claimed'`).get();
      if (already) return;
      database.prepare(`UPDATE runs SET user_id = ? WHERE user_id IS NULL`).run(userId);
      database.prepare(`UPDATE results SET user_id = ? WHERE user_id IS NULL`).run(userId);
      database.prepare(`UPDATE run_items SET user_id = ? WHERE user_id IS NULL`).run(userId);
      database.prepare(`UPDATE workers SET user_id = ? WHERE user_id IS NULL`).run(userId);
      database.prepare(`INSERT INTO meta (key, value) VALUES ('legacy_history_claimed', ?)`).run(userId);
      claimed = true;
    });
    tx();
    return claimed;
  },

  // --- Multi-user Stage 5: admin surface (MULTIUSER_PLAN.md §5.1) ---
  // Every function here is deliberately unscoped -- cross-tenant read is the
  // whole point, gated at the ROUTE layer (routes/admin.ts's hostname +
  // isSuperadmin hook), not here. Read-only in v1: no promote/demote/ban/edit.
  adminRepo: {
    // Each card's definition lives in shared/types.ts's AdminStats doc
    // comment -- this query set mirrors it exactly. Deliberately NOT one big
    // COUNT(*)-per-table pass: "tested" and "performed" are filtered/distinct
    // subsets of the tables, not raw row counts.
    stats(): AdminStats {
      const db = getDb();
      const scalar = (sql: string): number => (db.prepare(sql).get() as { n: number }).n;

      const users = scalar(`SELECT COUNT(*) AS n FROM users`);
      // "Ever connected" per the AdminStats doc comment -- a worker row
      // created by enrolment but never heartbeated (abandoned before
      // approval, or approved but never actually started) must not inflate
      // this, same as client/src/pages/Dashboard.tsx's own
      // machinesEverConnected filter.
      const machines = scalar(
        `SELECT COUNT(DISTINCT machine_id) AS n FROM workers WHERE last_heartbeat_at IS NOT NULL`
      );

      // Individual physical executions (items x repeats), not run/job rows --
      // this app's own "runs" vocabulary, see Dashboard.tsx's own totalRuns
      // and its header comment ("'Runs' ... means repeats (-r), not sweep
      // combinations or job rows"). A raw COUNT(*) FROM runs would count Test
      // entities instead, which is a different number entirely once anyone
      // uses sweep repeats > 1.
      const runRows = db
        .prepare(
          `SELECT runs.config AS config,
                  (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total
           FROM runs`
        )
        .all() as { config: string; items_total: number }[];
      let runs = 0;
      for (const row of runRows) {
        let repeats = 1;
        try {
          const cfg = JSON.parse(row.config) as TestConfig;
          if (typeof cfg.sweep?.repeats === "number") repeats = cfg.sweep.repeats;
        } catch {
          // corrupt config blob -- fall back to repeats=1, same as a run with
          // no sweep config at all
        }
        runs += row.items_total * repeats;
      }

      // Models tested + quants tested share the same tested-model universe:
      // distinct models referenced by at least one run (GROUP BY m.id collapses
      // the join's per-run duplicates before quant parsing).
      const modelsTested = scalar(`SELECT COUNT(DISTINCT model_id) AS n FROM runs`);
      const testedModelRows = db
        .prepare(
          `SELECT m.filename AS filename, m.hf_file AS hf_file, m.metadata AS metadata
           FROM runs r JOIN models m ON m.id = r.model_id
           GROUP BY m.id`
        )
        .all() as { filename: string | null; hf_file: string | null; metadata: string }[];
      const quants = new Set<string>();
      for (const row of testedModelRows) {
        let meta: ModelMetadata | undefined;
        try {
          meta = JSON.parse(row.metadata || "{}") as ModelMetadata;
        } catch {
          // corrupt metadata blob -- fall through to the filename parse below
        }
        const quant =
          typeof meta?.quant === "string" && meta.quant.trim()
            ? meta.quant.trim().toUpperCase()
            : parseQuant(row.filename ?? row.hf_file ?? "");
        if (quant) quants.add(quant);
      }

      // Terminal outcomes only -- the same finished set countUnfinishedItems
      // defines ('done','failed','failed_oom','failed_unsupported','cancelled');
      // still-'queued' or in-flight items haven't been performed yet.
      const tests = scalar(
        `SELECT COUNT(*) AS n FROM run_items WHERE status IN ('done','failed','failed_oom','failed_unsupported','cancelled')`
      );

      return { users, machines, modelsTested, quants: quants.size, tests, runs };
    },

    // Filterable global runs table (§5.1's own "the filterable global runs
    // table" spec) -- User/Machine/Backend/Status filters, each optional.
    // Capped at 500 rows: this is an operator glancing at recent activity,
    // not a full-history export (that's GET /api/admin/results/export,
    // which has no such cap).
    listTests(filters: AdminTestFilters): AdminTestSummary[] {
      const rows = getDb()
        .prepare(
          `SELECT runs.*, u.display_name AS user_display_name, w.display_name AS worker_display_name,
                  m.filename AS model_filename,
                  (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total,
                  (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'done') AS items_done,
                  (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status IN ('failed','failed_oom')) AS items_failed
           FROM runs
           LEFT JOIN users u ON u.id = runs.user_id
           LEFT JOIN workers w ON w.id = runs.worker_id
           LEFT JOIN models m ON m.id = runs.model_id
           WHERE (? IS NULL OR runs.user_id = ?)
             AND (? IS NULL OR runs.worker_id = ?)
             AND (? IS NULL OR runs.llama_cpp_backend = ?)
             AND (? IS NULL OR runs.status = ?)
           ORDER BY runs.started_at DESC
           LIMIT 500`
        )
        .all(
          filters.userId ?? null,
          filters.userId ?? null,
          filters.workerId ?? null,
          filters.workerId ?? null,
          filters.backend ?? null,
          filters.backend ?? null,
          filters.status ?? null,
          filters.status ?? null
        ) as (TestRow & { user_display_name: string | null; worker_display_name: string | null })[];
      return rows.map((row) => ({
        id: row.id,
        userId: row.user_id ?? null,
        userDisplayName: row.user_display_name,
        workerId: row.worker_id,
        workerDisplayName: row.worker_display_name,
        modelId: row.model_id,
        modelFilename: row.model_filename ?? null,
        llamaCppBackend: row.llama_cpp_backend,
        status: row.status,
        error: row.error,
        startedAt: row.started_at,
        completedAt: row.completed_at,
        itemsTotal: row.items_total ?? 0,
        itemsDone: row.items_done ?? 0,
        itemsFailed: row.items_failed ?? 0,
      }));
    },

    // Global user list -- id/display name/superadmin-relevant identity info
    // is intentionally NOT included (isSuperadmin is env-derived, not a DB
    // fact about a user row, see superadmin.ts); just enough to identify an
    // account in the admin UI.
    listUsers(): AdminUserSummary[] {
      const rows = getDb()
        .prepare(`SELECT id, display_name, avatar_url, created_at, last_login_at FROM users ORDER BY created_at ASC`)
        .all() as { id: string; display_name: string | null; avatar_url: string | null; created_at: number; last_login_at: number | null }[];
      return rows.map((r) => ({
        id: r.id,
        displayName: r.display_name,
        avatarUrl: r.avatar_url,
        createdAt: r.created_at,
        lastLoginAt: r.last_login_at,
      }));
    },
  },

  // Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- the AI assistant's one
  // cross-tenant read (§5.3). Consent is still enforced HERE, in SQL
  // (u.share_benchmarks = 1 / the consentWhere clause below) -- only opted-in
  // users' rows are ever included, and the caller's own contribution is
  // always excluded from their own results.
  //
  // BENCHMARKING_PLAN_V8.md §0.9 specified a hard k-anonymity floor of five
  // distinct contributors per aggregate. That floor is now operator-tunable
  // (COMMUNITY_MIN_CONTRIBUTORS, see communityMinContributors above) and
  // defaults to 1 -- an aggregate may describe a single opted-in machine.
  //
  // Why the default moved: the group key is (model, backend, test_type,
  // platform, gpu_model). Requiring five DISTINCT users on that exact
  // combination suppressed effectively every row until the database is very
  // large, which made the shared results base -- the entire point of the
  // hosted instance -- answer nothing. What actually protects a contributor
  // sits upstream of the floor and is still unconditional: explicit consent
  // (users.share_benchmarks), a projection carrying no identity, account id
  // or hostname, caller self-exclusion, and `build` deliberately kept out of
  // the group key. Every row also carries contributor_count, so a consumer
  // can say "one machine reported this" instead of implying a consensus.
  //
  // Set COMMUNITY_MIN_CONTRIBUTORS=5 to restore §0.9's original floor.
  communityRepo: {
    // Matches the plan's own §5.4 SQL with one addition: ri.test_type is
    // added to the SELECT/GROUP BY on top of the plan's literal query. pp and
    // tg throughput are different units entirely (prompt processing vs.
    // generation) -- averaging them together, as the plan's SQL does, would
    // produce an avg_tps with no meaning. `build` stays OUT of the group key
    // exactly as the plan specifies (§5.4 point 4): it's high-cardinality and
    // near-unique per user at any given moment, which would re-identify
    // through the k-threshold same as a narrow filter would.
    listAggregates(callerId: string, filters: CommunityAggregateFilters = {}): CommunityAggregateRow[] {
      const rows = getDb()
        .prepare(
          `SELECT
             r.model_id AS model_id, m.filename AS model_filename,
             r.llama_cpp_backend AS backend,
             ri.test_type AS test_type,
             json_extract(w.hardware_json, '$.platform')     AS platform,
             json_extract(w.hardware_json, '$.gpu[0].model') AS gpu_model,
             COUNT(DISTINCT r.user_id)     AS contributor_count,
             COUNT(DISTINCT r.id)          AS run_count,
             ROUND(AVG(ri.avg_tps), 1)     AS avg_tps,
             ROUND(AVG(ri.ram_peak_mib))   AS avg_ram_peak_mib,
             ROUND(AVG(ri.vram_peak_mib))  AS avg_vram_peak_mib
           FROM runs r
           JOIN results ri ON ri.run_id = r.id
           LEFT JOIN models m  ON m.id = r.model_id
           LEFT JOIN workers w ON w.id = r.worker_id
           JOIN users u ON u.id = r.user_id AND u.share_benchmarks = 1
           WHERE r.user_id IS NOT NULL AND r.user_id <> ?
             AND (? IS NULL OR r.model_id = ?)
             AND (? IS NULL OR r.llama_cpp_backend = ?)
             AND (? IS NULL OR json_extract(w.hardware_json, '$.platform') = ?)
             AND (? IS NULL OR json_extract(w.hardware_json, '$.gpu[0].model') = ?)
           GROUP BY r.model_id, r.llama_cpp_backend, ri.test_type, platform, gpu_model
           HAVING COUNT(DISTINCT r.user_id) >= ${communityMinContributors()}
           ORDER BY avg_tps DESC`
        )
        .all(
          callerId,
          filters.modelId ?? null,
          filters.modelId ?? null,
          filters.backend ?? null,
          filters.backend ?? null,
          filters.platform ?? null,
          filters.platform ?? null,
          filters.gpuModel ?? null,
          filters.gpuModel ?? null
        ) as {
          model_id: string;
          model_filename: string | null;
          backend: string;
          test_type: string;
          platform: string | null;
          gpu_model: string | null;
          contributor_count: number;
          run_count: number;
          avg_tps: number;
          avg_ram_peak_mib: number | null;
          avg_vram_peak_mib: number | null;
        }[];
      return rows.map((row) => ({
        modelId: row.model_id,
        modelFilename: row.model_filename,
        backend: row.backend,
        testType: row.test_type,
        platform: row.platform,
        gpuModel: row.gpu_model,
        contributorCount: row.contributor_count,
        runCount: row.run_count,
        avgTps: row.avg_tps,
        avgRamPeakMib: row.avg_ram_peak_mib,
        avgVramPeakMib: row.avg_vram_peak_mib,
      }));
    },

    // Each dimension is counted on its own here (not listAggregates' full
    // group key), so a value appears the moment communityMinContributors()
    // distinct opted-in users have shared it -- regardless of which
    // model/backend they happened to run it with. Same floor as
    // listAggregates, enforced the same way: HAVING, not a post-filter, so a
    // value below the floor never reaches the caller at all.
    listFacets(callerId: string): CommunityFacets {
      const db = getDb();
      const consentWhere = `r.user_id IS NOT NULL AND r.user_id <> ? AND u.share_benchmarks = 1`;

      const models = db
        .prepare(
          `SELECT r.model_id AS model_id, m.filename AS filename, COUNT(DISTINCT r.user_id) AS contributor_count
           FROM runs r JOIN users u ON u.id = r.user_id LEFT JOIN models m ON m.id = r.model_id
           WHERE ${consentWhere}
           GROUP BY r.model_id
           HAVING COUNT(DISTINCT r.user_id) >= ${communityMinContributors()}`
        )
        .all(callerId) as { model_id: string; filename: string | null; contributor_count: number }[];

      const backends = db
        .prepare(
          `SELECT r.llama_cpp_backend AS value, COUNT(DISTINCT r.user_id) AS contributor_count
           FROM runs r JOIN users u ON u.id = r.user_id
           WHERE ${consentWhere}
           GROUP BY r.llama_cpp_backend
           HAVING COUNT(DISTINCT r.user_id) >= ${communityMinContributors()}`
        )
        .all(callerId) as { value: string; contributor_count: number }[];

      const platforms = db
        .prepare(
          `SELECT json_extract(w.hardware_json, '$.platform') AS value, COUNT(DISTINCT r.user_id) AS contributor_count
           FROM runs r JOIN users u ON u.id = r.user_id LEFT JOIN workers w ON w.id = r.worker_id
           WHERE ${consentWhere} AND json_extract(w.hardware_json, '$.platform') IS NOT NULL
           GROUP BY value
           HAVING COUNT(DISTINCT r.user_id) >= ${communityMinContributors()}`
        )
        .all(callerId) as { value: string; contributor_count: number }[];

      const gpuModels = db
        .prepare(
          `SELECT json_extract(w.hardware_json, '$.gpu[0].model') AS value, COUNT(DISTINCT r.user_id) AS contributor_count
           FROM runs r JOIN users u ON u.id = r.user_id LEFT JOIN workers w ON w.id = r.worker_id
           WHERE ${consentWhere} AND json_extract(w.hardware_json, '$.gpu[0].model') IS NOT NULL
           GROUP BY value
           HAVING COUNT(DISTINCT r.user_id) >= ${communityMinContributors()}`
        )
        .all(callerId) as { value: string; contributor_count: number }[];

      return {
        models: models.map((m) => ({ modelId: m.model_id, value: m.filename ?? m.model_id, contributorCount: m.contributor_count })),
        backends: backends.map((b) => ({ value: b.value, contributorCount: b.contributor_count })),
        platforms: platforms.map((p) => ({ value: p.value, contributorCount: p.contributor_count })),
        gpuModels: gpuModels.map((g) => ({ value: g.value, contributorCount: g.contributor_count })),
      };
    },
  },

  // Operator-controlled feature gates -- see shared/types.ts's AppSettings
  // doc comment for what each field does. Backed by the `meta` key-value
  // table (schema.sql) rather than a dedicated table: a handful of
  // single-row/small-key-count settings, and `meta` already exists for
  // exactly this kind of thing.
  //
  // 24 kept as a literal here (not imported from shared/probeLadder.ts's own
  // PROBE_MAX_LOADS) so this documented meta-table default never silently
  // drifts if that module's own default is ever tuned for a reason unrelated
  // to "what should the admin-configurable cap start at" -- the two are
  // meant to agree, but this is the value that actually ships as the
  // default, deliberately independent of the ladder module's internals.
  appSettingsRepo: {
    get(): AppSettings {
      const db = getDb();
      const read = (key: string): string | undefined =>
        (db.prepare(`SELECT value FROM meta WHERE key = ?`).get(key) as { value: string } | undefined)?.value;
      const rawPolicy = read("worker_vram_discrepancy_policy");
      const rawProbeMaxLoads = read("probe_max_loads");
      const parsedProbeMaxLoads = rawProbeMaxLoads === undefined ? NaN : Number(rawProbeMaxLoads);
      return {
        communitySharingAllowed: read("community_sharing_allowed") === "1",
        communityUserChoiceAllowed: read("community_user_choice_allowed") !== "0",
        accountDeletionAllowed: read("account_deletion_allowed") !== "0",
        workerVramDiscrepancyPolicy: isVramDiscrepancyPolicy(rawPolicy)
          ? rawPolicy
          : "warn", // documented default -- the shipped v1 record-with-warning behavior
        probeMaxLoads: isValidProbeMaxLoads(parsedProbeMaxLoads)
          ? parsedProbeMaxLoads
          : DEFAULT_PROBE_MAX_LOADS, // documented default -- see that const's own comment
      };
    },

    // Single-field convenience for callers that only need the live cap (e.g.
    // measurements.ts's per-request validateProbeAttempts) without paying for
    // the other meta-table reads get() does on every call.
    getProbeMaxLoads(): number {
      return repo.appSettingsRepo.get().probeMaxLoads;
    },

    setCommunitySharingAllowed(allowed: boolean): AppSettings {
      getDb()
        .prepare(
          `INSERT INTO meta (key, value) VALUES ('community_sharing_allowed', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value`
        )
        .run(allowed ? "1" : "0");
      return repo.appSettingsRepo.get();
    },

    setCommunityUserChoiceAllowed(allowed: boolean): AppSettings {
      getDb()
        .prepare(
          `INSERT INTO meta (key, value) VALUES ('community_user_choice_allowed', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value`
        )
        .run(allowed ? "1" : "0");
      return repo.appSettingsRepo.get();
    },

    setAccountDeletionAllowed(allowed: boolean): AppSettings {
      getDb()
        .prepare(
          `INSERT INTO meta (key, value) VALUES ('account_deletion_allowed', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value`
        )
        .run(allowed ? "1" : "0");
      return repo.appSettingsRepo.get();
    },

    setWorkerVramDiscrepancyPolicy(policy: VramDiscrepancyPolicy): AppSettings {
      getDb()
        .prepare(
          `INSERT INTO meta (key, value) VALUES ('worker_vram_discrepancy_policy', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value`
        )
        .run(policy);
      return repo.appSettingsRepo.get();
    },

    // Callers validate with isValidProbeMaxLoads before reaching here (same
    // division of labor as the other setters above) -- this throws rather
    // than silently clamping so a bug upstream surfaces immediately instead
    // of quietly storing a wrong-but-in-range number.
    setProbeMaxLoads(value: number): AppSettings {
      if (!isValidProbeMaxLoads(value)) {
        throw new Error(`probeMaxLoads must be an integer in [1, 200], got ${value}`);
      }
      getDb()
        .prepare(
          `INSERT INTO meta (key, value) VALUES ('probe_max_loads', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value`
        )
        .run(String(value));
      return repo.appSettingsRepo.get();
    },
  },

  // N2 -- largest-usable-config verification. Verification is per (machine,
  // build, KV pair, placement): changing any of them invalidates it, which is
  // exactly the UNIQUE key a re-probe replaces through.
  limitsRepo: {
    upsert(input: VerifiedLimitInput): VerifiedLimit {
      const now = Date.now();
      const kvType = `${input.cache_type_k}/${input.cache_type_v}`;
      const database = getDb();
      // One idempotent transaction: a re-report replaces rather than
      // accumulating, so a worker retry can never leave two ceilings.
      const tx = database.transaction(() => {
        database
          .prepare(
            `INSERT INTO model_machine_limits
               (id, worker_id, model_id, llama_cpp_build, kv_type, placement_hash,
                verified_ctx_tokens, verified_ngl, margin_observed_frac, method_version, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(worker_id, model_id, llama_cpp_build, kv_type, placement_hash)
             DO UPDATE SET verified_ctx_tokens = excluded.verified_ctx_tokens,
                           verified_ngl = excluded.verified_ngl,
                           margin_observed_frac = excluded.margin_observed_frac,
                           method_version = excluded.method_version,
                           created_at = excluded.created_at`
          )
          .run(
            uuid(),
            input.worker_id,
            input.model_id,
            input.llama_cpp_build,
            kvType,
            input.placement_hash,
            input.verified_ctx_tokens,
            input.verified_ngl ?? null,
            input.margin_observed_frac ?? null,
            input.method_version ?? null,
            now
          );
      });
      tx();
      return database
        .prepare(
          `SELECT * FROM model_machine_limits
           WHERE worker_id = ? AND model_id = ? AND llama_cpp_build = ? AND kv_type = ? AND placement_hash = ?`
        )
        .get(input.worker_id, input.model_id, input.llama_cpp_build, kvType, input.placement_hash) as VerifiedLimit;
    },

    listForModelAndWorker(modelId: string, workerId: string): VerifiedLimit[] {
      return getDb()
        .prepare(
          `SELECT * FROM model_machine_limits WHERE model_id = ? AND worker_id = ? ORDER BY created_at DESC`
        )
        .all(modelId, workerId) as VerifiedLimit[];
    },

    listForModel(modelId: string): VerifiedLimit[] {
      return getDb()
        .prepare(`SELECT * FROM model_machine_limits WHERE model_id = ? ORDER BY created_at DESC`)
        .all(modelId) as VerifiedLimit[];
    },

    getById(id: string): VerifiedLimit | undefined {
      return getDb().prepare(`SELECT * FROM model_machine_limits WHERE id = ?`).get(id) as VerifiedLimit | undefined;
    },

    // Forgets a stale ceiling so ProfileCards can offer "Verify with a probe"
    // again -- the row otherwise lives forever (a re-probe would overwrite it
    // through the UNIQUE key, but nothing forces a re-probe without this).
    deleteById(id: string): boolean {
      return getDb().prepare(`DELETE FROM model_machine_limits WHERE id = ?`).run(id).changes > 0;
    },
  },

  // N2 -- the ladder behind a verified ceiling. Replace-whole-run rather than
  // per-row upsert: a worker retry re-runs the WHOLE ladder, so its second
  // report is the truth and the first one's rungs never happened.
  probeAttemptsRepo: {
    replaceForTest(input: ProbeAttemptsInput): number {
      const database = getDb();
      const now = Date.now();
      const tx = database.transaction(() => {
        database.prepare(`DELETE FROM probe_attempts WHERE run_id = ?`).run(input.run_id);
        const insert = database.prepare(
          `INSERT INTO probe_attempts
             (id, run_id, worker_id, model_id, seq, candidate_ctx, ngl,
              ok, oom, spill, vram_needed_mib, vram_free_mib, vram_peak_mib,
              ram_needed_mib, ram_free_mib, ram_peak_mib, vram_process_peak_mib, ram_total_peak_mib,
              vram_shared_peak_mib, gen_tps, error, created_at,
              reused_from_run_id, vram_discrepancy, gpu_layers_resident_est, gpu_layers_resident_exact)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        );
        input.attempts.forEach((a, seq) => {
          insert.run(
            uuid(),
            input.run_id,
            input.worker_id,
            input.model_id,
            seq,
            a.candidate_ctx,
            a.ngl ?? null,
            a.ok ? 1 : 0,
            a.oom ? 1 : 0,
            a.spill ? 1 : 0,
            a.vram_needed_mib ?? null,
            a.vram_free_mib ?? null,
            a.vram_peak_mib ?? null,
            a.ram_needed_mib ?? null,
            a.ram_free_mib ?? null,
            a.ram_peak_mib ?? null,
            a.vram_process_peak_mib ?? null,
            a.ram_total_peak_mib ?? null,
            a.vram_shared_peak_mib ?? null,
            a.gen_tps ?? null,
            a.error ?? null,
            now,
            a.reused_from_run_id ?? null,
            a.vram_discrepancy === undefined ? null : a.vram_discrepancy ? 1 : 0,
            a.gpu_layers_resident_est ?? null,
            a.gpu_layers_resident_exact === undefined ? null : a.gpu_layers_resident_exact ? 1 : 0
          );
        });
      });
      tx();
      return input.attempts.length;
    },

    listForTest(runId: string): ProbeAttemptRow[] {
      return getDb()
        .prepare(`SELECT * FROM probe_attempts WHERE run_id = ? ORDER BY seq ASC`)
        .all(runId) as ProbeAttemptRow[];
    },

    // N2 live progress -- posted once per rung AS it happens (see
    // POST /api/tests/:id/probe-attempt), unlike replaceForTest above which
    // only ever lands once, at the very end of the ladder. Upserts through
    // the same UNIQUE(run_id, seq) key replaceForTest's own comment relies on
    // for retry-safety, so a worker retry that re-sends an earlier seq
    // overwrites it rather than duplicating -- and the final replaceForTest
    // call, which still runs unchanged, harmlessly replaces these same rows
    // with identical final data.
    upsertOne(input: {
      run_id: string;
      worker_id: string;
      model_id: string;
      seq: number;
      attempt: ProbeAttemptReport;
    }): void {
      const a = input.attempt;
      getDb()
        .prepare(
          `INSERT INTO probe_attempts
             (id, run_id, worker_id, model_id, seq, candidate_ctx, ngl,
              ok, oom, spill, vram_needed_mib, vram_free_mib, vram_peak_mib,
              ram_needed_mib, ram_free_mib, ram_peak_mib, vram_process_peak_mib, ram_total_peak_mib,
              vram_shared_peak_mib, gen_tps, error, created_at,
              reused_from_run_id, vram_discrepancy, gpu_layers_resident_est, gpu_layers_resident_exact)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_id, seq) DO UPDATE SET
             candidate_ctx = excluded.candidate_ctx,
             ngl = excluded.ngl,
             ok = excluded.ok,
             oom = excluded.oom,
             spill = excluded.spill,
             vram_needed_mib = excluded.vram_needed_mib,
             vram_free_mib = excluded.vram_free_mib,
             vram_peak_mib = excluded.vram_peak_mib,
             ram_needed_mib = excluded.ram_needed_mib,
             ram_free_mib = excluded.ram_free_mib,
             ram_peak_mib = excluded.ram_peak_mib,
             vram_process_peak_mib = excluded.vram_process_peak_mib,
             ram_total_peak_mib = excluded.ram_total_peak_mib,
             vram_shared_peak_mib = excluded.vram_shared_peak_mib,
             gen_tps = excluded.gen_tps,
             error = excluded.error,
             created_at = excluded.created_at,
             reused_from_run_id = excluded.reused_from_run_id,
             vram_discrepancy = excluded.vram_discrepancy,
             gpu_layers_resident_est = excluded.gpu_layers_resident_est,
             gpu_layers_resident_exact = excluded.gpu_layers_resident_exact`
        )
        .run(
          uuid(),
          input.run_id,
          input.worker_id,
          input.model_id,
          input.seq,
          a.candidate_ctx,
          a.ngl ?? null,
          a.ok ? 1 : 0,
          a.oom ? 1 : 0,
          a.spill ? 1 : 0,
          a.vram_needed_mib ?? null,
          a.vram_free_mib ?? null,
          a.vram_peak_mib ?? null,
          a.ram_needed_mib ?? null,
          a.ram_free_mib ?? null,
          a.ram_peak_mib ?? null,
          a.vram_process_peak_mib ?? null,
          a.ram_total_peak_mib ?? null,
          a.vram_shared_peak_mib ?? null,
          a.gen_tps ?? null,
          a.error ?? null,
          Date.now(),
          a.reused_from_run_id ?? null,
          a.vram_discrepancy === undefined ? null : a.vram_discrepancy ? 1 : 0,
          a.gpu_layers_resident_est ?? null,
          a.gpu_layers_resident_exact === undefined ? null : a.gpu_layers_resident_exact ? 1 : 0
        );
    },
  },

  // N4 -- perplexity/KLD rows. Worker retries replace through the six-column
  // UNIQUE key; without it the f16 baseline pairing would face an ambiguous
  // "which one?".
  qualityRepo: {
    upsert(input: QualityRowInput): QualityRow {
      const database = getDb();
      const now = Date.now();
      const tx = database.transaction(() => {
        // Baseline pairing (N4): KLD is measured against the f16-KV,
        // same-ctx, same-dataset row of the same model+build -- mirroring
        // §0.8's twin discipline. "No baseline row => ppl only" is an
        // ENFORCED invariant here, not merely a description of today's
        // worker behavior: a reported kld_vs_baseline is only ever stored
        // when a matching f16/f16 row genuinely exists, and an f16/f16 row
        // itself (nothing to diff against) always lands ppl-only. Scoped by
        // (model, build, ctx, dataset) like orphanDependents below, not by
        // this row's own root_run_id -- a baseline measured once is reused by
        // every later comparison against that model+build, not re-measured
        // per chain.
        //
        // The value itself is populated ONLY from what the measurement
        // reported. It is deliberately NOT derived from the two perplexity
        // numbers: a log-perplexity ratio is not a KL divergence, and filling
        // a column named kld_vs_baseline with something that isn't one is
        // exactly the silent substitution this plan forbids everywhere else.
        // Until llama-perplexity's KL output is wired through (the plan's own
        // open question), a row with a real baseline still lands ppl-only
        // unless the worker itself reports a value -- which is the stated
        // fallback.
        const isBaselineItself = input.cache_type_k === "f16" && input.cache_type_v === "f16";
        let kld: number | null = null;
        if (!isBaselineItself) {
          const baselineExists = database
            .prepare(
              `SELECT 1 FROM quality_results
               WHERE model_id = ? AND llama_cpp_build = ? AND ctx_tokens = ?
                 AND cache_type_k = 'f16' AND cache_type_v = 'f16' AND dataset_hash = ?`
            )
            .get(input.model_id, input.llama_cpp_build, input.ctx_tokens, input.dataset_hash);
          if (baselineExists) kld = input.kld_vs_baseline ?? null;
        }

        database
          .prepare(
            `INSERT INTO quality_results
               (id, root_run_id, model_id, worker_id, llama_cpp_build, ctx_tokens,
                cache_type_k, cache_type_v, ppl, kld_vs_baseline, dataset_hash, method_version, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(root_run_id, model_id, ctx_tokens, cache_type_k, cache_type_v, dataset_hash)
             DO UPDATE SET ppl = excluded.ppl,
                           kld_vs_baseline = excluded.kld_vs_baseline,
                           worker_id = excluded.worker_id,
                           llama_cpp_build = excluded.llama_cpp_build,
                           method_version = excluded.method_version,
                           created_at = excluded.created_at`
          )
          .run(
            uuid(),
            input.root_run_id,
            input.model_id,
            input.worker_id,
            input.llama_cpp_build,
            input.ctx_tokens,
            input.cache_type_k,
            input.cache_type_v,
            input.ppl,
            kld,
            input.dataset_hash,
            input.method_version ?? null,
            now
          );
      });
      tx();
      return database
        .prepare(
          `SELECT * FROM quality_results
           WHERE root_run_id = ? AND model_id = ? AND ctx_tokens = ? AND cache_type_k = ? AND cache_type_v = ? AND dataset_hash = ?`
        )
        .get(
          input.root_run_id,
          input.model_id,
          input.ctx_tokens,
          input.cache_type_k,
          input.cache_type_v,
          input.dataset_hash
        ) as QualityRow;
    },

    listForModel(modelId: string): QualityRow[] {
      return getDb()
        .prepare(`SELECT * FROM quality_results WHERE model_id = ? ORDER BY created_at DESC`)
        .all(modelId) as QualityRow[];
    },

    listForRoot(rootRunId: string): QualityRow[] {
      return getDb()
        .prepare(`SELECT * FROM quality_results WHERE root_run_id = ? ORDER BY created_at DESC`)
        .all(rootRunId) as QualityRow[];
    },

    // Deleting a baseline marks dependent KLD rows orphaned rather than
    // silently re-baselining them (N4's honesty rule).
    orphanDependents(modelId: string, build: string, ctxTokens: number, datasetHash: string): number {
      const info = getDb()
        .prepare(
          `UPDATE quality_results SET kld_vs_baseline = NULL
           WHERE model_id = ? AND llama_cpp_build = ? AND ctx_tokens = ? AND dataset_hash = ?
             AND NOT (cache_type_k = 'f16' AND cache_type_v = 'f16')`
        )
        .run(modelId, build, ctxTokens, datasetHash);
      return info.changes;
    },
  },
};

// N7 -- a validated bundle row, exactly as shared/exchange.ts hands it over.
export interface ImportedResultRow {
  config_hash: string;
  test_type: TestType;
  n_prompt: number;
  n_gen: number;
  n_depth: number;
  n_threads: number;
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  n_gpu_layers_draft: number;
  n_cpu_moe: number;
  engine: string | null;
  concurrency: number | null;
  method_version: number | null;
  avg_tps: number;
  stddev_tps: number;
  sample_count: number | null;
  suspect_count: number | null;
  vram_peak_mib: number | null;
  ram_peak_mib: number;
  ttft_ms_p50: number | null;
  ttft_ms_p95: number | null;
  ttft_n: number | null;
  e2e_ms_mean: number | null;
  gpu_temp_c_max: number | null;
  gpu_clock_mhz_min: number | null;
  caveat_flags: string[];
  created_at: number;
}

// N2 storage shapes.
export interface VerifiedLimitInput {
  worker_id: string;
  model_id: string;
  llama_cpp_build: string;
  cache_type_k: string;
  cache_type_v: string;
  placement_hash: string;
  verified_ctx_tokens: number;
  /** Layers on GPU at the verified rung -- see ProbeResultInput.verified_ngl. */
  verified_ngl?: number | null;
  margin_observed_frac?: number | null;
  method_version?: number | null;
}

export interface VerifiedLimit {
  id: string;
  worker_id: string;
  model_id: string;
  llama_cpp_build: string;
  kv_type: string;
  placement_hash: string;
  verified_ctx_tokens: number;
  verified_ngl: number | null;
  margin_observed_frac: number | null;
  method_version: number | null;
  created_at: number;
}

export interface ProbeAttemptsInput {
  run_id: string;
  worker_id: string;
  model_id: string;
  attempts: ProbeAttemptReport[];
}

// SQLite has no boolean: ok/oom/spill come back as 0/1 and callers coerce.
export interface ProbeAttemptRow {
  id: string;
  run_id: string;
  worker_id: string;
  model_id: string;
  seq: number;
  candidate_ctx: number;
  ngl: number | null;
  ok: number;
  oom: number;
  spill: number;
  vram_needed_mib: number | null;
  vram_free_mib: number | null;
  vram_peak_mib: number | null;
  ram_needed_mib: number | null;
  ram_free_mib: number | null;
  ram_peak_mib: number | null;
  vram_process_peak_mib: number | null;
  ram_total_peak_mib: number | null;
  vram_shared_peak_mib: number | null;
  gen_tps: number | null;
  error: string | null;
  created_at: number;
  reused_from_run_id: string | null;
  vram_discrepancy: number | null;
  gpu_layers_resident_est: number | null;
  gpu_layers_resident_exact: number | null;
}

// N4 storage shapes.
export interface QualityRowInput {
  root_run_id: string;
  model_id: string;
  worker_id: string;
  llama_cpp_build: string;
  ctx_tokens: number;
  cache_type_k: string;
  cache_type_v: string;
  ppl: number;
  kld_vs_baseline?: number | null;
  dataset_hash: string;
  method_version?: number | null;
}

export interface QualityRow {
  id: string;
  root_run_id: string;
  model_id: string;
  worker_id: string;
  llama_cpp_build: string;
  ctx_tokens: number;
  cache_type_k: string;
  cache_type_v: string;
  ppl: number | null;
  kld_vs_baseline: number | null;
  dataset_hash: string;
  method_version: number | null;
  created_at: number;
}

// Builds the *physical-column* subset of ResultRow -- deliberately excludes
// backend_type/backend_device_name, which aren't physical columns on
// `results` at all (see ResultRow's own doc comment: they're a JOIN done in
// getResultsForTest, only known once a row is read back with its parent run,
// never at insert time here).
function buildResultRow(
  runId: string,
  modelId: string,
  idx: number,
  r: IngestResultInput
): Omit<ResultRow, "backend_type" | "backend_device_name"> {
  return {
    id: uuid(),
    run_id: runId,
    idx,
    model_id: modelId,
    test_type: r.test_type,
    n_prompt: r.n_prompt,
    n_gen: r.n_gen,
    n_threads: r.n_threads,
    n_gpu_layers: r.n_gpu_layers,
    batch_size: r.batch_size,
    ubatch_size: r.ubatch_size,
    cache_type_k: r.cache_type_k,
    cache_type_v: r.cache_type_v,
    flash_attn: r.flash_attn,
    mtp: r.mtp,
    // Defaults to 0 for a version-skewed worker that predates this field --
    // see shared/types.ts's IngestResultInput.n_gpu_layers_draft and
    // schema.sql's matching column default.
    n_gpu_layers_draft: r.n_gpu_layers_draft ?? 0,
    // Same reasoning as n_gpu_layers_draft above.
    n_cpu_moe: r.n_cpu_moe ?? 0,
    avg_tps: r.avg_tps,
    stddev_tps: r.stddev_tps,
    ram_peak_mib: r.ram_peak_mib,
    vram_peak_mib: r.vram_peak_mib,
    ram_avg_mib: r.ram_avg_mib,
    vram_avg_mib: r.vram_avg_mib,
    ram_free_before_mib: r.ram_free_before_mib,
    vram_free_before_mib: r.vram_free_before_mib,
    system_memory_total_mb: r.system_memory_total_mb,
    gpu_memory_total_mb: r.gpu_memory_total_mb,
    gpu_memory_total_accuracy: r.gpu_memory_total_accuracy,
    gpu_memory_total_source: r.gpu_memory_total_source,
    gpu_memory_free_start_accuracy: r.gpu_memory_free_start_accuracy,
    gpu_memory_free_start_source: r.gpu_memory_free_start_source,
    gpu_memory_model_avg_accuracy: r.gpu_memory_model_avg_accuracy,
    gpu_memory_model_avg_source: r.gpu_memory_model_avg_source,
    gpu_memory_model_peak_accuracy: r.gpu_memory_model_peak_accuracy,
    gpu_memory_model_peak_source: r.gpu_memory_model_peak_source,
    // Version-skew tolerant like every optional IngestResultInput field --
    // absent on an older worker, null from a current one with nothing to
    // measure on that backend (e.g. no per-process reading on Metal).
    gpu_memory_used_avg_mib: r.gpu_memory_used_avg_mib ?? null,
    gpu_memory_used_peak_mib: r.gpu_memory_used_peak_mib ?? null,
    // accuracy is always a concrete level, never null -- "unavailable" is
    // itself the valid value for "nothing measured", matching every other
    // accuracy field on this row (see shared/types.ts's GpuMemoryAccuracyLevel).
    gpu_memory_used_avg_accuracy: r.gpu_memory_used_avg_accuracy ?? "unavailable",
    gpu_memory_used_avg_source: r.gpu_memory_used_avg_source ?? null,
    gpu_memory_used_peak_accuracy: r.gpu_memory_used_peak_accuracy ?? "unavailable",
    gpu_memory_used_peak_source: r.gpu_memory_used_peak_source ?? null,
    gpu_memory_process_avg_mib: r.gpu_memory_process_avg_mib ?? null,
    gpu_memory_process_peak_mib: r.gpu_memory_process_peak_mib ?? null,
    gpu_memory_process_avg_accuracy: r.gpu_memory_process_avg_accuracy ?? "unavailable",
    gpu_memory_process_avg_source: r.gpu_memory_process_avg_source ?? null,
    gpu_memory_process_peak_accuracy: r.gpu_memory_process_peak_accuracy ?? "unavailable",
    gpu_memory_process_peak_source: r.gpu_memory_process_peak_source ?? null,
    ram_total_used_avg_mib: r.ram_total_used_avg_mib ?? null,
    ram_total_used_peak_mib: r.ram_total_used_peak_mib ?? null,
    gpu_layers_loaded: r.gpu_layers_loaded,
    total_model_layers: r.total_model_layers,
    gpu_layers_loaded_draft: r.gpu_layers_loaded_draft ?? null,
    total_model_layers_draft: r.total_model_layers_draft ?? null,
    // Version-skew tolerant like every optional IngestResultInput field:
    // absent on an older worker, null from a current one with nothing to
    // estimate (no offload claim, or no buffer report to derive from).
    gpu_layers_resident_est: r.gpu_layers_resident_est ?? null,
    // Same version-skew tolerance -- the draft companion's own estimate,
    // null on every non-MTP row (no draft buffer lines to derive from).
    gpu_layers_resident_est_draft: r.gpu_layers_resident_est_draft ?? null,
    sample_count: r.sample_count,
    suspect_count: r.suspect_count,
    suspect_samples: r.suspect_samples,
    repeat_samples: r.repeat_samples,
    spec_drafted: r.spec_drafted,
    spec_accepted: r.spec_accepted,
    method_version: r.method_version ?? null,
    n_depth: r.n_depth ?? 0,
    prompt_offset: r.prompt_offset ?? null,
    spec_type: r.spec_type ?? null,
    spec_n_max: r.spec_n_max ?? null,
    spec_n_min: r.spec_n_min ?? null,
    caveat_flags: r.caveat_flags ?? [],
    concurrency: r.concurrency ?? null,
    gpu_temp_c_max: r.gpu_temp_c_max ?? null,
    gpu_clock_mhz_min: r.gpu_clock_mhz_min ?? null,
    gpu_clock_samples: r.gpu_clock_samples ?? null,
    cpu_isa: r.cpu_isa ?? null,
    ttft_ms_p50: r.ttft_ms_p50 ?? null,
    ttft_ms_p95: r.ttft_ms_p95 ?? null,
    ttft_n: r.ttft_n ?? null,
    e2e_ms_mean: r.e2e_ms_mean ?? null,
    speedup: null,
    speedup_status: null,
    raw_json_path: undefined,
    created_at: Date.now(),
  };
}

function configHashInputFromResult(
  row: Omit<ResultRow, "backend_type" | "backend_device_name">,
  r: IngestResultInput,
  engine: EngineKind
): ConfigHashInput {
  return {
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_depth: row.n_depth ?? 0,
    threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    n_cpu_moe: row.n_cpu_moe,
    engine,
    concurrency: row.concurrency ?? undefined,
  };
}

function deriveEngineForResult(
  row: Pick<ResultRow, "mtp">,
  run: Test
): EngineKind {
  if (row.mtp === "on") return "server";
  if (run.kind === "runtime" || run.kind === "probe") return "server";
  return engineFromItem({ mtp: row.mtp });
}

interface TwinCandidateRow {
  id: string;
  avg_tps: number | null;
  suspect_count: number | null;
  sample_count: number | null;
  spec_drafted: number | null;
  spec_accepted: number | null;
  prompt_offset: number | null;
  method_version: number | null;
  speedup_status: string | null;
  mtp: string;
}

// §0.8 twin join -- a speculative {server, spec} row pairs with its
// {server, off} baseline on (root_run_id, config_hash, test_type, n_depth),
// where config_hash excludes the spec fields. Both sides must also match on
// prompt_offset and method_version; any mismatch or unverifiable input makes
// the pair `unverified` (an offset mismatch additionally writes
// spec_pair_prompt_mismatch), no counterpart at all is `unavailable`.
function markSpeedupStatuses(
  database: ReturnType<typeof getDb>,
  insertedRowId: string,
  hash: string,
  engine: EngineKind,
  wallClockFallback: boolean
): void {
  const inserted = database
    .prepare(
      `SELECT id, run_id, test_type, COALESCE(n_depth, 0) AS n_depth, mtp, avg_tps,
              suspect_count, sample_count, spec_drafted, spec_accepted, prompt_offset, method_version, speedup_status
       FROM results WHERE id = ?`
    )
    .get(insertedRowId) as (TwinCandidateRow & { run_id: string; test_type: string; n_depth: number }) | undefined;
  if (!inserted) return;
  const rootId = (
    database.prepare(`SELECT COALESCE(root_run_id, id) AS root FROM runs WHERE id = ?`).get(inserted.run_id) as {
      root: string;
    }
  ).root;

  const insertedIsSpec = inserted.mtp === "on";
  const insertedIsServerBaseline = engine === "server" && inserted.mtp !== "on";
  if (!insertedIsSpec && !insertedIsServerBaseline) return;

  const update = database.prepare(`UPDATE results SET speedup = ?, speedup_status = ? WHERE id = ?`);
  const writeFlags = database.prepare(`UPDATE results SET caveat_flags = ? WHERE id = ?`);
  const readFlags = database.prepare(`SELECT caveat_flags FROM results WHERE id = ?`);

  const findTwin = (mtpFilter: string): TwinCandidateRow | undefined =>
    database
      .prepare(
        `SELECT r.id, r.avg_tps, r.suspect_count, r.sample_count, r.spec_drafted, r.spec_accepted,
                r.prompt_offset, r.method_version, r.speedup_status, r.mtp
         FROM results r JOIN runs ON runs.id = r.run_id
         WHERE COALESCE(runs.root_run_id, runs.id) = ?
           AND r.config_hash IS ? AND r.test_type = ? AND COALESCE(r.n_depth, 0) = ?
           AND r.mtp ${mtpFilter}
         ORDER BY r.created_at DESC LIMIT 1`
      )
      .get(rootId, hash, inserted.test_type, inserted.n_depth) as TwinCandidateRow | undefined;

  const spec = insertedIsSpec ? inserted : findTwin("= 'on'");
  const baseline = insertedIsSpec ? findTwin("!= 'on'") : inserted;
  if (!spec || !baseline) {
    if (wallClockFallback) update.run(null, "unverified", inserted.id);
    else if (inserted.speedup_status == null && insertedIsSpec) update.run(null, "unavailable", inserted.id);
    return;
  }

  let status: "ok" | "unverified" = "ok";
  const flagsToWrite: string[] = [];
  const suspectOnly = (r: TwinCandidateRow): boolean =>
    r.sample_count != null && r.sample_count > 0 && (r.suspect_count ?? 0) >= (r.sample_count ?? 0);
  if (suspectOnly(spec) || suspectOnly(baseline)) status = "unverified";
  if ((spec.spec_drafted != null && spec.spec_drafted === 0) || (spec.spec_accepted != null && spec.spec_accepted === 0)) {
    status = "unverified";
  }
  if (spec.prompt_offset !== baseline.prompt_offset) {
    status = "unverified";
    flagsToWrite.push("spec_pair_prompt_mismatch");
  }
  if ((spec.method_version ?? null) !== (baseline.method_version ?? null)) status = "unverified";
  if (spec.speedup_status === "unverified" || baseline.speedup_status === "unverified" || wallClockFallback) {
    status = "unverified";
  }

  const ratio =
    baseline.avg_tps != null && spec.avg_tps != null && baseline.avg_tps > 0 && spec.avg_tps > 0
      ? spec.avg_tps / baseline.avg_tps
      : null;
  for (const target of [spec.id, baseline.id]) {
    update.run(ratio, ratio == null ? "unavailable" : status, target);
  }
  if (flagsToWrite.length > 0) {
    const current = (readFlags.get(spec.id) as { caveat_flags: string | null }).caveat_flags;
    const merged = [...new Set([...parseCaveatFlags(current), ...flagsToWrite])];
    writeFlags.run(JSON.stringify(merged), spec.id);
  }
}