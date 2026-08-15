// Sentinel error message the server uses whenever it can't reach a worker at
// all (connection refused, DNS failure, timeout, ...) -- shared so the
// client can recognize it and render a clean "Inaccessible" state instead of
// displaying it as a generic error string. See server/src/worker-errors.ts
// for where this gets produced.
export const WORKER_INACCESSIBLE_MESSAGE = "worker is inaccessible";

// Well-known values used for auto-detection's own guess (see
// worker/src/hardware.ts's detectBackend) and as UI suggestions -- NOT an
// exhaustive allowlist. Backend itself is a plain string (below): any value
// is accepted end-to-end, matched against release asset names generically
// (see server/src/github-releases.ts), so a future llama.cpp release adding
// a new backend variant works without a code change here.
export const KNOWN_BACKENDS = [
  "cpu",
  "vulkan",
  "cuda",
  "rocm",
  "sycl",
  "opencl-adreno",
  "openvino",
] as const;

export type Backend = string;

// How a GPU memory figure was obtained, and how much to trust it -- see
// worker/src/vram.ts's per-backend readGpuMemory dispatcher, which is the
// only thing that ever assigns these. "unavailable" is a real, valid state
// (not an absence): whenever a GPU memory value is null, its accuracy is
// always "unavailable" and its source always null, never the other way
// around. Never fabricate a value to avoid reporting "unavailable" -- that's
// the one hard rule every backend's collection code follows.
export const GPU_MEMORY_ACCURACY_LEVELS = ["exact", "high", "estimated", "unavailable"] as const;
export type GpuMemoryAccuracyLevel = (typeof GPU_MEMORY_ACCURACY_LEVELS)[number];

// "exact" = a first-party vendor tool's per-process reading (e.g. nvidia-smi
// --query-compute-apps matched against the benchmark's own PID). "high" = a
// real driver/kernel reading that exists but isn't process-isolated (whole-
// adapter usage, e.g. Windows' GPU Adapter Memory perf counter or Linux's
// amdgpu sysfs vram_used). "estimated"/"budget"/"unified" sources are for
// backends with no direct usage reading at all (Metal's unified memory).
export const GPU_MEMORY_MEASUREMENT_SOURCES = [
  "process_gpu_usage",
  "driver_reported_memory",
  "backend_allocation_tracking",
  "memory_budget_estimate",
  "unified_memory_estimate",
] as const;
export type GpuMemoryMeasurementSource = (typeof GPU_MEMORY_MEASUREMENT_SOURCES)[number];

export type TestType = "pp" | "tg" | "pg";

export type ModelSource = "local" | "huggingface";

export interface ModelMetadata {
  arch?: string;
  quant?: string;
  trained_ctx?: number;
  // Transformer layer count (GGUF's <architecture>.block_count), read from
  // the file header on download -- see worker/src/gguf.ts. Undefined for
  // models registered before this existed, or for manually-registered
  // "local" models the app never had file bytes to parse.
  n_layer?: number;
  // Total parameter count (raw element count across every tensor in the
  // GGUF, as Hugging Face itself computes it from the file's tensor shapes)
  // -- fetched from HF's model API at download time, see
  // server/src/hf.ts's getHfGgufMeta. Deliberately NOT derived from repo/
  // filename naming: some families (e.g. Gemma 3n's "E2B"/"E4B") name
  // themselves after an "effective"/marketing parameter count that's much
  // smaller than the real total. Undefined for models registered before
  // this existed, local models with no hf_repo, or repos HF hasn't computed
  // gguf metadata for.
  param_count?: number;
  // Nextn/MTP (multi-token-prediction) layer count, GGUF's
  // <architecture>.nextn_predict_layers, read at the same time as n_layer --
  // see worker/src/gguf.ts. >0 means this file has a usable MTP head:
  // either baked into a normal base model (Qwen/DeepSeek/GLM-style, this
  // file is directly benchmarkable with --spec-type draft-mtp and no
  // companion), or -- when mtp_role below is "draft" -- this *is* the
  // companion file itself (Gemma-4-style), meant to be paired with a base
  // model via --model-draft rather than run standalone.
  mtp_layers?: number;
  // Set to "draft" when this model was registered from a path/filename
  // containing "mtp" (the real-world convention for standalone MTP/drafter
  // downloads, e.g. unsloth's "MTP/mtp-gemma-4-12B-it.gguf") -- see
  // server/src/routes/workers.ts's download route. Such a model isn't
  // directly benchmarkable on its own; the client excludes it from the main
  // model picker and offers it only as an MTP-model companion choice.
  mtp_role?: "draft";
  [key: string]: unknown;
}

export interface Model {
  id: string;
  filename: string;
  size_bytes: number;
  source: ModelSource;
  hf_repo?: string;
  hf_file?: string;
  metadata: ModelMetadata;
  created_at: number;
}

// Real-world unsloth convention for a standalone MTP/drafter file: either at
// repo root ("mtp-gemma-4-E2B-it.gguf") or inside an "MTP/" folder
// ("MTP/mtp-gemma-4-E2B-it-Q4_0.gguf") -- "mtp" as its own path segment /
// filename-prefix token, not merely a substring, so a legitimately
// self-sufficient base model that happens to advertise MTP support in its
// own name (e.g. a real "Qwen3.6-27B-MTP-Q4_K_M.gguf") doesn't false-positive.
const MTP_FILENAME_PATTERN = /(^|\/)mtp[-_.]/i;

// Re-derives whether a model is a standalone MTP/draft companion file (not
// directly benchmarkable on its own) from its own stored metadata + filename,
// rather than trusting a persisted metadata.mtp_role flag alone. This keeps
// models registered before a detection-logic fix (or that otherwise ended up
// with stale/missing metadata.mtp_role) correctly classified everywhere --
// grouping, the New Run model picker, and the trigger route's run-eligibility
// check -- with no backfill migration needed, since n_layer/mtp_layers/
// filename are the only real inputs metadata.mtp_role was ever derived from
// in the first place. See worker/src/gguf.ts and server/src/routes/
// workers.ts's download route for where these fields come from.
//
// Unlike the content-based check below, the filename check does NOT require
// n_layer to be absent: a real drafter file can still report a small nonzero
// n_layer of its own (observed live -- Gemma-4's MTP file reports n_layer 4,
// not absent -- which is exactly why that file wasn't being detected before
// this function existed).
export function isMtpDraftModel(m: { metadata: ModelMetadata; hf_file?: string; filename: string }): boolean {
  if (m.metadata.mtp_role === "draft") return true;
  const { n_layer, mtp_layers } = m.metadata;
  if (typeof mtp_layers === "number" && mtp_layers > 0 && typeof n_layer !== "number") return true;
  return MTP_FILENAME_PATTERN.test(m.hf_file ?? m.filename);
}

export interface RegisterModelInput {
  id?: string;
  filename?: string;
  size_bytes?: number;
  source: ModelSource;
  hf_repo?: string;
  hf_file?: string;
  metadata?: ModelMetadata;
}

export interface SweepConfig {
  model_id: string;
  n_prompt: number[];
  n_gen: number[];
  threads: number[];
  n_gpu_layers: number[];
  batch_size: number[];
  ubatch_size: number[];
  cache_type_k: string[];
  cache_type_v: string[];
  flash_attn: string[];
  // "on"/"off" -- whether to run this combo's benchmark via llama-server
  // with --spec-type draft-mtp instead of the normal llama-bench path. See
  // worker/src/serverBench.ts; llama-bench itself has no MTP support at all.
  mtp: string[];
  // GPU layers offloaded for the MTP/draft companion model -- llama-server's
  // -ngld/--n-gpu-layers-draft, swept independently of the base model's own
  // -ngl. See shared/sweep.ts's SweepItem.n_gpu_layers_draft for why it's
  // still a required field even on combos where it's a no-op.
  n_gpu_layers_draft: number[];
  repeats: number;
}

export interface RunConfig {
  model_id: string;
  // Which registered model to pass as llama-server's --model-draft when any
  // sweep item has mtp "on" and the base model's own mtp_layers isn't set
  // (i.e. a Gemma-4-style base model that needs a separate companion file
  // rather than Qwen-style in-file MTP). A single fixed choice for the whole
  // run, not swept -- unlike flash_attn, "which draft model" isn't something
  // that makes sense to toggle per combination.
  mtp_model_id?: string;
  // Index into the worker's HardwareInfo.gpu array (see worker/src/index.ts's
  // /run handler) -- restricts this run to that one GPU on a multi-GPU
  // worker, forwarded to llama-bench/llama-server as `-sm none -mg <index>`
  // (see worker/src/bench.ts and worker/src/serverBench.ts's buildArgs).
  // Undefined means "let llama.cpp use its own default" (split across every
  // visible GPU) -- same as before this field existed. Meaningless (ignored)
  // on a single-GPU or cpu-only worker.
  main_gpu?: number;
  sweep: Omit<SweepConfig, "model_id">;
}

// "partial" -- some sweep items succeeded and some failed, distinct from a
// wholesale "failed" (nothing at all completed) or unstarted "running".
// "cancelled" -- the user stopped the run before every item finished,
// distinct from "failed" (which means something actually went wrong).
// "scheduled" -- triggered while its worker already had another run
// running (or queued) against it; sits in that worker's FIFO queue and
// flips to "running" on its own once the run ahead of it finalizes (see
// server/src/routes/runs.ts's dispatchScheduledRun).
export type RunStatus = "running" | "scheduled" | "done" | "partial" | "failed" | "cancelled";

export interface Run {
  id: string;
  worker_name: string;
  llama_cpp_build: string;
  llama_cpp_backend: Backend;
  // The specific GPU/CPU model llama.cpp itself reported (e.g. "AMD Radeon
  // RX 6600 XT", "NVIDIA RTX 4090", "Apple M4 Max", "CPU") -- distinct from
  // llama_cpp_backend above, which is only the backend *kind*. Two-tier: set
  // from the worker's own hardware detection as soon as the run starts
  // (always available), then upgraded once the first non-MTP item's
  // llama-bench JSON output reports its exact gpu_info string (see
  // worker/src/bench.ts) -- a run made entirely of MTP items never gets the
  // upgrade and keeps the first-tier value, since llama-server never prints
  // an equivalent device-enumeration line at any verbosity (confirmed live).
  // Undefined only for a run predating this field.
  backend_device_name?: string;
  model_id: string;
  model_filename?: string;
  config: RunConfig;
  status: RunStatus;
  error?: string;
  started_at: number;
  completed_at?: number;
  // Cheap aggregate counts over this run's run_items, for list views that
  // want a live "x/y done" chip without fetching the full item list.
  // items_total is 0 (not undefined) for runs predating run_items -- treat
  // 0 the same as "nothing to show a chip for".
  items_total?: number;
  items_done?: number;
  items_failed?: number;
  items_cancelled?: number;
}

export interface ResultRow {
  id: string;
  run_id: string;
  // Matches the run_item this result came from, so a merged live/final table
  // in the client can join a result back to its item without relying on
  // sweep-parameter matching (a degenerate sweep could contain duplicate
  // parameter tuples). Not part of IngestResultInput -- the server already
  // knows idx as a route/repo parameter, no need to round-trip it through
  // the worker payload.
  idx: number;
  model_id: string;
  // The backend/device that actually produced this result -- denormalized
  // from the parent run (runs.llama_cpp_backend/backend_device_name, see
  // Run's own doc comments) via a JOIN in server/src/db/repo.ts's
  // getResultsForRun, not stored per-row: a run's backend can't vary between
  // its own items. backend_device_name is null exactly when the run's
  // two-tier detection (see Run) hasn't captured one at all.
  backend_type: Backend;
  backend_device_name: string | null;
  test_type: TestType;
  n_prompt: number;
  n_gen: number;
  n_threads: number;
  // The requested -ngl value for this combination -- already equivalent to
  // the spec's "gpu_layers_requested", so that's not a separate field. See
  // gpu_layers_loaded/total_model_layers below for what actually happened.
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  // See SweepConfig.n_gpu_layers_draft -- the requested value for this
  // combination's MTP/draft companion model, 0 on every row where it wasn't
  // applicable (mtp "off", a Qwen-style self-sufficient model, or a row
  // predating this column). Not narrowed further here since -- unlike
  // gpu_layers_loaded/total_model_layers below -- llama-server prints no
  // equivalent "draft offloaded X/Y" runtime line to confirm what actually
  // happened.
  n_gpu_layers_draft: number;
  avg_tps: number;
  stddev_tps: number;
  // Already equivalent to the spec's system_memory_model_peak_mb --
  // sampled RAM of the benchmark process, not renamed/duplicated since
  // that would touch 6+ files for zero functional gain.
  ram_peak_mib: number;
  // Already equivalent to the spec's gpu_memory_model_peak_mb, minus the
  // accuracy_level/measurement_source metadata -- see
  // gpu_memory_model_peak_accuracy/_source below for that.
  vram_peak_mib: number | null;
  // Already equivalent to the spec's system_memory_model_avg_mb.
  ram_avg_mib: number;
  // Already equivalent to the spec's gpu_memory_model_avg_mb -- see
  // gpu_memory_model_avg_accuracy/_source below for its metadata.
  vram_avg_mib: number | null;
  // Already equivalent to the spec's system_memory_free_start_mb.
  ram_free_before_mib: number | null;
  // Already equivalent to the spec's gpu_memory_free_start_mb -- see
  // gpu_memory_free_start_accuracy/_source below for its metadata.
  vram_free_before_mib: number | null;
  // Total system RAM (on Apple Silicon, total unified memory) -- a static
  // hardware fact, not sampled, so unlike the pairs above it has no
  // avg/peak/free_start variants and no accuracy/source metadata (the spec
  // scopes accuracy/provenance to GPU memory metrics only).
  system_memory_total_mb: number | null;
  // Total VRAM capacity (Metal: total unified memory, per the spec's
  // explicit "use total unified memory" instruction for that backend) --
  // see worker/src/vram.ts's per-backend readGpuMemory. Null + accuracy
  // "unavailable" on a cpu-backend run.
  gpu_memory_total_mb: number | null;
  // accuracy is always one of the 4 concrete levels, never absent -- it's
  // "unavailable" (not null) when the paired _mb value is null. Only source
  // (and the value itself) can genuinely be null.
  gpu_memory_total_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_total_source: GpuMemoryMeasurementSource | null;
  gpu_memory_free_start_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_free_start_source: GpuMemoryMeasurementSource | null;
  gpu_memory_model_avg_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_model_avg_source: GpuMemoryMeasurementSource | null;
  gpu_memory_model_peak_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_model_peak_source: GpuMemoryMeasurementSource | null;
  // Actual GPU layers loaded and the model's real total layer count, both
  // read from llama.cpp's own runtime output (worker/src/index.ts's
  // parseOffloadLayers, scraping "load_tensors: offloaded X/Y layers to
  // GPU") -- never inferred/calculated. total_model_layers is deliberately
  // NOT sourced from models.metadata.n_layer: confirmed live that llama.cpp's
  // runtime Y equals the GGUF's real layer count + 1 (the output layer), a
  // different number under a different counting convention. Both null when
  // the model failed to load before this line was ever printed, or on a
  // build too old to support the verbosity flag this requires.
  gpu_layers_loaded: number | null;
  total_model_layers: number | null;
  // The MTP/draft companion model's OWN actual offload, read from its own
  // separate "load_tensors: offloaded X/Y layers to GPU" line -- llama-server
  // prints one such line per model it loads, and an MTP item loads two (the
  // base model and its --model-draft companion). Only ever populated by the
  // llama-server/MTP path (worker/src/serverBench.ts via
  // worker/src/index.ts's parseOffloadLayers, which disambiguates the two
  // lines by total layer count -- a draft/companion model is always far
  // smaller than the base model it's paired with). Both undefined on every
  // non-MTP row (nothing to report) and on an MTP row where the draft
  // model's own line wasn't captured (e.g. a Qwen-style in-file MTP model
  // with no separate --model-draft load at all). gpu_layers_loaded/
  // total_model_layers above are always the *base* model's figures on an MTP
  // row, never the draft's -- these are the draft's own, so the two aren't
  // interchangeable.
  gpu_layers_loaded_draft?: number | null;
  total_model_layers_draft?: number | null;
  // Only ever populated by the llama-server/MTP path (worker/src/serverBench.ts)
  // -- the llama-bench-CLI path (bench.ts) does its own internal repeat
  // averaging and never sees individual readings, so these stay undefined
  // there. sample_count is every repeat that produced a computable reading
  // for this test_type (clean + suspect); suspect_count is the subset of
  // those flagged implausible (see MAX_PLAUSIBLE_*_TOKENS_PER_SECOND in
  // serverBench.ts) and excluded from avg_tps/stddev_tps *unless* every
  // single reading was suspect, in which case avg_tps/stddev_tps falls back
  // to averaging the suspect readings anyway rather than silently omitting
  // this test_type's row entirely -- suspect_samples holds their raw values
  // either way, so a real number is always visible instead of vanishing.
  sample_count?: number;
  suspect_count?: number;
  suspect_samples?: number[];
  // Every individual repeat's computable reading (clean + suspect, in repeat
  // order) for this test_type -- only populated by the llama-server/MTP path,
  // same as sample_count/suspect_count/suspect_samples above. avg_tps/
  // stddev_tps are still the single number most call sites want; this is for
  // seeing per-run variance directly (e.g. in the CSV/JSON export) instead of
  // only the aggregate. suspect_samples is a subset of this array (the
  // flagged-implausible ones specifically); this array has no such flag per
  // element, just the raw per-repeat values in the order they were measured.
  repeat_samples?: number[];
  // Only populated on the tg row of an MTP item (worker/src/serverBench.ts) --
  // llama-server's own /metrics speculative-decoding counters, the only
  // direct evidence the --spec-type draft-mtp draft head actually contributed
  // accepted tokens to this item's tg number rather than silently no-opping
  // (e.g. a bad/incompatible --model-draft file). Both undefined when
  // /metrics was unreachable or reported no counters at all -- a distinct,
  // worse condition than a present-but-zero spec_drafted, which means the
  // draft head loaded but never actually drafted anything.
  spec_drafted?: number;
  spec_accepted?: number;
  raw_json_path?: string;
  created_at: number;
}

export interface IngestResultInput {
  test_type: TestType;
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
  // Optional (unlike every other field here): an older worker that predates
  // this column simply won't send it. Defaulted to 0 server-side (see
  // server/src/db/repo.ts's buildResultRow) rather than required, so a
  // version-skewed worker's real results still get saved instead of the
  // whole item being rejected over one missing/new metric.
  n_gpu_layers_draft?: number;
  avg_tps: number;
  stddev_tps: number;
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  ram_avg_mib: number;
  vram_avg_mib: number | null;
  ram_free_before_mib: number | null;
  vram_free_before_mib: number | null;
  // See the matching fields on ResultRow above -- backend_type/
  // backend_device_name are deliberately NOT here, since the server derives
  // both from the parent run (a JOIN, see repo.ts's getResultsForRun) rather
  // than trusting a per-item copy the worker would have to keep in sync.
  system_memory_total_mb: number | null;
  gpu_memory_total_mb: number | null;
  gpu_memory_total_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_total_source: GpuMemoryMeasurementSource | null;
  gpu_memory_free_start_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_free_start_source: GpuMemoryMeasurementSource | null;
  gpu_memory_model_avg_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_model_avg_source: GpuMemoryMeasurementSource | null;
  gpu_memory_model_peak_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_model_peak_source: GpuMemoryMeasurementSource | null;
  gpu_layers_loaded: number | null;
  total_model_layers: number | null;
  // See the matching fields on ResultRow above -- the MTP/draft companion
  // model's own actual offload, distinct from gpu_layers_loaded/
  // total_model_layers above (always the base model's).
  gpu_layers_loaded_draft?: number | null;
  total_model_layers_draft?: number | null;
  sample_count?: number;
  suspect_count?: number;
  suspect_samples?: number[];
  repeat_samples?: number[];
  spec_drafted?: number;
  spec_accepted?: number;
}

// --- Per-item live progress (one llama-bench process per sweep combo) ---

export type RunItemStatus =
  | "queued"
  | "loading"
  | "processing"
  | "generating"
  | "benchmarking"
  | "done"
  | "failed"
  | "failed_oom"
  | "cancelled";

const TERMINAL_RUN_ITEM_STATUSES = ["done", "failed", "failed_oom", "cancelled"] as const;
export type TerminalRunItemStatus = (typeof TERMINAL_RUN_ITEM_STATUSES)[number];

export function isTerminalRunItemStatus(status: RunItemStatus): status is TerminalRunItemStatus {
  return (TERMINAL_RUN_ITEM_STATUSES as readonly string[]).includes(status);
}

export interface RunItem {
  id: string;
  run_id: string;
  idx: number;
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
  status: RunItemStatus;
  detail?: string;
  ram_mib?: number | null;
  vram_mib?: number | null;
  ram_peak_mib?: number | null;
  vram_peak_mib?: number | null;
  ram_avg_mib?: number | null;
  vram_avg_mib?: number | null;
  ram_free_before_mib?: number | null;
  vram_free_before_mib?: number | null;
  // Measured per-repeat throughput from the most recently completed repeat
  // (see worker/src/index.ts's repeat-marker parsing) -- refined every
  // repeat, undefined until the first repeat of the current phase finishes.
  live_tps?: number | null;
  error?: string;
  started_at?: number;
  completed_at?: number;
}

// Fire-and-forget progress ticks -- worker sends these best-effort with a
// short timeout while an item is loading/running; losing one is harmless
// since the next tick or the terminal update below catches the DB up.
export interface RunItemTickInput {
  status: "loading" | "processing" | "generating" | "benchmarking";
  detail?: string;
  ram_mib?: number;
  vram_mib?: number | null;
  live_tps?: number | null;
  // Free-memory baseline, captured once right before the process spawns --
  // only meaningful on the very first tick for an item. The server persists
  // it with COALESCE-on-existing-value (same pattern as started_at) so later
  // ticks that omit it don't clobber it.
  ram_free_before_mib?: number;
  vram_free_before_mib?: number | null;
}

// Terminal per-item outcome -- worker sends these with retry-with-backoff
// (same posture as the old whole-run ingest they replace), since they're
// what permanently records history: a "done" result becomes a `results` row,
// a failure closes out that item with no result.
export interface RunItemTerminalInput {
  status: "done" | "failed" | "failed_oom" | "cancelled";
  error?: string;
  // Tier-2 device-name upgrade (see Run.backend_device_name) -- only ever
  // sent alongside status "done" from the llama-bench path, when that item's
  // llama-bench JSON output included a gpu_info string (worker/src/index.ts's
  // runSweepItem). Absent on every other item; the server applies it as a
  // plain UPDATE onto the parent run, not per-item.
  backend_device_name?: string;
  // Peaks are worth recording even on a failure -- "climbed to 3.8GB then
  // died" is the whole story for an OOM. `results` (the tps metrics that
  // become `results` rows) only applies when status is "done" -- an array
  // since one llama-bench process can produce up to two rows for a single
  // sweep item (a pp row and a tg row) when both n_prompt and n_gen are set
  // without -pg, see worker/src/bench.ts's parseLlamaBench.
  ram_peak_mib?: number;
  vram_peak_mib?: number | null;
  ram_avg_mib?: number;
  vram_avg_mib?: number | null;
  results?: IngestResultInput[];
}

export type RunItemUpdateInput = RunItemTickInput | RunItemTerminalInput;

export function isTerminalRunItemInput(input: RunItemUpdateInput): input is RunItemTerminalInput {
  return isTerminalRunItemStatus(input.status);
}

export interface TriggerPayload {
  model_id: string;
  worker_name: string;
  // See RunConfig.mtp_model_id -- same field, forwarded through the trigger
  // request rather than looked up server-side from anything else.
  mtp_model_id?: string;
  // See RunConfig.main_gpu -- same field, forwarded through unchanged.
  main_gpu?: number;
  sweep: Omit<SweepConfig, "model_id">;
}

// --- llama.cpp build management ---

export interface LlamaCppAsset {
  name: string;
  download_url: string;
  size_bytes: number;
}

export interface LlamaCppRelease {
  tag: string;
  published_at: string;
  assets: LlamaCppAsset[];
}

export interface InstalledBuild {
  tag: string;
  asset_name: string;
  installed_at: number;
  active: boolean;
}

export interface HardwareInfo {
  platform: string;
  arch: string;
  cpu: { manufacturer: string; brand: string; flags: string[]; cores: number };
  gpu: { vendor: string; model: string }[];
  // Total system RAM in bytes (on Apple Silicon, total *unified* memory --
  // shared with the GPU, not a separate pool) -- see worker/src/hardware.ts's
  // detectHardware, si.mem().total. Optional so a worker process still
  // running old code (hasn't been restarted since this field was added)
  // keeps reporting a valid snapshot, just without this one value -- see
  // that same file's /hardware handler for why a restart is required at all.
  mem_total_bytes?: number;
}

// Which of a worker's detected GPUs (HardwareInfo.gpu) a given backend build
// can actually address as a --main-gpu/-mg target (see worker/src/bench.ts's
// buildArgs, which forwards a picked index straight through as `-mg
// <index>`). CUDA only ever enumerates NVIDIA devices and ROCm only AMD --
// so on a mixed-vendor box (e.g. an Intel iGPU alongside an NVIDIA card),
// HardwareInfo.gpu's raw index (whatever order systeminformation happens to
// enumerate controllers in) does NOT correspond to what the backend itself
// calls index 0/1/2..., and blindly forwarding it can silently select the
// wrong physical device or an out-of-range one entirely. Every other backend
// string (vulkan, sycl, openvino, ... or anything future/unrecognized -- see
// KNOWN_BACKENDS above, Backend accepts any string) is left unfiltered since
// this app has no specific vendor-visibility mapping for it and wrongly
// hiding a real device would be worse than not filtering at all. Used by
// both client/src/pages/NewRun.tsx (building the GPU picker and the index it
// sends as main_gpu) and worker/src/index.ts (resolving backend_device_name
// for whatever main_gpu a run actually picked) so both sides agree on what
// index N means.
export function backendVisibleGpus<T extends { vendor: string; model: string }>(
  gpu: T[],
  backend: Backend
): T[] {
  let filtered: T[];
  if (backend === "cuda") {
    filtered = gpu.filter((g) => /nvidia/i.test(g.vendor));
  } else if (backend === "rocm") {
    filtered = gpu.filter((g) => /amd|advanced micro devices/i.test(g.vendor));
  } else {
    return gpu;
  }
  // Defensive fallback -- e.g. a GPU whose vendor string systeminformation
  // reports unexpectedly shouldn't leave a picker (or backend_device_name)
  // with zero options; better to show/target something than nothing.
  return filtered.length > 0 ? filtered : gpu;
}

export interface WorkerCurrentRun {
  run_id: string;
  model_filename: string;
  started_at: number;
}

export interface WorkerLlamaCppInfo {
  worker_name: string;
  platform: string;
  arch: string;
  backend: Backend;
  installed: InstalledBuild[];
  active_tag: string | null;
  available: LlamaCppRelease[];
  update_available: boolean;
  // Best-effort, diagnostic only -- undefined if the worker predates this
  // field or /hardware couldn't be reached. Nothing here gates the
  // "available" list above; that's still filtered by platform/arch/backend
  // alone (see github-releases.ts) since current llama.cpp releases don't
  // ship CPU-tiered variants for these flags to choose between anyway.
  hardware?: HardwareInfo;
  // Same best-effort story as hardware above, sourced from the worker's
  // /health instead: undefined if that fetch failed rather than the main
  // /llama-cpp one, rather than failing the whole response over it.
  busy?: boolean;
  current_run?: WorkerCurrentRun | null;
  // True while the worker has finished its current sweep item and is
  // holding before starting the next one (see worker/src/index.ts's
  // /run/pause) -- only meaningful while busy is true.
  paused?: boolean;
}

// --- Hugging Face model search ---

export interface HfRepoSearchResult {
  id: string;
  downloads: number;
  likes: number;
  // Repo creation date reported by HF's search API (ISO string), null if
  // that field was missing from the response -- powers the "Newest" sort in
  // the client's HF search UI.
  created_at: string | null;
}

export interface HfFileEntry {
  path: string;
  size_bytes: number;
  quant: string | null;
}

export interface DownloadProgress {
  bytes: number;
  total: number | null;
  started_at: number;
}

// Worker -> server callback reporting a model download's terminal outcome --
// mirrors RunItemTerminalInput's role for /run (see worker/src/index.ts's
// POST /models/download, which now acks fast and runs the actual transfer in
// the background rather than blocking the whole HTTP round trip on it).
// Retried with backoff by the worker (safeReportDownloadResult), so
// repo.registerModel's upsert-on-id semantics matter: a retried "ok" payload
// must be safe to apply twice.
export interface ModelDownloadCallbackInput {
  worker: string;
  hf_repo: string;
  hf_file: string;
  ok: boolean;
  error?: string;
  sha256?: string;
  size_bytes?: number;
  n_layer?: number | null;
  mtp_layers?: number | null;
}
