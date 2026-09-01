// Sentinel error message the server uses whenever it can't reach a worker at
// all (connection refused, DNS failure, timeout, ...) -- shared so the
// client can recognize it and render a clean "Inaccessible" state instead of
// displaying it as a generic error string. See server/src/worker-errors.ts
// for where this gets produced.
export const WORKER_INACCESSIBLE_MESSAGE = "worker is inaccessible";

import type { EngineKind } from "./engineSpec.js";
import type { TensorLayerBreakdown } from "./vramEstimate.js";
import type { ProbeGranularity, ProbeMode } from "./probeLadder.js";

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

// Manually-maintained snapshot of AMD GPU model tokens ROCm officially
// supports, checked live against AMD's own docs this session (Aug 2026):
// rocm.docs.amd.com's Windows "system requirements" page (HIP SDK 7.2) and
// Linux "compatibility matrix" (ROCm 7.0) -- these two lists actually differ
// (ROCm-on-Windows supports a narrower set than Linux), but this app doesn't
// know a worker's exact driver/ROCm version, so the two are merged into one
// permissive union rather than trying to be precise per-platform. This WILL
// go stale as AMD updates support -- re-check those two pages if a "should
// this default to rocm" question comes up for a model not listed here.
const ROCM_SUPPORTED_AMD_GPU_MARKERS = [
  // RDNA3 / RDNA4 consumer (Radeon RX)
  "7600", "7650 gre", "7700 xt", "7800 xt", "7900 xt", "7900 xtx",
  "9060", "9070",
  // Radeon PRO / workstation
  "w6800", "w7700", "w7800", "w7900", "v620", "v710", "r9700", "r9600",
  // Instinct (data center)
  "mi100", "mi200", "mi300", "mi325x", "mi350x", "mi355x",
  // Ryzen AI Max APUs
  "ryzen ai max",
];

function isRocmSupportedAmdGpu(model: string): boolean {
  const lower = model.toLowerCase();
  return ROCM_SUPPORTED_AMD_GPU_MARKERS.some((marker) => lower.includes(marker));
}

// Best-effort default when a worker's config.json doesn't pin a `backend`.
// Only a guess -- an explicit value in config.json always overrides this,
// see worker/src/index.ts's startup logic. Picks among KNOWN_BACKENDS above;
// a user who wants something this heuristic doesn't know about (or gets
// wrong for their exact hardware) can always set `backend` explicitly
// instead -- see server/src/github-releases.ts's generic asset matching,
// which accepts any backend string, not just these.
//
// macOS is a hard special case, confirmed against a real llama.cpp release's
// asset list (ggml-org/llama.cpp): it ships exactly "macos-arm64"/"macos-x64"
// with no vulkan/cuda/rocm variant whatsoever -- Metal is just baked into
// that one build (macOS only has the one GPU API to begin with, unlike
// Windows/Linux which support several competing ones). Reporting "vulkan"
// there (as an earlier version of this function did, based only on "a GPU
// exists") pointed a Mac at zero installable assets. So macOS always
// resolves to `cpu` regardless of GPU vendor -- the bucket its one real
// build actually falls into.
//
// win32/linux: NVIDIA gets `cuda` (broad CUDA compatibility across
// virtually every NVIDIA GPU, unlike AMD's much narrower ROCm support below
// -- note current ubuntu releases ship no cuda asset at all, so this guess
// can come up empty on a Linux+NVIDIA worker specifically until upstream
// adds one back). AMD gets `rocm` only when the specific detected model is
// one ROCm actually supports (see isRocmSupportedAmdGpu above); otherwise
// `vulkan`, same as any other detected GPU (Intel, unrecognized AMD, ...) --
// the broadest cross-vendor option that reliably works. No GPU at all falls
// back to `cpu`.
//
// Originally worker-only (worker/src/hardware.ts, still re-exported from
// there for that file's own callers) -- moved here so client/src/pages/
// NewRun.tsx can run the exact same per-GPU logic client-side, instantly,
// when a user picks a specific GPU: see backendVisibleGpus below for the
// inverse operation (given a backend, which already-detected GPUs can it
// see) and shared/types.ts's own TriggerPayload.main_gpu_backend for how a
// mismatch between the two gets resolved into an install+run override.
export function detectBackend(platform: string, gpu: { vendor: string; model: string }[]): Backend {
  if (platform === "darwin") return "cpu";
  const text = gpu.map((g) => `${g.vendor} ${g.model}`.toLowerCase()).join(" ");
  if (text.includes("nvidia")) return "cuda";
  const amdGpu = gpu.find((g) => /amd|advanced micro devices/i.test(g.vendor));
  if (amdGpu && isRocmSupportedAmdGpu(amdGpu.model)) return "rocm";
  if (gpu.length > 0) return "vulkan";
  return "cpu";
}

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

// M6 sensor provenance -- worst-source tracking across an item's samples,
// exactly like the memory streams above.
export const SENSOR_MEASUREMENT_SOURCES = [
  "sensor_nvidia_smi",
  "sensor_rocm_smi",
  "sensor_amdgpu_hwmon",
  "sensor_lhm",
] as const;
export type SensorMeasurementSource = (typeof SENSOR_MEASUREMENT_SOURCES)[number];

export const WORKER_CAPABILITIES = ["benchmark", "probe-v1", "quality-v1", "curve-v1"] as const;
export type WorkerCapability = (typeof WORKER_CAPABILITIES)[number];

// BENCHMARKING_PLAN_V8.md §0.1 -- increments whenever measurement semantics
// change (warmup added, TTFT derived→streamed, suspect filter applied to both
// sides of a pair, stddev floor). Stamped by the worker on every ingested row;
// rows of different vintages are surfaced together but never averaged together
// in scoring or curves.
export const METHOD_VERSION = 1;

// N1's choreographed context-curve points (cold timed prefill + warm-repeat
// statistics) are a real semantics change under §0.1, so they stamp this
// instead -- which is also what keeps ordinary runtime rows' warm-biased TTFT
// out of curves without a dedicated marker column.
export const CURVE_METHOD_VERSION = 2;

// N2 added 'probe'; N4's quality measurement gets its own kind too, for the
// same reason -- a single perplexity measurement is not a sweep, and folding
// it into the 'standalone' (NULL) bucket would make it collide with the
// duplicate-trigger guard against unrelated standalone runs on the same
// (user, model, worker) triple.
export const RUN_KINDS = ["tuning", "refine", "sweep", "runtime", "probe", "quality"] as const;
export type RunKind = (typeof RUN_KINDS)[number];

// §0.5 budgets.
export const MAX_SWEEP_ITEMS = 20_000;
export const WARN_SWEEP_ITEMS = 4_096;
export const MAX_AXIS_VALUES = 64;
export const MIN_REPEATS = 1;
export const MAX_REPEATS = 25;
export const MAX_CHAIN_DEPTH = 3;
export const MAX_ACTIVE_ROOTS_PER_USER = 3;
export const CHAIN_WALL_CLOCK_MS = 48 * 60 * 60 * 1000;

// §0.10 closed registry of caveat flags. Flagged rows are kept, never deleted;
// consumers decide exclusions (e.g. scoring's TG reference-depth rule skips
// `swa` rows).
export const CAVEAT_FLAGS = [
  "swa",
  "context_unverified",
  "kv_estimate_rough",
  "spec_pair_prompt_mismatch",
  "thermally_throttled",
  "cache_evicted",
  "context_shift",
] as const;
export type CaveatFlag = (typeof CAVEAT_FLAGS)[number];

export function parseCaveatFlags(text: string | null | undefined): CaveatFlag[] {
  if (!text) return [];
  try {
    const parsed = JSON.parse(text) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((f): f is CaveatFlag => typeof f === "string" && (CAVEAT_FLAGS as readonly string[]).includes(f));
  } catch {
    return [];
  }
}

export const SPEEDUP_STATUSES = ["ok", "unverified", "unavailable"] as const;
export type SpeedupStatus = (typeof SPEEDUP_STATUSES)[number];

export type TestType = "pp" | "tg" | "pg";

export type ModelSource = "local" | "huggingface";

export interface ModelMetadata {
  arch?: string;
  quant?: string;
  trained_ctx?: number;
  // GGUF's <arch>.context_length -- read alongside n_layer at download time.
  // Anchors M2's target-context clamp and N1's sizing-ladder ceiling.
  // Undefined for models registered before this existed.
  // Transformer layer count (GGUF's <architecture>.block_count), read from
  // the file header on download -- see worker/src/gguf.ts. Undefined for
  // models registered before this existed, or for manually-registered
  // "local" models the app never had file bytes to parse.
  n_layer?: number;
  // KV geometry consumed by shared/vramEstimate.ts's maxAffordableContext
  // (M1) -- all read from the GGUF header at download time where present.
  n_head_kv?: number;
  head_dim_k?: number;
  head_dim_v?: number;
  n_embd?: number;
  n_head?: number;
  sliding_window?: number;
  // SWA geometry beyond the plain window size -- see worker/src/gguf.ts's
  // GgufInfo.sliding_window_pattern/shared_kv_layers for what each means and
  // shared/vramEstimate.ts's swaGlobalLayersInSuffix/sharedKvLayers handling
  // for how they refine the SWA VRAM estimate beyond its Gemma-3-style
  // hardcoded fallback. Undefined for models registered before these
  // existed, or read from a header that writes sliding_window_pattern as a
  // per-layer array rather than a scalar (see that field's own doc comment).
  sliding_window_pattern?: number;
  shared_kv_layers?: number;
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
  // Real per-tensor weight-byte breakdown, read from the GGUF file's own
  // tensor_info section (worker/src/gguf.ts's readGgufInfo) -- see
  // shared/vramEstimate.ts's TensorLayerBreakdown for the shape and
  // placeWeightBytes for how it's consumed. Drives the accurate replacement
  // for the old flat file-size/layer-count VRAM estimate: undefined for a
  // model registered before this existed (falls back to that flat average),
  // or a "local" entry never hash-matched/downloaded through the app.
  tensor_layer_bytes?: TensorLayerBreakdown;
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
  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4) -- populated by the ROUTE
  // from req.user.id, never by the caller directly (undefined in
  // single-tenant mode, matching every other userId in this app). Not a
  // second positional parameter on registerModel itself -- keeping that
  // function's signature exactly as-is is the plan's own explicit
  // requirement (§4.3: "listModels/getModel/registerModel keep their
  // current signatures -- the catalog is global").
  created_by?: string;
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
  // Optional COUPLED (K,V) pairs, e.g. shared/goals.ts's KV_PRESET_PAIRS.
  // When present and non-empty, expandSweep iterates these pairs directly
  // instead of the full cross product of cache_type_k x cache_type_v --
  // needed because a curated preset list is deliberately NOT rectangular
  // (e.g. it can include q8_0/f16 without also including every other
  // K x V combination those two axes would otherwise imply). cache_type_k/
  // cache_type_v should still be set to the pairs' own K/V values for
  // callers that only read the plain axis arrays (e.g. display labels);
  // they no longer independently drive expansion once pairs are given.
  cache_type_pairs?: Array<readonly [string, string]>;
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
  // llama.cpp's --n-cpu-moe N -- see shared/sweep.ts's SweepItem.n_cpu_moe.
  n_cpu_moe: number[];
  // KV-prefill depth (llama-bench's -d) -- llama-bench only; sweep validation
  // rejects n_depth > 0 for any server-engine pair (§0.2's depth rule).
  // Optional so legacy payloads keep validating; expandSweep defaults [0].
  n_depth?: number[];
  // N5 -- concurrent streaming slots (server engine only; llama-bench has no
  // request lifecycle). Optional so legacy payloads keep validating.
  concurrency?: number[];
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
  // Index into backendVisibleGpus(hardware.gpu, effectiveBackend) -- NOT a
  // raw index into HardwareInfo.gpu -- where effectiveBackend is
  // main_gpu_backend below when set, otherwise the worker's own default
  // backend. Restricts this run to that one GPU on a multi-GPU worker,
  // forwarded to llama-bench/llama-server as `-sm none -mg <index>` (see
  // worker/src/bench.ts and worker/src/serverBench.ts's buildArgs).
  // Undefined means "let llama.cpp use its own default" (split across every
  // visible GPU) -- same as before this field existed. Meaningless (ignored)
  // on a single-GPU or cpu-only worker.
  main_gpu?: number;
  // Set only when main_gpu names a GPU the worker's own default backend
  // can't see at all (e.g. an Intel iGPU on a worker whose default backend
  // is cuda) and the user confirmed switching for this run -- see
  // client/src/pages/NewRun.tsx's GPU picker and server/src/routes/runs.ts's
  // ensureBuildForBackend. Tells the worker to run this sweep against a
  // *different* installed build than whatever's currently its persisted
  // "active" one, without changing that persisted default -- see worker/
  // src/index.ts's /run handler, main_gpu_build_tag on the wire payload it
  // actually receives (this field alone isn't enough; the server resolves
  // and installs the specific tag before dispatch). Undefined means "use
  // the worker's own default backend," same as before this field existed.
  main_gpu_backend?: Backend;
  sweep: Omit<SweepConfig, "model_id">;
  // M2 -- the questionnaire's answers, stored verbatim so results stay
  // reproducible without reference to whatever the defaults were that day.
  // A skipped questionnaire leaves this undefined (byte-identical legacy
  // payload).
  goals?: import("./goals.js").GoalsConfig;
  // N6 -- steady-state option: discard the first M repeats at scoring time.
  // Recorded in configuration (never post-hoc trimming outside it); only
  // offered when repeats >= M + 3 so post-discard n still satisfies n >= 3.
  discard_first_repeats?: number;
  // N2 -- the probe spec this run verifies, stored verbatim so the ingestion
  // route derives the KV pair and placement from the RUN, never the payload.
  probe?: ProbeTriggerSpec;
  // N4 -- the quality spec this run measures, stored verbatim as a record of
  // intent (the ingestion route independently validates the worker's own
  // reported ctx/kv/dataset -- see QualityTriggerSpec's doc comment).
  quality?: QualityTriggerSpec;
  // N1 -- the context-curve point(s) this runtime run measures. One run
  // measures every uncovered ladder cell the "Measure missing points"
  // trigger priced together, so a click never fans out into N separate
  // runs that would collide with the §0.5 duplicate-trigger guard after
  // the first.
  curve_point?: CurvePointSpec;
  // N5 -- the concurrency ladder this runtime run walks.
  knee?: KneeSpec;
  // §0.5 -- chain depth of this run under its root (roots are depth 1).
  chain_depth?: number;
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
  // §0.5 -- denormalized at creation, immutable. Points at the run itself
  // for standalone runs; chain children point at their root. Chain-scoped
  // reads are indexed equality predicates (idx_runs_root).
  root_run_id?: string | null;
  // §0.5 -- 'tuning' | 'refine' | 'sweep' | 'runtime', NULL = standalone.
  // N2 adds 'probe'. Undefined on runs predating the column.
  kind?: RunKind | null;
  // N3 -- groups model-vs-model comparison members; null otherwise.
  comparison_id?: string | null;
  worker_name: string;
  // Which `workers` row dispatched this run -- see MULTIUSER_PLAN.md §1.2.
  // Undefined for runs predating the workers table, or whose worker was
  // later deleted (ON DELETE SET NULL). worker_name above stays the
  // point-in-time snapshot the export reads directly, independent of this.
  worker_id?: string;
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
  // KV-prefill depth (llama-bench -d) -- always 0 on server-engine rows
  // (§0.2's depth rule). NULL on rows predating the column; reads as 0.
  n_depth?: number | null;
  n_threads: number;
  // §0.1 -- stamped by the worker on every ingested row. NULL = pre-v8-core
  // methodology. Rows of different vintages are never averaged together in
  // scoring or curves.
  method_version?: number | null;
  // §0.8 -- the canonical configuration hash (§0.4), which excludes spec
  // fields; the exclusion is what makes twin joins work. Written by the
  // server wherever the twin join needs it (speculative + baseline pairs).
  config_hash?: string | null;
  prompt_offset?: number | null;
  spec_type?: string | null;
  spec_n_max?: number | null;
  spec_n_min?: number | null;
  speedup?: number | null;
  speedup_status?: SpeedupStatus | null;
  caveat_flags?: CaveatFlag[];
  // N5 -- concurrent streaming slots this row measured (1 = solo). NULL on
  // ordinary rows.
  concurrency?: number | null;
  // M6 -- adapter peak temperature and lowest sampled sclk, plus the sample
  // series itself (~6s cadence, JSON number[]; prunable 30 days after
  // ingest -- detection already ran at ingest time, aggregates persist).
  gpu_temp_c_max?: number | null;
  gpu_clock_mhz_min?: number | null;
  gpu_clock_samples?: number[] | null;
  // N6 -- llama.cpp startup-banner ISA provenance for CPU-bound rows.
  cpu_isa?: string | null;
  // MEASURED streamed time-to-first-token, milliseconds. Null on llama-bench
  // rows (no request lifecycle) and on rows predating the streamed clock --
  // the UI falls back to the derived n_prompt/pp estimate there and says so.
  // ttft_n is the sample count: a choreographed curve point (N1) is
  // single-shot by construction and renders as such, never as a p50/p95.
  ttft_ms_p50?: number | null;
  ttft_ms_p95?: number | null;
  ttft_n?: number | null;
  e2e_ms_mean?: number | null;
  // N7 -- non-null on a row that arrived through an import. Such rows are
  // badged wherever they appear and never merge into local profile scoring
  // unless the importer opted in for that specific bundle.
  imported_bundle_id?: string | null;
  import_opt_in?: boolean;
  // Denormalized from the parent run at insert time so the curve read path
  // (N1) can be an index range scan over its exact grouping keys
  // (idx_results_curve) rather than a five-way JOIN. NULL on legacy rows.
  worker_id?: string | null;
  llama_cpp_build?: string | null;
  engine?: EngineKind | null;
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
  // The requested --n-cpu-moe value -- see shared/sweep.ts's
  // SweepItem.n_cpu_moe. Request-only, like every other field on this row
  // except n_gpu_layers/n_gpu_layers_draft: llama.cpp prints no equivalent
  // confirmation line for how many layers' MoE experts actually landed on
  // CPU, only the "offloaded X/Y layers to GPU" line those two fields read.
  n_cpu_moe: number;
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
  // Whole-adapter ("every process on the GPU combined") VRAM usage sampled
  // around the test -- avg and peak over the same interval that produced
  // vram_avg_mib/vram_peak_mib above. Always whole-GPU, never isolated to
  // the benchmark process; distinct from the gpu_memory_process_* pair
  // below, which is the benchmark process's own reading over that same
  // interval. All optional + undefined on rows from workers predating these
  // fields (older rows fall back to the vram_avg/vram_peak hybrid in the UI
  // and read "n/a" in the CSV) -- see the accuracy/source fields for the
  // honest labeling of what each stream actually measures.
  gpu_memory_used_avg_mib?: number | null;
  gpu_memory_used_peak_mib?: number | null;
  gpu_memory_used_avg_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_used_avg_source?: GpuMemoryMeasurementSource | null;
  gpu_memory_used_peak_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_used_peak_source?: GpuMemoryMeasurementSource | null;
  // The benchmark process's own VRAM usage (worker/src/vram.ts's per-backend
  // per-process readers: nvidia-smi --query-compute-apps, Windows' WDDM
  // "GPU Process Memory" counter, rocm-smi --showpids / amdgpu fdinfo).
  // Null + accuracy "unavailable" whenever the backend has no per-process
  // reading (Metal) or the driver never caught up to the process.
  gpu_memory_process_avg_mib?: number | null;
  gpu_memory_process_peak_mib?: number | null;
  gpu_memory_process_avg_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_process_avg_source?: GpuMemoryMeasurementSource | null;
  gpu_memory_process_peak_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_process_peak_source?: GpuMemoryMeasurementSource | null;
  // Whole-system RAM usage (every process combined, si.mem().active) sampled
  // over the same interval as ram_avg_mib/ram_peak_mib (which remain the
  // benchmark process's own figures). Plain numbers -- RAM has no
  // accuracy/source metadata, same as every other RAM figure on this row.
  ram_total_used_avg_mib?: number | null;
  ram_total_used_peak_mib?: number | null;
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
  // Worker-derived ESTIMATE of how many of gpu_layers_loaded's layers were
  // actually VRAM-resident -- computed on EVERY run (not just suspicious
  // ones) whenever llama.cpp's own post-allocation buffer report is
  // available (see worker/src/bench.ts's parseModelBufferSizes), or from the
  // VRAM-discrepancy heuristic on a run where that report was missing and
  // the claim looked wrong. llama.cpp's own count above reflects buffer
  // *assignment*, never residency, and Windows' NVIDIA driver can silently
  // back those assignments with system RAM (CUDA sysmem fallback) -- this is
  // the "~" figure shown next to such a claim so logs/UI/CSV can say
  // "claimed 33/33, ~=30 actually resident" instead of endorsing the claim.
  // Null whenever nothing was claimed (ngl=0 / cpu backend) or no buffer
  // report/baseline existed to derive a figure from. Base model only -- the
  // draft companion has its own gpu_layers_resident_est_draft.
  gpu_layers_resident_est?: number | null;
  // The MTP/draft companion model's own actually-resident estimate -- see
  // gpu_layers_resident_est above for the general meaning; this is the draft
  // model's own figure, derived only from its separately-attributed buffer
  // lines (the external VRAM sample can't split base vs draft). Optional +
  // undefined on every non-MTP row, same as gpu_layers_loaded_draft.
  gpu_layers_resident_est_draft?: number | null;
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
  // Optional (version-skew tolerant): KV-prefill depth, llama-bench only.
  n_depth?: number;
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
  // See ResultRow.n_cpu_moe above. Optional like n_gpu_layers_draft above --
  // an older worker that predates this column simply won't send it,
  // defaulted to 0 server-side (server/src/db/repo.ts's buildResultRow)
  // rather than required.
  n_cpu_moe?: number;
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
  // Whole-adapter and per-process VRAM usage avg/peak, plus whole-system RAM
  // usage avg/peak -- see the matching fields on ResultRow above. All
  // optional (an older worker that predates these fields simply won't send
  // them), and each value is nullable when the stream couldn't be measured
  // on that backend (e.g. no per-process reading on Metal).
  gpu_memory_used_avg_mib?: number | null;
  gpu_memory_used_peak_mib?: number | null;
  gpu_memory_used_avg_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_used_avg_source?: GpuMemoryMeasurementSource | null;
  gpu_memory_used_peak_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_used_peak_source?: GpuMemoryMeasurementSource | null;
  gpu_memory_process_avg_mib?: number | null;
  gpu_memory_process_peak_mib?: number | null;
  gpu_memory_process_avg_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_process_avg_source?: GpuMemoryMeasurementSource | null;
  gpu_memory_process_peak_accuracy?: GpuMemoryAccuracyLevel;
  gpu_memory_process_peak_source?: GpuMemoryMeasurementSource | null;
  ram_total_used_avg_mib?: number | null;
  ram_total_used_peak_mib?: number | null;
  gpu_layers_loaded: number | null;
  total_model_layers: number | null;
  // See the matching fields on ResultRow above -- the MTP/draft companion
  // model's own actual offload, distinct from gpu_layers_loaded/
  // total_model_layers above (always the base model's).
  gpu_layers_loaded_draft?: number | null;
  total_model_layers_draft?: number | null;
  // Optional (like n_gpu_layers_draft/n_cpu_moe above) so a version-skewed
  // worker that predates this field simply won't send it -- and null when a
  // current worker had nothing to estimate (no offload claim, or no buffer
  // report to derive the figure from). Sent on every ordinary row whenever
  // an actual-resident estimate was computable -- see
  // ResultRow.gpu_layers_resident_est and shared/vramEstimate.ts.
  gpu_layers_resident_est?: number | null;
  // See the matching fields on ResultRow above -- the draft companion's own
  // actually-resident estimate, sent only by the llama-server/MTP path and
  // only when the draft model's own buffer lines were captured. Optional
  // like gpu_layers_resident_est above (version-skew tolerant).
  gpu_layers_resident_est_draft?: number | null;
  sample_count?: number;
  suspect_count?: number;
  suspect_samples?: number[];
  repeat_samples?: number[];
  spec_drafted?: number;
  spec_accepted?: number;
  // §0.1 -- worker-stamped methodology version. Absent on an older worker;
  // the server stores NULL then (pre-v8-core methodology).
  method_version?: number;
  // §0.8 -- pinned prompt-content offset each server row records (the
  // working offset from serverBench's retry ladder). Both sides of a twin
  // join must match on it.
  prompt_offset?: number | null;
  // §0.8 -- speculative configuration actually in force for this row.
  spec_type?: string | null;
  spec_n_max?: number | null;
  spec_n_min?: number | null;
  // Set by the llama-server path when a tg reading was recovered through
  // the wall-clock fallback rather than the server's own timers -- any
  // wall-clock-fallback side makes its speedup `unverified` (§0.8).
  wall_clock_fallback?: boolean;
  // M6 -- worker-derived at item end from its own sample buffer; the server
  // stores these verbatim and never re-derives them (one writer per column).
  gpu_temp_c_max?: number | null;
  gpu_clock_mhz_min?: number | null;
  gpu_clock_samples?: number[] | null;
  // N6 -- llama.cpp startup-banner ISA provenance for CPU-bound rows.
  cpu_isa?: string | null;
  // N5 -- concurrent streaming slots this row measured.
  concurrency?: number;
  // Measured streamed TTFT / end-to-end latency (llama-server path only).
  ttft_ms_p50?: number | null;
  ttft_ms_p95?: number | null;
  ttft_n?: number | null;
  e2e_ms_mean?: number | null;
  // §0.10 -- worker-derived caveat flags from the closed registry.
  caveat_flags?: CaveatFlag[];
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
  | "cancelled"
  // §0.7 -- an unsupported flag disables its axis (items become skipped with
  // a reason) rather than failing every item. Never measured; scoring's
  // eligibility gates treat skipped items as non-disqualifying.
  | "skipped";

const TERMINAL_RUN_ITEM_STATUSES = ["done", "failed", "failed_oom", "cancelled", "skipped"] as const;
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
  n_depth: number;
  concurrency: number;
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
  // "skipped" (§0.7): an unsupported flag disables its axis rather than
  // failing every item -- never measured, so scoring's eligibility gates
  // treat it as non-disqualifying.
  status: TerminalRunItemStatus;
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
  // The `workers.id` (UUID) to run against -- machines are identified by id,
  // not name, from Stage 1 onward (MULTIUSER_PLAN.md §1.16). A machine's
  // display name can be renamed without breaking anything that referenced it.
  worker_id: string;
  // See RunConfig.mtp_model_id -- same field, forwarded through the trigger
  // request rather than looked up server-side from anything else.
  mtp_model_id?: string;
  // See RunConfig.main_gpu -- same field, forwarded through unchanged.
  main_gpu?: number;
  // See RunConfig.main_gpu_backend -- same field, forwarded through
  // unchanged.
  main_gpu_backend?: Backend;
  sweep: Omit<SweepConfig, "model_id">;
  // M2 -- questionnaire answers, echoed on the created root run's config.
  goals?: import("./goals.js").GoalsConfig;
  // N6 -- steady-state discard, recorded in the stored configuration.
  discard_first_repeats?: number;
  // §0.5 -- run kind. Absent = standalone (NULL), byte-identical to legacy
  // payloads. 'probe' rides the `probe` block below; 'runtime' rides
  // `runtime_spec`.
  kind?: RunKind;
  // N2 -- probe-run trigger block (kind must be "probe"). Refused unless the
  // target worker advertises `probe-v1`.
  probe?: ProbeTriggerSpec;
  // N4 -- quality-run trigger block (kind must be "quality"). Refused unless
  // the target worker advertises `quality-v1`.
  quality?: QualityTriggerSpec;
  // N1 -- one context-curve point per runtime-kind run (kind must be
  // "runtime"). Refused unless the target worker advertises `curve-v1`.
  curve_point?: CurvePointSpec;
  // N5 -- the concurrency ladder (kind must be "runtime"). Rows land as
  // ordinary results with `concurrency` set; the knee is derived on read,
  // never a stored verdict.
  knee?: KneeSpec;
  // N3 -- groups comparison members created through this same route; the
  // caller supplies a stable id shared by every member.
  comparison_id?: string;
  // §0.5 chains -- links this run into an existing root's chain (tune→refine
  // →sweep); the server resolves the parent's root and enforces depth ≤ 3.
  parent_run_id?: string;
  // N2 batching -- attaches this probe run as a sibling under an existing
  // probe root (kind must be "probe"; the target run must be kind "probe"
  // too), so several search modes fired from one "Run test" click collapse
  // into one row on the Runs list instead of each becoming its own root.
  // Deliberately separate from parent_run_id: a batch is a flat sibling
  // group, not a tuning->refine->sweep chain, so it carries none of that
  // path's chain_depth bookkeeping or its MAX_CHAIN_DEPTH limit -- a 5-mode
  // batch would otherwise hit that cap on its 4th member.
  probe_batch_root_id?: string;
}

export interface ProbeTriggerSpec {
  candidate_ctx: number;
  placement: { ngl: number; n_cpu_moe?: number; slots?: number };
  kv_pair: [string, string];
  // N2's search strategy -- see shared/probeLadder.ts. Both optional so an
  // older client keeps working: absent means the single-axis context search
  // over slider stops, which is what a probe did before modes existed.
  mode?: ProbeMode;
  granularity?: ProbeGranularity;
}

// N4 -- what a quality run was triggered to measure, echoed onto the run's
// own config the same way probe/curve_point/knee are (a record of intent for
// the UI/logs). The ingestion route (POST .../quality-result) still validates
// the worker's own reported ctx/kv/dataset independently -- see its own doc
// comment for why that's the §0.12-observable boundary, not this spec.
export interface QualityTriggerSpec {
  ctx_tokens: number;
  kv_pair: [string, string];
  dataset_hash: string;
  dataset_license?: string | null;
}

export interface CurvePointSpec {
  // A single number for one point (kept for back-compat with stored
  // configs); an array measures every listed context within this ONE run,
  // one run_item per context, so pricing/enqueueing "missing points" never
  // has to split across runs.
  effective_ctx: number | number[];
  n_gen?: number;
  repeats?: number;
  placement?: { ngl: number; n_cpu_moe?: number };
  kv_pair?: [string, string];
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
  // CUDA runtime redistributables ("cudart-llama-bin-win-cuda-12.4-x64.zip"
  // etc.) keyed by the llama asset they belong to -- these zips carry the
  // cublas/cudart DLLs a Windows CUDA build needs next llama-bench.exe, and
  // without them a CUDA binary can't even load (it doesn't "fall back to
  // CPU", it dies with a missing-DLL error). Deliberately NOT part of
  // `assets` above: everything that iterates that list (install pickers,
  // assetMatchesWorker, the client's per-release rows) expects exactly one
  // installable binary per backend variant, and cudart is a companion
  // download resolved server-side at enqueue time (see buildInstallPayload).
  // Absent when a release ships no matching cudart zip (every non-Windows
  // platform today).
  cudart_assets?: Record<string, LlamaCppAsset>;
}

export interface InstalledBuild {
  tag: string;
  asset_name: string;
  installed_at: number;
  active: boolean;
  // Resolved local paths to the installed build's llama-bench/llama-server
  // binaries -- see worker/src/llama-builds.ts's InstalledBuildInfo, which
  // has always had these; they just weren't previously serialized onto the
  // wire-facing shape. The pull-queue worker (MULTIUSER_PLAN.md §1.9) needs
  // them to resolve a QueueJob's `llama_cpp_build` tag to an actual binary to
  // run, without a live round trip back to the server. There is no
  // `InstalledBuild.backend` field -- matching an installed build to a
  // requested backend is done by tag/asset_name (see
  // assetMatchesWorker), never by a stored backend string.
  bench_path?: string;
  server_path?: string;
  // Set when this build was installed together with its matching CUDA
  // runtime redistributable (cudart-*.zip extracted alongside the binaries) --
  // informational only, see InstallBuildJobPayload.cudart_name.
  cudart_name?: string;
}

// Walks model_dir recursively -- see worker/src/index.ts's listModelDirFiles.
// Promoted here (was worker-local) so the server can type the
// WorkerStatePush.model_files it now receives directly from the worker's own
// heartbeat/queue poll instead of fetching a worker's file list live.
export interface ModelDirFile {
  path: string;
  size_bytes: number;
  // SHA-256 of the file contents, computed by the worker on heartbeat.
  // Optional: absent for workers running old code that doesn't hash.
  sha256?: string;
  // Local model state, set by the worker's local-cache flow.
  // Absent for workers running old code (no state machine yet).
  state?: LocalModelState;
  // Hugging Face match metadata, resolved via the hash-lookup API.
  // Absent when no match found or worker is old code.
  // deleted is true once the worker has confirmed (via periodic
  // re-verification, see worker/src/model-scanner.ts) that this match's HF
  // source has since been removed -- absent/false for a live match or a
  // worker running old code.
  hf_match?: { repo_id: string; filename: string; revision: string; deleted?: boolean } | null;
  // Per-file GGUF header metadata (see worker/src/gguf.ts's readGgufInfo) --
  // read once at download/reconciliation time and carried on the heartbeat so
  // the server can populate a model's catalog metadata for files that were
  // dropped into model_dir by hand (never downloaded through the app). Null
  // (not absent) when the field was checked and came back empty; absent for
  // workers running old code that never reads GGUF headers per file.
  n_layer?: number | null;
  mtp_layers?: number | null;
  quant?: string | null;
  param_count?: number | null;
  // Trained context + KV-cache geometry, from the same GGUF header walk
  // (see ModelMetadata's matching fields for what each feeds downstream).
  // Same optional-null convention as the fields above: null = header key
  // absent (e.g. no sliding-window attention), absent = old worker code.
  trained_ctx?: number | null;
  n_head_kv?: number | null;
  head_dim_k?: number | null;
  head_dim_v?: number | null;
  n_embd?: number | null;
  n_head?: number | null;
  sliding_window?: number | null;
  // See ModelMetadata's matching fields for what each means.
  sliding_window_pattern?: number | null;
  shared_kv_layers?: number | null;
  // Real per-tensor weight-byte breakdown -- see ModelMetadata's matching
  // field. Same optional-null convention: null = tensor_info walk ran but
  // couldn't build one (e.g. n_layer unresolved), absent = old worker code.
  tensor_layer_bytes?: TensorLayerBreakdown | null;
}

export interface HardwareInfo {
  platform: string;
  arch: string;
  cpu: { manufacturer: string; brand: string; flags: string[]; cores: number };
  // vram_mb/vram_dynamic -- see worker/src/hardware.ts's detectHardware
  // (si.graphics().controllers[].vram/vramDynamic) for what these mean.
  // Optional for the same cross-version reason as mem_total_bytes below.
  gpu: { vendor: string; model: string; vram_mb?: number | null; vram_dynamic?: boolean }[];
  // Total system RAM in bytes (on Apple Silicon, total *unified* memory --
  // shared with the GPU, not a separate pool) -- see worker/src/hardware.ts's
  // detectHardware, si.mem().total. Optional so a worker process still
  // running old code (hasn't been restarted since this field was added)
  // keeps reporting a valid snapshot, just without this one value -- see
  // that same file's /hardware handler for why a restart is required at all.
  mem_total_bytes?: number;
  // Best-effort NVIDIA driver probe (nvidia-smi's banner, see worker/src/
  // hardware.ts) -- present only on boxes with an NVIDIA GPU where the probe
  // succeeded. `cuda_version` is the MAXIMUM CUDA toolkit version this
  // DRIVER can run (not what's installed system-wide) -- exactly the figure
  // needed to pick between llama.cpp's cuda-12.x/cuda-13.x build variants,
  // since a CUDA build only runs if its toolkit version is <= what the
  // driver supports. Optional so older workers keep validating.
  nvidia_driver?: { version: string; cuda_version: string } | null;
}

// --- CUDA asset-variant matching (which cudart/download variant fits a box) ---

export interface CudaVariant {
  major: number;
  minor: number;
}

// "llama-b10612-bin-win-cuda-12.4-x64.zip" -> {major:12, minor:4}; null for
// every non-CUDA asset (cpu/vulkan/rocm/...). Tolerant to where "cuda"
// appears in the name, but requires a major.minor pair -- bare "cuda" tokens
// don't exist in real release names.
export function extractCudaVariant(assetName: string): CudaVariant | null {
  const m = /(?:^|[-_.])cuda-(\d+)\.(\d+)(?:[-_.]|$)/i.exec(assetName);
  return m ? { major: Number(m[1]), minor: Number(m[2]) } : null;
}

// Whether an installed driver can run a CUDA build of the given toolkit
// version. driverCudaVersion is nvidia-smi's "CUDA Version: X.Y" -- the max
// toolkit the driver itself supports (NOT whatever runtime happens to be
// installed). Within one major version, newer-minor builds still run on
// older-minor drivers via CUDA's forward-minor compatibility guarantees, so
// the comparison that matters is: build's (major, minor) <= driver's.
// Unknown/unparseable driver info conservatively reports true -- better to
// offer everything than to wrongly hide installable builds; the server-side
// sort (sortAssetsForWorker) keeps the safest variant first in that case.
export function cudaDriverSupports(driverCudaVersion: string | null | undefined, v: CudaVariant): boolean {
  if (!driverCudaVersion) return true;
  const m = /^(\d+)\.(\d+)/.exec(driverCudaVersion.trim());
  if (!m) return true;
  const dmaj = Number(m[1]);
  const dmin = Number(m[2]);
  return v.major < dmaj || (v.major === dmaj && v.minor <= dmin);
}

// Orders a release's already platform/backend-filtered assets best-first for
// one specific machine:
//   1. CUDA variants the driver can run -- newest toolkit first, or OLDEST
//      first when no driver info is known (an unprobed box is likelier to be
//      old/misconfigured than bleeding-edge, and cuda-12.x loads on any
//      driver >= ~527 while cuda-13.x needs >= ~580, so oldest-first is the
//      safe default)
//   2. everything else, original order preserved
//   3. CUDA variants too new for the driver, oldest first
// Callers that take assets[0] (install pickers, auto-install at run time)
// then get the right variant with no further logic: e.g. with a CUDA-12.7
// driver, cuda-13.3 drops below cuda-12.4 instead of being picked blind.
// With no CUDA assets at all this is the identity (a fresh array copy).
export function sortAssetsForWorker<T extends LlamaCppAsset>(assets: T[], driverCudaVersion?: string | null): T[] {
  const known = typeof driverCudaVersion === "string" && /^(\d+)\.(\d+)/.test(driverCudaVersion.trim());
  const rank = (a: T): number => {
    const v = extractCudaVariant(a.name);
    if (!v) return 1;
    if (!known) return 0; // unknown driver: every variant is a candidate
    return cudaDriverSupports(driverCudaVersion, v) ? 0 : 2;
  };
  // Newest-first once the driver is known, oldest-first when it isn't.
  // Within the incompatible tail (rank 2) it's always oldest-first -- if
  // nothing can run, the closest-to-running variant is the least useless
  // one to look at.
  const dirFor = (r: number): 1 | -1 => (r === 0 && !known ? 1 : r === 0 ? -1 : 1);
  return [...assets].sort((x, y) => {
    const rx = rank(x);
    const ry = rank(y);
    if (rx !== ry) return rx - ry;
    if (rx === 1) return 0;
    const dir = dirFor(rx);
    const vx = extractCudaVariant(x.name)!;
    const vy = extractCudaVariant(y.name)!;
    if (vx.major !== vy.major) return vx.major > vy.major ? dir : -dir;
    if (vx.minor !== vy.minor) return vx.minor > vy.minor ? dir : -dir;
    return 0;
  });
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

// Cached free-VRAM reading -- worker/src/index.ts's collectState() computes
// this on every heartbeat/queue poll while idle (skipped while busy: not
// meaningful for pre-flight fit-checking, and avoids a second concurrent
// readGpuMemory call alongside MemorySampler's own). Server persists it on
// the worker row (workers.vram_json) and GET /api/workers/:id/vram just
// serves that cached value -- no server->worker proxy exists post-Stage-1
// (MULTIUSER_PLAN.md §1.11), so this can be a poll-cycle stale (up to
// HEARTBEAT_INTERVAL_MS) rather than truly live. Consumed by NewRun.tsx's
// pre-flight VRAM-fit banner (see shared/vramEstimate.ts). Deliberately NOT
// the same type as worker/src/sampler.ts's FreeMemoryBaseline (which this
// mirrors field-for-field) -- keeping the wire contract independent means
// sampler.ts itself needs no changes, same reasoning as HardwareInfo above
// being defined twice rather than shared across the worker/client boundary.
export interface WorkerVramInfo {
  ok: true;
  backend: Backend;
  ram_free_before_mib: number;
  vram_free_before_mib: number | null;
  vram_free_before_accuracy: GpuMemoryAccuracyLevel;
  vram_free_before_source: GpuMemoryMeasurementSource | null;
  system_memory_total_mib: number | null;
  gpu_memory_total_mib: number | null;
  gpu_memory_total_accuracy: GpuMemoryAccuracyLevel;
  gpu_memory_total_source: GpuMemoryMeasurementSource | null;
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

// Worker -> server callback reporting a model download's terminal outcome --
// mirrors RunItemTerminalInput's role for /run (see worker/src/index.ts's
// POST /models/download, which now acks fast and runs the actual transfer in
// the background rather than blocking the whole HTTP round trip on it).
// Retried with backoff by the worker (safeReportDownloadResult), so
// repo.registerModel's upsert-on-id semantics matter: a retried "ok" payload
// must be safe to apply twice.
export interface ModelDownloadCallbackInput {
  worker: string;
  // Needed for worker-auth.ts's Stage 1 (WORKER_SHARED_TOKEN) fallback path,
  // which resolves worker identity from the body's machine_id -- every other
  // worker->server call already includes this (WorkerStatePush.machine_id);
  // this route was the one place it got missed, which silently broke the
  // download callback for any worker still on shared-token auth (a Stage 3
  // session-token worker never hits this fallback at all, since
  // authenticateWorker tries the session first). Optional only so an
  // old-not-yet-updated worker binary doesn't fail request validation.
  machine_id?: string;
  hf_repo: string;
  hf_file: string;
  ok: boolean;
  error?: string;
  sha256?: string;
  size_bytes?: number;
  n_layer?: number | null;
  mtp_layers?: number | null;
  // Quant code read from the downloaded file's own general.file_type header
  // (see worker/src/gguf.ts's quantLabelFromFileType), independent of
  // hf_file's naming -- some real files (e.g. unsloth's MTP drafters, named
  // like "mtp-gemma-4-E2B-it.gguf") carry no quant token in their filename
  // at all, so ModelMetadata.quant would otherwise stay unset for them.
  quant?: string | null;
  // Total parameter count read from the downloaded file's own tensor_info
  // section (see worker/src/gguf.ts's readGgufInfo) -- per-FILE and
  // authoritative, unlike HF's repo-level gguf.total which reports one
  // (often the first) file's count for an entire repo and is simply wrong
  // for a multi-model repo (e.g. Ex0bit's PRISM-DQ repos shipping
  // 0.8B/2B/4B/9B files under one repo id). When present the server stores
  // it straight into ModelMetadata.param_count instead of calling
  // getHfGgufMeta.
  param_count?: number | null;
  // Trained context + KV geometry from the same GGUF header read -- stored
  // straight into ModelMetadata (feeds M2's target-context clamp, N1's
  // sizing ladder, and the Benchmark page's model card). Optional/null so an
  // older worker binary that doesn't send them keeps validating.
  trained_ctx?: number | null;
  n_head_kv?: number | null;
  head_dim_k?: number | null;
  head_dim_v?: number | null;
  n_embd?: number | null;
  n_head?: number | null;
  sliding_window?: number | null;
  // See ModelMetadata's matching fields for what each means.
  sliding_window_pattern?: number | null;
  shared_kv_layers?: number | null;
  // Real per-tensor weight-byte breakdown from the same GGUF header read --
  // see ModelMetadata.tensor_layer_bytes.
  tensor_layer_bytes?: TensorLayerBreakdown | null;
}

// --- Pull queue (MULTIUSER_PLAN.md Stage 1) ---
//
// The worker pulls all work over one outbound connection and pushes all
// state back -- no server->worker HTTP. A machine is identified by a
// server-assigned `id` (see the `workers` table); Stage 3 layers real
// enrolment/ownership on top of the same row, so this type only carries what
// Stage 1 actually populates.

export interface Worker {
  id: string;
  machineId: string;
  displayName: string;
  hostname: string | null;
  backend: Backend | null;
  platform: string | null;
  arch: string | null;
  hardware: HardwareInfo | null;
  // §0.7/N2 version-skew gate -- what this worker binary supports
  // ("benchmark", "probe-v1", "quality-v1", "curve-v1"). Absent for workers
  // predating the column (reads as ["benchmark"] only).
  capabilities: string[];
  installedBuilds: InstalledBuild[];
  modelFiles: ModelDirFile[];
  // Last-reported free-VRAM reading (see WorkerVramInfo below) -- null until
  // the worker's first idle heartbeat, or if it's currently busy (not
  // recomputed while running a benchmark).
  vram: WorkerVramInfo | null;
  // M6 -- sensor availability declared up front.
  sensors?: { clock: boolean; temp: boolean; source: SensorMeasurementSource | null } | null;
  // N6 -- llama.cpp startup-banner ISA provenance.
  cpuIsa?: string | null;
  status: "offline" | "idle" | "busy"; // DERIVED, never stored -- see server/src/liveness.ts
  lastHeartbeatAt: number | null;
  activeJobId: string | null;
  // The two pieces of live detail a UI needs while activeJobId is set --
  // both derived from the SAME worker_jobs row (its run_id and progress_json,
  // extended on every heartbeat, see repo.ts's queueRepo.extendLeaseAndGetFlags)
  // so a card/dashboard needs no extra round trip to show "what" and "how
  // far along". Undefined whenever activeJobId is null. activeRunId is only
  // ever set for a 'benchmark' job -- an install/download job has no run to
  // point at, just activeJobProgress's phase/bytes.
  activeRunId?: string;
  activeJobProgress?: ActiveJobReport;
  // Concurrently-running download_model jobs (worker/src/index.ts's
  // downloadJobPool) -- separate from activeJobProgress above, which is
  // reserved for the ONE serial (benchmark/build) job slot. A worker can
  // have several of these at once (up to its configured
  // max_concurrent_downloads) while activeJobProgress is undefined (status
  // stays "idle" -- see WorkerStatePush.status's own doc comment).
  activeDownloads?: ActiveJobReport[];
  createdAt: number;
  updatedAt: number;
}

// What a worker reports of itself on every heartbeat/queue poll -- replaces
// the old live `GET /health` + `GET /llama-cpp` + `GET /hardware` fan-out
// with one self-reported snapshot. Validated and capped server-side (see
// server/src/validate-worker-state.ts) before being persisted or trusted.
export interface WorkerStatePush {
  machine_id: string;
  capabilities: string[];
  hostname: string;
  // The worker's own configured/detected backend (WorkerConfig.backend,
  // worker/src/index.ts -- explicit override or detectBackend's guess). Not
  // derivable server-side from `hardware` alone: a box with an NVIDIA GPU
  // could still be pinned to `cpu`, and this is what `workers.backend`
  // (schema.sql) actually stores.
  backend: Backend;
  hardware: HardwareInfo;
  installed_builds: InstalledBuild[];
  model_files: ModelDirFile[];
  status: "idle" | "busy";
  // M6 -- sensor availability per backend, reported up front so machine
  // cards can declare "clock · temp available" before any flag ever appears.
  // Absent for workers predating the field (reads as unavailable).
  sensors?: { clock: boolean; temp: boolean; source: SensorMeasurementSource | null } | null;
  // N6 -- llama.cpp startup-banner ISA provenance (the running build's own
  // dispatch), parsed once per binary identity. Absent for older workers.
  cpu_isa?: string | null;
  // See WorkerVramInfo's doc comment -- omitted (not just null) while busy,
  // since collectState() skips the read entirely rather than reporting a
  // stale/meaningless-while-busy figure.
  vram?: WorkerVramInfo | null;
}

// Reported alongside a heartbeat while a job is active -- the replacement
// for the deleted `/models/download/progress` polling route, and for the
// old worker-side `busy` health flag: this is what actually drives the run
// page's live phase/progress display now.
export interface ActiveJobReport {
  job_id: string;
  phase: "downloading" | "extracting" | "loading" | "benchmarking" | "finalizing";
  bytes?: number;
  total_bytes?: number;
  item_idx?: number;
  items_total?: number;
  detail?: string;
}

export interface HeartbeatResponse {
  worker_id: string;
  // discard_job_ids is a subset of cancel_job_ids (every discarded job is
  // also a cancelled one) -- Cancel Download vs Pause on the Models page:
  // both abort the same in-flight fetch, but only Cancel also tells the
  // worker to delete the .part file instead of keeping it for resume (see
  // worker/src/index.ts's discard handling).
  control: { cancel_job_ids: string[]; discard_job_ids: string[]; pause: boolean };
  lease_until: number;
  // Operator-controlled settings (see AppSettings) piggybacked on every
  // beat -- the worker's ~10s heartbeat cadence doubles as the settings
  // propagation channel, so a policy flip on the supervise dashboard takes
  // effect without touching the worker or restarting anything. Absent
  // (older server) just means "keep whatever you had".
  app_settings?: AppSettings;
}

// Multi-user Stage 3 (MULTIUSER_PLAN.md §3.1/§3.4/§3.5) -- RFC 8628-shaped
// device flow. Shared between server/src/routes/device.ts (producer) and
// worker/src/vps-client.ts (consumer) so both sides agree on the wire shape.
export interface DeviceStartResponse {
  device_code: string;
  user_code: string;
  verification_uri: string;
  interval: number;
  expires_in: number;
}

export interface DeviceTokenSuccess {
  session_token: string;
  refresh_token: string;
}

export interface DeviceTokenError {
  error: "authorization_pending" | "expired_token";
}

export interface RefreshResponse {
  session_token: string;
  refresh_token: string;
  expires_at: number;
}

export interface InstallBuildJobPayload {
  tag: string;
  asset_name: string;
  download_url: string;
  // GitHub-reported size of asset_name's zip -- lets the worker show an
  // accurate progress bar before the first response header arrives.
  // Optional: an older server queueing a fresh install simply omits it.
  size_bytes?: number;
  // CUDA builds need their matching cudart redistributable (the cublas/
  // cudart DLLs) extracted next to the binaries or they can't load at all
  // (see LlamaCppRelease.cudart_assets). The server resolves the pairing
  // from its cached GitHub data -- never from a browser-supplied URL --
  // and the worker re-validates the host before downloading. Absent for
  // every non-CUDA build and any platform without a cudart zip.
  cudart_name?: string;
  cudart_url?: string;
  cudart_size_bytes?: number;
}

export interface DownloadModelJobPayload {
  hf_repo: string;
  hf_file: string;
}

export interface BenchmarkJob {
  run_id: string;
  model: Model;
  mtp_model?: Model;
  // The WHOLE sweep -- the worker loops it (one job per run, MULTIUSER_PLAN.md §1.1).
  sweep: Omit<SweepConfig, "model_id">;
  main_gpu?: number;
  llama_cpp_build: string; // tag; resolved to a path from installed_builds
  llama_cpp_backend: Backend;
  // Set on runtime-kind runs: "context_curve" executes N1's server-path
  // choreography for exactly one point; "knee" executes N5's load-driver
  // ladder. Absent = ordinary sweep execution.
  mode?: "sweep" | "context_curve" | "knee";
  curve_point?: CurvePointSpec;
  knee_spec?: KneeSpec;
}

// N5 -- concurrency-knee ladder. Prompt/gen cells are fixed from the M2
// target; slots is Expert-editable ({1,2,4,8} default).
export interface KneeSpec {
  n_prompt: number;
  n_gen: number;
  repeats?: number;
  slots?: number[];
}

// N2 -- one probe's whole ladder. Success per rung = no OOM, no spill, gen
// tok/s above floor. The search itself (which axis moves, how far, when it
// stops) lives in shared/probeLadder.ts; this payload only carries what that
// module needs as inputs.
export interface RunProbeJobPayload {
  run_id: string;
  model_id: string;
  // The full record, exactly as BenchmarkJob carries it: the worker resolves
  // a path from filename/source/hf_* and cannot do that from an id alone.
  model: Model;
  candidateCtx: number;
  placement: { ngl: number; nCpuMoe?: number; slots: number };
  kvPair: [string, string];
  llama_cpp_build: string;
  llama_cpp_backend: Backend;
  main_gpu?: number;
  trained_ctx?: number | null;
  gpu_total_mib?: number | null;
  // Absent means the pre-modes behavior -- see ProbeTriggerSpec.
  mode?: ProbeMode;
  granularity?: ProbeGranularity;
}

// One rung of the ladder, as reported by the worker and persisted verbatim
// into probe_attempts. Every field past `spill` is optional: a worker built
// before these existed still validates, and its rows simply read "—".
export interface ProbeAttemptReport {
  candidate_ctx: number;
  ok: boolean;
  oom: boolean;
  spill: boolean;
  // The placement this rung actually loaded at. The ladder moves ngl as well
  // as context, so a row is only interpretable alongside the ngl it used.
  ngl?: number | null;
  vram_peak_mib?: number | null;
  // MemorySampler's own RSS peak for the probe process -- the real measured
  // RAM usage, alongside vram_peak_mib.
  ram_peak_mib?: number | null;
  gen_tps?: number | null;
  // What computeDualPoolFit PREDICTED this rung would need, and what the
  // machine actually had free just before the load -- the predicted-vs-real
  // pair that until now only ever reached a log line.
  vram_needed_mib?: number | null;
  vram_free_mib?: number | null;
  ram_needed_mib?: number | null;
  ram_free_mib?: number | null;
  error?: string;
  /** N2 batch dedup -- set when this rung was reused from an earlier sibling
   * run's own measurement instead of actually being loaded (see
   * shared/api-v8.ts's ProbeDedupPoint). */
  reused_from_run_id?: string | null;
}

export type ProbeResultStatus = "verified" | "failed" | "failed_oom" | "stopped";

export interface ProbeResultInput {
  status: ProbeResultStatus;
  verified_ctx_tokens: number | null;
  /**
   * The layer count the winning rung actually loaded at. The ladder searches
   * this axis in most modes, so it is routinely NOT the ngl the probe was
   * requested with -- and the stored ceiling is keyed on the placement it was
   * verified at, not the one that was asked for. Optional so a worker
   * predating this field still reports successfully (the server then falls
   * back to the requested placement, exactly as it always did).
   */
  verified_ngl?: number | null;
  margin_observed_frac?: number | null;
  method_version?: number;
  attempts: ProbeAttemptReport[];
  error?: string;
}

// N4's "bundled default corpus with a pinned hash" -- every worker
// provisions this file into its corpora/ directory at startup (see
// worker/assets/default-corpus.txt and worker/src/index.ts's
// ensureDefaultQualityCorpus), so a quality run never needs a manually
// placed dataset for the common case. Original composition written for this
// project, not derived from any existing work -- licensing is unambiguous.
export const DEFAULT_QUALITY_DATASET_HASH =
  "sha256:ca4eb3719b9edc2ada8fce839fc31ea8664ba06a025692b48fd2255dfdf2a232";
export const DEFAULT_QUALITY_DATASET_LABEL = "Bundled default corpus";
export const DEFAULT_QUALITY_DATASET_LICENSE = "Original composition for this project (public domain / CC0)";

// N4 -- one llama-perplexity measurement of quality as a labeled synthetic
// proxy. Never feeds ranking until the calibration gate passes.
export interface MeasureQualityJobPayload {
  run_id: string;
  model_id: string;
  // See RunProbeJobPayload.model -- same reason.
  model: Model;
  ctxTokens: number;
  kvPair: [string, string];
  datasetHash: string;
  datasetLicense?: string | null;
  llama_cpp_build: string;
  llama_cpp_backend: Backend;
  main_gpu?: number;
}

export interface QualityResultInput {
  ppl: number;
  kld_vs_baseline?: number | null;
  dataset_hash: string;
  ctx_tokens: number;
  cache_type_k: string;
  cache_type_v: string;
  method_version?: number;
}

// activate_build/delete_build aren't in the plan's own Appendix A QueueJob
// union, but WorkerCard.tsx's Activate/Delete build buttons are real,
// actively-used functionality (not a minor edge feature) -- the worker's
// inbound HTTP server (the old /llama-cpp/activate, DELETE /llama-cpp/:tag)
// is gone under the pull model, so these have to become queue jobs like
// everything else, or the feature silently breaks. Kept minimal on purpose:
// both payloads are just the tag the worker already knows how to resolve
// against its own installed-build registry (worker/src/llama-builds.ts).
export interface ActivateBuildJobPayload {
  tag: string;
}

export interface DeleteBuildJobPayload {
  tag: string;
}

export type QueueJob =
  | { job_id: string; type: "benchmark"; payload: BenchmarkJob }
  | { job_id: string; type: "install_build"; payload: InstallBuildJobPayload }
  | { job_id: string; type: "activate_build"; payload: ActivateBuildJobPayload }
  | { job_id: string; type: "delete_build"; payload: DeleteBuildJobPayload }
  | { job_id: string; type: "download_model"; payload: DownloadModelJobPayload }
  | { job_id: string; type: "delete_model_file"; payload: { filename: string } }
  // Triggers a full model directory reconciliation on the worker
  | { job_id: string; type: "refresh_models"; payload: Record<string, never> }
  // No payload -- just a signal. Queued (not sent via HeartbeatResponse.control
  // like cancel/pause) so it takes its place behind whatever job is already
  // running instead of yanking the process mid-benchmark.
  | { job_id: string; type: "shutdown_worker"; payload: Record<string, never> }
  | { job_id: string; type: "run_probe"; payload: RunProbeJobPayload }
  | { job_id: string; type: "measure_quality"; payload: MeasureQualityJobPayload };

// --- Multi-user Stage 2: auth (MULTIUSER_PLAN.md §2) ---

// The shape req.user / GET /api/auth/status exposes -- never the raw `users`
// row (no created_at noise the client doesn't need). isSuperadmin is derived
// per-request from env + linked identities (§5.1), never a stored/cached
// column. shareBenchmarks IS exposed (unlike the rest of the raw row) --
// Settings' own toggle (§5.4) needs to know the caller's current value.
export interface AuthUser {
  id: string;
  displayName: string | null;
  avatarUrl: string | null;
  isSuperadmin: boolean;
  shareBenchmarks: boolean;
}

export interface AuthStatus {
  user: AuthUser | null;
  // Lets the client decide whether to show a login gate at all -- Stage 1
  // and Stage 2 are independently deployable (MULTIUSER_PLAN.md §2.3), so
  // these routes exist regardless of AUTH_ENABLED, and `user` alone can't
  // distinguish "nobody is logged in" from "logins aren't required here."
  authEnabled: boolean;
  // Operator-controlled feature gates (see AppSettings below) -- always
  // present, computed regardless of authEnabled, so Settings.tsx can decide
  // what to show/allow without a second round trip.
  appSettings: AppSettings;
}

// Operator-controlled toggles, set from the admin/supervise dashboard (see
// routes/admin.ts's GET/POST /api/admin/settings) and stored in the `meta`
// key-value table (server/src/db/schema.sql) rather than a dedicated table --
// there are only ever these two flags. Absent key reads as the documented
// default on each field below (repo.ts's appSettingsRepo.get()).
// How a worker's own post-run VRAM-discrepancy detection (finalizeSweepItemResult's
// claimed-vs-actual offload check) translates into item outcomes:
//   warn                 -- record results normally, attach the warning (original behavior)
//   retry_once_then_fail -- re-run the item once (heals transient VRAM contention);
//                           a reproduced discrepancy marks the item failed, and the
//                           run-scoped persistent memo makes later items fail
//                           immediately instead of each paying their own retry
//   fail                 -- never record discrepancy results; mark the item failed
// Only the unambiguous signature triggers the hard actions: llama.cpp's own
// post-allocation buffer report contradicting the claim with ~0 bytes on the
// GPU buffer (~0 layers actually resident). Softer shortfalls (partial
// residency, or VRAM-sample-based detections without a buffer report) always
// stay at warn level regardless of policy -- too estimate-noisy to fail a run over.
export type VramDiscrepancyPolicy = "warn" | "retry_once_then_fail" | "fail";

// Single whitelist for every boundary that accepts one of these from the
// outside -- the admin settings POST, the meta-table read (a hand-edited DB
// row must degrade to the documented default, not poison heartbeats), and
// the worker's heartbeat ingestion.
export function isVramDiscrepancyPolicy(value: unknown): value is VramDiscrepancyPolicy {
  return value === "warn" || value === "retry_once_then_fail" || value === "fail";
}

// Single validity check for every boundary that accepts a probeMaxLoads
// value from the outside -- the admin settings POST, the meta-table read (a
// hand-edited DB row must degrade to the documented default, not poison a
// probe run), and the worker's heartbeat ingestion. Upper bound (200) is
// deliberately generous above the shipped default (24): high enough that no
// realistic ladder search ever hits it, low enough that a fat-fingered
// six-figure value on the admin dashboard can't turn one probe into an
// effectively unbounded sequence of real model loads (each one is a
// multi-second llama-server spawn).
export function isValidProbeMaxLoads(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 200;
}

export interface AppSettings {
  // Whether users are allowed to turn ON the "contribute to community
  // benchmark database" toggle in Settings at all -- default false (starts
  // greyed out). Also enforced server-side (POST /api/auth/share-benchmarks,
  // and the AI assistant's community tools in routes/ai.ts) so a direct API
  // call can't bypass a disabled toggle.
  communitySharingAllowed: boolean;
  // Whether the self-service "delete my account" flow is exposed at all --
  // default true (matches the shipped v1 behavior). When false, Settings
  // hides the whole Danger zone section, and DELETE /api/auth/account
  // refuses server-side too.
  accountDeletionAllowed: boolean;
  // What workers do when a benchmark claims full GPU offload but provably
  // ran from system RAM -- default "warn" (record anyway with the warning,
  // the shipped v1 behavior). Set from the supervise dashboard; reaches
  // workers via HeartbeatResponse.app_settings.
  workerVramDiscrepancyPolicy: VramDiscrepancyPolicy;
  // Hard cap on how many real model loads a single N2 probe run may perform
  // (shared/probeLadder.ts's own PROBE_MAX_LOADS is the ladder's built-in
  // default of this same number) -- default 24. Set from the supervise
  // dashboard; reaches workers via HeartbeatResponse.app_settings and is
  // passed as nextLadderRung's `maxLoads` override, and is independently
  // enforced server-side when a worker reports probe attempts
  // (routes/measurements.ts's validateProbeAttempts) so a misbehaving or
  // stale worker can never report more loads than the current setting
  // allows.
  probeMaxLoads: number;
}

// GET /api/sessions -- one row per live session for the caller's own
// account. token_hash/refresh_hash/prev_refresh_hash never leave the server.
export interface SessionInfo {
  id: string;
  label: string | null;
  isWorker: boolean;
  createdAt: number;
  lastSeenAt: number | null;
  current: boolean;
}

// GET /api/auth/identities -- Settings' "Connected accounts" list for the
// caller's own account (§2.4). provider_user_id is never exposed to the
// client -- it's an internal key, not display data.
export interface IdentityInfo {
  provider: string;
  providerLogin: string | null;
  createdAt: number;
}

// A candidate machine's `hostname` and/or hardware (CPU/GPU/RAM) look like
// another machine the same user already owns -- surfaced so the approval
// screen can offer merging into that existing worker instead of silently
// creating an indistinguishable duplicate. This is the common case after a
// user deletes a worker's install folder and re-runs setup: worker/
// config.json's machine_id (worker/src/index.ts) is generated once and lives
// ONLY in that folder, so wiping it makes the next enrolment look like a
// brand-new machine even though it's the same box -- see
// server/src/db/repo.ts's workerRepo.findPossibleDuplicate.
// hostnameMatch/hardwareMatch let the UI say *why* this candidate was
// flagged -- hostname alone is a weak signal (a generic default like
// "DESKTOP-XXXXXXX" collides across genuinely different machines), so a
// human deciding whether to merge benefits from knowing which signal(s) fired.
export interface PossibleDuplicateWorker {
  id: string;
  displayName: string;
  lastHeartbeatAt: number | null;
  hostnameMatch: boolean;
  hardwareMatch: boolean;
}

// GET /api/device/status?user_code=... -- the browser's own poll while the
// "Add machine"/"/device" screen (MULTIUSER_PLAN.md §3.1 step 4) waits for a
// human to look at and approve a code. `machine` is present only in the
// "pending" state -- there's nothing left to show once approved, and nothing
// resolved yet for a code that was never issued or already expired.
export type DeviceStatusResponse =
  | { state: "not_found" }
  | { state: "approved" }
  | {
      state: "pending";
      machine: { hostname: string | null; platform: string | null; arch: string | null; gpu: string | null };
      possibleDuplicate: PossibleDuplicateWorker | null;
    };

// POST /api/device/approve -- normally approves outright. If the candidate
// machine looks like another machine this user already owns (see
// PossibleDuplicateWorker above) and the request passed neither
// confirm_duplicate nor merge_into, approval is held pending an explicit
// choice from the human: "add it anyway" (confirm_duplicate) or "merge it
// into <duplicateOf>" (merge_into: duplicateOf.id) instead of silently
// creating a duplicate. `merged` is only ever true on the merge path -- see
// server/src/db/repo.ts's workerRepo.mergeEnrolment.
export type DeviceApproveResponse =
  | { ok: true; machine: { hostname: string | null }; merged?: true }
  | { ok: false; needsConfirmation: true; duplicateOf: PossibleDuplicateWorker };

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1) -- the admin surface's own
// wire shapes, reachable only from the admin origin (supervise.*) by a
// superadmin-listed identity. Deliberately separate from the main app's
// Run/Worker types rather than reusing them with optional fields bolted on:
// this is cross-tenant data (every user's runs, not the caller's own), and a
// shared shape would risk a field meant only for admin eyes (userDisplayName,
// workerDisplayName) leaking into a response type the main app also uses.
export interface AdminStats {
  // Unique accounts -- one row per provider-independent user (schema.sql's
  // users table), so the row count IS the distinct-user count.
  users: number;
  // Unique machine configurations ever connected -- one row per machine_id.
  machines: number;
  // Unique models ACTUALLY TESTED -- distinct models referenced by at least
  // one run, NOT the full registered-models list (a downloaded-but-never-run
  // model must not inflate this).
  modelsTested: number;
  // Unique quant variants among ONLY the tested models above -- metadata.quant
  // where present, else parsed off the model filename (same pattern as
  // server/src/hf.ts's parseQuant / client/src/modelGrouping.ts's extractQuant).
  quants: number;
  // Individual sweep tests that reached a FINAL outcome (done/failed/
  // failed_oom/cancelled) -- pre-created-but-still-'queued'/in-flight items
  // must not count as performed yet. Includes partial runs' items per the
  // "at least partially completed" reading.
  tests: number;
  // Every run row ever created (runs are inserted at trigger time).
  runs: number;
}

export interface AdminRunSummary {
  id: string;
  userId: string | null;
  userDisplayName: string | null;
  workerId: string | null;
  workerDisplayName: string | null;
  modelId: string;
  modelFilename: string | null;
  llamaCppBackend: string;
  status: string;
  error: string | null;
  startedAt: number;
  completedAt: number | null;
  itemsTotal: number;
  itemsDone: number;
  itemsFailed: number;
}

// GET /api/admin/runs's own filter querystring -- every field optional,
// unset means "don't filter on this."
export interface AdminRunFilters {
  userId?: string;
  workerId?: string;
  backend?: string;
  status?: string;
}

// GET /api/admin/users
export interface AdminUserSummary {
  id: string;
  displayName: string | null;
  avatarUrl: string | null;
  createdAt: number;
  lastLoginAt: number | null;
}

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- the AI assistant's one
// cross-tenant read (§5.3). No minimum contributor count is enforced (the
// original k>=5 anonymity floor was deliberately dropped -- collecting five
// identical hardware/model/backend combos was unrealistic in practice), so a
// row may describe as few as a single opted-in contributor. There is no
// field here and no code path that can carry a UUID or per-user identifier,
// but contributorCount itself should always be shown/considered by callers.
export interface CommunityAggregateRow {
  modelId: string;
  modelFilename: string | null;
  backend: string;
  testType: string;
  platform: string | null;
  gpuModel: string | null;
  contributorCount: number;
  runCount: number;
  avgTps: number;
  avgRamPeakMib: number | null;
  avgVramPeakMib: number | null;
}

export interface CommunityAggregateFilters {
  modelId?: string;
  backend?: string;
  platform?: string;
  gpuModel?: string;
}

// One (dimension, value) pair that currently has at least one opted-in
// contributor -- lets a caller discover valid CommunityAggregateFilters
// values instead of guessing one blind. No minimum contributor count is
// enforced (see CommunityAggregateRow's own doc comment).
export interface CommunityFacetValue {
  value: string;
  contributorCount: number;
}

export interface CommunityFacets {
  models: (CommunityFacetValue & { modelId: string })[];
  backends: CommunityFacetValue[];
  platforms: CommunityFacetValue[];
  gpuModels: CommunityFacetValue[];
  // When this facets snapshot was computed (ISO string). Absent for rows
  // predating the field -- the client treats that as "stale, refresh".
  lastUpdated?: string;
}

// Hugging Face GGUF index entry -- one row per (sha256, repo_id, filename)
// combination in the server's hf_gguf_index table. The composite primary key
// is (sha256, repo_id, filename) so mirrored content across repos keeps
// distinct rows; sha256 is the lookup target (the worker sends hashes, the
// server returns metadata). See server/src/hf-index.ts for the full flow.
export interface HfGgufIndexEntry {
  sha256: string;
  repo_id: string;
  filename: string;
  revision: string;
  file_size: number;
  last_seen: number;
  // Set when server/src/hf-index.ts confirmed this file/repo no longer
  // exists on Hugging Face. Rows are soft-deleted, never hard-removed, so a
  // match that's since been removed can still be reported as such (rather
  // than silently vanishing) -- see hf-index.ts's module doc comment.
  deleted_at: number | null;
}

// Worker-local persistent cache of model file hashes and their HF
// mappings. See worker/src/local-cache.ts for the full scan/validate/
// hash/resolve flow. sha256 and hf_model_id are optional: a file that
// hasn't been hashed yet has neither, and a hashed file with no HF
// match has sha256 but no hf_model_id.
export interface LocalModelCacheEntry {
  path: string;
  size: number;
  mtime: number;
  sha256?: string;
  hf_model_id?: string;
  last_verified: number;
}

// Local model state machine -- see worker/src/local-cache.ts for the full
// flow. Each state is a distinct UI signal on the Models page.
export const LOCAL_MODEL_STATES = [
  "detected",   // File discovered locally, hash not yet computed
  "hashing",    // SHA-256 calculation in progress
  "verified",   // Exact Hugging Face match found
  "unknown",    // No matching SHA-256 found in the HF index
  "modified",   // File changed since last verification
  "corrupted",  // Download verification failed
  "missing",    // Cached file no longer exists on disk
] as const;

export type LocalModelState = (typeof LOCAL_MODEL_STATES)[number];
