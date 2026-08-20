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
  // GGUF's <architecture>.expert_count, read at the same time as n_layer --
  // see worker/src/gguf.ts. Present and >0 only for a Mixture-of-Experts
  // architecture; absent for a dense model or a model registered before this
  // existed, same "absent means unknown/not applicable" convention as
  // mtp_layers above. Gates NewRun.tsx's --n-cpu-moe control (see
  // shared/sweep.ts's n_cpu_moe once that axis exists).
  expert_count?: number;
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
}

// Walks model_dir recursively -- see worker/src/index.ts's listModelDirFiles.
// Promoted here (was worker-local) so the server can type the
// WorkerStatePush.model_files it now receives directly from the worker's own
// heartbeat/queue poll instead of fetching a worker's file list live.
export interface ModelDirFile {
  path: string;
  size_bytes: number;
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
  hf_repo: string;
  hf_file: string;
  ok: boolean;
  error?: string;
  sha256?: string;
  size_bytes?: number;
  n_layer?: number | null;
  mtp_layers?: number | null;
  expert_count?: number | null;
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
  installedBuilds: InstalledBuild[];
  modelFiles: ModelDirFile[];
  // Last-reported free-VRAM reading (see WorkerVramInfo below) -- null until
  // the worker's first idle heartbeat, or if it's currently busy (not
  // recomputed while running a benchmark).
  vram: WorkerVramInfo | null;
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
  model_files: (ModelDirFile & { n_layer?: number | null; mtp_layers?: number | null })[];
  status: "idle" | "busy";
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
  control: { cancel_job_ids: string[]; pause: boolean };
  lease_until: number;
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
  | { job_id: string; type: "delete_model_file"; payload: { filename: string } };

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
  users: number;
  machines: number;
  runs: number;
  tests: number;
  models: number;
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
// cross-tenant read (§5.3). Every row returned by repo.communityRepo already
// satisfies contributor_count >= 5 (SQL-enforced, see repo.ts) -- there is no
// field here and no code path that can carry a UUID or per-user identifier.
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

// One (dimension, value) pair that currently clears k=5 on its own -- lets a
// caller discover valid CommunityAggregateFilters values without being able
// to fish for one that would turn out to describe fewer than five people
// (§5.4's own "or it leaks the existence of a single rare GPU").
export interface CommunityFacetValue {
  value: string;
  contributorCount: number;
}

export interface CommunityFacets {
  models: (CommunityFacetValue & { modelId: string })[];
  backends: CommunityFacetValue[];
  platforms: CommunityFacetValue[];
  gpuModels: CommunityFacetValue[];
}
