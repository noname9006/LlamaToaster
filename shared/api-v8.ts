// BENCHMARKING_PLAN_V8.md -- the response shapes the v8 read paths return.
// Declared in shared/ (not in the route files) so client and server agree on
// one contract instead of the client re-describing what it hopes to receive.

import type { GoalsConfig } from "./goals.js";
import type { ScoringResult } from "./scoring.js";
import type { CurvePoint, KneeResult, LadderCell } from "./curves.js";
import type { PricedRate } from "./pricing.js";
import type { ComparisonMemberRow, ParetoPoint } from "./comparison.js";
import type { ImportRowVerdict } from "./exchange.js";

// N2 -- a verified ceiling, per (machine, build, KV pair, placement).
export interface VerifiedLimitDto {
  id: string;
  worker_id: string;
  model_id: string;
  llama_cpp_build: string;
  kv_type: string;
  placement_hash: string;
  verified_ctx_tokens: number;
  /** Layers on GPU at the verified rung; null from a probe predating this. */
  verified_ngl: number | null;
  margin_observed_frac: number | null;
  method_version: number | null;
  created_at: number;
}

// N2 -- one rung of a probe's ladder, as GET /api/runs/:id/probe-attempts
// returns it. ok/oom/spill are SQLite integers (0/1), not booleans.
export interface ProbeAttemptDto {
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
  /** WHOLE-ADAPTER VRAM peak (every process on the GPU combined), despite the
   * name -- see shared/types.ts's ProbeAttemptReport.vram_peak_mib. Pair with
   * vram_process_peak_mib below for the llama.cpp-only figure. */
  vram_peak_mib: number | null;
  ram_needed_mib: number | null;
  ram_free_mib: number | null;
  /** This process's own RAM peak (per-pid RSS) -- pair with ram_total_peak_mib
   * below for the whole-system figure. */
  ram_peak_mib: number | null;
  /** This process's own (llama.cpp-only) peak VRAM usage, as distinct from
   * vram_peak_mib's whole-adapter reading. Null wherever the backend/platform
   * never attributed a reading to this pid during the whole load (not a
   * measured 0), including every worker predating this reading. */
  vram_process_peak_mib: number | null;
  /** Whole-system RAM peak (every process combined), the RAM-side counterpart
   * to vram_peak_mib's whole-adapter reading. Null from a worker predating
   * this reading. */
  ram_total_peak_mib: number | null;
  /** This process's peak system-RAM-backed GPU allocation (Windows WDDM
   * "Shared Usage" or Linux amdgpu GTT) -- the direct measured figure for
   * "how much silently spilled into system RAM", alongside vram_peak_mib's
   * dedicated-only reading. Null wherever no such counter/file exists at all
   * (not a measured 0), including every worker predating this reading. */
  vram_shared_peak_mib: number | null;
  gen_tps: number | null;
  /** See ProbeAttemptReport.pp_tps -- prefill has its own placement cliff,
   * several layers below the weights one, so neither of these is derivable
   * from gen_tps. Null from a worker predating them. */
  pp_tps: number | null;
  ttft_ms: number | null;
  /** SQLite boolean (1/0/null), same convention as vram_discrepancy above. */
  prefill_cliff: number | null;
  /** Which evidence decided vram_discrepancy, and the measured slope in
   * layers of system RAM per layer added. Null on the inference path (no
   * shared-memory counter) and from workers predating the check. */
  host_backed_method: "slope" | "ratio" | null;
  host_backed_slope: number | null;
  error: string | null;
  created_at: number;
  /** Set when this rung was never actually loaded, but reused verbatim from
   * an earlier sibling run under the same N2 batch that already measured
   * this exact (candidate_ctx, ngl) point -- see GET .../probe-dedup. Names
   * the sibling run it came from; null for every genuinely measured rung. */
  reused_from_run_id: string | null;
  /** SQLite integer (0/1), null from a worker/row predating the check. True
   * means observed vram_peak_mib came in far below vram_needed_mib for the
   * claimed ngl -- the silent sysmem-fallback signature (see
   * shared/vramEstimate.ts's top comment; confirmed on both NVIDIA/CUDA and
   * AMD/Vulkan), not just this estimate over-budgeting. */
  vram_discrepancy: number | null;
  /** How many of `ngl`'s claimed layers actually landed in a GPU buffer, per
   * llama.cpp's own post-allocation buffer-size report -- null when ngl<=0,
   * a worker predating this check, or the load failed before tensor loading
   * finished. */
  gpu_layers_resident_est: number | null;
  /** SQLite integer (0/1). True means gpu_layers_resident_est is an EXACT
   * count from llama.cpp's own per-layer "assigned to device" lines; false
   * means it's the coarser buffer-byte-ratio estimate. Meaningless when
   * gpu_layers_resident_est is null. */
  gpu_layers_resident_exact: number | null;
}

// N2 batch dedup -- one already-measured (ctx, ngl) point from an earlier
// sibling run under the same batch root, as GET /api/runs/:id/probe-dedup
// returns it. The worker consults this before spawning a candidate load so
// several search modes fired together don't each re-measure the same point.
export interface ProbeDedupPoint {
  candidate_ctx: number;
  ngl: number | null;
  ok: boolean;
  oom: boolean;
  spill: boolean;
  gen_tps: number | null;
  vram_needed_mib: number | null;
  vram_free_mib: number | null;
  vram_peak_mib: number | null;
  ram_needed_mib: number | null;
  ram_free_mib: number | null;
  ram_peak_mib: number | null;
  /** See ProbeAttemptDto.vram_shared_peak_mib. */
  vram_shared_peak_mib: number | null;
  /** See ProbeAttemptDto.vram_discrepancy; coerced to false for a sibling row
   * predating the check, same as ok/oom/spill's own boolean coercion above. */
  vram_discrepancy: boolean;
  /** See ProbeAttemptDto.gpu_layers_resident_est. */
  gpu_layers_resident_est: number | null;
  /** See ProbeAttemptDto.gpu_layers_resident_exact; coerced to false for a
   * sibling row predating the check or where gpu_layers_resident_est is
   * null. */
  gpu_layers_resident_exact: boolean;
  /** Which sibling run actually measured this point. */
  source_run_id: string;
}

// N4 -- one llama-perplexity measurement, scoped by model (perplexity is
// deterministic given model+build+ctx+kv+corpus, not machine-specific the
// way a verified context ceiling is).
export interface QualityRowDto {
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

// M3 -- GET /api/runs/:id/profiles
export interface ProfilesResponse {
  run_id: string;
  root_run_id: string;
  scoring: ScoringResult;
  method_versions_present: (number | null)[];
  verified_limits: VerifiedLimitDto[];
  // N4 -- every quality measurement recorded for this model, so a card whose
  // KV pair has been measured can show it without a separate fetch.
  quality_results: QualityRowDto[];
  goals: GoalsConfig;
  goals_overridden: boolean;
  target_ctx_clamped: boolean;
}

// §0.6 -- GET /api/models/:id/rates
export interface ModelRatesResponse {
  model_id: string;
  pp: PricedRate;
  tg: PricedRate;
}

// N1 -- GET /api/models/:id/curve
export interface CurveResponse {
  model_id: string;
  engine: string | null;
  method_version: number | null;
  points: CurvePoint[];
  ladder: LadderCell[];
  all_points_covered: boolean;
}

// N5 -- GET /api/runs/:id/knee
export interface KneeResponse extends KneeResult {
  run_id: string;
  root_run_id: string;
  spec: { n_prompt: number; n_gen: number; slots?: number[]; repeats?: number } | null;
}

// N6 -- GET /api/runs/:id/sustained
export interface SustainedResponse {
  run_id: string;
  flagged_items: number[];
  denominator: number;
  ratio: number;
  offer_rerun: boolean;
  rerun_estimate: string;
  rerun_seconds: number | null;
  steady_state: {
    discard_first_repeats: number;
    available: boolean;
    reason: string | null;
  };
}

// N3 -- GET /api/comparisons/:id
export interface ComparisonResponse {
  comparison_id: string;
  members: ComparisonMemberRow[];
  pareto: ParetoPoint[];
  drifted_members: string[];
  quality_disclaimer: string;
}

// N7 -- POST /api/import
export interface ImportResponse {
  bundle_id: string;
  run_id: string;
  imported_rows: number;
  rejected_rows: ImportRowVerdict[];
  method_versions: (number | null)[];
  mixed_vintages: boolean;
  opted_into_scoring: boolean;
  notice: string;
}
