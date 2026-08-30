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
  vram_peak_mib: number | null;
  ram_needed_mib: number | null;
  ram_free_mib: number | null;
  ram_peak_mib: number | null;
  gen_tps: number | null;
  error: string | null;
  created_at: number;
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
