// BENCHMARKING_PLAN_V8.md M3 -- goal-parameterized profile cards served over
// stored results. Re-scoring is pure post-processing: this route never
// enqueues anything and never re-measures. Changing the goal is a different
// query string over the same rows.

import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { resolveAuthUser } from "../auth-middleware.js";
import type { Model, ResultRow, RunConfig } from "../../../shared/types.js";
import { scoreProfiles, tupleKey, type ScoringRow, type ScoringResult } from "../../../shared/scoring.js";
import type { ConfigHashInput } from "../../../shared/configHash.js";
import { maxAffordableContext, residentWeightsMibFromPeak } from "../../../shared/vramEstimate.js";
import type { MaxCtxEstimate } from "../../../shared/vramEstimate.js";
import { normalizeGoals, defaultGoals, type GoalsConfig } from "../../../shared/goals.js";
import type { VerifiedLimit, QualityRow } from "../db/repo.js";

export interface ProfilesResponse {
  run_id: string;
  root_run_id: string;
  scoring: ScoringResult;
  /** Every method_version present in the chain -- surfaced, never averaged (§0.1). */
  method_versions_present: (number | null)[];
  /** N2 -- verified ceilings for this model+machine, keyed by the KV pair they were probed at. */
  verified_limits: VerifiedLimit[];
  /** N4 -- every quality measurement recorded for this model (deterministic given model+build+ctx+kv+corpus, not machine-specific). */
  quality_results: QualityRow[];
  /** The goals actually used, after query overrides and the trained-context clamp. */
  goals: GoalsConfig;
  /** True when the caller overrode the run's stored goals for this read only. */
  goals_overridden: boolean;
  target_ctx_clamped: boolean;
}

// M2's target is clamped by trained context inline; scoring must see the same
// clamped number the questionnaire showed.
function clampTarget(goals: GoalsConfig, model: Model | undefined): { goals: GoalsConfig; clamped: boolean } {
  const trained = model?.metadata.trained_ctx;
  if (goals.target_ctx != null && typeof trained === "number" && trained > 0 && goals.target_ctx > trained) {
    return { goals: { ...goals, target_ctx: trained }, clamped: true };
  }
  return { goals, clamped: false };
}

function goalsFromQuery(
  stored: GoalsConfig | undefined,
  query: Record<string, string | undefined>
): { goals: GoalsConfig; overridden: boolean } {
  const hasOverride = ["goal", "target_ctx", "workload", "speed_floor_frac", "kv_tolerance"].some(
    (k) => query[k] !== undefined
  );
  if (!hasOverride) return { goals: stored ?? defaultGoals(), overridden: false };
  const base = stored ?? defaultGoals();
  const merged = normalizeGoals({
    goal: query.goal ?? base.goal,
    workload: query.workload ?? base.workload,
    kv_tolerance: query.kv_tolerance ?? base.kv_tolerance,
    speed_floor_frac:
      query.speed_floor_frac !== undefined ? Number(query.speed_floor_frac) : base.speed_floor_frac,
    target_ctx:
      query.target_ctx === undefined
        ? base.target_ctx
        : query.target_ctx === "unverified" || query.target_ctx === ""
          ? null
          : Number(query.target_ctx),
  });
  return { goals: merged ?? base, overridden: true };
}

// Placement identity for M1's weights interpolation -- deliberately NARROWER
// than a scoring tuple key: the VRAM-resident weight share depends only on
// which layers/experts are offloaded and the KV type, never on n_prompt/
// n_gen/threads/batch_size, so a row measured at a different prompt size for
// the SAME placement is still a valid weights sample.
function placementKey(k: {
  cache_type_k: unknown;
  cache_type_v: unknown;
  n_gpu_layers: unknown;
  n_cpu_moe: unknown;
}): string {
  return [k.cache_type_k, k.cache_type_v, k.n_gpu_layers, k.n_cpu_moe].join("|");
}

// M1 per-placement estimate. weightsMib is never invented: it comes from a
// measured peak for that exact placement minus that item's own KV cost
// (shared/vramEstimate.ts's residentWeightsMibFromPeak). When the current
// chain never measured this placement, M1's own stated fallback applies:
// interpolate from any prior run's peak for this model+machine at the same
// placement; only when neither exists does the estimate degrade to
// confidence "unknown" (with a conservative floor candidate, if trainedCtx
// is known).
function buildMaxCtxFor(
  model: Model | undefined,
  rowsByTuple: Map<string, ResultRow>,
  crossRunRows: ResultRow[],
  trainedCtx: number | null
): (axes: ConfigHashInput) => MaxCtxEstimate | null {
  const meta = model?.metadata ?? {};

  // Built once per request, not once per scored tuple: the widest-context
  // row per placement, mirroring rowsByTuple's own widest-context tie-break.
  const crossRunByPlacement = new Map<string, ResultRow>();
  const ctxOf = (r: ResultRow): number => (r.n_depth ?? 0) + r.n_prompt + r.n_gen;
  for (const row of crossRunRows) {
    if (row.vram_peak_mib == null || row.gpu_memory_total_mb == null) continue;
    const key = placementKey(row);
    const held = crossRunByPlacement.get(key);
    if (!held || ctxOf(row) > ctxOf(held)) crossRunByPlacement.set(key, row);
  }

  return (axes) => {
    const geometry = {
      nLayer: meta.n_layer ?? 0,
      nHeadKv: meta.n_head_kv ?? 0,
      headDimK: meta.head_dim_k,
      headDimV: meta.head_dim_v,
      nEmbd: meta.n_embd,
      nHead: meta.n_head,
      cacheTypeK: String(axes.cache_type_k ?? ""),
      cacheTypeV: String(axes.cache_type_v ?? ""),
      slidingWindow: meta.sliding_window,
    };
    // Falls through to the cross-run sample not only when this chain never
    // measured the placement at all, but also when it did and the row it
    // has simply carries no usable VRAM reading (e.g. a CPU-only item, or a
    // measurement where the GPU read failed) -- either way this chain alone
    // can't answer, and a prior run for the same model+machine might.
    const tupleSample = rowsByTuple.get(tupleKeyFromAxes(axes));
    const sample =
      tupleSample?.gpu_memory_total_mb != null
        ? tupleSample
        : crossRunByPlacement.get(
            placementKey({
              cache_type_k: axes.cache_type_k,
              cache_type_v: axes.cache_type_v,
              n_gpu_layers: axes.n_gpu_layers,
              n_cpu_moe: axes.n_cpu_moe,
            })
          );
    if (!sample || sample.gpu_memory_total_mb == null) {
      return {
        tokens: 0,
        confidence: "unknown",
        binding: null,
        conservativeFloorTokens: trainedCtx != null && trainedCtx > 0 ? Math.floor(trainedCtx) : null,
      };
    }
    const weightsMib = residentWeightsMibFromPeak({
      ...geometry,
      vramPeakMib: sample.vram_peak_mib,
      contextTokens: (sample.n_depth ?? 0) + sample.n_prompt + sample.n_gen,
      parallelSlots: sample.concurrency ?? 1,
    });
    return maxAffordableContext({
      ...geometry,
      totalMib: sample.gpu_memory_total_mb,
      weightsMib,
      parallelSlots: sample.concurrency ?? 1,
      trainedCtx,
    });
  };
}

// The tuple key scoring will compute for these axes -- one function, so the
// estimate map and the scored tuples can never disagree about identity.
function tupleKeyFromAxes(axes: ConfigHashInput): string {
  return tupleKey({
    idx: 0,
    test_type: "tg",
    n_prompt: axes.n_prompt ?? 0,
    n_gen: axes.n_gen ?? 0,
    n_depth: 0,
    n_threads: axes.threads ?? 0,
    n_gpu_layers: axes.n_gpu_layers ?? 0,
    batch_size: axes.batch_size ?? 0,
    ubatch_size: axes.ubatch_size ?? 0,
    cache_type_k: axes.cache_type_k ?? "",
    cache_type_v: axes.cache_type_v ?? "",
    flash_attn: axes.flash_attn ?? "",
    mtp: axes.engine === "server" ? "on" : "off",
    n_gpu_layers_draft: axes.n_gpu_layers_draft ?? 0,
    n_cpu_moe: axes.n_cpu_moe ?? 0,
    concurrency: axes.concurrency ?? 1,
    engine: axes.engine,
    avg_tps: 0,
    stddev_tps: 0,
    vram_peak_mib: null,
    ram_peak_mib: 0,
    gpu_memory_total_mb: null,
    system_memory_total_mb: null,
  });
}

function toScoringRow(row: ResultRow, idxOffset: number): ScoringRow {
  return {
    idx: idxOffset + row.idx,
    test_type: row.test_type,
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_depth: row.n_depth ?? 0,
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
    concurrency: row.concurrency ?? 1,
    engine: row.engine ?? null,
    method_version: row.method_version ?? null,
    avg_tps: row.avg_tps,
    stddev_tps: row.stddev_tps,
    sample_count: row.sample_count,
    suspect_count: row.suspect_count,
    repeat_samples: row.repeat_samples,
    caveat_flags: row.caveat_flags,
    vram_peak_mib: row.vram_peak_mib,
    ram_peak_mib: row.ram_peak_mib,
    gpu_memory_total_mb: row.gpu_memory_total_mb,
    system_memory_total_mb: row.system_memory_total_mb,
  };
}

export async function profilesRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Params: { id: string }; Querystring: Record<string, string | undefined> }>(
    "/api/runs/:id/profiles",
    async (request, reply) => {
      const authed = resolveAuthUser(request);
      const userId = authed?.user.id;
      const run = repo.getRun(userId, request.params.id);
      if (!run) return reply.code(404).send({ error: "run not found" });

      const rootRunId = run.root_run_id ?? run.id;
      const chainRuns = repo.listChainScoringRuns(userId, rootRunId);
      // A run that is itself excluded from scoring (a probe, a comparison
      // member) still answers here -- with an empty universe and the tallies
      // that say why, never a silent 404.
      const model = repo.getModel(run.model_id);

      const rows: ScoringRow[] = [];
      const itemStatusByIdx: Record<number, string> = {};
      const rowsByTuple = new Map<string, ResultRow>();
      const versionsPresent = new Set<number | null>();
      let offset = 0;
      let repeats: number | undefined;
      let discardFirst = 0;
      for (const chainRun of chainRuns) {
        const config = chainRun.config as RunConfig;
        repeats = repeats ?? config?.sweep?.repeats;
        discardFirst = Math.max(discardFirst, config?.discard_first_repeats ?? 0);
        const results = repo.getResultsForRun(chainRun.id);
        for (const item of repo.getRunItems(chainRun.id)) {
          itemStatusByIdx[offset + item.idx] = item.status;
        }
        for (const result of results) {
          // N7 -- imported rows are badged and never merge into local profile
          // scoring unless the importer opted in for that bundle.
          if (result.imported_bundle_id && !result.import_opt_in) continue;
          versionsPresent.add(result.method_version ?? null);
          rows.push(toScoringRow(result, offset));
          // Keep the widest-context measured row per tuple: it is the one
          // whose peak carries the most KV, so subtracting that KV leaves the
          // best-conditioned weights figure.
          const key = tupleKey(toScoringRow(result, offset));
          const held = rowsByTuple.get(key);
          const ctxOf = (r: ResultRow): number => (r.n_depth ?? 0) + r.n_prompt + r.n_gen;
          if (!held || (result.vram_peak_mib != null && ctxOf(result) > ctxOf(held))) {
            rowsByTuple.set(key, result);
          }
        }
        // Item indices are per-run; the chain flattens several runs into one
        // scoring universe, so they get namespaced rather than colliding.
        offset += 10_000;
      }

      const storedGoals = (run.config as RunConfig)?.goals;
      const fromQuery = goalsFromQuery(storedGoals, request.query);
      const clamped = clampTarget(fromQuery.goals, model);

      // §0.3's "TG reference depth": a pinned depth, default 50 % of the
      // stated target (else 50 % of trained context); the nearest measured
      // depth is what actually gets compared.
      const anchor = clamped.goals.target_ctx ?? model?.metadata.trained_ctx ?? null;
      const referenceDepthTokens = anchor != null ? Math.floor(anchor / 2) : null;

      // M1 -- when this chain never measured a given placement, interpolate
      // from any prior run's peak for this model+machine at that same
      // placement, rather than degrading straight to "unknown".
      const crossRunRows = run.worker_id ? repo.listCurveRows(userId, run.model_id, { workerId: run.worker_id }) : [];
      const trainedCtx = typeof model?.metadata.trained_ctx === "number" ? model.metadata.trained_ctx : null;

      const scoring = scoreProfiles({
        rows,
        itemStatusByIdx,
        goals: clamped.goals,
        repeats,
        discardFirstRepeats: discardFirst,
        referenceDepthTokens,
        maxCtxFor: buildMaxCtxFor(model, rowsByTuple, crossRunRows, trainedCtx),
      });

      const verified = run.worker_id
        ? repo.limitsRepo.listForModelAndWorker(run.model_id, run.worker_id)
        : [];
      const qualityResults = repo.qualityRepo.listForModel(run.model_id);

      const body: ProfilesResponse = {
        run_id: run.id,
        root_run_id: rootRunId,
        scoring,
        method_versions_present: [...versionsPresent],
        verified_limits: verified,
        quality_results: qualityResults,
        goals: clamped.goals,
        goals_overridden: fromQuery.overridden,
        target_ctx_clamped: clamped.clamped,
      };
      return body;
    }
  );
}
