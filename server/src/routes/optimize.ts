// Optimization flow routes (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md):
//   worker -> server  POST /api/tests/:id/fit-points   fit map answers, streamed
//                     POST /api/tests/:id/kv-support   KV support detection rows
//   browser          GET  /api/tests/:id/fit-points    the map + KV rows of a fit run
//                     GET  /api/flows/:flowId          every run of one optimization session
//
// Worker posts use the dual-mode worker auth (enrolled session, or the shared
// deployment secret for an unowned machine) and are bound BY PATH: the run
// named in the URL must have been dispatched to the posting machine. Fit
// points are predictions about that one machine, so -- unlike probe ceilings
// -- they feed no cross-tenant claim; the stricter session-only rule isn't
// needed and would just break shared-secret workers.

import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { repo } from "../db/repo.js";
import { getDb } from "../db/migrate.js";
import { FitPointLimitError, fitRepo } from "../db/fitRepo.js";
import { authenticateWorker } from "../worker-auth.js";
import { assertOwnsWorker, sessionScope, type UserScope } from "../auth-middleware.js";
import { BadRequestError, ConflictError, ForbiddenError, NotFoundError } from "../errors.js";
import type { FitPoint, FitVerdict } from "../../../shared/fitParams.js";
import { KV_CACHE_TYPES } from "../../../shared/fitParams.js";
import { MAX_FIT_CTX_STOPS } from "../../../shared/optimizeFlow.js";
import type { KvSupportRow, KvSupportStatus } from "../../../shared/kvSupport.js";
import type { Test, TestConfig } from "../../../shared/types.js";

const MAX_POINTS_PER_POST = 400;
const MAX_KV_ROWS = 256;
/** Every (ctx stop, FA, K, V) a valid spec can ask for -- points are keyed by
 * exactly that, so a run past this is a broken or hostile worker. */
export const MAX_POINTS_PER_RUN = MAX_FIT_CTX_STOPS * 2 * KV_CACHE_TYPES.length ** 2;
const VERDICTS: readonly FitVerdict[] = ["full", "partial", "cpu_only", "doesnt_fit", "error"];
const KV_STATUSES: readonly KvSupportStatus[] = ["ok", "cpu_fallback", "invalid", "fa_disabled", "cuda_slow"];
const CACHE_TYPES = new Set<string>(KV_CACHE_TYPES);
const MAX_MIB = 64 * 1024 * 1024; // 64 TiB -- sanity bound only

function int(v: unknown, field: string, min: number, max: number, nullable = true): number | null {
  if (v === null || v === undefined) {
    if (nullable) return null;
    throw new BadRequestError(`${field} is required`);
  }
  if (typeof v !== "number" || !Number.isInteger(v) || v < min || v > max) {
    throw new BadRequestError(`${field} must be an integer in [${min}, ${max}]`);
  }
  return v;
}

function mib(v: unknown, field: string): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v !== "number" || !Number.isFinite(v) || v < -MAX_MIB || v > MAX_MIB) {
    throw new BadRequestError(`${field} must be a finite MiB figure or null`);
  }
  return v;
}

function str(v: unknown, field: string, max: number): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v !== "string") throw new BadRequestError(`${field} must be a string or null`);
  // Control characters never belong here; this text reaches the UI.
  return v.replace(/[\x00-\x1f\x7f]/g, " ").slice(0, max);
}

export function validateFitPoint(raw: unknown, field: string): FitPoint {
  if (!raw || typeof raw !== "object") throw new BadRequestError(`${field} must be an object`);
  const p = raw as Record<string, unknown>;
  if (p.fa !== "on" && p.fa !== "off") throw new BadRequestError(`${field}.fa must be on or off`);
  if (typeof p.ctk !== "string" || !CACHE_TYPES.has(p.ctk) || typeof p.ctv !== "string" || !CACHE_TYPES.has(p.ctv)) {
    throw new BadRequestError(`${field}.ctk/ctv must be non-f32 llama.cpp cache types`);
  }
  if (typeof p.verdict !== "string" || !(VERDICTS as readonly string[]).includes(p.verdict)) {
    throw new BadRequestError(`${field}.verdict must be one of ${VERDICTS.join(", ")}`);
  }
  if (p.reason != null && p.reason !== "vram" && p.reason !== "ram") throw new BadRequestError(`${field}.reason must be vram, ram or null`);
  return {
    ctx: int(p.ctx, `${field}.ctx`, 256, 4_194_304, false)!,
    fa: p.fa,
    ctk: p.ctk,
    ctv: p.ctv,
    margin_mib: int(p.margin_mib, `${field}.margin_mib`, 0, 1_000_000, false)!,
    verdict: p.verdict as FitVerdict,
    reason: (p.reason as FitPoint["reason"]) ?? null,
    ngl: int(p.ngl, `${field}.ngl`, -1, 4096),
    layers_gpu: int(p.layers_gpu, `${field}.layers_gpu`, 0, 4096),
    layers_total: int(p.layers_total, `${field}.layers_total`, 0, 4096),
    overflow_layers: int(p.overflow_layers ?? 0, `${field}.overflow_layers`, 0, 4096, false)!,
    ot: str(p.ot, `${field}.ot`, 64_000),
    dev_used_mib: mib(p.dev_used_mib, `${field}.dev_used_mib`),
    dev_free_mib: mib(p.dev_free_mib, `${field}.dev_free_mib`),
    host_used_mib: mib(p.host_used_mib, `${field}.host_used_mib`),
    total_mib: mib(p.total_mib, `${field}.total_mib`),
    kv_mib: mib(p.kv_mib, `${field}.kv_mib`),
    compute_mib: mib(p.compute_mib, `${field}.compute_mib`),
    need_all_gpu_mib: mib(p.need_all_gpu_mib, `${field}.need_all_gpu_mib`),
    splits: int(p.splits, `${field}.splits`, 0, 1_000_000),
    fa_disabled: p.fa_disabled === true,
    n_expert: int(p.n_expert, `${field}.n_expert`, 0, 1_000_000),
    n_expert_used: int(p.n_expert_used, `${field}.n_expert_used`, 0, 1_000_000),
    inferred: p.inferred === true,
    raw_args: str(p.raw_args, `${field}.raw_args`, 64_000),
    error: str(p.error, `${field}.error`, 2000),
  };
}

export function validateKvRow(raw: unknown, field: string): KvSupportRow {
  if (!raw || typeof raw !== "object") throw new BadRequestError(`${field} must be an object`);
  const r = raw as Record<string, unknown>;
  if (typeof r.ctk !== "string" || !CACHE_TYPES.has(r.ctk) || typeof r.ctv !== "string" || !CACHE_TYPES.has(r.ctv)) {
    throw new BadRequestError(`${field}.ctk/ctv must be non-f32 llama.cpp cache types`);
  }
  if (r.fa !== "on" && r.fa !== "off") throw new BadRequestError(`${field}.fa must be on or off`);
  if (typeof r.status !== "string" || !(KV_STATUSES as readonly string[]).includes(r.status)) {
    throw new BadRequestError(`${field}.status must be one of ${KV_STATUSES.join(", ")}`);
  }
  return {
    ctk: r.ctk,
    ctv: r.ctv,
    fa: r.fa,
    status: r.status as KvSupportStatus,
    splits: int(r.splits, `${field}.splits`, 0, 1_000_000),
    baseline_splits: int(r.baseline_splits, `${field}.baseline_splits`, 0, 1_000_000),
    detail: str(r.detail, `${field}.detail`, 500),
  };
}

async function resolveFitRunForWorker(request: FastifyRequest<{ Params: { id: string } }>): Promise<Test> {
  const worker = await authenticateWorker(request);
  const run = repo.getTest(undefined, request.params.id);
  if (!run) throw new NotFoundError("run not found");
  if (!run.worker_id || run.worker_id !== worker.id) throw new ForbiddenError("this run was dispatched to a different machine");
  if (run.kind !== "fit") throw new BadRequestError("that run is not a fit-map run");
  // A finished, failed or cancelled run is closed; a lease re-claim puts it
  // back to running before the new attempt posts.
  if (run.status !== "running") throw new ConflictError(`that run is ${run.status}, not running`);
  return run;
}

export const getFitPointsHandler =
  (scope: UserScope) => async (request: FastifyRequest<{ Params: { id: string } }>, reply: FastifyReply) => {
    const run = repo.getTest(scope(request), request.params.id);
    if (!run) throw new NotFoundError("run not found");
    if (run.worker_id) assertOwnsWorker(scope(request), run.worker_id);
    const kv =
      run.worker_id && run.llama_cpp_build && run.llama_cpp_backend
        ? fitRepo.listKv({ worker_id: run.worker_id, build: run.llama_cpp_build, backend: run.llama_cpp_backend, model_id: run.model_id })
        : [];
    return reply.send({ points: fitRepo.listPoints(run.id), kv, spec: (run.config as TestConfig).fit_map ?? null });
  };

export const getFlowHandler =
  (scope: UserScope) => async (request: FastifyRequest<{ Params: { flowId: string } }>, reply: FastifyReply) => {
    const flowId = request.params.flowId;
    if (typeof flowId !== "string" || !/^[A-Za-z0-9_-]{6,64}$/.test(flowId)) throw new BadRequestError("invalid flow id");
    const userId = scope(request);
    const ids = getDb()
      .prepare(
        `SELECT id FROM runs WHERE json_extract(config, '$.flow_id') = ? AND (? IS NULL OR user_id = ?)
         ORDER BY started_at ASC LIMIT 100`
      )
      .all(flowId, userId ?? null, userId ?? null) as { id: string }[];
    const tests = ids.map((r) => repo.getTest(userId, r.id)).filter((t): t is Test => !!t);
    return reply.send({ tests });
  };

export async function optimizeRoutes(app: FastifyInstance): Promise<void> {
  const postFitPoints = async (
    request: FastifyRequest<{ Params: { id: string }; Body: { points?: unknown } }>,
    reply: FastifyReply
  ) => {
    const run = await resolveFitRunForWorker(request);
    const raw = request.body?.points;
    if (!Array.isArray(raw)) throw new BadRequestError("points must be an array");
    if (raw.length > MAX_POINTS_PER_POST) throw new BadRequestError(`at most ${MAX_POINTS_PER_POST} points per post`);
    const points = raw.map((p, i) => validateFitPoint(p, `points[${i}]`));
    let stored: number;
    try {
      stored = fitRepo.upsertPoints({
        run_id: run.id,
        worker_id: run.worker_id ?? null,
        model_id: run.model_id,
        build: run.llama_cpp_build ?? null,
        points,
        max_rows: MAX_POINTS_PER_RUN,
      });
    } catch (e) {
      if (e instanceof FitPointLimitError) throw new BadRequestError(e.message);
      throw e;
    }
    return reply.send({ ok: true, stored });
  };
  app.post("/api/tests/:id/fit-points", { bodyLimit: 4 * 1024 * 1024 }, postFitPoints);

  const postKvSupport = async (
    request: FastifyRequest<{ Params: { id: string }; Body: { rows?: unknown } }>,
    reply: FastifyReply
  ) => {
    const run = await resolveFitRunForWorker(request);
    const raw = request.body?.rows;
    if (!Array.isArray(raw)) throw new BadRequestError("rows must be an array");
    if (raw.length > MAX_KV_ROWS) throw new BadRequestError(`at most ${MAX_KV_ROWS} rows`);
    const rows = raw.map((r, i) => validateKvRow(r, `rows[${i}]`));
    if (!run.worker_id || !run.llama_cpp_build || !run.llama_cpp_backend) throw new BadRequestError("that run has no machine/build attached");
    const stored = fitRepo.upsertKv({
      worker_id: run.worker_id,
      build: run.llama_cpp_build,
      backend: run.llama_cpp_backend,
      model_id: run.model_id,
      run_id: run.id,
      rows,
    });
    return reply.send({ ok: true, stored });
  };
  app.post("/api/tests/:id/kv-support", { bodyLimit: 1024 * 1024 }, postKvSupport);

  app.get("/api/tests/:id/fit-points", getFitPointsHandler(sessionScope));
  app.get("/api/flows/:flowId", getFlowHandler(sessionScope));
}
