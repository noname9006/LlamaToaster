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
import { DEFAULT_FIT_MARGIN_MIB, isFitMargin, KV_CACHE_TYPES } from "../../../shared/fitParams.js";
import { MAX_FIT_CTX_STOPS } from "../../../shared/optimizeFlow.js";
import type { KvSupportRow, KvSupportStatus } from "../../../shared/kvSupport.js";
import type { Backend, FitCheckJobPayload, FitCheckResult, Model, Test, TestConfig, Worker } from "../../../shared/types.js";
import { queueEvents } from "../queue-events.js";
import { assetMatchesWorker } from "../github-releases.js";

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

  // docs/plans/HEADER_FIT_PLAN.md -- the model list's verdicts from
  // llama-fit-params itself. Asks the machine to size the given models at one
  // context (from the header alone for a model it does not hold); answers
  // already on file for the same build, context and margin -- sized against
  // about the same free VRAM, within a day -- are not asked again.
  app.post<{ Params: { id: string }; Body: { model_ids?: unknown; ctx?: unknown; margin_mib?: unknown } }>(
    "/api/workers/:id/fit-check",
    { bodyLimit: 64 * 1024 },
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(sessionScope(request), worker.id);
      if (worker.status === "offline") throw new ConflictError("that machine is offline");
      if (!(worker.capabilities ?? []).includes("fit-check-v1")) {
        throw new ConflictError("that machine runs an older worker build -- update it for llama.cpp fit checks");
      }
      const body = request.body ?? {};
      const ctx = Number(body.ctx);
      if (!Number.isInteger(ctx) || ctx < 512 || ctx > 4_194_304) throw new BadRequestError("ctx must be an integer in [512, 4194304]");
      const margin = body.margin_mib === undefined ? DEFAULT_FIT_MARGIN_MIB : Number(body.margin_mib);
      if (!isFitMargin(margin)) throw new BadRequestError("margin_mib must be 512, 1024 or 1536");
      if (!Array.isArray(body.model_ids) || body.model_ids.length === 0 || body.model_ids.length > MAX_FIT_CHECK_MODELS) {
        throw new BadRequestError(`model_ids must list 1..${MAX_FIT_CHECK_MODELS} models`);
      }
      const backend = worker.backend;
      const build = newestInstalledBuild(worker, backend);
      if (!backend || !build) throw new ConflictError("no llama.cpp build installed on that machine for its backend");

      const freeNow = worker.vram?.vram_free_before_mib ?? null;
      const known = new Map(fitRepo.listFitChecks({ worker_id: worker.id, ctx, margin_mib: margin }).map((r) => [r.model_id, r]));
      const fresh = (id: string) => {
        const r = known.get(id);
        if (!r || r.llama_cpp_build !== build || r.created_at == null) return false;
        if (Date.now() - r.created_at > FIT_CHECK_FRESH_MS) return false;
        if (r.error) return false;
        return freeNow == null || r.vram_free_mib == null || Math.abs(r.vram_free_mib - freeNow) <= FIT_CHECK_FREE_DRIFT_MIB;
      };
      const models: Model[] = [];
      for (const raw of body.model_ids) {
        if (typeof raw !== "string") throw new BadRequestError("model_ids must be strings");
        const model = repo.getModelForWorker(raw, worker.id);
        if (!model) throw new BadRequestError(`unknown model ${raw}`);
        if (!fresh(model.id)) models.push(model);
      }
      if (models.length === 0) return reply.send({ ok: true, queued: 0 });
      // One check job at a time per machine: a list re-render must not stack
      // copies behind a long benchmark.
      const pending = getDb()
        .prepare(`SELECT COUNT(*) AS n FROM worker_jobs WHERE worker_id = ? AND job_type = 'fit_check' AND status IN ('pending','claimed')`)
        .get(worker.id) as { n: number };
      if (pending.n > 0) return reply.send({ ok: true, queued: 0, pending: true });
      const memTotalMib = worker.hardware?.mem_total_bytes ? Math.floor(worker.hardware.mem_total_bytes / 1048576) : null;
      const payload: FitCheckJobPayload = {
        ctx,
        margin_mib: margin,
        llama_cpp_build: build,
        llama_cpp_backend: backend,
        ram_budget_mib: memTotalMib != null ? Math.max(0, memTotalMib - margin) : null,
        models,
      };
      repo.queueRepo.enqueueJob(worker.id, { type: "fit_check", payload });
      queueEvents.emit(worker.id);
      return reply.code(202).send({ ok: true, queued: models.length });
    }
  );

  app.get<{ Params: { id: string }; Querystring: { ctx?: string; margin_mib?: string } }>(
    "/api/workers/:id/fit-checks",
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(sessionScope(request), worker.id);
      const ctx = Number(request.query.ctx);
      if (!Number.isInteger(ctx) || ctx <= 0) throw new BadRequestError("ctx must be a positive integer");
      const margin = request.query.margin_mib === undefined ? DEFAULT_FIT_MARGIN_MIB : Number(request.query.margin_mib);
      if (!isFitMargin(margin)) throw new BadRequestError("margin_mib must be 512, 1024 or 1536");
      const pending = getDb()
        .prepare(`SELECT COUNT(*) AS n FROM worker_jobs WHERE worker_id = ? AND job_type = 'fit_check' AND status IN ('pending','claimed')`)
        .get(worker.id) as { n: number };
      return reply.send({ results: fitRepo.listFitChecks({ worker_id: worker.id, ctx, margin_mib: margin }), pending: pending.n > 0 });
    }
  );

  // Worker -> server: answers of a fit_check job, posted as they come in.
  app.post<{ Body: { results?: unknown } }>("/api/worker/fit-checks", { bodyLimit: 2 * 1024 * 1024 }, async (request, reply) => {
    const worker = await authenticateWorker(request);
    const raw = request.body?.results;
    if (!Array.isArray(raw) || raw.length > MAX_FIT_CHECK_MODELS) throw new BadRequestError(`results must be an array of at most ${MAX_FIT_CHECK_MODELS}`);
    const results = raw.map((r, i) => validateFitCheckResult(r, `results[${i}]`));
    return reply.send({ ok: true, stored: fitRepo.upsertFitChecks(worker.id, results) });
  });
}

const MAX_FIT_CHECK_MODELS = 200;
const FIT_CHECK_FRESH_MS = 24 * 3600 * 1000;
const FIT_CHECK_FREE_DRIFT_MIB = 256;

function newestInstalledBuild(worker: Worker, backend: Backend | null | undefined): string | null {
  if (!backend || !worker.platform || !worker.arch) return null;
  const tagNum = (tag: string) => Number(/^b(\d+)/.exec(tag)?.[1] ?? -1);
  const matching = worker.installedBuilds
    .filter((b) => assetMatchesWorker(b.asset_name, worker.platform!, worker.arch!, backend))
    .sort((a, b) => tagNum(b.tag) - tagNum(a.tag));
  return matching[0]?.tag ?? null;
}

function validateFitCheckResult(raw: unknown, field: string): FitCheckResult {
  if (!raw || typeof raw !== "object") throw new BadRequestError(`${field} must be an object`);
  const r = raw as Record<string, unknown>;
  const str = (v: unknown, f: string, max = 256): string => {
    if (typeof v !== "string" || v.length === 0 || v.length > max) throw new BadRequestError(`${field}.${f} must be a non-empty string`);
    return v;
  };
  const optNum = (v: unknown, f: string): number | null => {
    if (v === null || v === undefined) return null;
    if (typeof v !== "number" || !Number.isFinite(v)) throw new BadRequestError(`${field}.${f} must be a number`);
    return v;
  };
  const ctx = r.ctx;
  if (typeof ctx !== "number" || !Number.isInteger(ctx) || ctx <= 0) throw new BadRequestError(`${field}.ctx must be a positive integer`);
  const margin = r.margin_mib;
  if (!isFitMargin(margin)) throw new BadRequestError(`${field}.margin_mib must be 512, 1024 or 1536`);
  if (r.source !== null && r.source !== "local" && r.source !== "header") throw new BadRequestError(`${field}.source must be local, header or null`);
  const error = r.error === null || r.error === undefined ? null : str(r.error, "error", 2000);
  const point = r.point === null || r.point === undefined ? null : validateFitPoint(r.point, `${field}.point`);
  if (!point && !error) throw new BadRequestError(`${field} needs a point or an error`);
  const fetched = optNum(r.fetched_bytes, "fetched_bytes");
  return {
    model_id: str(r.model_id, "model_id", 128),
    ctx,
    margin_mib: margin,
    llama_cpp_build: str(r.llama_cpp_build, "llama_cpp_build", 128),
    source: (r.source as FitCheckResult["source"]) ?? null,
    vram_free_mib: optNum(r.vram_free_mib, "vram_free_mib"),
    point,
    error,
    ...(fetched != null ? { fetched_bytes: fetched } : {}),
  };
}
