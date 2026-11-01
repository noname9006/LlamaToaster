// The Serve page's API (docs/plans/MODEL_DAEMON_PLAN.md). The server never talks to
// a worker: commands are queue jobs, and state arrives on telemetry. Live state is
// kept in memory rather than rewritten to a row every few seconds; a server restart
// only means the page waits one telemetry tick.

import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { daemonRepo, MAX_ACTIVE_KEYS } from "../db/daemonRepo.js";
import { queueEvents } from "../queue-events.js";
import { BadRequestError, ConflictError, HttpError, NotFoundError } from "../errors.js";
import { assertOwnsWorker, resolveAuthUser } from "../auth-middleware.js";
import { authenticateWorker } from "../worker-auth.js";
import { parseDaemonStatus, parseDaemonUsage } from "../validate-daemon.js";
import { chargeIngest } from "../ingest-budget.js";
import {
  ALIAS_PATTERN,
  DEFAULT_GATEWAY,
  defaultAlias,
  validateGateway,
  validateLoadSettings,
  validateSampling,
} from "../../../shared/daemonSettings.js";
import { contextUsed, ppSpeed, speedCurve, summarizeUsage, tgSpeed } from "../../../shared/daemonUsage.js";
import type { DaemonLoadSpec, DaemonStatus, DaemonTelemetryResponse, DaemonView } from "../../../shared/daemonTypes.js";

// A viewer polls every few seconds. Past this with no poll, the worker stops reading
// memory, because a GPU read disturbs inference timing.
const WATCH_WINDOW_MS = 12_000;
const USAGE_WINDOW_DAYS = 30;
const MAX_PROFILE_ROWS = 20_000;
const MAX_CHART_POINTS = 1500;
const MAX_BODY_BYTES = 1024 * 1024;
const MMPROJ_FILENAME = /^[A-Za-z0-9._ ()/\\-]{1,512}$/;

// One usage row with its indexes, for the per-owner write budget.
const USAGE_ROW_BYTES = 400;

interface Live {
  status: DaemonStatus;
  at: number;
}

const live = new Map<string, Live>();
const watchedUntil = new Map<string, number>();
// What was last written for a machine's load, so a report repeating it writes nothing:
// these arrive every few seconds for as long as a model is loaded.
const lastLoadWritten = new Map<string, string>();
const LIVE_STATE_MAX_AGE_MS = 60 * 60 * 1000;
const SWEEP_EVERY_MS = 10 * 60 * 1000;
let lastSweepAt = 0;

/** Forgets what no machine has refreshed in an hour, including machines since deleted. */
export function sweepLiveState(now: number): void {
  lastSweepAt = now;
  for (const [id, state] of live) {
    if (now - state.at <= LIVE_STATE_MAX_AGE_MS) continue;
    live.delete(id);
    lastLoadWritten.delete(id);
  }
  for (const [id, until] of watchedUntil) {
    if (until <= now) watchedUntil.delete(id);
  }
}

export function liveStateSizes(): { live: number; watched: number; loads: number } {
  return { live: live.size, watched: watchedUntil.size, loads: lastLoadWritten.size };
}

function ownedWorker(userId: string | undefined, id: string) {
  const worker = repo.workerRepo.getWorker(id);
  if (!worker) throw new NotFoundError("unknown machine");
  assertOwnsWorker(userId, worker.id);
  return worker;
}

function capableOnlineWorker(userId: string | undefined, id: string) {
  const worker = ownedWorker(userId, id);
  if (worker.status === "offline") throw new ConflictError("that machine is offline -- start the worker and try again");
  if (!worker.capabilities?.includes("daemon-v1")) {
    throw new ConflictError("this worker is too old to serve models -- update it first");
  }
  return worker;
}

function supersede(workerId: string): void {
  const { claimedLoadIds } = daemonRepo.supersedeJobs(workerId);
  for (const id of claimedLoadIds) repo.queueRepo.requestCancel(id);
}

export async function daemonRoutes(app: FastifyInstance): Promise<void> {
  // --- Worker side ---
  app.post("/api/worker/daemon/telemetry", { logLevel: "silent", bodyLimit: MAX_BODY_BYTES }, async (req): Promise<DaemonTelemetryResponse> => {
    const worker = await authenticateWorker(req);
    const body = (req.body ?? {}) as { config_version?: unknown; config_model_id?: unknown; status?: unknown; usage?: unknown };
    const status = parseDaemonStatus(body.status);
    if (Date.now() - lastSweepAt > SWEEP_EVERY_MS) sweepLiveState(Date.now());
    if (status) {
      live.set(worker.id, { status, at: Date.now() });
      if (status.load_hash && status.model_id && status.settings) {
        // Only a new load adds a row. Over budget it waits for a later report,
        // which repeats it while the model stays loaded.
        const load = {
          load_hash: status.load_hash,
          model_id: status.model_id,
          build_tag: status.build_tag,
          settings: status.settings,
          actual: status.actual,
        };
        const signature = `${load.load_hash}:${JSON.stringify(load.actual)}`;
        try {
          const known = daemonRepo.hasLoad(worker.id, load.load_hash);
          if (!known) {
            chargeIngest(repo.catalogBudgetScopeForWorker(worker.id), USAGE_ROW_BYTES + JSON.stringify(load.settings).length);
          }
          if (!known || lastLoadWritten.get(worker.id) !== signature) {
            daemonRepo.recordLoad(worker.id, load);
            lastLoadWritten.set(worker.id, signature);
          }
        } catch (err) {
          if (!(err instanceof HttpError) || err.statusCode !== 429) throw err;
        }
      }
    }
    const parsed = parseDaemonUsage(body.usage);
    // Over budget, the rows stay spooled on the worker and come back next hour. The
    // rest of this call still answers: config and revocations must keep flowing.
    let acked: string[] = [];
    try {
      if (parsed.rows.length > 0) chargeIngest(repo.catalogBudgetScopeForWorker(worker.id), parsed.rows.length * USAGE_ROW_BYTES);
      daemonRepo.insertUsage(worker.id, parsed.rows);
      acked = parsed.acked;
    } catch (err) {
      if (!(err instanceof HttpError) || err.statusCode !== 429) throw err;
    }

    const config = daemonRepo.buildConfig(worker.id, status?.model_id ?? null);
    // The model part matters when a load's config was skipped as older: the stored
    // one then still carries sampling for whatever was loaded before.
    const behind =
      config != null &&
      (body.config_version !== config.version || (status?.model_id != null && body.config_model_id !== status.model_id));
    return {
      watched: (watchedUntil.get(worker.id) ?? 0) > Date.now(),
      acked,
      ...(behind ? { config } : {}),
    };
  });

  // --- Page side ---
  app.get<{ Params: { id: string } }>("/api/workers/:id/daemon", { logLevel: "silent" }, async (request): Promise<DaemonView> => {
    const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
    watchedUntil.set(worker.id, Date.now() + WATCH_WINDOW_MS);
    const state = live.get(worker.id);
    const job = daemonRepo.latestJob(worker.id);
    const pending = job && (job.status === "pending" || job.status === "claimed");
    return {
      status: state?.status ?? null,
      reported_at: state?.at ?? null,
      capable: worker.capabilities?.includes("daemon-v1") ?? false,
      gateway: daemonRepo.getGateway(worker.id)?.gateway ?? DEFAULT_GATEWAY,
      keys: daemonRepo.listKeys(worker.id),
      pending_job: pending ? { id: job.id, type: job.type, status: job.status } : null,
      last_error: job?.status === "failed" ? job.error : null,
    };
  });

  app.post<{
    Params: { id: string };
    Body: { model_id?: unknown; alias?: unknown; settings?: unknown; sampling?: unknown; mmproj_filename?: unknown; draft_model_id?: unknown };
  }>("/api/workers/:id/daemon/load", async (request, reply) => {
    const userId = resolveAuthUser(request)?.user.id;
    const worker = capableOnlineWorker(userId, request.params.id);
    const body = request.body ?? {};
    if (typeof body.model_id !== "string") throw new BadRequestError("model_id is required");
    // Answers exactly like an unknown id for a model this account cannot see.
    const model = repo.isModelVisibleTo(userId, body.model_id) ? repo.getModelForWorker(body.model_id, worker.id) : undefined;
    if (!model) throw new NotFoundError("unknown model");
    const { values: settings, errors } = validateLoadSettings(body.settings);
    const { sampling, errors: samplingErrors } = validateSampling(body.sampling);
    const problems = [...errors, ...samplingErrors];
    const filename = model.hf_file ?? model.filename ?? model.id;
    const alias = typeof body.alias === "string" && body.alias.trim() ? body.alias.trim() : defaultAlias(filename);
    if (!ALIAS_PATTERN.test(alias)) problems.push("API identifier may use letters, digits and . _ : / - only");
    let mmproj: string | null = null;
    if (body.mmproj_filename != null && body.mmproj_filename !== "") {
      // The worker checks this resolves inside its model folder too.
      if (typeof body.mmproj_filename !== "string" || !MMPROJ_FILENAME.test(body.mmproj_filename) || body.mmproj_filename.includes("..")) {
        problems.push("the projector file name is invalid");
      } else mmproj = body.mmproj_filename;
    }
    let draft = null;
    if (body.draft_model_id != null && body.draft_model_id !== "") {
      draft = typeof body.draft_model_id === "string" && repo.isModelVisibleTo(userId, body.draft_model_id) ? repo.getModelForWorker(body.draft_model_id, worker.id) ?? null : null;
      if (!draft) problems.push("unknown draft model");
    }
    if (problems.length > 0) throw new BadRequestError(problems.join("; "));

    daemonRepo.saveModelSettings(worker.id, model.id, { settings, sampling, alias, mmproj_filename: mmproj, draft_model_id: draft?.id ?? null });
    if (!daemonRepo.getGateway(worker.id)) daemonRepo.bumpConfig(worker.id, DEFAULT_GATEWAY);
    const config = daemonRepo.buildConfig(worker.id, model.id);
    if (!config) throw new ConflictError("could not prepare this machine's endpoint");
    const spec: DaemonLoadSpec = { model_id: model.id, model_name: filename, model, alias, settings, sampling, mmproj_filename: mmproj, draft_model: draft };

    supersede(worker.id);
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "daemon_load", payload: { spec, config, requested_at: Date.now() } });
    queueEvents.emit(worker.id);
    return reply.code(202).send({ ok: true, job_id: jobId });
  });

  app.post<{ Params: { id: string } }>("/api/workers/:id/daemon/unload", async (request, reply) => {
    const worker = capableOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
    supersede(worker.id);
    const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "daemon_unload", payload: {} });
    queueEvents.emit(worker.id);
    return reply.code(202).send({ ok: true, job_id: jobId });
  });

  app.get<{ Params: { id: string; modelId: string } }>("/api/workers/:id/daemon/models/:modelId/settings", async (request) => {
    const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
    return { settings: daemonRepo.getModelSettings(worker.id, request.params.modelId) };
  });

  // Applies on the next request, with no restart: telemetry delivers it.
  app.put<{ Params: { id: string }; Body: { model_id?: unknown; sampling?: unknown } }>("/api/workers/:id/daemon/sampling", async (request) => {
    const userId = resolveAuthUser(request)?.user.id;
    const worker = ownedWorker(userId, request.params.id);
    const body = request.body ?? {};
    if (typeof body.model_id !== "string") throw new BadRequestError("model_id is required");
    if (!repo.isModelVisibleTo(userId, body.model_id)) throw new NotFoundError("unknown model");
    const { sampling, errors } = validateSampling(body.sampling);
    if (errors.length > 0) throw new BadRequestError(errors.join("; "));
    daemonRepo.saveSampling(worker.id, body.model_id, sampling);
    if (daemonRepo.getGateway(worker.id)) daemonRepo.bumpConfig(worker.id);
    return { ok: true, sampling };
  });

  app.put<{ Params: { id: string }; Body: unknown }>("/api/workers/:id/daemon/gateway", async (request) => {
    const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
    const { gateway, errors } = validateGateway(request.body, daemonRepo.countActiveKeys(worker.id));
    if (errors.length > 0) throw new BadRequestError(errors.join("; "));
    daemonRepo.bumpConfig(worker.id, gateway);
    return { ok: true, gateway };
  });

  app.post<{ Params: { id: string }; Body: { name?: unknown } }>("/api/workers/:id/daemon/keys", async (request, reply) => {
    const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
    const raw = request.body?.name;
    const name = typeof raw === "string" && raw.trim() ? raw.trim().slice(0, 64) : "API key";
    if (daemonRepo.countActiveKeys(worker.id) >= MAX_ACTIVE_KEYS) throw new ConflictError(`revoke a key first: a machine holds at most ${MAX_ACTIVE_KEYS}`);
    const { info, key } = daemonRepo.createKey(worker.id, name);
    return reply.code(201).send({ ...info, key });
  });

  app.delete<{ Params: { id: string; keyId: string } }>("/api/workers/:id/daemon/keys/:keyId", async (request) => {
    const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
    if (!daemonRepo.revokeKey(worker.id, request.params.keyId)) throw new NotFoundError("unknown key");
    return { ok: true, gateway: daemonRepo.getGateway(worker.id)?.gateway ?? DEFAULT_GATEWAY };
  });

  app.get<{ Params: { id: string }; Querystring: { model_id?: string; load_hash?: string; include_concurrent?: string } }>(
    "/api/workers/:id/daemon/usage",
    { logLevel: "silent" },
    async (request) => {
      const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
      const modelId = request.query.model_id;
      if (!modelId) throw new BadRequestError("model_id is required");
      const loads = daemonRepo.listLoads(worker.id, modelId);
      const loadHash = request.query.load_hash || null;
      const includeConcurrent = request.query.include_concurrent === "1";
      const rows = daemonRepo.listUsage(worker.id, modelId, {
        since: Date.now() - USAGE_WINDOW_DAYS * 86_400_000,
        loadHash,
        limit: MAX_PROFILE_ROWS,
      });
      // Context figures come from the load being viewed, when one is, or the live one.
      const state = live.get(worker.id)?.status;
      const viewed = loadHash ? loads.find((l) => l.load_hash === loadHash) : loads[0];
      const actual = viewed?.actual ?? (state?.model_id === modelId ? state.actual : null);
      const kvMib = state?.load_hash && state.load_hash === viewed?.load_hash ? state.memory?.breakdown.kv_gpu ?? null : null;
      const profile = summarizeUsage(rows, { nCtx: actual?.n_ctx, slots: actual?.n_slots, kvMib });
      // Slots share throughput, so concurrent requests are not this model's speed at
      // this context. They are still counted in the profile above.
      const single = includeConcurrent ? rows : rows.filter((r) => r.concurrency <= 1);
      const points = single
        .filter((r) => r.status >= 200 && r.status < 300)
        .slice(0, MAX_CHART_POINTS)
        .map((r) => ({ at: r.at, ctx: contextUsed(r), pp: ppSpeed(r), tg: tgSpeed(r) }))
        .filter((p) => p.ctx != null && (p.pp != null || p.tg != null));
      return {
        loads,
        profile,
        curve: speedCurve(single.filter((r) => r.status >= 200 && r.status < 300)),
        points,
        concurrent_excluded: includeConcurrent ? 0 : rows.length - single.length,
        window_days: USAGE_WINDOW_DAYS,
      };
    }
  );

  app.delete<{ Params: { id: string }; Querystring: { model_id?: string } }>("/api/workers/:id/daemon/usage", async (request) => {
    const worker = ownedWorker(resolveAuthUser(request)?.user.id, request.params.id);
    const modelId = request.query.model_id;
    if (!modelId) throw new BadRequestError("model_id is required");
    return { ok: true, removed: daemonRepo.clearUsage(worker.id, modelId) };
  });
}
