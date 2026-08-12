import type { FastifyInstance } from "fastify";
import { v4 as uuid } from "uuid";
import { repo } from "../db/repo.js";
import type {
  TriggerPayload,
  Run,
  RunConfig,
  Backend,
  RunItemUpdateInput,
  InstalledBuild,
  Model,
} from "../../../shared/types.js";
import {
  isTerminalRunItemInput,
  isMtpDraftModel,
  GPU_MEMORY_ACCURACY_LEVELS,
  GPU_MEMORY_MEASUREMENT_SOURCES,
} from "../../../shared/types.js";
import { expandSweep } from "../../../shared/sweep.js";
import { getWorkerForRun, getDefaultWorker, type WorkerDef } from "../config.js";
import { describeWorkerError } from "../worker-errors.js";
import { getReleases, filterReleasesForWorker } from "../github-releases.js";

// The worker now acks /run as soon as it validates the request and kicks the
// benchmark off in the background (see worker/src/index.ts) rather than
// blocking until the whole thing finishes, so this only needs to cover a
// normal request/response round trip -- not the bench's own duration, which
// is governed entirely by the worker's own bench_timeout_ms instead.
const TRIGGER_FETCH_TIMEOUT_MS = Number(
  process.env.TRIGGER_TIMEOUT_MS ?? 20_000
);

const NUMERIC_SWEEP_FIELDS = [
  "n_prompt",
  "n_gen",
  "threads",
  "n_gpu_layers",
  "batch_size",
  "ubatch_size",
] as const;
const STRING_SWEEP_FIELDS = ["cache_type_k", "cache_type_v", "flash_attn", "mtp"] as const;

function validateSweep(sweep: unknown): string | null {
  if (!sweep || typeof sweep !== "object") return "sweep must be an object";
  const s = sweep as Record<string, unknown>;
  for (const field of NUMERIC_SWEEP_FIELDS) {
    const v = s[field];
    if (
      !Array.isArray(v) ||
      v.length === 0 ||
      !v.every((x) => typeof x === "number" && Number.isFinite(x))
    ) {
      return `sweep.${field} must be a non-empty array of numbers`;
    }
  }
  for (const field of STRING_SWEEP_FIELDS) {
    const v = s[field];
    if (
      !Array.isArray(v) ||
      v.length === 0 ||
      !v.every((x) => typeof x === "string" && x.length > 0)
    ) {
      return `sweep.${field} must be a non-empty array of non-empty strings`;
    }
  }
  if (typeof s.repeats !== "number" || !Number.isInteger(s.repeats) || s.repeats < 1) {
    return "sweep.repeats must be a positive integer";
  }
  return null;
}

const TICK_STATUSES = new Set(["loading", "processing", "generating", "benchmarking"]);
const TERMINAL_STATUSES = new Set(["done", "failed", "failed_oom", "cancelled"]);
const VALID_TEST_TYPES = new Set(["pp", "tg", "pg"]);
const NUMERIC_RESULT_FIELDS = [
  "n_prompt",
  "n_gen",
  "n_threads",
  "n_gpu_layers",
  "batch_size",
  "ubatch_size",
  "avg_tps",
  "stddev_tps",
  "ram_peak_mib",
  "ram_avg_mib",
] as const;
const STRING_RESULT_FIELDS = ["cache_type_k", "cache_type_v", "flash_attn", "mtp"] as const;

function validateIngestResult(value: unknown): string | null {
  if (!value || typeof value !== "object") return "must be an object";
  const row = value as Record<string, unknown>;
  if (!VALID_TEST_TYPES.has(row.test_type as string)) {
    return `test_type must be one of ${[...VALID_TEST_TYPES].join("/")}`;
  }
  for (const field of NUMERIC_RESULT_FIELDS) {
    if (typeof row[field] !== "number" || !Number.isFinite(row[field] as number)) {
      return `${field} must be a number`;
    }
  }
  for (const field of ["vram_peak_mib", "vram_avg_mib", "ram_free_before_mib", "vram_free_before_mib"] as const) {
    if (row[field] !== null && (typeof row[field] !== "number" || !Number.isFinite(row[field] as number))) {
      return `${field} must be a number or null`;
    }
  }
  for (const field of STRING_RESULT_FIELDS) {
    if (typeof row[field] !== "string") {
      return `${field} must be a string`;
    }
  }
  // Optional -- only ever populated by the llama-server/MTP path (see
  // shared/types.ts's ResultRow/IngestResultInput doc comments).
  for (const field of ["sample_count", "suspect_count", "spec_drafted", "spec_accepted"] as const) {
    if (row[field] !== undefined && (typeof row[field] !== "number" || !Number.isFinite(row[field] as number))) {
      return `${field} must be a number`;
    }
  }
  for (const field of ["suspect_samples", "repeat_samples"] as const) {
    const v = row[field];
    if (v !== undefined && (!Array.isArray(v) || !v.every((x) => typeof x === "number" && Number.isFinite(x)))) {
      return `${field} must be an array of numbers`;
    }
  }
  for (const field of [
    "system_memory_total_mb",
    "gpu_memory_total_mb",
    "gpu_layers_loaded",
    "total_model_layers",
  ] as const) {
    if (row[field] !== null && (typeof row[field] !== "number" || !Number.isFinite(row[field] as number))) {
      return `${field} must be a number or null`;
    }
  }
  // accuracy is always a concrete level, never null/absent -- "unavailable"
  // is itself the valid value for "couldn't be determined", see
  // shared/types.ts's GpuMemoryAccuracyLevel doc comment.
  for (const field of [
    "gpu_memory_total_accuracy",
    "gpu_memory_free_start_accuracy",
    "gpu_memory_model_avg_accuracy",
    "gpu_memory_model_peak_accuracy",
  ] as const) {
    if (!GPU_MEMORY_ACCURACY_LEVELS.includes(row[field] as (typeof GPU_MEMORY_ACCURACY_LEVELS)[number])) {
      return `${field} must be one of ${GPU_MEMORY_ACCURACY_LEVELS.join("/")}`;
    }
  }
  // source, unlike accuracy, is null exactly when its paired value is null
  // (accuracy "unavailable").
  for (const field of [
    "gpu_memory_total_source",
    "gpu_memory_free_start_source",
    "gpu_memory_model_avg_source",
    "gpu_memory_model_peak_source",
  ] as const) {
    if (
      row[field] !== null &&
      !GPU_MEMORY_MEASUREMENT_SOURCES.includes(row[field] as (typeof GPU_MEMORY_MEASUREMENT_SOURCES)[number])
    ) {
      return `${field} must be one of ${GPU_MEMORY_MEASUREMENT_SOURCES.join("/")} or null`;
    }
  }
  return null;
}

// Covers both tiers of worker->server item updates: best-effort ticks
// (loading/processing/generating/benchmarking) and the retried terminal
// outcome (done/failed/failed_oom). See shared/types.ts's
// RunItemTickInput/RunItemTerminalInput for the two shapes this discriminates.
function validateRunItemUpdate(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return "payload must be an object";
  const p = payload as Record<string, unknown>;
  const status = p.status;
  if (typeof status !== "string" || !(TICK_STATUSES.has(status) || TERMINAL_STATUSES.has(status))) {
    return `status must be one of ${[...TICK_STATUSES, ...TERMINAL_STATUSES].join("/")}`;
  }
  if (p.detail !== undefined && typeof p.detail !== "string") return "detail must be a string";
  if (p.error !== undefined && typeof p.error !== "string") return "error must be a string";
  if (p.ram_mib !== undefined && (typeof p.ram_mib !== "number" || !Number.isFinite(p.ram_mib))) {
    return "ram_mib must be a number";
  }
  if (
    p.vram_mib !== undefined &&
    p.vram_mib !== null &&
    (typeof p.vram_mib !== "number" || !Number.isFinite(p.vram_mib))
  ) {
    return "vram_mib must be a number or null";
  }
  if (
    p.ram_peak_mib !== undefined &&
    (typeof p.ram_peak_mib !== "number" || !Number.isFinite(p.ram_peak_mib))
  ) {
    return "ram_peak_mib must be a number";
  }
  if (
    p.vram_peak_mib !== undefined &&
    p.vram_peak_mib !== null &&
    (typeof p.vram_peak_mib !== "number" || !Number.isFinite(p.vram_peak_mib))
  ) {
    return "vram_peak_mib must be a number or null";
  }
  if (p.ram_avg_mib !== undefined && (typeof p.ram_avg_mib !== "number" || !Number.isFinite(p.ram_avg_mib))) {
    return "ram_avg_mib must be a number";
  }
  for (const field of [
    "vram_avg_mib",
    "ram_free_before_mib",
    "vram_free_before_mib",
    "live_tps",
  ] as const) {
    if (p[field] !== undefined && p[field] !== null && (typeof p[field] !== "number" || !Number.isFinite(p[field] as number))) {
      return `${field} must be a number or null`;
    }
  }
  if (p.backend_device_name !== undefined && typeof p.backend_device_name !== "string") {
    return "backend_device_name must be a string";
  }
  if (status === "done" && p.results !== undefined) {
    if (!Array.isArray(p.results)) return "results must be an array";
    for (const [i, entry] of p.results.entries()) {
      const err = validateIngestResult(entry);
      if (err) return `results[${i}].${err}`;
    }
  }
  return null;
}

const WORKER_HEALTH_TIMEOUT_MS = 5_000;
const LLAMA_CPP_INFO_TIMEOUT_MS = 10_000;
const BUILD_ACTIVATE_TIMEOUT_MS = 15_000;
// Matches workers.ts's WORKER_INSTALL_TIMEOUT_MS -- a real llama.cpp release
// archive download+extract can legitimately take minutes.
const BUILD_INSTALL_TIMEOUT_MS = 10 * 60 * 1000;

// deviceName is the Tier-1 backend_device_name fallback (see shared/types.ts's
// Run.backend_device_name) -- always sourced from the one /health fetch at
// the top of ensureActiveBuild below, which runs before any of its 3 return
// paths branch, so all 3 carry it through rather than just the fast path.
type BuildResolution = { build: string; backend: Backend; deviceName?: string } | { error: string };

async function activateOnWorker(worker: WorkerDef, tag: string): Promise<{ ok: true } | { error: string }> {
  try {
    const res = await fetch(`${worker.url}/llama-cpp/activate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tag }),
      signal: AbortSignal.timeout(BUILD_ACTIVATE_TIMEOUT_MS),
    });
    if (!res.ok) {
      const text = await res.text();
      return { error: `failed to activate llama.cpp ${tag}: ${res.status} ${text}` };
    }
    return { ok: true };
  } catch (err) {
    return { error: `failed to activate llama.cpp ${tag}: ${describeWorkerError(err)}` };
  }
}

// The orchestrator's WORKERS env var declares a static llama_cpp_build per
// worker, but the Workers page can switch a worker's active build at runtime
// (see routes/workers.ts), and a worker can also simply have nothing active
// yet (fresh install, or its "active" build was deleted). Rather than
// silently falling back to the possibly-stale/placeholder declared value,
// self-heal at trigger time: activate the most recently installed build if
// one exists, or download+install+activate the latest available release for
// this worker's platform/arch/backend if nothing is installed at all. Only
// a genuinely unreachable worker, a busy worker with no active build, or a
// worker with no installable release at all should still fail the trigger.
async function ensureActiveBuild(worker: WorkerDef): Promise<BuildResolution> {
  let health: { busy?: boolean; active_build?: unknown; backend?: unknown; backend_device_name?: unknown };
  try {
    const res = await fetch(`${worker.url}/health`, { signal: AbortSignal.timeout(WORKER_HEALTH_TIMEOUT_MS) });
    if (!res.ok) throw new Error(`worker responded ${res.status}`);
    health = (await res.json()) as typeof health;
  } catch (err) {
    return { error: describeWorkerError(err) };
  }
  const deviceName = typeof health.backend_device_name === "string" ? health.backend_device_name : undefined;
  if (typeof health.active_build === "string" && typeof health.backend === "string") {
    return { build: health.active_build, backend: health.backend as Backend, deviceName };
  }
  if (health.busy) {
    return { error: "worker is busy running another benchmark and has no active llama.cpp build" };
  }

  let info: { platform: string; arch: string; backend: Backend; installed: InstalledBuild[]; active_tag: string | null };
  try {
    const res = await fetch(`${worker.url}/llama-cpp`, { signal: AbortSignal.timeout(LLAMA_CPP_INFO_TIMEOUT_MS) });
    if (!res.ok) throw new Error(`worker responded ${res.status}`);
    info = (await res.json()) as typeof info;
  } catch (err) {
    return { error: describeWorkerError(err) };
  }

  // Worker already sorts `installed` by installed_at descending (see
  // worker/src/llama-builds.ts's listInstalledBuilds), so [0] is "the
  // latest downloaded build" with no new sorting logic needed here.
  if (info.installed.length > 0) {
    const latest = info.installed[0];
    const activated = await activateOnWorker(worker, latest.tag);
    if ("error" in activated) return activated;
    return { build: latest.tag, backend: info.backend, deviceName };
  }

  // Nothing installed at all -- find the latest release with an asset that
  // actually matches this worker (mirrors WorkerCard.tsx's client-side
  // filter, which the equivalent computation in workers.ts's GET
  // /api/workers/:name/llama-cpp route skips -- don't repeat that gap here).
  let releases;
  try {
    releases = await getReleases();
  } catch (err) {
    return {
      error: `no llama.cpp build installed on this worker, and GitHub releases are unreachable: ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }
  const available = filterReleasesForWorker(releases, info.platform, info.arch, worker.backend).filter(
    (r) => r.assets.length > 0
  );
  if (available.length === 0) {
    return {
      error: "no llama.cpp build installed on this worker, and no matching release is available to auto-download",
    };
  }
  const release = available[0];
  const asset = release.assets[0];

  try {
    const res = await fetch(`${worker.url}/llama-cpp/install`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tag: release.tag, asset_name: asset.name, download_url: asset.download_url }),
      signal: AbortSignal.timeout(BUILD_INSTALL_TIMEOUT_MS),
    });
    if (!res.ok) {
      const text = await res.text();
      return { error: `auto-install of llama.cpp ${release.tag} failed: ${res.status} ${text}` };
    }
  } catch (err) {
    return { error: `auto-install of llama.cpp ${release.tag} failed: ${describeWorkerError(err)}` };
  }

  const activated = await activateOnWorker(worker, release.tag);
  if ("error" in activated) return activated;
  return { build: release.tag, backend: info.backend, deviceName };
}

type SendRunResult =
  | { ok: true }
  // `error`/`detail` are the reply shape a route handler sends straight to
  // the client; `full` is the more verbose message recorded against every
  // run_item via repo.failAllRunItems -- kept distinct so the two dispatch
  // call sites below (an HTTP route with a client waiting, and the
  // fire-and-forget queue dispatcher below with no client at all) can each
  // use only the half that applies to them.
  | { error: string; detail: string; full: string };

// POSTs an already-created run to the worker that should execute it. Shared
// by the trigger route's immediate-dispatch path and dispatchScheduledRun's
// queued-run path below -- both create/already have the run+run_items rows
// and just need this one HTTP round trip to actually kick it off.
async function sendRunToWorker(
  worker: WorkerDef,
  runId: string,
  model: Model,
  mtpModel: Model | undefined,
  sweep: TriggerPayload["sweep"],
  build: string,
  backend: Backend
): Promise<SendRunResult> {
  try {
    const res = await fetch(`${worker.url}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        run_id: runId,
        model_id: model.id,
        model,
        mtp_model: mtpModel,
        sweep,
        llama_cpp_build: build,
        llama_cpp_backend: backend,
      }),
      signal: AbortSignal.timeout(TRIGGER_FETCH_TIMEOUT_MS),
    });
    if (!res.ok) {
      const text = await res.text();
      return { error: "worker rejected run", detail: text, full: `worker rejected run: ${res.status} ${text}` };
    }
    return { ok: true };
  } catch (err) {
    const message = describeWorkerError(err);
    return { error: "could not reach worker", detail: message, full: message };
  }
}

// Called whenever a worker might have just freed up (a run finalized, or a
// stale "running" row was reconciled away) to see whether anything is
// waiting behind it in that worker's queue (see repo.hasActiveRunForWorker/
// getNextScheduledRunForWorker -- a trigger against a busy worker joins a
// FIFO queue instead of being rejected). Deliberately fire-and-forget from
// every call site: resolving a build can itself take minutes (installing a
// fresh llama.cpp release, same as ensureActiveBuild's own doc comment
// above), which is far too long to hold open the HTTP request that happened
// to be the one to notice the worker was free.
async function dispatchScheduledRun(app: FastifyInstance, workerName: string): Promise<void> {
  const next = repo.getNextScheduledRunForWorker(workerName);
  if (!next) return;
  const worker = getWorkerForRun(workerName);
  if (!worker) {
    app.log.error({ run_id: next.id, worker: workerName }, "queued run's worker is no longer configured");
    repo.failAllRunItems(next.id, `worker "${workerName}" is no longer configured`);
    return;
  }
  const model = repo.getModel(next.config.model_id);
  if (!model) {
    app.log.error({ run_id: next.id, model_id: next.config.model_id }, "queued run's model no longer exists");
    repo.failAllRunItems(next.id, "model this run was queued for no longer exists");
    return;
  }
  const mtpModel = next.config.mtp_model_id ? repo.getModel(next.config.mtp_model_id) : undefined;

  const live = await ensureActiveBuild(worker);
  if ("error" in live) {
    app.log.error({ run_id: next.id, worker: workerName, error: live.error }, "queued run dispatch failed");
    repo.failAllRunItems(next.id, live.error);
    return;
  }
  repo.markRunRunning(next.id, {
    llama_cpp_build: live.build,
    llama_cpp_backend: live.backend,
    backend_device_name: live.deviceName,
    started_at: Date.now(),
  });
  app.log.info({ run_id: next.id, worker: workerName, build: live.build }, "queued run dispatching");

  const sent = await sendRunToWorker(worker, next.id, model, mtpModel, next.config.sweep, live.build, live.backend);
  if ("error" in sent) {
    app.log.error({ run_id: next.id, worker: workerName, error: sent.full }, "queued run rejected by worker");
    repo.failAllRunItems(next.id, sent.full);
  }
}

export async function runsRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/runs", async () => {
    return { runs: repo.listRuns() };
  });

  app.get<{ Params: { id: string } }>(
    "/api/runs/:id",
    // Polled every ~1-2s by every open Runs/RunDetail tab while any run is
    // active -- logLevel: "silent" suppresses Fastify's default per-request
    // log pair for just this route; index.ts's polling-summary hook counts
    // these instead and reports volume once a minute.
    { logLevel: "silent" },
    async (request, reply) => {
      const data = repo.getRunWithResults(request.params.id);
      if (!data) return reply.code(404).send({ error: "run not found" });
      // Best-effort: only meaningful while the run is actually in flight, and
      // a worker that's slow or unreachable shouldn't block viewing the rest
      // of the run's data -- same posture as workers.ts's other diagnostic
      // /health reads.
      let paused: boolean | undefined;
      if (data.run.status === "running") {
        const worker = getWorkerForRun(data.run.worker_name);
        if (worker) {
          try {
            const res = await fetch(`${worker.url}/health`, { signal: AbortSignal.timeout(WORKER_HEALTH_TIMEOUT_MS) });
            if (res.ok) {
              const health = (await res.json()) as {
                paused?: unknown;
                busy?: unknown;
                current_run?: { run_id?: unknown } | null;
              };
              if (typeof health.paused === "boolean") paused = health.paused;

              // The worker has no persistent state of its own -- a restart
              // (crash, manual, pm2) or an exhausted safeItemTerminal retry
              // on the run's last item (see worker/src/index.ts) can leave
              // this row "running" forever with the worker no longer aware
              // of it at all. Reconcile against what the worker actually
              // reports rather than trusting our own stale row.
              const workerRunId =
                health.current_run && typeof health.current_run === "object"
                  ? (health.current_run as { run_id?: unknown }).run_id
                  : undefined;
              const staleAgainstWorker =
                health.busy === false || (typeof workerRunId === "string" && workerRunId !== data.run.id);
              if (staleAgainstWorker) {
                const reconciled = repo.reconcileStaleRun(data.run.id, "Cancelled");
                if (reconciled) {
                  data.run = reconciled;
                  data.items = repo.getRunItems(data.run.id);
                  paused = undefined;
                  // This worker just turned out to be free -- give its
                  // queue a chance too, same as the normal finalize path
                  // below (a run stuck "running" forever with nothing ever
                  // reaching the terminal item-update route would otherwise
                  // never unblock whatever's queued behind it).
                  void dispatchScheduledRun(app, reconciled.worker_name);
                }
              }
            }
          } catch {
            /* worker unreachable -- leave paused undefined and don't reconcile,
               a transient network blip shouldn't cancel a run that's actually
               still running fine */
          }
        }
      }
      return { ...data, paused };
    }
  );

  app.post<{ Body: TriggerPayload }>(
    "/api/runs/trigger",
    async (request, reply) => {
      const body = request.body;
      if (!body || !body.model_id || !body.sweep) {
        return reply.code(400).send({ error: "model_id and sweep are required" });
      }
      const sweepError = validateSweep(body.sweep);
      if (sweepError) {
        return reply.code(400).send({ error: sweepError });
      }
      const model = repo.getModel(body.model_id);
      if (!model) {
        return reply.code(400).send({ error: "unknown model_id" });
      }
      // A standalone MTP/draft companion file (see shared/types.ts's
      // isMtpDraftModel) isn't a loadable base model on its own -- the
      // client's model picker already excludes these, but a direct API call
      // could still name one here, so re-check server-side too rather than
      // letting every item in the run fail at the llama.cpp load step. Uses
      // the live-recomputed check rather than trusting metadata.mtp_role
      // alone, so a model registered before a detection-logic fix is still
      // correctly rejected.
      if (isMtpDraftModel(model)) {
        return reply.code(400).send({
          error: "model_id refers to a standalone MTP/draft companion file, not a benchmarkable base model",
        });
      }

      // MTP needs either the base model's own head (Qwen/DeepSeek/GLM-style,
      // baked into the GGUF -- see shared/types.ts's ModelMetadata.mtp_layers)
      // or an explicit companion drafter model (Gemma-4-style, --model-draft).
      // Checked here, before any run/run_items rows exist, same posture as
      // ensureActiveBuild below -- a sweep that can't actually run shouldn't
      // leave a permanently-misleading "failed" Run behind.
      let mtpModel: Model | undefined;
      if (body.sweep.mtp.includes("on")) {
        const modelMtpCapable = typeof model.metadata.mtp_layers === "number" && model.metadata.mtp_layers > 0;
        if (!modelMtpCapable) {
          if (!body.mtp_model_id) {
            return reply.code(400).send({
              error:
                'sweep includes mtp:"on" but this model has no built-in MTP head -- pick an MTP/draft companion model first',
            });
          }
          mtpModel = repo.getModel(body.mtp_model_id);
          if (!mtpModel) {
            return reply.code(400).send({ error: "unknown mtp_model_id" });
          }
          if (!isMtpDraftModel(mtpModel)) {
            return reply.code(400).send({ error: "mtp_model_id does not refer to a registered MTP/draft model" });
          }
        }
      }

      const requestedWorkerName = resolveWorkerName(body.worker_name);
      const worker = requestedWorkerName
        ? getWorkerForRun(requestedWorkerName)
        : getDefaultWorker();
      if (!worker) {
        request.log.warn(
          { worker_name: requestedWorkerName },
          "run trigger rejected: unknown or unconfigured worker"
        );
        return reply.code(400).send({
          error: requestedWorkerName
            ? `unknown worker: ${requestedWorkerName}`
            : "no workers configured",
        });
      }
      // A worker only ever runs one benchmark at a time (see worker/src/
      // index.ts's `busy` flag) -- rather than rejecting this trigger
      // outright the way a direct hit against a busy worker's own /run
      // would, queue it: create the run/run_items rows now (so the UI can
      // show it immediately) with status 'scheduled', skip contacting the
      // worker or resolving a build (both are only meaningful once this
      // run's turn actually comes), and let dispatchScheduledRun pick it up
      // once whatever's ahead of it on this worker finishes. Checking DB
      // state (not the worker's live /health) also naturally chains: a
      // second trigger against an already-queued worker queues behind the
      // first, giving a real FIFO instead of just one deep.
      if (repo.hasActiveRunForWorker(worker.name)) {
        const scheduledRun: Run = {
          id: uuid(),
          worker_name: worker.name,
          llama_cpp_build: worker.llama_cpp_build,
          llama_cpp_backend: worker.backend,
          model_id: body.model_id,
          config: {
            model_id: body.model_id,
            mtp_model_id: mtpModel?.id,
            sweep: body.sweep,
          } as RunConfig,
          status: "scheduled",
          started_at: Date.now(),
        };
        repo.createRun(scheduledRun);
        repo.createRunItems(scheduledRun.id, expandSweep(body.sweep));
        request.log.info(
          { run_id: scheduledRun.id, worker: worker.name, model_id: body.model_id },
          "run scheduled: worker busy"
        );
        return reply.code(201).send({ run: scheduledRun });
      }

      const live = await ensureActiveBuild(worker);
      if ("error" in live) {
        request.log.warn(
          { worker: worker.name, model_id: body.model_id, error: live.error },
          "run trigger rejected: could not ensure an active llama.cpp build"
        );
        return reply.code(502).send({ error: live.error });
      }
      request.log.debug(
        { worker: worker.name, model_id: body.model_id, sweep: body.sweep, build: live.build },
        "trigger requested"
      );
      const run: Run = {
        id: uuid(),
        worker_name: worker.name,
        llama_cpp_build: live.build,
        llama_cpp_backend: live.backend,
        backend_device_name: live.deviceName,
        model_id: body.model_id,
        config: {
          model_id: body.model_id,
          mtp_model_id: mtpModel?.id,
          sweep: body.sweep,
        } as RunConfig,
        status: "running",
        started_at: Date.now(),
      };
      repo.createRun(run);
      const items = expandSweep(body.sweep);
      repo.createRunItems(run.id, items);
      request.log.info(
        { run_id: run.id, worker: worker.name, model_id: body.model_id, items: items.length },
        "run created"
      );

      const sent = await sendRunToWorker(worker, run.id, model, mtpModel, body.sweep, live.build, live.backend);
      if ("error" in sent) {
        request.log.error({ run_id: run.id, error: sent.full }, "worker rejected run");
        repo.failAllRunItems(run.id, sent.full);
        return reply.code(502).send({ error: sent.error, detail: sent.detail });
      }
      request.log.debug({ run_id: run.id }, "worker accepted run, items reporting individually");

      return reply.code(201).send({ run });
    }
  );

  // Replaces the old whole-run /ingest: the worker now reports one sweep
  // combination at a time (see shared/sweep.ts's expandSweep), so a crash on
  // one item doesn't lose the rest. Two tiers share this route -- a
  // best-effort progress tick (loading/processing/generating/benchmarking)
  // and a terminal outcome (done/failed/failed_oom) -- discriminated by
  // `status`, matching shared/types.ts's RunItemTickInput/RunItemTerminalInput.
  app.post<{ Params: { id: string; idx: string }; Body: RunItemUpdateInput }>(
    "/api/runs/:id/items/:idx",
    // Same rationale as GET /api/runs/:id above -- the worker ticks this
    // every couple seconds for progress. logLevel: "silent" only suppresses
    // Fastify's automatic per-request pair; the two explicit app.log calls
    // below (validation rejection, terminal outcome) still print since they
    // go through the top-level logger, not this route's silenced child one.
    { logLevel: "silent" },
    async (request, reply) => {
      const { id } = request.params;
      const idx = Number(request.params.idx);
      if (!Number.isInteger(idx) || idx < 0) {
        return reply.code(400).send({ error: "idx must be a non-negative integer" });
      }
      const validationError = validateRunItemUpdate(request.body);
      if (validationError) {
        app.log.warn(
          { run_id: id, idx, error: validationError },
          "item update rejected: invalid payload"
        );
        return reply.code(400).send({ error: validationError });
      }
      const body = request.body;

      if (isTerminalRunItemInput(body)) {
        app.log[body.status === "done" ? "info" : "error"](
          { run_id: id, idx, status: body.status, error: body.error },
          "item terminal update received"
        );
        const run = repo.recordRunItemTerminal(id, idx, body);
        if (!run) return reply.code(404).send({ error: "run not found" });
        // recordRunItemTerminal only flips the run off 'running' once every
        // item is terminal -- a status other than 'running' here means this
        // was the update that just finished the whole run, so this worker
        // may have something queued behind it (see dispatchScheduledRun).
        if (run.status !== "running") {
          void dispatchScheduledRun(app, run.worker_name);
        }
        return reply.code(200).send({ ok: true, run_status: run.status });
      }

      repo.updateRunItemTick(id, idx, body);
      return reply.code(200).send({ ok: true });
    }
  );
}

function resolveWorkerName(name?: string): string | undefined {
  return name ?? process.env.DEFAULT_WORKER_NAME ?? undefined;
}
