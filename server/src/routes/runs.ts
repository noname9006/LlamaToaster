import type { FastifyInstance } from "fastify";
import { v4 as uuid } from "uuid";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { repo } from "../db/repo.js";
import { queueEvents } from "../queue-events.js";
import { safeEqual, hashToken } from "../session.js";
import { userOrIpKeyGenerator, resolveAuthUser, assertOwnsWorker } from "../auth-middleware.js";
import { BadRequestError, ConflictError, ForbiddenError, NotFoundError, UnauthorizedError } from "../errors.js";
import type {
  TriggerPayload,
  Run,
  RunConfig,
  Backend,
  RunItemUpdateInput,
  Model,
  Worker,
  InstallBuildJobPayload,
  BenchmarkJob,
} from "../../../shared/types.js";
import {
  isTerminalRunItemInput,
  isMtpDraftModel,
  backendVisibleGpus,
  GPU_MEMORY_ACCURACY_LEVELS,
  GPU_MEMORY_MEASUREMENT_SOURCES,
} from "../../../shared/types.js";
import { expandSweep } from "../../../shared/sweep.js";
import { getReleases, filterReleasesForWorker, assetMatchesWorker, buildInstallPayload } from "../github-releases.js";

const NUMERIC_SWEEP_FIELDS = [
  "n_prompt",
  "n_gen",
  "threads",
  "n_gpu_layers",
  "batch_size",
  "ubatch_size",
  "n_gpu_layers_draft",
  "n_cpu_moe",
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
  // Also optional (unlike the rest of NUMERIC_RESULT_FIELDS above): a worker
  // running a version that predates this column won't send it at all. A
  // whole item's real, already-computed results shouldn't be thrown away
  // just because one newer, defaultable field is missing -- see
  // shared/types.ts's IngestResultInput.n_gpu_layers_draft doc comment.
  if (
    row.n_gpu_layers_draft !== undefined &&
    (typeof row.n_gpu_layers_draft !== "number" || !Number.isFinite(row.n_gpu_layers_draft as number))
  ) {
    return "n_gpu_layers_draft must be a number";
  }
  // Same optional-field reasoning as n_gpu_layers_draft above.
  if (row.n_cpu_moe !== undefined && (typeof row.n_cpu_moe !== "number" || !Number.isFinite(row.n_cpu_moe as number))) {
    return "n_cpu_moe must be a number";
  }
  // Same optional-field reasoning as n_gpu_layers_draft/n_cpu_moe above --
  // plus nullable: a current worker sends a number on every row where an
  // actual-resident estimate was computable, null otherwise. See
  // shared/types.ts's ResultRow.gpu_layers_resident_est.
  if (
    row.gpu_layers_resident_est !== undefined &&
    row.gpu_layers_resident_est !== null &&
    (typeof row.gpu_layers_resident_est !== "number" || !Number.isFinite(row.gpu_layers_resident_est as number))
  ) {
    return "gpu_layers_resident_est must be a number or null";
  }
  // Same optional/nullable reasoning as gpu_layers_resident_est above -- the
  // MTP/draft companion's own estimate, see shared/types.ts's
  // ResultRow.gpu_layers_resident_est_draft.
  if (
    row.gpu_layers_resident_est_draft !== undefined &&
    row.gpu_layers_resident_est_draft !== null &&
    (typeof row.gpu_layers_resident_est_draft !== "number" ||
      !Number.isFinite(row.gpu_layers_resident_est_draft as number))
  ) {
    return "gpu_layers_resident_est_draft must be a number or null";
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

type ResolvedBuild =
  | { tag: string; deviceName?: string; alreadyInstalled: true }
  | { tag: string; deviceName?: string; alreadyInstalled: false; installPayload: InstallBuildJobPayload }
  | { error: string };

// Resolves which build a run should execute against, reading entirely from
// the worker's own last-reported state (its heartbeat cache, see
// repo.workerRepo) instead of a live HTTP round trip -- there is no outbound
// HTTP to workers anywhere in the server under the pull model
// (MULTIUSER_PLAN.md §1.12). Matches installed builds by tag/asset name,
// never by a stored "backend" field on InstalledBuild -- it doesn't exist,
// see assetMatchesWorker.
async function resolveBuildForRun(
  worker: Worker,
  backend: Backend,
  mainGpu: number | undefined
): Promise<ResolvedBuild> {
  const { platform, arch } = worker;
  if (!platform || !arch) {
    // Can't happen once the caller's offline check has passed --
    // recordHeartbeat always sets platform/arch/hardware together in the
    // same write. Defensive only.
    return { error: "worker has not reported its platform/hardware yet" };
  }

  // Tier-1 backend_device_name fallback (see shared/types.ts's
  // Run.backend_device_name) -- from the worker's own cached hardware, not a
  // live fetch.
  let deviceName: string | undefined;
  if (worker.hardware) {
    const visible = backendVisibleGpus(worker.hardware.gpu, backend);
    const picked = mainGpu != null ? visible[mainGpu] : visible[0];
    deviceName = picked?.model || picked?.vendor || undefined;
  }

  const alreadyInstalled = worker.installedBuilds.find((b) =>
    assetMatchesWorker(b.asset_name, platform, arch, backend)
  );
  if (alreadyInstalled) {
    return { tag: alreadyInstalled.tag, deviceName, alreadyInstalled: true };
  }

  // Nothing installed for this backend -- find the latest release with an
  // asset that actually matches this worker (mirrors WorkerCard.tsx's
  // client-side filter).
  let releases;
  try {
    releases = await getReleases();
  } catch (err) {
    return {
      error: `no llama.cpp build installed for backend "${backend}" on this worker, and GitHub releases are unreachable: ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }
  // driver's max CUDA version (HardwareInfo.nvidia_driver) orders variants
  // best-first, so assets[0] below is the newest toolkit this box can
  // actually load -- see shared/types.ts's sortAssetsForWorker.
  const matching = filterReleasesForWorker(
    releases,
    platform,
    arch,
    backend,
    worker.hardware?.nvidia_driver?.cuda_version ?? null
  ).filter((r) => r.assets.length > 0);
  if (matching.length === 0) {
    return { error: `no installable llama.cpp release found for backend "${backend}" on ${platform}/${arch}` };
  }
  const release = matching[0];
  const asset = release.assets[0];
  return {
    tag: release.tag,
    deviceName,
    alreadyInstalled: false,
    // buildInstallPayload carries the matching cudart redistributable for
    // CUDA builds -- identical to the Workers page's manual-install route.
    installPayload: buildInstallPayload(release, asset),
  };
}

// MULTIUSER_PLAN.md §1.10: the worker pushes its per-run log on job
// completion instead of the server proxying a live GET to the worker's own
// /logs -- no outbound HTTP to workers anywhere in the server, and the log
// survives the worker being wiped. Stored gzipped exactly as the worker sent
// it; served back with Content-Encoding: gzip so the browser decompresses it
// rather than spending server CPU on it.
const LOG_DIR = process.env.LOG_DIR ?? join(process.cwd(), "data", "run-logs");
const MAX_LOG_BYTES = 5 * 1024 * 1024;

function runLogPath(runId: string): string {
  // Run ids are server-generated UUIDs (see createRun's uuid()), never
  // user-controlled path segments -- no traversal risk from this value, but
  // the param still comes off the URL, so it's validated as a run id (an
  // existing `runs` row) before ever reaching this function regardless.
  return join(LOG_DIR, `run-${runId}.log.gz`);
}

export async function runsRoutes(app: FastifyInstance): Promise<void> {
  // Raw gzip bytes, not JSON -- Fastify has no built-in parser for this
  // content type. Scoped to this plugin's own routes only (Fastify's
  // content-type parsers are subject to the same encapsulation as routes).
  app.addContentTypeParser("application/octet-stream", { parseAs: "buffer" }, (_req, body, done) => {
    done(null, body);
  });

  app.get("/api/runs", async (request) => {
    // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3): scoped to the caller's own
    // runs once authenticated; every run in single-tenant mode (AUTH_ENABLED
    // off), unchanged from before this scoping existed.
    const authed = resolveAuthUser(request);
    return { runs: repo.listRuns(authed?.user.id) };
  });

  app.get<{ Params: { id: string } }>(
    "/api/runs/:id",
    // Polled every ~1-2s by every open Runs/RunDetail tab while any run is
    // active -- logLevel: "silent" suppresses Fastify's default per-request
    // log pair for just this route; index.ts's polling-summary hook counts
    // these instead and reports volume once a minute.
    { logLevel: "silent" },
    async (request, reply) => {
      const authed = resolveAuthUser(request);
      const data = repo.getRunWithResults(authed?.user.id, request.params.id);
      if (!data) return reply.code(404).send({ error: "run not found" });
      // No live worker probe anymore -- liveness/staleness are handled
      // server-side now: a dead worker's claimed job is caught by the lease
      // reaper (index.ts's reapExpiredLeases), not by this route reaching
      // out on every poll. `paused` reflects the same per-worker flag the
      // heartbeat handler delivers control from (MULTIUSER_PLAN.md §1.6/§1.7/§1.14).
      const paused =
        data.run.status === "running" && data.run.worker_id
          ? repo.workerRepo.getPauseRequested(data.run.worker_id)
          : undefined;
      return { ...data, paused };
    }
  );

  // Reads the log the worker pushed on job completion (POST below) -- no
  // outbound call to any worker. 404 both when the run doesn't exist and
  // when it exists but never got a log pushed (a run that failed before
  // producing one, or predates this feature).
  app.get<{ Params: { id: string } }>("/api/runs/:id/log", async (request, reply) => {
    const authed = resolveAuthUser(request);
    const run = repo.getRun(authed?.user.id, request.params.id);
    if (!run) return reply.code(404).send({ error: "run not found" });
    const path = runLogPath(run.id);
    if (!existsSync(path)) return reply.code(404).send({ error: "no log file for this run" });
    const gzipped = readFileSync(path);
    reply.header("content-type", "text/plain; charset=utf-8");
    reply.header("content-encoding", "gzip");
    reply.header("content-disposition", `attachment; filename="run-${run.id}.log"`);
    return reply.send(gzipped);
  });

  // Worker -> server push of a completed run's log file (MULTIUSER_PLAN.md
  // §1.10). Dual-mode worker auth (Stage 3 session first, Stage 1 shared
  // secret as fallback -- same posture as worker-auth.ts's authenticateWorker,
  // MULTIUSER_PLAN.md §3.4/§4.3) duplicated inline rather than calling that
  // function directly: it expects to read machine_id out of a JSON req.body,
  // but this route's body is raw gzip bytes, so the shared-secret fallback
  // here resolves the machine from the X-Machine-Id header instead.
  app.post<{ Params: { id: string } }>(
    "/api/runs/:id/log",
    { bodyLimit: MAX_LOG_BYTES },
    async (request, reply) => {
      const auth = request.headers.authorization;
      const token = auth?.startsWith("Bearer ") ? auth.slice("Bearer ".length) : undefined;
      if (!token) throw new UnauthorizedError("missing worker token");

      let worker: Worker | undefined;
      const session = repo.sessionRepo.getByTokenHash(hashToken(token));
      if (session && session.expiresAt >= Date.now() && session.isWorker && session.workerId) {
        worker = repo.workerRepo.getWorker(session.workerId);
      }
      if (!worker) {
        const expected = process.env.WORKER_SHARED_TOKEN;
        if (!expected || !safeEqual(token, expected)) throw new UnauthorizedError("invalid worker token");
        const machineId = request.headers["x-machine-id"];
        if (typeof machineId !== "string" || !machineId) {
          throw new BadRequestError("X-Machine-Id header is required");
        }
        worker = repo.workerRepo.getByMachineId(machineId);
        if (!worker) throw new UnauthorizedError("unknown machine");
      }

      const run = repo.getRun(undefined, request.params.id);
      if (!run) throw new NotFoundError("run not found");
      if (run.worker_id !== worker.id) {
        throw new ForbiddenError("this machine did not execute this run");
      }

      const body = request.body;
      if (!Buffer.isBuffer(body) || body.length === 0) {
        throw new BadRequestError("request body must be non-empty gzip bytes");
      }
      mkdirSync(LOG_DIR, { recursive: true });
      writeFileSync(runLogPath(run.id), body);
      return reply.code(200).send({ ok: true });
    }
  );

  app.post<{ Body: TriggerPayload }>(
    "/api/runs/trigger",
    // §2.6: 30/hour, keyed on user (falls back to IP when AUTH_ENABLED is
    // off / caller has no session -- there's no user to attribute it to yet).
    { config: { rateLimit: { max: 30, timeWindow: "1 hour", keyGenerator: userOrIpKeyGenerator } } },
    async (request, reply) => {
      // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3): the run's owner. undefined
      // in single-tenant mode (AUTH_ENABLED off), matching every other
      // scoped call in this route.
      const authed = resolveAuthUser(request);
      const userId = authed?.user.id;
      const body = request.body;
      if (!body || !body.model_id || !body.sweep) {
        return reply.code(400).send({ error: "model_id and sweep are required" });
      }
      const sweepError = validateSweep(body.sweep);
      if (sweepError) {
        return reply.code(400).send({ error: sweepError });
      }
      // expandSweep silently drops any flash_attn:"off" combo paired with a
      // quantized cache type (llama.cpp refuses those outright -- see its
      // own comment) -- a sweep built entirely out of such combos would
      // otherwise create a Run with zero run_items and nothing to ever mark
      // it finished, rather than surfacing why upfront.
      if (expandSweep(body.sweep).length === 0) {
        return reply.code(400).send({
          error:
            "sweep has no valid combinations -- flash_attn:\"off\" can't be paired with a quantized cache_type_k/cache_type_v (llama.cpp requires flash attention for a quantized KV cache); add flash_attn:\"on\" or use cache_type_k/v:\"f32\"/\"f16\"/\"bf16\"",
        });
      }
      if (body.main_gpu !== undefined && (!Number.isInteger(body.main_gpu) || body.main_gpu < 0)) {
        return reply.code(400).send({ error: "main_gpu must be a non-negative integer" });
      }
      if (body.main_gpu_backend !== undefined && typeof body.main_gpu_backend !== "string") {
        return reply.code(400).send({ error: "main_gpu_backend must be a string" });
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

      if (!body.worker_id) {
        return reply.code(400).send({ error: "worker_id is required" });
      }

      // --n-cpu-moe only has anything to offload on a Mixture-of-Experts
      // model (see shared/types.ts's ModelMetadata.expert_count) -- checked
      // here for the same reason as the MTP guard above: NewRun.tsx's
      // slider is disabled for a non-MoE model, but that's a UX nicety, not
      // a guarantee, so a direct API call needs the same rejection
      // server-side rather than every item in the run silently no-opping
      // the flag against a model with no experts to keep on CPU.
      if (body.sweep.n_cpu_moe.some((n) => n > 0)) {
        const modelIsMoe = typeof model.metadata.expert_count === "number" && model.metadata.expert_count > 0;
        if (!modelIsMoe) {
          return reply.code(400).send({
            error: 'sweep includes n_cpu_moe > 0 but this model isn\'t detected as a Mixture-of-Experts model',
          });
        }
      }

      const worker = repo.workerRepo.getWorker(body.worker_id);
      if (!worker) {
        request.log.warn({ worker_id: body.worker_id }, "run trigger rejected: unknown machine");
        throw new BadRequestError("unknown machine");
      }
      assertOwnsWorker(userId, worker.id);
      if (worker.status === "offline") {
        request.log.warn({ worker_id: worker.id }, "run trigger rejected: machine offline");
        throw new ConflictError("that machine is offline -- start the worker and try again");
      }

      const targetBackend: Backend | null = body.main_gpu_backend ?? worker.backend;
      if (!targetBackend) {
        throw new BadRequestError("worker has not reported a backend yet");
      }

      // Resolved ONCE, from the worker's own cached state -- no live HTTP
      // round trip (MULTIUSER_PLAN.md §1.12).
      const resolved = await resolveBuildForRun(worker, targetBackend, body.main_gpu);
      if ("error" in resolved) {
        request.log.warn(
          { worker: worker.id, model_id: body.model_id, error: resolved.error },
          "run trigger rejected: could not resolve a llama.cpp build"
        );
        throw new BadRequestError(resolved.error);
      }

      // EXPLICIT field list -- never spread the request body into an insert.
      const run: Run = {
        id: uuid(),
        worker_id: worker.id,
        worker_name: worker.displayName, // point-in-time snapshot; the export reads this directly
        llama_cpp_build: resolved.tag,
        llama_cpp_backend: targetBackend,
        backend_device_name: resolved.deviceName,
        model_id: body.model_id,
        config: {
          model_id: body.model_id,
          mtp_model_id: mtpModel?.id,
          main_gpu: body.main_gpu,
          main_gpu_backend: body.main_gpu_backend,
          sweep: body.sweep,
        } as RunConfig,
        // Flipped to 'running' by queueRepo.claimNextJob once the worker
        // actually claims the benchmark job (MULTIUSER_PLAN.md §1.13) --
        // never here, regardless of whether the worker is currently idle or
        // busy with something else. The worker_jobs queue is what serializes
        // execution now; there's no more "queue in the DB if busy" branch.
        status: "scheduled",
        started_at: Date.now(),
      };
      repo.createRun(userId, run);
      const items = expandSweep(body.sweep);
      repo.createRunItems(userId, run.id, items);
      request.log.info(
        { run_id: run.id, worker: worker.id, model_id: body.model_id, items: items.length },
        "run created"
      );

      if (!resolved.alreadyInstalled) {
        repo.queueRepo.enqueueJob(worker.id, {
          type: "install_build",
          payload: resolved.installPayload,
          runId: run.id,
        });
      }
      const benchmarkPayload: BenchmarkJob = {
        run_id: run.id,
        model,
        mtp_model: mtpModel,
        sweep: body.sweep,
        main_gpu: body.main_gpu,
        llama_cpp_build: resolved.tag,
        llama_cpp_backend: targetBackend,
      };
      repo.queueRepo.enqueueJob(worker.id, { type: "benchmark", payload: benchmarkPayload, runId: run.id });
      queueEvents.emit(worker.id);

      return reply.code(201).send({ run });
    }
  );

  // Replaces the old whole-run /ingest: the worker now reports one sweep
  // combination at a time (see shared/sweep.ts's expandSweep), so a crash on
  // one item doesn't lose the rest. Two tiers share this route -- a
  // best-effort progress tick (loading/processing/generating/benchmarking)
  // and a terminal outcome (done/failed/failed_oom) -- discriminated by
  // `status`, matching shared/types.ts's RunItemTickInput/RunItemTerminalInput.
  //
  // Multi-user Stage 4 fix (MULTIUSER_PLAN.md §4.3): this route had NO worker
  // authentication at all before this -- not /api/worker/*-prefixed, not in
  // PUBLIC_PATHS, not in WORKER_AUTHENTICATED_ROUTES, so any caller (or, once
  // AUTH_ENABLED ships, any logged-in user) could report fabricated results
  // against any run. Dual-mode like the /log route above: a Stage 3 session
  // resolves a specific worker.id directly; the Stage 1 shared-secret
  // fallback trusts the RUN's own worker_id instead of requiring a
  // machine_id in this route's tick/terminal payload (RunItemUpdateInput has
  // no such field, and never did) -- the same "the shared secret means SOME
  // worker" trust model Stage 1 already had everywhere else, just now
  // actually checked here too.
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

      const auth = request.headers.authorization;
      const token = auth?.startsWith("Bearer ") ? auth.slice("Bearer ".length) : undefined;
      if (!token) throw new UnauthorizedError("missing worker token");

      const run = repo.getRun(undefined, id);
      if (!run) return reply.code(404).send({ error: "run not found" });

      let authorizedAsWorkerId: string | null | undefined;
      const session = repo.sessionRepo.getByTokenHash(hashToken(token));
      if (session && session.expiresAt >= Date.now() && session.isWorker && session.workerId) {
        authorizedAsWorkerId = session.workerId;
      } else {
        const expected = process.env.WORKER_SHARED_TOKEN;
        if (!expected || !safeEqual(token, expected)) throw new UnauthorizedError("invalid worker token");
        authorizedAsWorkerId = run.worker_id;
      }
      if (!run.worker_id || authorizedAsWorkerId !== run.worker_id) {
        throw new ForbiddenError("this machine did not execute this run");
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
        const updatedRun = repo.recordRunItemTerminal(id, idx, body);
        if (!updatedRun) return reply.code(404).send({ error: "run not found" });
        // No dispatchScheduledRun anymore -- there's no "next queued run" to
        // hand off explicitly. The worker's own pull loop just goes back to
        // long-polling for its next job once this one's execution finishes
        // (MULTIUSER_PLAN.md §1.9); the queue itself is what serializes it.
        return reply.code(200).send({ ok: true, run_status: updatedRun.status });
      }

      repo.updateRunItemTick(id, idx, body);
      return reply.code(200).send({ ok: true });
    }
  );

  // Target the RUN, not the machine (a change from the old worker-scoped
  // /api/workers/:name/stop|pause|resume) -- the right shape once one
  // account owns several machines (MULTIUSER_PLAN.md §1.14).
  app.post<{ Params: { id: string } }>("/api/runs/:id/stop", async (request, reply) => {
    const authed = resolveAuthUser(request);
    const run = repo.getRun(authed?.user.id, request.params.id);
    if (!run) throw new NotFoundError("run not found");

    const jobs = repo.queueRepo.getNonTerminalJobsForRun(run.id);
    repo.queueRepo.cancelPendingJobsForRun(run.id); // never-claimed jobs -- nothing to signal, cancel outright

    const worker = run.worker_id ? repo.workerRepo.getWorker(run.worker_id) : undefined;
    const anyClaimed = jobs.some((j) => j.status === "claimed");

    if (!anyClaimed || !worker || worker.status === "offline") {
      // Nothing is actually executing (or the machine is unreachable) --
      // reconcile immediately instead of waiting on a lease/heartbeat that
      // may never arrive.
      const reconciled = repo.reconcileStaleRun(authed?.user.id, run.id, "stopped by user");
      return { run: reconciled ?? run };
    }

    // Delivered on the worker's next heartbeat (≤10s) -- the UI shows
    // "stopping…" until the worker's own terminal item reports land.
    repo.queueRepo.requestCancelForRun(run.id);
    queueEvents.emit(worker.id);
    return { run };
  });

  app.post<{ Params: { id: string } }>("/api/runs/:id/pause", async (request) => {
    const authed = resolveAuthUser(request);
    const run = repo.getRun(authed?.user.id, request.params.id);
    if (!run) throw new NotFoundError("run not found");
    if (!run.worker_id) throw new ConflictError("run has no assigned machine");
    repo.workerRepo.setPauseRequested(run.worker_id, true);
    return { ok: true };
  });

  app.post<{ Params: { id: string } }>("/api/runs/:id/resume", async (request) => {
    const authed = resolveAuthUser(request);
    const run = repo.getRun(authed?.user.id, request.params.id);
    if (!run) throw new NotFoundError("run not found");
    if (!run.worker_id) throw new ConflictError("run has no assigned machine");
    repo.workerRepo.setPauseRequested(run.worker_id, false);
    return { ok: true };
  });
}
