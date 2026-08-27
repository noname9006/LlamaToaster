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
  RunProbeJobPayload,
  MeasureQualityJobPayload,
} from "../../../shared/types.js";
import { MIN_PROBE_CTX, MAX_PROBE_CTX, DATASET_HASH_RE } from "./measurements.js";
import { recheckComparisonMember } from "./comparisons.js";
import {
  isTerminalRunItemInput,
  isMtpDraftModel,
  backendVisibleGpus,
  GPU_MEMORY_ACCURACY_LEVELS,
  GPU_MEMORY_MEASUREMENT_SOURCES,
  RUN_KINDS,
} from "../../../shared/types.js";
import type { RunKind } from "../../../shared/types.js";
import { expandSweep, validateDepthRule, validateConcurrencyRule } from "../../../shared/sweep.js";
import {
  MAX_SWEEP_ITEMS,
  WARN_SWEEP_ITEMS,
  MAX_AXIS_VALUES,
  MIN_REPEATS,
  MAX_REPEATS,
  MAX_CHAIN_DEPTH,
  MAX_ACTIVE_ROOTS_PER_USER,
  CHAIN_WALL_CLOCK_MS,
} from "../../../shared/types.js";
import { CACHE_TYPE_VALUES, isKnownCacheType } from "../../../shared/engineSpec.js";
import type { GoalsConfig } from "../../../shared/goals.js";
import { normalizeGoals } from "../../../shared/goals.js";
import { getReleases, filterReleasesForWorker, assetMatchesWorker, buildInstallPayload } from "../github-releases.js";
import {
  checkComparisonFairness,
  gridSignature,
  MAX_COMPARISON_MEMBERS,
  type ComparisonFairnessFacts,
} from "../../../shared/comparison.js";

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
  // §0.5 -- n_depth is an optional axis (legacy payloads omit it entirely).
  const depth = s.n_depth;
  if (depth !== undefined) {
    if (!Array.isArray(depth) || !depth.every((x) => typeof x === "number" && Number.isFinite(x) && x >= 0)) {
      return "sweep.n_depth must be an array of non-negative numbers";
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
  for (const [field, values] of [
    ["cache_type_k", s.cache_type_k],
    ["cache_type_v", s.cache_type_v],
  ] as const) {
    if (Array.isArray(values)) {
      for (const v of values) {
        if (!isKnownCacheType(v)) {
          return `sweep.${field} contains "${String(v)}" -- allowed: ${CACHE_TYPE_VALUES.join(", ")}`;
        }
      }
    }
  }
  for (const field of NUMERIC_SWEEP_FIELDS) {
    const v = s[field] as number[];
    if (Array.isArray(v) && v.length > MAX_AXIS_VALUES) {
      return `sweep.${field} has more than ${MAX_AXIS_VALUES} values`;
    }
  }
  if (typeof s.repeats !== "number" || !Number.isInteger(s.repeats) || s.repeats < MIN_REPEATS || s.repeats > MAX_REPEATS) {
    return `sweep.repeats must be an integer between ${MIN_REPEATS} and ${MAX_REPEATS}`;
  }
  return null;
}

const TICK_STATUSES = new Set(["loading", "processing", "generating", "benchmarking"]);
const TERMINAL_STATUSES = new Set(["done", "failed", "failed_oom", "cancelled", "skipped"]);
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
  // Whole-adapter/per-process VRAM usage + whole-system RAM usage avg/peak
  // -- all optional (an older worker that predates these fields won't send
  // them) and nullable (a stream the backend couldn't measure), same
  // version-skew posture as n_gpu_layers_draft/n_cpu_moe above.
  for (const field of [
    "gpu_memory_used_avg_mib",
    "gpu_memory_used_peak_mib",
    "gpu_memory_process_avg_mib",
    "gpu_memory_process_peak_mib",
    "ram_total_used_avg_mib",
    "ram_total_used_peak_mib",
  ] as const) {
    if (
      row[field] !== undefined &&
      row[field] !== null &&
      (typeof row[field] !== "number" || !Number.isFinite(row[field] as number))
    ) {
      return `${field} must be a number, null, or absent`;
    }
  }
  for (const field of [
    "gpu_memory_used_avg_accuracy",
    "gpu_memory_used_peak_accuracy",
    "gpu_memory_process_avg_accuracy",
    "gpu_memory_process_peak_accuracy",
  ] as const) {
    if (
      row[field] !== undefined &&
      !GPU_MEMORY_ACCURACY_LEVELS.includes(row[field] as (typeof GPU_MEMORY_ACCURACY_LEVELS)[number])
    ) {
      return `${field} must be one of ${GPU_MEMORY_ACCURACY_LEVELS.join("/")} or absent`;
    }
  }
  for (const field of [
    "gpu_memory_used_avg_source",
    "gpu_memory_used_peak_source",
    "gpu_memory_process_avg_source",
    "gpu_memory_process_peak_source",
  ] as const) {
    if (
      row[field] !== undefined &&
      row[field] !== null &&
      !GPU_MEMORY_MEASUREMENT_SOURCES.includes(row[field] as (typeof GPU_MEMORY_MEASUREMENT_SOURCES)[number])
    ) {
      return `${field} must be one of ${GPU_MEMORY_MEASUREMENT_SOURCES.join("/")}, null, or absent`;
    }
  }
  // --- BENCHMARKING_PLAN_V8.md §0 additions (all optional -- version-skew
  // tolerant, an older worker simply doesn't send them) ---
  if (
    row.n_depth !== undefined &&
    (typeof row.n_depth !== "number" || !Number.isInteger(row.n_depth) || (row.n_depth as number) < 0)
  ) {
    return "n_depth must be a non-negative integer";
  }
  if (row.method_version !== undefined && (typeof row.method_version !== "number" || !Number.isInteger(row.method_version))) {
    return "method_version must be an integer";
  }
  if (
    row.prompt_offset !== undefined &&
    row.prompt_offset !== null &&
    (typeof row.prompt_offset !== "number" || !Number.isInteger(row.prompt_offset))
  ) {
    return "prompt_offset must be an integer or null";
  }
  for (const field of ["spec_type", "cpu_isa"] as const) {
    if (row[field] !== undefined && row[field] !== null && typeof row[field] !== "string") {
      return `${field} must be a string or null`;
    }
  }
  for (const field of ["spec_n_max", "spec_n_min", "concurrency", "gpu_temp_c_max", "gpu_clock_mhz_min"] as const) {
    if (row[field] !== undefined && row[field] !== null && (typeof row[field] !== "number" || !Number.isFinite(row[field] as number))) {
      return `${field} must be a number or null`;
    }
  }
  if (row.wall_clock_fallback !== undefined && typeof row.wall_clock_fallback !== "boolean") {
    return "wall_clock_fallback must be a boolean";
  }
  if (row.gpu_clock_samples !== undefined && row.gpu_clock_samples !== null) {
    if (
      !Array.isArray(row.gpu_clock_samples) ||
      !row.gpu_clock_samples.every((x: unknown) => typeof x === "number" && Number.isFinite(x))
    ) {
      return "gpu_clock_samples must be an array of numbers or null";
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
      if (!body || !body.model_id || !body.worker_id) {
        return reply.code(400).send({ error: "model_id and worker_id are required" });
      }
      if (!body.model_id || !body.sweep) {
        return reply.code(400).send({ error: "model_id and sweep are required" });
      }

      // §0.5 -- run kind. Absent/undefined = NULL standalone, byte-identical
      // to every legacy payload.
      const kind = body.kind ?? null;
      if (body.kind !== undefined && !RUN_KINDS.includes(body.kind)) {
        return reply.code(400).send({ error: `kind must be one of ${RUN_KINDS.join("/")}` });
      }
      if (body.kind === "probe" && !body.probe) {
        return reply.code(400).send({ error: "probe runs require a probe block" });
      }
      if (body.probe && body.kind !== "probe") {
        return reply.code(400).send({ error: "the probe block requires kind \"probe\"" });
      }
      if (body.kind === "quality" && !body.quality) {
        return reply.code(400).send({ error: "quality runs require a quality block" });
      }
      if (body.quality && body.kind !== "quality") {
        return reply.code(400).send({ error: "the quality block requires kind \"quality\"" });
      }
      if (body.curve_point && body.kind !== "runtime") {
        return reply.code(400).send({ error: "the curve_point block requires kind \"runtime\"" });
      }
      if (body.knee && body.kind !== "runtime") {
        return reply.code(400).send({ error: "the knee block requires kind \"runtime\"" });
      }
      if (body.knee && body.curve_point) {
        return reply.code(400).send({
          error: "a runtime run measures either a curve point or a concurrency ladder, not both",
        });
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
      const expanded = expandSweep(body.sweep);
      if (expanded.length === 0) {
        return reply.code(400).send({
          error:
            "sweep has no valid combinations -- flash_attn:\"off\" can't be paired with a quantized cache_type_k/cache_type_v (llama.cpp requires flash attention for a quantized KV cache); add flash_attn:\"on\" or use cache_type_k/v:\"f32\"/\"f16\"/\"bf16\"",
        });
      }
      // §0.2 depth rule -- rejection copy names the fix.
      const depthError = validateDepthRule(expanded);
      if (depthError) return reply.code(400).send({ error: depthError });
      const concurrencyError = validateConcurrencyRule(expanded);
      if (concurrencyError) return reply.code(400).send({ error: concurrencyError });

      // §0.5 budgets -- hard 400 past MAX_SWEEP_ITEMS, 201-with-warning past
      // WARN_SWEEP_ITEMS.
      let budgetWarning: string | undefined;
      if (expanded.length > MAX_SWEEP_ITEMS) {
        return reply.code(400).send({
          error: `sweep expands to ${expanded.length} items -- the hard cap is ${MAX_SWEEP_ITEMS}. Reduce axis values or split into multiple runs.`,
        });
      }
      if (expanded.length > WARN_SWEEP_ITEMS) {
        budgetWarning = `large sweep: ${expanded.length} items exceeds the ${WARN_SWEEP_ITEMS}-item warning threshold and may take a long time`;
      }

      // M2 -- goals are intent stored as configuration; normalized once here
      // so the echo on the created root is exactly what scoring later reads.
      const goals: GoalsConfig | undefined =
        body.goals !== undefined ? normalizeGoals(body.goals) : undefined;

      // N6 -- steady-state discard, recorded in configuration; only legal
      // when post-discard n still satisfies the n >= 3 gate.
      const discardFirst = body.discard_first_repeats ?? 0;
      if (!Number.isInteger(discardFirst) || discardFirst < 0 || discardFirst > 25) {
        return reply.code(400).send({ error: "discard_first_repeats must be an integer between 0 and 25" });
      }
      if (discardFirst > 0 && body.sweep.repeats < discardFirst + 3) {
        return reply.code(400).send({
          error: `discard_first_repeats=${discardFirst} requires repeats >= ${discardFirst + 3} so post-discard n still satisfies the n >= 3 stability floor`,
        });
      }

      if (body.main_gpu !== undefined && (!Number.isInteger(body.main_gpu) || body.main_gpu < 0)) {
        return reply.code(400).send({ error: "main_gpu must be a non-negative integer" });
      }
      if (body.main_gpu_backend !== undefined && typeof body.main_gpu_backend !== "string") {
        return reply.code(400).send({ error: "main_gpu_backend must be a string" });
      }
      if (
        body.comparison_id !== undefined &&
        (typeof body.comparison_id !== "string" || body.comparison_id.length === 0 || body.comparison_id.length > 128)
      ) {
        return reply.code(400).send({ error: "comparison_id must be a short non-empty string" });
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

      // N3 fairness preconditions -- BLOCKING, at trigger and re-checked per
      // member (server/src/routes/comparisons.ts). Comparison members ride
      // ordinary sweeps with a shared comparison_id, and the interesting
      // variable is the model file, so same worker / build / backend / GPU /
      // flags / repeats are all held fixed.
      let comparisonReference: Run | undefined;
      if (body.comparison_id) {
        const existingMembers = repo.listComparisonMembers(body.comparison_id, userId);
        if (existingMembers.length >= MAX_COMPARISON_MEMBERS) {
          return reply.code(400).send({
            error: `a comparison groups at most ${MAX_COMPARISON_MEMBERS} models`,
          });
        }
        comparisonReference = existingMembers[0];
        if (comparisonReference && comparisonReference.model_id === body.model_id) {
          return reply.code(400).send({
            error: "that model is already a member of this comparison -- the variable being compared is the model file",
          });
        }
      }

      if (!body.worker_id) {
        return reply.code(400).send({ error: "worker_id is required" });
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

      // N3 -- comparison members must be "already registered on the target
      // worker (file present in its local cache, not merely bookmarked)":
      // the interesting variable is the model file, so every member needs to
      // actually be loadable right away, not queued behind a download of
      // unknown length on the most expensive object in the product. models.id
      // IS the file's sha256 (comparisons.ts's own convention), so this is a
      // direct membership check against the worker's own reported files.
      // Skipped -- not enforced closed -- for a worker that has never
      // reported ANY file hash at all (an older, pre-hashing build): the
      // version-skew posture everywhere else in this file is to degrade
      // gracefully for an old worker rather than block it outright.
      if (body.comparison_id) {
        const workerHashes = (worker.modelFiles ?? []).map((f) => f.sha256).filter((h): h is string => h != null);
        const workerCanVerify = workerHashes.length > 0;
        if (workerCanVerify && !workerHashes.includes(model.id)) {
          return reply.code(400).send({
            error: `comparison members must already be present in the target worker's local cache -- ${
              model.filename ?? model.id
            } was not found there. Download it to that machine first, then start the comparison.`,
          });
        }
      }

      // §0.7/N2/N4/N1 version-skew gates -- dispatch refuses to enqueue a job
      // type an older worker would choke on mid-fleet.
      const capabilities = new Set(worker.capabilities ?? []);
      if (body.kind === "probe" && !capabilities.has("probe-v1")) {
        throw new ConflictError(
          "that machine runs an older worker build — update it to probe on this model."
        );
      }
      if (body.kind === "quality" && !capabilities.has("quality-v1")) {
        throw new ConflictError(
          "that machine runs an older worker build — update it to measure quality on this model."
        );
      }
      if (body.curve_point && !capabilities.has("curve-v1")) {
        throw new ConflictError(
          "that machine runs an older worker build — update it to measure context curves on this model."
        );
      }
      if (body.knee && !capabilities.has("curve-v1")) {
        throw new ConflictError(
          "that machine runs an older worker build — update it to measure a concurrency knee on this model."
        );
      }

      // N2 payload bounds mirror the sweep-axis table -- out-of-bounds yields
      // 400, never a silent clamp.
      if (body.probe) {
        const p = body.probe;
        if (!Number.isFinite(p.candidate_ctx) || p.candidate_ctx < 256 || p.candidate_ctx > 4_194_304) {
          return reply.code(400).send({ error: "probe.candidate_ctx must be between 256 and 4194304" });
        }
        if (!Number.isInteger(p.placement.ngl) || p.placement.ngl < 0 || p.placement.ngl > 1024) {
          return reply.code(400).send({ error: "probe.placement.ngl must be an integer in [0, 1024]" });
        }
        if (
          p.placement.n_cpu_moe !== undefined &&
          (!Number.isInteger(p.placement.n_cpu_moe) || p.placement.n_cpu_moe < 0 || p.placement.n_cpu_moe > 1024)
        ) {
          return reply.code(400).send({ error: "probe.placement.n_cpu_moe must be an integer in [0, 1024]" });
        }
        const slots = p.placement.slots ?? 1;
        if (!Number.isInteger(slots) || slots < 1 || slots > 64) {
          return reply.code(400).send({ error: "probe.placement.slots must be an integer in [1, 64]" });
        }
        for (const t of p.kv_pair) {
          if (!isKnownCacheType(t)) {
            return reply.code(400).send({ error: `probe.kv_pair contains "${String(t)}" -- allowed: ${CACHE_TYPE_VALUES.join(", ")}` });
          }
        }
      }
      // N4 -- ctx_tokens bounds mirror the probe-context table (same MIN/
      // MAX_PROBE_CTX the ingestion route validates against); dataset_hash
      // must already be pinned since a perplexity number without its corpus
      // hash is meaningless.
      if (body.quality) {
        const q = body.quality;
        if (!Number.isFinite(q.ctx_tokens) || q.ctx_tokens < MIN_PROBE_CTX || q.ctx_tokens > MAX_PROBE_CTX) {
          return reply.code(400).send({ error: `quality.ctx_tokens must be between ${MIN_PROBE_CTX} and ${MAX_PROBE_CTX}` });
        }
        for (const t of q.kv_pair) {
          if (!isKnownCacheType(t)) {
            return reply.code(400).send({ error: `quality.kv_pair contains "${String(t)}" -- allowed: ${CACHE_TYPE_VALUES.join(", ")}` });
          }
        }
        if (typeof q.dataset_hash !== "string" || !DATASET_HASH_RE.test(q.dataset_hash)) {
          return reply.code(400).send({ error: "quality.dataset_hash must match sha256:<64 lowercase hex>" });
        }
      }
      if (body.knee) {
        const k = body.knee;
        if (!Number.isInteger(k.n_prompt) || k.n_prompt < 1 || k.n_prompt > 4_194_304) {
          return reply.code(400).send({ error: "knee.n_prompt must be an integer in [1, 4194304]" });
        }
        if (!Number.isInteger(k.n_gen) || k.n_gen < 1 || k.n_gen > 1_048_576) {
          return reply.code(400).send({ error: "knee.n_gen must be an integer in [1, 1048576]" });
        }
        if (k.slots !== undefined) {
          if (
            !Array.isArray(k.slots) ||
            k.slots.length === 0 ||
            k.slots.length > MAX_AXIS_VALUES ||
            !k.slots.every((n) => Number.isInteger(n) && n >= 1 && n <= 64)
          ) {
            return reply.code(400).send({ error: "knee.slots must be 1..64 integers, at most " + MAX_AXIS_VALUES });
          }
        }
        if (
          k.repeats !== undefined &&
          (!Number.isInteger(k.repeats) || k.repeats < MIN_REPEATS || k.repeats > MAX_REPEATS)
        ) {
          return reply.code(400).send({
            error: `knee.repeats must be an integer between ${MIN_REPEATS} and ${MAX_REPEATS}`,
          });
        }
      }
      let curveContexts: number[] = [];
      if (body.curve_point) {
        const c = body.curve_point;
        curveContexts = Array.isArray(c.effective_ctx) ? c.effective_ctx : [c.effective_ctx];
        if (curveContexts.length === 0) {
          return reply.code(400).send({ error: "curve_point.effective_ctx must list at least one context" });
        }
        if (curveContexts.length > MAX_AXIS_VALUES) {
          return reply.code(400).send({
            error: `curve_point.effective_ctx lists ${curveContexts.length} contexts -- at most ${MAX_AXIS_VALUES} per run`,
          });
        }
        const trainedCtx = typeof model.metadata.trained_ctx === "number" ? model.metadata.trained_ctx : null;
        for (const ctx of curveContexts) {
          if (!Number.isFinite(ctx) || ctx < 256 || ctx > 4_194_304) {
            return reply.code(400).send({ error: "curve_point.effective_ctx must be between 256 and 4194304" });
          }
          if (trainedCtx != null && ctx > trainedCtx) {
            return reply.code(400).send({
              error: `curve_point.effective_ctx exceeds the model's trained context (${trainedCtx}) -- nothing beyond trained ctx can appear as measured`,
            });
          }
        }
      }

      // §0.5 duplicate-trigger guard -- same (user, model, worker) triple with
      // the same non-NULL kind already running/scheduled. The 409 names the
      // active root and kind.
      const blocking = repo.findBlockingRun(userId, body.model_id, worker.id, kind);
      if (blocking) {
        throw new ConflictError(
          `you already have a ${blocking.kind ?? "standalone"} run for this model on that machine (${blocking.id}). Stop it first or wait for it to finish.`
        );
      }

      // §0.5 chain quotas -- ≤ 3 active roots per user (probes exempt; a
      // comparison group counts once), depth ≤ 3 via the parent link.
      if (body.parent_run_id !== undefined && typeof body.parent_run_id !== "string") {
        return reply.code(400).send({ error: "parent_run_id must be a string" });
      }
      let rootRunId: string | null = null;
      let chainDepth = 0;
      if (body.parent_run_id) {
        const parent = repo.getRun(undefined, body.parent_run_id);
        if (!parent) return reply.code(400).send({ error: "unknown parent_run_id" });
        const parentOwnerId = repo.getRunOwnerId(body.parent_run_id);
        if (parentOwnerId != null && userId != null && parentOwnerId !== userId) {
          return reply.code(403).send({ error: "parent run belongs to another user" });
        }
        rootRunId = parent.root_run_id ?? parent.id;
        chainDepth = repo.getChainDepth(rootRunId) + 1;
        if (chainDepth > MAX_CHAIN_DEPTH) {
          return reply.code(400).send({ error: `chain depth ${chainDepth} exceeds the maximum of ${MAX_CHAIN_DEPTH}` });
        }
      }
      // §0.5 -- "≤ 3 active roots per user"; probes are exempt (minutes-long
      // by construction) and a comparison group counts once, not once per
      // member. On a single-tenant instance every run carries user_id NULL,
      // which is one bucket -- the same accounting, one implicit user.
      if (kind !== "probe") {
        const activeRoots = repo.countActiveRoots(userId);
        const groupHasActiveMember = body.comparison_id
          ? await repo.hasActiveComparisonMember(body.comparison_id)
          : false;
        if (!groupHasActiveMember && activeRoots >= MAX_ACTIVE_ROOTS_PER_USER) {
          throw new ConflictError(
            `you already have ${activeRoots} active runs (limit ${MAX_ACTIVE_ROOTS_PER_USER} per account). Stop one first or wait for it to finish.`
          );
        }
      }
      repo.cancelExpiredRoots();

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

      if (comparisonReference) {
        const candidateFacts: ComparisonFairnessFacts = {
          worker_id: worker.id,
          llama_cpp_build: resolved.tag,
          llama_cpp_backend: targetBackend,
          backend_device_name: resolved.deviceName ?? null,
          repeats: body.sweep.repeats,
          method_version: null,
          grid_signature: gridSignature(body.sweep as unknown as Record<string, unknown>),
        };
        const violations = checkComparisonFairness(
          {
            worker_id: comparisonReference.worker_id ?? null,
            llama_cpp_build: comparisonReference.llama_cpp_build ?? null,
            llama_cpp_backend: comparisonReference.llama_cpp_backend ?? null,
            backend_device_name: comparisonReference.backend_device_name ?? null,
            repeats: (comparisonReference.config as RunConfig)?.sweep?.repeats ?? null,
            method_version: null,
            grid_signature: gridSignature(
              (comparisonReference.config as RunConfig)?.sweep as unknown as Record<string, unknown>
            ),
          },
          candidateFacts
        );
        if (violations.length > 0) {
          request.log.warn(
            {
              comparison_member_failed: true,
              member_run_id: null,
              reason: violations[0].field,
              comparison_id: body.comparison_id,
            },
            "comparison_member_failed"
          );
          return reply.code(400).send({ error: violations[0].message });
        }
      }

      // EXPLICIT field list -- never spread the request body into an insert.
      const runId = uuid();
      const runConfig: RunConfig = {
        model_id: body.model_id,
        mtp_model_id: mtpModel?.id,
        main_gpu: body.main_gpu,
        main_gpu_backend: body.main_gpu_backend,
        sweep: body.sweep,
      };
      if (goals) runConfig.goals = goals;
      if (discardFirst > 0) runConfig.discard_first_repeats = discardFirst;
      if (rootRunId) runConfig.chain_depth = chainDepth;
      // N2/N1/N5 -- the spec each of these run kinds executes is stored on the
      // run itself, so the worker-authed ingestion routes derive KV pair,
      // placement and context from the RUN, never from the reported payload.
      if (body.probe) runConfig.probe = body.probe;
      if (body.quality) runConfig.quality = body.quality;
      if (body.curve_point) runConfig.curve_point = body.curve_point;
      if (body.knee) runConfig.knee = body.knee;
      const run: Run = {
        id: runId,
        // §0.5 -- denormalized at creation; points at the run itself for
        // standalone runs.
        root_run_id: rootRunId ?? runId,
        kind: kind as RunKind | null,
        comparison_id: body.comparison_id ?? null,
        worker_id: worker.id,
        worker_name: worker.displayName, // point-in-time snapshot; the export reads this directly
        llama_cpp_build: resolved.tag,
        llama_cpp_backend: targetBackend,
        backend_device_name: resolved.deviceName,
        model_id: body.model_id,
        config: runConfig,
        // Flipped to 'running' by queueRepo.claimNextJob once the worker
        // actually claims the benchmark job (MULTIUSER_PLAN.md §1.13) --
        // never here, regardless of whether the worker is currently idle or
        // busy with something else. The worker_jobs queue is what serializes
        // execution now; there's no more "queue in the DB if busy" branch.
        status: "scheduled",
        started_at: Date.now(),
      };
      repo.createRun(userId, run);
      // A context-curve run gets one item PER MEASURED CONTEXT, not the
      // sweep's own cross-product -- the sweep here only supplies the
      // placement/KV template (expanded[0]); n_prompt is overwritten per
      // item with the context it actually measures, one per curve point, so
      // status/progress tracks each ladder cell individually.
      const items = body.curve_point
        ? curveContexts.map((ctx, i) => ({ ...expanded[0], idx: i, n_prompt: ctx }))
        : expanded;
      repo.createRunItems(userId, run.id, items);
      request.log.info(
        { run_id: run.id, worker: worker.id, model_id: body.model_id, items: items.length, kind },
        "run created"
      );

      if (!resolved.alreadyInstalled) {
        repo.queueRepo.enqueueJob(worker.id, {
          type: "install_build",
          payload: resolved.installPayload,
          runId: run.id,
        });
      }
      if (body.kind === "probe" && body.probe) {
        // N2 -- a probe is its own job type, not a sweep: the engine is pinned
        // to llama-server (the tok/s floor is meaningless on llama-bench,
        // which has no request lifecycle) and it performs at most three loads
        // ever, so it never rides the sweep executor.
        const probePayload: RunProbeJobPayload = {
          run_id: run.id,
          model_id: body.model_id,
          model,
          candidateCtx: body.probe.candidate_ctx,
          placement: {
            ngl: body.probe.placement.ngl,
            nCpuMoe: body.probe.placement.n_cpu_moe,
            slots: body.probe.placement.slots ?? 1,
          },
          kvPair: body.probe.kv_pair,
          llama_cpp_build: resolved.tag,
          llama_cpp_backend: targetBackend,
          main_gpu: body.main_gpu,
          trained_ctx: typeof model.metadata.trained_ctx === "number" ? model.metadata.trained_ctx : null,
          gpu_total_mib: worker.vram?.ok ? worker.vram.gpu_memory_total_mib : null,
        };
        repo.queueRepo.enqueueJob(worker.id, { type: "run_probe", payload: probePayload, runId: run.id });
      } else if (body.kind === "quality" && body.quality) {
        // N4 -- a quality measurement is its own job type too, same reason as
        // a probe: one llama-perplexity invocation, not a sweep.
        const qualityPayload: MeasureQualityJobPayload = {
          run_id: run.id,
          model_id: body.model_id,
          model,
          ctxTokens: body.quality.ctx_tokens,
          kvPair: body.quality.kv_pair,
          datasetHash: body.quality.dataset_hash,
          datasetLicense: body.quality.dataset_license,
          llama_cpp_build: resolved.tag,
          llama_cpp_backend: targetBackend,
          main_gpu: body.main_gpu,
        };
        repo.queueRepo.enqueueJob(worker.id, { type: "measure_quality", payload: qualityPayload, runId: run.id });
      } else {
        const benchmarkPayload: BenchmarkJob = {
          run_id: run.id,
          model,
          mtp_model: mtpModel,
          sweep: body.sweep,
          main_gpu: body.main_gpu,
          llama_cpp_build: resolved.tag,
          llama_cpp_backend: targetBackend,
          ...(body.curve_point
            ? { mode: "context_curve" as const, curve_point: body.curve_point }
            : body.knee
              ? { mode: "knee" as const, knee_spec: body.knee }
              : {}),
        };
        repo.queueRepo.enqueueJob(worker.id, { type: "benchmark", payload: benchmarkPayload, runId: run.id });
      }
      queueEvents.emit(worker.id);

      return reply.code(201).send({ run, warning: budgetWarning });
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
        const wasTerminal = run.status !== "running" && run.status !== "scheduled";
        const updatedRun = repo.recordRunItemTerminal(id, idx, body);
        if (!updatedRun) return reply.code(404).send({ error: "run not found" });
        const nowTerminal = updatedRun.status !== "running" && updatedRun.status !== "scheduled";
        if (!wasTerminal && nowTerminal) {
          // §0.12's minimum event set -- fired exactly once, from the write
          // that actually finalizes the run (not from GET /sustained, which
          // the UI polls repeatedly and would otherwise re-log this every
          // page view).
          const { ratio, flagged, denominator } = repo.thermallyFlaggedRatio(id);
          request.log.info(
            { thermally_flagged_ratio: ratio, run_id: id, flagged, denominator },
            "thermally_flagged_ratio"
          );
          // N3 -- re-checked PER MEMBER at the moment its own results land,
          // not only when someone happens to open the comparison view: a
          // build swapped mid-group is exactly the silent confound this
          // exists to catch, and it can only be seen once the member has
          // actually run. recheckComparisonMember logs comparison_member_failed
          // itself on any violation.
          if (updatedRun.comparison_id) {
            recheckComparisonMember(updatedRun.comparison_id, updatedRun, request.log);
          }
        }
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
