// BENCHMARKING_PLAN_V8.md N2 + N4 result ingestion.
//
// Security, stated once and enforced identically in both handlers:
//
//  * An ENROLLED WORKER SESSION is required -- the per-worker bearer
//    credential established when the machine was approved. The shared
//    deployment secret is refused here (the chain-advance rule refuses it for
//    the same reason), because these rows feed *verified* claims other
//    tenants see.
//  * Authorization binds BY PATH, not payload: worker_id comes from the run
//    named in the URL, so a spoofed identity cannot poison another machine's
//    verified ceiling.
//  * Payload bounds mirror the sweep-axis table; out-of-bounds is a 400,
//    never a silent clamp, and a malformed reading is never stored as data.

import type { FastifyInstance, FastifyRequest } from "fastify";
import { repo } from "../db/repo.js";
import { hashToken } from "../session.js";
import { BadRequestError, ForbiddenError, NotFoundError, UnauthorizedError } from "../errors.js";
import { resolveAuthUser, assertOwnsWorker } from "../auth-middleware.js";
import type {
  ProbeAttemptReport,
  ProbeResultInput,
  QualityResultInput,
  Run,
  RunConfig,
  Worker,
} from "../../../shared/types.js";
import { METHOD_VERSION } from "../../../shared/types.js";
import { isKnownCacheType, CACHE_TYPE_VALUES } from "../../../shared/engineSpec.js";
import { placementHash } from "../../../shared/configHash.js";

export const MIN_PROBE_CTX = 256;
export const MAX_PROBE_CTX = 4_194_304;
// Sanity bound on a reported verified_ngl. No real model comes close; this
// only stops a malformed report from poisoning a placement hash.
export const MAX_PROBE_NGL = 4096;
// The ladder's own hard ceiling on loads. Was 3 when the ladder was a single
// multiplicative retry; the coarse-to-fine search brackets then bisects, which
// needs more rungs. Mirrors appSettingsRepo's documented probeMaxLoads
// default (repo.ts's DEFAULT_PROBE_MAX_LOADS) -- kept exported as "the
// shipped default" for callers/tests that want it, but the actual
// per-request bound below is the LIVE admin-configurable value
// (appSettingsRepo.getProbeMaxLoads()), not this constant, so an operator's
// override on the supervise dashboard is actually enforced and a
// misbehaving worker can't report more loads than that live setting allows.
export const MAX_PROBE_ATTEMPTS = 24;
export const DATASET_HASH_RE = /^sha256:[0-9a-f]{64}$/;

// The enrolled-worker-session rule. Deliberately NOT authenticateWorker():
// that function is dual-mode and falls back to WORKER_SHARED_TOKEN, which is
// exactly what must not be accepted on these two routes.
function requireEnrolledWorkerSession(request: FastifyRequest): Worker {
  const auth = request.headers.authorization;
  const token = auth?.startsWith("Bearer ") ? auth.slice("Bearer ".length) : undefined;
  if (!token) throw new UnauthorizedError("missing worker token");
  const session = repo.sessionRepo.getByTokenHash(hashToken(token));
  if (!session || session.expiresAt < Date.now() || !session.isWorker || !session.workerId) {
    throw new UnauthorizedError(
      "this route requires an enrolled worker session -- the shared deployment secret is not accepted for verified measurements"
    );
  }
  const worker = repo.workerRepo.getWorker(session.workerId);
  if (!worker) throw new UnauthorizedError("worker session points at a machine that no longer exists");
  repo.sessionRepo.touch(session);
  return worker;
}

// The run named in the URL is the authority on which machine may report.
function resolveRunForWorker(runId: string, worker: Worker): Run {
  const run = repo.getRun(undefined, runId);
  if (!run) throw new NotFoundError("run not found");
  if (!run.worker_id || run.worker_id !== worker.id) {
    throw new ForbiddenError("this run was dispatched to a different machine");
  }
  return run;
}

function requireFiniteInt(value: unknown, field: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value) || value < min || value > max) {
    throw new BadRequestError(`${field} must be an integer in [${min}, ${max}]`);
  }
  return value;
}

// An optional measured/estimated MiB (or tok/s) reading. Absent and null both
// mean "this worker didn't report it" and store as NULL; a present value must
// be a finite non-negative number, never NaN/Infinity/a string -- a malformed
// reading is rejected, never stored as data.
function optionalNonNegative(value: unknown, field: string): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new BadRequestError(`${field} must be a non-negative finite number or null`);
  }
  return value;
}

// Validates the ladder rungs before any of them reach the database. Bounds
// mirror the trigger route's own probe-axis table (runs.ts): context within
// the probe range, ngl within the sweep axis' [0, 1024].
function validateProbeAttempts(attempts: unknown): ProbeAttemptReport[] {
  if (attempts === undefined) return [];
  if (!Array.isArray(attempts)) throw new BadRequestError("attempts must be an array");
  // Live admin-configurable cap (see shared/types.ts's AppSettings.probeMaxLoads),
  // not the MAX_PROBE_ATTEMPTS constant above -- an operator's override on the
  // supervise dashboard must actually be enforced here, not just advertised.
  const maxProbeAttempts = repo.appSettingsRepo.getProbeMaxLoads();
  if (attempts.length > maxProbeAttempts) {
    throw new BadRequestError(`a probe performs at most ${maxProbeAttempts} loads`);
  }
  return attempts.map((raw, i) => {
    const a = (raw ?? {}) as Record<string, unknown>;
    if (typeof a.ok !== "boolean" || typeof a.oom !== "boolean" || typeof a.spill !== "boolean") {
      throw new BadRequestError(`attempts[${i}] must carry boolean ok/oom/spill`);
    }
    return {
      candidate_ctx: requireFiniteInt(a.candidate_ctx, `attempts[${i}].candidate_ctx`, MIN_PROBE_CTX, MAX_PROBE_CTX),
      ok: a.ok,
      oom: a.oom,
      spill: a.spill,
      ngl: a.ngl === undefined || a.ngl === null ? null : requireFiniteInt(a.ngl, `attempts[${i}].ngl`, 0, 1024),
      vram_needed_mib: optionalNonNegative(a.vram_needed_mib, `attempts[${i}].vram_needed_mib`),
      vram_free_mib: optionalNonNegative(a.vram_free_mib, `attempts[${i}].vram_free_mib`),
      vram_peak_mib: optionalNonNegative(a.vram_peak_mib, `attempts[${i}].vram_peak_mib`),
      ram_needed_mib: optionalNonNegative(a.ram_needed_mib, `attempts[${i}].ram_needed_mib`),
      ram_free_mib: optionalNonNegative(a.ram_free_mib, `attempts[${i}].ram_free_mib`),
      ram_peak_mib: optionalNonNegative(a.ram_peak_mib, `attempts[${i}].ram_peak_mib`),
      gen_tps: optionalNonNegative(a.gen_tps, `attempts[${i}].gen_tps`),
      error: typeof a.error === "string" ? a.error : undefined,
    } satisfies ProbeAttemptReport;
  });
}

export async function measurementRoutes(app: FastifyInstance): Promise<void> {
  // --- N2: probe result -----------------------------------------------------
  app.post<{ Params: { id: string }; Body: ProbeResultInput }>(
    "/api/runs/:id/probe-result",
    async (request, reply) => {
      const worker = requireEnrolledWorkerSession(request);
      const run = resolveRunForWorker(request.params.id, worker);
      if (run.kind !== "probe") throw new BadRequestError("that run is not a probe run");

      const spec = (run.config as RunConfig).probe;
      if (!spec) throw new BadRequestError("that probe run has no stored probe spec");

      const body = request.body ?? ({} as ProbeResultInput);
      if (body.status !== "verified" && body.status !== "failed" && body.status !== "failed_oom") {
        throw new BadRequestError("status must be verified/failed/failed_oom");
      }
      const attempts = validateProbeAttempts(body.attempts);

      // The KV pair and placement come from the RUN's stored spec, never the
      // payload: verification is per (machine, build, KV pair, placement).
      const [cacheK, cacheV] = spec.kv_pair;
      if (!isKnownCacheType(cacheK) || !isKnownCacheType(cacheV)) {
        throw new BadRequestError(`probe kv_pair must be drawn from ${CACHE_TYPE_VALUES.join(", ")}`);
      }

      // Every rung, not just the winner -- written before the ceiling upsert
      // so a probe that reached no usable context still leaves the evidence
      // of what it tried. Replaces the run's previous rows, so a worker retry
      // reports one ladder rather than two interleaved ones.
      repo.probeAttemptsRepo.replaceForRun({
        run_id: run.id,
        worker_id: worker.id,
        model_id: run.model_id,
        attempts,
      });

      let stored = null;
      if (body.status === "verified") {
        const verifiedCtx = requireFiniteInt(
          body.verified_ctx_tokens,
          "verified_ctx_tokens",
          MIN_PROBE_CTX,
          MAX_PROBE_CTX
        );
        if (
          body.margin_observed_frac !== undefined &&
          body.margin_observed_frac !== null &&
          (typeof body.margin_observed_frac !== "number" ||
            !Number.isFinite(body.margin_observed_frac) ||
            body.margin_observed_frac < 0 ||
            body.margin_observed_frac > 1)
        ) {
          throw new BadRequestError("margin_observed_frac must be a fraction in [0, 1] or null");
        }
        // The ladder moves ngl in every mode but fixed_offload/custom, so the
        // placement that was REQUESTED is routinely not the placement the
        // ceiling was verified at -- keying the row on the request would file
        // "N tokens loaded here" under a placement that may well have failed
        // outright (max_gpu's very first rung is every layer, and that is the
        // one that usually doesn't fit). Key on what the worker actually
        // loaded, falling back to the request only for a worker predating the
        // field.
        const verifiedNgl =
          body.verified_ngl === undefined || body.verified_ngl === null
            ? null
            : requireFiniteInt(body.verified_ngl, "verified_ngl", 0, MAX_PROBE_NGL);
        stored = repo.limitsRepo.upsert({
          worker_id: worker.id,
          model_id: run.model_id,
          llama_cpp_build: run.llama_cpp_build ?? "",
          cache_type_k: cacheK,
          cache_type_v: cacheV,
          placement_hash: placementHash({
            ngl: verifiedNgl ?? spec.placement.ngl,
            n_cpu_moe: spec.placement.n_cpu_moe,
            slots: spec.placement.slots,
          }),
          verified_ctx_tokens: verifiedCtx,
          verified_ngl: verifiedNgl,
          margin_observed_frac: body.margin_observed_frac ?? null,
          method_version: body.method_version ?? METHOD_VERSION,
        });
      }

      // The run itself finalizes through the ordinary item path, so a probe
      // shows up in the Runs list badged like anything else.
      repo.recordRunItemTerminal(run.id, 0, {
        status: body.status === "verified" ? "done" : body.status,
        error:
          body.status === "verified"
            ? undefined
            : (body.error ?? "probe did not reach a usable context at this placement"),
      });

      request.log.info(
        {
          probe_finished: true,
          outcome: body.status,
          loads_used: attempts.length,
          run_id: run.id,
          worker_id: worker.id,
        },
        "probe_finished"
      );
      return reply.code(200).send({ ok: true, limit: stored, attempts_stored: attempts.length });
    }
  );

  // The ladder behind a run's verified ceiling. Read-side authorization
  // mirrors /api/models/:id/verified-limits: the caller must own the machine
  // the run was dispatched to, since these rows describe that machine's
  // memory. Worker sessions never read this -- it exists for the UI.
  app.get<{ Params: { id: string } }>("/api/runs/:id/probe-attempts", async (request, reply) => {
    const run = repo.getRun(undefined, request.params.id);
    if (!run) throw new NotFoundError("run not found");
    if (!run.worker_id) throw new NotFoundError("that run has no machine attached");
    assertOwnsWorker(resolveAuthUser(request)?.user.id, run.worker_id);
    return reply.send({ attempts: repo.probeAttemptsRepo.listForRun(run.id) });
  });

  // --- N4: quality result ---------------------------------------------------
  app.post<{ Params: { id: string }; Body: QualityResultInput }>(
    "/api/runs/:id/quality-result",
    async (request, reply) => {
      const worker = requireEnrolledWorkerSession(request);
      const run = resolveRunForWorker(request.params.id, worker);
      if (run.kind !== "quality") throw new BadRequestError("that run is not a quality run");

      const body = request.body ?? ({} as QualityResultInput);
      // Validation BEFORE any write -- malformed readings are a 400, never
      // stored as data.
      if (typeof body.dataset_hash !== "string" || !DATASET_HASH_RE.test(body.dataset_hash)) {
        throw new BadRequestError("dataset_hash must match sha256:<64 lowercase hex>");
      }
      const ctxTokens = requireFiniteInt(body.ctx_tokens, "ctx_tokens", MIN_PROBE_CTX, MAX_PROBE_CTX);
      if (typeof body.ppl !== "number" || !Number.isFinite(body.ppl) || body.ppl <= 0) {
        throw new BadRequestError("ppl must be a finite number greater than 0");
      }
      if (
        body.kld_vs_baseline !== undefined &&
        body.kld_vs_baseline !== null &&
        (typeof body.kld_vs_baseline !== "number" ||
          !Number.isFinite(body.kld_vs_baseline) ||
          body.kld_vs_baseline < 0)
      ) {
        throw new BadRequestError("kld_vs_baseline must be a finite number >= 0, or null");
      }
      if (!isKnownCacheType(body.cache_type_k) || !isKnownCacheType(body.cache_type_v)) {
        throw new BadRequestError(`cache_type_k/cache_type_v must be drawn from ${CACHE_TYPE_VALUES.join(", ")}`);
      }

      const row = repo.qualityRepo.upsert({
        // Quality work is run-scoped like everything else, and the root is
        // what groups it.
        root_run_id: run.root_run_id ?? run.id,
        model_id: run.model_id,
        worker_id: worker.id,
        llama_cpp_build: run.llama_cpp_build ?? "",
        ctx_tokens: ctxTokens,
        cache_type_k: body.cache_type_k,
        cache_type_v: body.cache_type_v,
        ppl: body.ppl,
        kld_vs_baseline: body.kld_vs_baseline ?? null,
        dataset_hash: body.dataset_hash,
        method_version: body.method_version ?? METHOD_VERSION,
      });

      // A quality run is a single measurement, same shape as a probe -- it
      // always finalizes its one run_item (idx 0) here, the same place a
      // probe's own result route does. Without this the run never reaches a
      // terminal status: nothing else ever reports this item done.
      repo.recordRunItemTerminal(run.id, 0, { status: "done" });

      request.log.info(
        { quality_job_finished: true, outcome: "ok", run_id: run.id, worker_id: worker.id },
        "quality_job_finished"
      );
      return reply.code(200).send({ ok: true, quality: row });
    }
  );
}
