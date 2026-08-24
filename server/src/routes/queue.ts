import type { FastifyInstance } from "fastify";
import type { IncomingMessage } from "node:http";
import { repo } from "../db/repo.js";
import { queueEvents } from "../queue-events.js";
import { authenticateWorker } from "../worker-auth.js";
import { parseWorkerState, parseActiveJobReport, parseActiveDownloads } from "../validate-worker-state.js";
import { LEASE_MS } from "../liveness.js";
import { NotFoundError } from "../errors.js";
import { getHfGgufMeta } from "../hf.js";
import { log } from "../log.js";
import type { QueueJob, HeartbeatResponse, WorkerStatePush, ModelMetadata, ModelDirFile } from "../../../shared/types.js";
import { isMtpDraftModel } from "../../../shared/types.js";

// 25s keeps the hanging long-poll comfortably under Caddy's/any intermediary's
// 30s idle default. 10s heartbeat (worker/src/index.ts) gives cancel/pause a
// worst-case ~10s latency -- fine for a Stop button on a multi-hour job.
// See MULTIUSER_PLAN.md §1.3.
const LONG_POLL_MS = 25_000;
const BACKSTOP_MS = 5_000; // safety net only; the queueEvents emit is the primary wake-up
const WORKER_BODY_LIMIT = 1_000_000; // 1 MB -- see MULTIUSER_PLAN.md §1.8
const HF_META_TIMEOUT_MS = 15_000;

// A model dropped into a worker's model_dir by hand (hash-matched via the
// catalog rather than downloaded through this app -- see
// registerHashVerifiedModelFiles below) never goes through
// workers.ts's download-callback route, so it misses that route's inline
// getHfGgufMeta call and would otherwise sit with no param_count forever,
// showing "—" on the Models page until someone clicks the manual backfill
// button. Fetches in the background (never awaited by the heartbeat/queue
// handlers that call this) since HF can take up to HF_META_TIMEOUT_MS and a
// heartbeat response must stay fast; deduped per model id so a slow lookup
// isn't re-fired every ~10s until it resolves.
const paramLookupInFlight = new Set<string>();
function backfillParamCountInBackground(modelId: string, hfRepo: string): void {
  if (paramLookupInFlight.has(modelId)) return;
  paramLookupInFlight.add(modelId);
  getHfGgufMeta(hfRepo, HF_META_TIMEOUT_MS)
    .then(({ param_count }) => {
      if (typeof param_count === "number") {
        repo.updateModelMetadata(modelId, { param_count });
      }
    })
    .catch((err) => {
      log.warn(`[queue] background param_count lookup failed for ${hfRepo}: ${err instanceof Error ? err.message : String(err)}`);
    })
    .finally(() => {
      paramLookupInFlight.delete(modelId);
    });
}

// A worker's reconciliation just told us, per model_dir file, whether its
// SHA-256 matched the server's HF index (state "verified" + hf_match -- see
// worker/src/model-scanner.ts's resolveHfMetadata). Any such file whose hash
// isn't in the catalog yet gets registered here, keyed by sha256 exactly like
// download-callback registrations are (routes/workers.ts). This is what makes
// a model dropped into a worker's model_dir by hand -- never downloaded
// through the app -- show up on the Models page under the machine that has it.
// Existence-checked first because heartbeats arrive every ~10s: re-upserting
// an already-correct row each beat would be pure write churn. Still
// re-registers when the stored hf_repo/hf_file disagree with this beat's
// hf_match (e.g. a row written by a since-fixed reconstruction bug) so a
// worker code fix self-heals previously-corrupted catalog rows on their next
// scan instead of leaving them wrong forever. Never throws -- catalog
// bookkeeping must not break heartbeat ingestion.
function registerHashVerifiedModelFiles(files: WorkerStatePush["model_files"]): void {
  for (const f of files) {
    if (!f.sha256 || f.state !== "verified" || !f.hf_match || f.hf_match.deleted) continue;
    try {
      const existing = repo.getModel(f.sha256);

      // The file's own GGUF header metadata rides along on the heartbeat (see
      // ModelDirFile's doc comments and worker/src/model-scanner.ts's
      // backfillGgufMetadata) -- the per-file authoritative source for a
      // hand-dropped file's quant/param_count/layer-count, adopted here the
      // way the download-callback route adopts it for an in-app download.
      // Without this, a hand-dropped file would keep its "?" quant and a
      // filename/HF-derived (often wrong -- see getHfGgufMeta's repo-level
      // total) param count forever.
      const patch = ggufMetadataPatch(f);
      const merged = { ...(existing?.metadata ?? {}), ...patch };
      if (isMtpDraftModel({ metadata: merged, hf_file: f.hf_match.filename, filename: f.hf_match.filename })) {
        merged.mtp_role = "draft";
      }
      const identityOk =
        existing && existing.hf_repo === f.hf_match.repo_id && existing.hf_file === f.hf_match.filename;
      if (!identityOk) {
        repo.registerModel({
          id: f.sha256,
          filename: f.hf_match.filename,
          size_bytes: f.size_bytes,
          source: "huggingface",
          hf_repo: f.hf_match.repo_id,
          hf_file: f.hf_match.filename,
          metadata: merged,
        });
      }
      // Adoption-only: adjust metadata without touching identity. Identity
      // updates are ownership-gated (registerModel's ON CONFLICT WHERE -- a
      // foreign-owner row can block the whole statement, metadata included),
      // but filling in a derived metadata field must stay open:
      // updateModelMetadata is that always-open path. Re-reads the row so the
      // check sees the post-registration truth in both branches, and passes
      // the fully-merged shape (a superset of what's stored) -- equivalent to
      // applying just the diff and immune to field-by-field drift.
      const fresh = repo.getModel(f.sha256);
      if (fresh && JSON.stringify(fresh.metadata) !== JSON.stringify(merged)) {
        repo.updateModelMetadata(f.sha256, merged);
      }
      // Unlike workers.ts's download-callback route, a scan-discovered match
      // used to have no file-derived param_count -- fetch one from HF's
      // repo-level API in the background for any row (fresh or pre-existing)
      // still missing it. Skipped when the worker supplied its own count.
      if (typeof merged.param_count !== "number") {
        backfillParamCountInBackground(f.sha256, f.hf_match.repo_id);
      }
    } catch {
      // duplicate registration race or a transient DB hiccup -- the next
      // beat simply retries; nothing downstream depends on this insert.
    }
  }
}

// Maps a verified model file's heartbeat-reported GGUF header fields onto the
// ModelMetadata shape (see ModelDirFile / worker/src/gguf.ts). Only fields
// with a real value are included, so calling it with a file whose header
// couldn't be parsed produces an empty patch (a no-op for the caller).
function ggufMetadataPatch(f: ModelDirFile): Partial<ModelMetadata> {
  const patch: Partial<ModelMetadata> = {};
  if (typeof f.n_layer === "number") patch.n_layer = f.n_layer;
  if (typeof f.mtp_layers === "number" && f.mtp_layers > 0) patch.mtp_layers = f.mtp_layers;
  if (typeof f.expert_count === "number" && f.expert_count > 0) patch.expert_count = f.expert_count;
  if (typeof f.param_count === "number") patch.param_count = f.param_count;
  if (typeof f.quant === "string" && f.quant) patch.quant = f.quant;
  return patch;
}

// Resolves as soon as a job is claimable for this worker (woken by
// queueEvents) or LONG_POLL_MS elapses, whichever comes first. Resolves
// `null` (never claims) if the client disconnects first -- claiming a job
// onto a socket that's already gone would strand it "claimed" with nobody to
// execute it until the lease reaper eventually rescues it (MULTIUSER_PLAN.md §1.4).
function waitForJob(workerId: string, raw: IncomingMessage): Promise<QueueJob | null> {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v: QueueJob | null) => {
      if (done) return;
      done = true;
      queueEvents.off(workerId, onEvent);
      raw.off("close", onClose);
      clearInterval(backstop);
      clearTimeout(deadline);
      resolve(v);
    };
    const tryClaim = () => {
      const job = repo.queueRepo.claimNextJob(workerId);
      if (job) finish(job);
    };
    const onEvent = () => tryClaim();
    const onClose = () => finish(null);
    const backstop = setInterval(tryClaim, BACKSTOP_MS);
    const deadline = setTimeout(() => finish(null), LONG_POLL_MS);
    queueEvents.on(workerId, onEvent);
    raw.on("close", onClose);
    tryClaim(); // race: a job enqueued between the caller's own sync claim and this call
  });
}

export async function queueRoutes(app: FastifyInstance): Promise<void> {
  // Only called while the worker is idle; hangs up to LONG_POLL_MS waiting
  // for work. See MULTIUSER_PLAN.md §1.3/§1.4.
  app.post("/api/worker/queue", { bodyLimit: WORKER_BODY_LIMIT }, async (req, reply) => {
    const worker = await authenticateWorker(req);
    const state = parseWorkerState(req.body);
    repo.workerRepo.recordHeartbeat(worker.id, state, null);
    registerHashVerifiedModelFiles(state.model_files);

    // A worker that says it is busy must never be handed a second job --
    // it should be heartbeating, not queue-polling, while active.
    if (state.status === "busy") return reply.code(204).send();

    const claimed = repo.queueRepo.claimNextJob(worker.id);
    if (claimed) return reply.send(claimed);

    const job = await waitForJob(worker.id, req.raw);
    return job ? reply.send(job) : reply.code(204).send();
  });

  // Called every 10s, ALWAYS (idle and busy) -- returns fast. Carries
  // control directives (cancel/pause) and, while a job is active, its live
  // progress. See MULTIUSER_PLAN.md §1.5. logLevel silent: same treatment as
  // the existing high-frequency polling routes in index.ts.
  app.post(
    "/api/worker/heartbeat",
    { logLevel: "silent", bodyLimit: WORKER_BODY_LIMIT },
    async (req): Promise<HeartbeatResponse> => {
      const worker = await authenticateWorker(req);
      const state = parseWorkerState(req.body);
      const active = parseActiveJobReport(req.body);
      const activeDownloads = parseActiveDownloads(req.body);

      repo.workerRepo.recordHeartbeat(worker.id, state, active?.job_id ?? null);
      registerHashVerifiedModelFiles(state.model_files);

      const control: HeartbeatResponse["control"] = {
        cancel_job_ids: [],
        discard_job_ids: [],
        pause: repo.workerRepo.getPauseRequested(worker.id),
      };
      // Same lease-extend-and-check-cancel dance for both the one serial job
      // and every concurrently-active download (worker/src/index.ts's
      // downloadJobPool) -- extendLeaseAndGetFlags is generic per job id, not
      // tied to job type, so a download's Pause/Cancel buttons
      // (routes/workers.ts) reach the worker through this exact same
      // cancel_job_ids channel; discard_job_ids additionally tells it to
      // delete the .part file for a Cancel rather than keep it for resume.
      for (const report of active ? [active, ...activeDownloads] : activeDownloads) {
        const flags = repo.queueRepo.extendLeaseAndGetFlags(report.job_id, worker.id, report);
        if (!flags) {
          // The server no longer considers this job live for this worker
          // (lease expired and reassigned, or the run was deleted/reconciled).
          // Tell the worker to stop rather than let two executions of the
          // same job both report results.
          control.cancel_job_ids.push(report.job_id);
        } else if (flags.cancel_requested) {
          control.cancel_job_ids.push(report.job_id);
          if (flags.discard_requested) control.discard_job_ids.push(report.job_id);
        }
      }
      // Operator-controlled settings ride along on every beat (see
      // HeartbeatResponse.app_settings) -- the heartbeat cadence doubles as
      // the settings-propagation channel, so flipping a policy on the
      // supervise dashboard reaches an already-running worker within one
      // interval without any restart.
      return {
        worker_id: worker.id,
        control,
        lease_until: Date.now() + LEASE_MS,
        app_settings: repo.appSettingsRepo.get(),
      };
    }
  );

  // Reported once by the worker after it finishes executing a claimed job
  // (any type: benchmark, install_build, download_model, delete_model_file)
  // -- the ONLY thing that ever moves a job out of 'claimed' on success.
  // Without this, a completed job's lease would simply expire with nothing
  // extending it, and the reaper (index.ts) would requeue it as if the
  // worker had crashed -- re-handing an already-finished benchmark job back
  // out for a second execution. A 'benchmark' job's run_items/results are
  // already recorded via the existing per-item POST /api/runs/:id/items/:idx
  // reports throughout execution; this call is purely about the job
  // bookkeeping row, not run data.
  app.post<{ Params: { jobId: string }; Body: { ok?: boolean; error?: string } }>(
    "/api/worker/jobs/:jobId/complete",
    { bodyLimit: 100_000 },
    async (req) => {
      const worker = await authenticateWorker(req);
      const job = repo.queueRepo.getJob(req.params.jobId);
      if (!job || job.workerId !== worker.id) throw new NotFoundError("unknown job");
      // Idempotent: the worker retries this call with backoff (same posture
      // as its existing item-terminal reporting), so a job already marked
      // terminal by an earlier attempt (or by the reaper, in a rare race) is
      // a harmless no-op, not an error.
      if (job.status !== "claimed") return { ok: true };

      if (req.body?.ok === false) {
        const error = typeof req.body.error === "string" ? req.body.error.slice(0, 2000) : "job failed";
        repo.queueRepo.markJobFailed(job.id, error);
        if (job.runId) {
          // undefined userId -- this is the worker's own job-completion
          // report, not a browser request; see repo.ts's reconcileStaleRun
          // doc comment (MULTIUSER_PLAN.md §4.3).
          repo.reconcileStaleRun(undefined, job.runId, error);
          // See the matching comment in index.ts's reapExpiredLeases -- an
          // install_build failure can leave a sibling benchmark job pending.
          repo.queueRepo.cancelPendingJobsForRun(job.runId);
        }
      } else {
        repo.queueRepo.markJobCompleted(job.id);
      }
      return { ok: true };
    }
  );
}
