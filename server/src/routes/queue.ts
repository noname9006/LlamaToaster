import type { FastifyInstance } from "fastify";
import type { IncomingMessage } from "node:http";
import { repo } from "../db/repo.js";
import { queueEvents } from "../queue-events.js";
import { authenticateWorker } from "../worker-auth.js";
import { parseWorkerState, parseActiveJobReport, parseActiveDownloads } from "../validate-worker-state.js";
import { LEASE_MS } from "../liveness.js";
import { NotFoundError } from "../errors.js";
import type { QueueJob, HeartbeatResponse } from "../../../shared/types.js";

// 25s keeps the hanging long-poll comfortably under Caddy's/any intermediary's
// 30s idle default. 10s heartbeat (worker/src/index.ts) gives cancel/pause a
// worst-case ~10s latency -- fine for a Stop button on a multi-hour job.
// See MULTIUSER_PLAN.md §1.3.
const LONG_POLL_MS = 25_000;
const BACKSTOP_MS = 5_000; // safety net only; the queueEvents emit is the primary wake-up
const WORKER_BODY_LIMIT = 1_000_000; // 1 MB -- see MULTIUSER_PLAN.md §1.8

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
      return { worker_id: worker.id, control, lease_until: Date.now() + LEASE_MS };
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
