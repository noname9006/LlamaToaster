import { repo } from "./db/repo.js";
import { queueEvents } from "./queue-events.js";
import { OFFLINE_AFTER_MS } from "./liveness.js";

// Replaces the deleted push-model self-healing path (the old /health probe
// inside GET /api/runs/:id, and the /stop proxy's 409 handling) -- a crashed
// or silently-dead worker's claimed job would otherwise sit "claimed"
// forever with its run stuck "running". See MULTIUSER_PLAN.md §1.7.
export const REAP_INTERVAL_MS = 15_000;
export const MAX_ATTEMPTS = 2;

export interface ReaperLogger {
  warn: (obj: Record<string, unknown>, msg: string) => void;
}

interface ReapableJob {
  id: string;
  worker_id: string;
  run_id: string | null;
  attempts: number;
  cancel_requested: number;
}

// Marks a job permanently abandoned: terminal-fails it, reconciles its run
// (if any) instead of leaving it stuck, and cancels any pending sibling job
// for the same run (an install_build/benchmark pair sharing one run_id) so
// it can't fire later against an already-finalized run.
function abandonJob(job: ReapableJob, reason: string, log: ReaperLogger, event: string): void {
  repo.queueRepo.markJobFailed(job.id, reason);
  if (job.run_id) {
    // undefined userId -- the reaper has no request/user context, and a
    // stale job's run needs reconciling regardless of who owns it (see
    // repo.ts's reconcileStaleTest doc comment, MULTIUSER_PLAN.md §4.3).
    repo.reconcileStaleTest(undefined, job.run_id, reason);
    repo.queueRepo.cancelPendingJobsForTest(job.run_id);
  }
  log.warn({ job_id: job.id, run_id: job.run_id, cancel_requested: !!job.cancel_requested }, event);
}

export function reapExpiredLeases(log: ReaperLogger): void {
  const expired = repo.queueRepo.listExpiredLeases(Date.now());
  for (const job of expired) {
    // A cancelled job must never be requeued, regardless of attempts: the
    // worker that had it never confirmed the stop before its lease expired
    // (e.g. it crashed right after a Stop was requested), and
    // returnToPending doesn't clear cancel_requested (nor should it --
    // clearing it would silently resume a run the user explicitly asked to
    // stop). Without this branch, claimNextJob's own `cancel_requested = 0`
    // filter means such a job would go back to 'pending' and then never be
    // claimable by anyone again -- a permanently stuck job with its run
    // stuck 'running' forever. Finalizing it here the same way a
    // max-attempts abandonment is handled fixes both: the job resolves to a
    // real terminal status, and the run gets reconciled instead of hanging.
    if (job.cancel_requested || job.attempts >= MAX_ATTEMPTS) {
      const reason = job.cancel_requested
        ? "worker did not confirm the stop before it stopped reporting"
        : "worker stopped reporting while running this job";
      abandonJob(job, reason, log, "job abandoned after lease expiry");
    } else {
      repo.queueRepo.returnToPending(job.id);
      queueEvents.emit(job.worker_id);
      log.warn({ job_id: job.id, attempts: job.attempts + 1 }, "job lease expired, requeued");
    }
  }

  // A job a previous lease-expiry already returned to 'pending' (attempts >=
  // 1) is only ever reclaimable by its OWN worker_id -- jobs aren't
  // stealable across workers. listExpiredLeases above only ever looks at
  // 'claimed' rows, so if that worker never comes back online (a genuine
  // crash, not a brief pm2/systemd restart blip), such a job would sit
  // 'pending' forever with nothing left to give it a second look -- its run
  // stuck 'running' forever too (MULTIUSER_PLAN.md §1.17's "kill -9 mid-run"
  // exit criterion). Swept here, gated on the worker's own last_heartbeat_at
  // (the same liveness definition every other "is this worker alive" check
  // in the app uses) rather than a fixed delay.
  const stuck = repo.queueRepo.listStuckPendingJobsForOfflineWorkers(Date.now() - OFFLINE_AFTER_MS);
  for (const job of stuck) {
    abandonJob(
      job,
      "worker went offline before reclaiming this job after its first lease expired",
      log,
      "stuck pending job abandoned -- its worker is offline"
    );
  }
}

// How long a terminal (completed/failed/cancelled) worker_jobs row is kept
// before pruning -- see repo.ts's pruneCompletedOlderThan doc comment.
export const TERMINAL_JOB_RETENTION_DAYS = 7;

// §1.7's own spec: "The same scheduler also runs (from Stage 3/4 onward):
// expired-enrolment pruning, terminal-job pruning (pruneCompletedOlderThan,
// 7 days), and expired-session pruning -- which v3.3 never had, so
// `sessions` would grow forever despite having an expires_at index." All
// three were missing entirely until this function existed -- sessions,
// worker_jobs, and abandoned-enrolment workers rows all grew unboundedly.
// Called from the same interval as reapExpiredLeases (index.ts); kept as a
// separate exported function (rather than folded into reapExpiredLeases
// itself) so existing tests that call reapExpiredLeases directly for its
// own job-lease behavior are unaffected by this addition.
// M6's own retention rule -- see repo.ts's pruneOldGpuClockSamples doc
// comment for what stays behind (everything except the raw sample series).
export const GPU_CLOCK_SAMPLE_RETENTION_DAYS = 30;

export function runMaintenanceSweep(log: ReaperLogger): void {
  reapExpiredLeases(log);
  const prunedSessions = repo.sessionRepo.pruneExpired();
  const prunedJobs = repo.queueRepo.pruneCompletedOlderThan(TERMINAL_JOB_RETENTION_DAYS);
  const prunedEnrolments = repo.workerRepo.pruneExpiredEnrolments();
  const prunedGpuClockSamples = repo.pruneOldGpuClockSamples(GPU_CLOCK_SAMPLE_RETENTION_DAYS);
  if (prunedSessions || prunedJobs || prunedEnrolments || prunedGpuClockSamples) {
    log.warn(
      { prunedSessions, prunedJobs, prunedEnrolments, prunedGpuClockSamples },
      "maintenance sweep pruned expired sessions/terminal jobs/stale enrolment codes/old GPU clock samples"
    );
  }
}
