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
  error?: (obj: Record<string, unknown>, msg: string) => void;
}

// Runs one unit of background bookkeeping, logging (never throwing) on
// failure: this runs from a bare setInterval, where a throw would crash the
// whole server, and one bad step or job must not starve the rest (audit
// 2026-09-27 M7).
function guarded<T>(log: ReaperLogger, what: Record<string, unknown>, fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch (err) {
    (log.error ?? log.warn)({ ...what, err }, "maintenance sweep step failed");
    return fallback;
  }
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
  for (const job of expired) guarded(log, { step: "reap expired lease", job_id: job.id }, () => reapOne(job), undefined);
  function reapOne(job: ReapableJob): void {
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
    guarded(
      log,
      { step: "abandon stuck pending job", job_id: job.id },
      () =>
        abandonJob(
          job,
          "worker went offline before reclaiming this job after its first lease expired",
          log,
          "stuck pending job abandoned -- its worker is offline"
        ),
      undefined
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

// A run whose job shows no progress for this long while its machine isn't
// paused is stuck: the worker is alive (heartbeats keep the lease) but
// nothing moves -- no changed progress report, no item tick, no item result.
// llama-bench ticks on every repetition, so this is far above any real gap;
// generous on purpose, for very long single items.
export const STALL_TIMEOUT_MS = 12 * 60 * 60 * 1000;
// A pause this old was almost certainly forgotten. Ending it stops the run
// (finished items and their results stay) and frees the machine and the
// user's active-run slot.
export const PAUSE_TIMEOUT_MS = 7 * 24 * 60 * 60 * 1000;

export function stopStalledRuns(log: ReaperLogger): void {
  const now = Date.now();
  for (const job of repo.queueRepo.listStalledRunJobs(now - STALL_TIMEOUT_MS)) {
    guarded(
      log,
      { step: "stop stalled run", job_id: job.id },
      () =>
        abandonJob(
          { ...job, attempts: 0, cancel_requested: 0 },
          "stopped: no progress for 12 hours",
          log,
          "stalled run stopped -- no progress while not paused"
        ),
      undefined
    );
  }
  for (const { worker_id, run_ids } of repo.queueRepo.listLongPausedWorkers(now - PAUSE_TIMEOUT_MS)) {
    guarded(
      log,
      { step: "end long pause", worker_id },
      () => {
        for (const runId of run_ids) {
          repo.queueRepo.cancelPendingJobsForTest(runId);
          const claimed = repo.queueRepo.getNonTerminalJobsForTest(runId).some((j) => j.status === "claimed");
          // A claimed job gets the stop on the worker's next heartbeat (its
          // pause loop exits on stop) and finishes the run itself.
          if (claimed) repo.queueRepo.requestCancelForTest(runId);
          else repo.reconcileStaleTest(undefined, runId, "stopped: paused for 7 days");
        }
        repo.workerRepo.setPauseRequested(worker_id, false);
        queueEvents.emit(worker_id);
        log.warn({ worker_id, run_ids }, "pause left for 7 days ended -- its runs were stopped");
      },
      undefined
    );
  }
}

// How long a run may sit running/scheduled after its last job ended before
// it's finalized as-is -- covers a worker whose last item reports were still
// retrying when it reported the job done.
export const ORPHANED_RUN_GRACE_MS = 10 * 60 * 1000;

// A run whose jobs have all ended but which never reached a final status (an
// item's terminal report was lost) -- finalized with what it has, the
// unreported items marked cancelled (audit 2026-10-18 A2).
export function finalizeOrphanedRuns(log: ReaperLogger): void {
  for (const runId of repo.listOrphanedActiveTests(ORPHANED_RUN_GRACE_MS)) {
    guarded(
      log,
      { step: "finalize orphaned run", run_id: runId },
      () => {
        repo.reconcileStaleTest(undefined, runId, "the machine finished without reporting every item");
        log.warn({ run_id: runId }, "run finalized -- its jobs had ended but some items never reported");
      },
      undefined
    );
  }
}

export function runMaintenanceSweep(log: ReaperLogger): void {
  const step = <T>(name: string, fn: () => T, fallback: T) => guarded(log, { step: name }, fn, fallback);
  step("reap expired leases", () => reapExpiredLeases(log), undefined);
  step("stop stalled and long-paused runs", () => stopStalledRuns(log), undefined);
  step("finalize orphaned runs", () => finalizeOrphanedRuns(log), undefined);
  const prunedSessions = step("prune sessions", () => repo.sessionRepo.pruneExpired(), 0);
  const prunedJobs = step("prune terminal jobs", () => repo.queueRepo.pruneCompletedOlderThan(TERMINAL_JOB_RETENTION_DAYS), 0);
  const prunedEnrolments =
    step("prune enrolments", () => repo.workerRepo.pruneExpiredEnrolments(), 0) +
    step("prune re-enrolments", () => repo.workerRepo.pruneExpiredReenrolments(), 0) +
    step("prune challenges", () => repo.workerRepo.pruneExpiredChallenges(), 0) +
    step("prune link intents", () => repo.sessionRepo.pruneExpiredLinkIntents(), 0);
  const prunedGpuClockSamples = step(
    "prune GPU clock samples",
    () => repo.pruneOldGpuClockSamples(GPU_CLOCK_SAMPLE_RETENTION_DAYS),
    0
  );
  step("prune catalog row budget", () => repo.pruneOldCatalogRowBudget(), undefined);
  if (prunedSessions || prunedJobs || prunedEnrolments || prunedGpuClockSamples) {
    log.warn(
      { prunedSessions, prunedJobs, prunedEnrolments, prunedGpuClockSamples },
      "maintenance sweep pruned expired sessions/terminal jobs/stale enrolment codes/old GPU clock samples"
    );
  }
}
