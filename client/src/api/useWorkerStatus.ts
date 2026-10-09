import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./client";
import type { Worker } from "../types";

// Shared by the Dashboard (compact per-machine chips) and the Workers page
// (full cards) so both read the same machine list the same way. One
// GET /api/workers read carries everything -- hardware, installed builds,
// derived status, and (while a job is active) its run id / live progress --
// no more separate per-worker fetch (MULTIUSER_PLAN.md §1.16).
//
// Polls on a self-rescheduling timer (same shape as Runs.tsx/RunDetail.tsx)
// rather than fetching once on mount -- a worker's install/activate actions
// (WorkerCard.tsx) just enqueue a job and return immediately, so the one
// refetch that used to follow a POST fired while the job was still
// queued/running and nothing ever polled again, leaving the page stuck
// showing "Queued: activate ..." long after the build actually finished.
// Server-side data is already fresh within one heartbeat (~10s); this just
// needs to keep asking for it.
const POLL_MS = 5000;
// A 401 means no session (signed out, or it expired). Polling every 5s would
// just repeat the rejection, so back off doubling each time up to this cap.
// Still retries, so a fresh sign-in in another tab is picked up without a
// reload.
const UNAUTHORIZED_MAX_MS = 60_000;

export function useWorkerStatuses() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loaded, setLoaded] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);

  const refresh = useCallback(async () => {
    const list = await api.listWorkers();
    setWorkers(list);
    setLoaded(true);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unauthorizedStreak = 0;
    async function poll() {
      let delay = POLL_MS;
      try {
        const list = await api.listWorkers();
        unauthorizedStreak = 0;
        if (cancelled) return;
        setWorkers(list);
        setLoaded(true);
      } catch (err) {
        // Transient (server restart, brief network drop) -- the finally below
        // reschedules regardless, so the loop self-heals on the next tick and
        // the page keeps showing its last good data rather than blanking.
        if (err instanceof ApiError && err.status === 401) {
          unauthorizedStreak += 1;
          delay = Math.min(POLL_MS * 2 ** unauthorizedStreak, UNAUTHORIZED_MAX_MS);
        }
        // Swallowed rather than allowed to propagate: poll() is invoked
        // un-awaited (`void poll()` below, and again from its own setTimeout),
        // so an escaping rejection has nowhere to go and surfaces as an
        // "Uncaught (in promise)" on EVERY failed poll -- once per interval,
        // for as long as the server is unreachable.
      } finally {
        if (!cancelled) timerRef.current = window.setTimeout(poll, delay);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  const order = workers.map((w) => w.id);
  const status = Object.fromEntries(workers.map((w) => [w.id, w]));

  return { order, status, workers, loaded, refresh };
}
