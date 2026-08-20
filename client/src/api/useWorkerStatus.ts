import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./client";
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
    async function poll() {
      try {
        const list = await api.listWorkers();
        if (cancelled) return;
        setWorkers(list);
        setLoaded(true);
      } finally {
        if (!cancelled) timerRef.current = window.setTimeout(poll, POLL_MS);
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
