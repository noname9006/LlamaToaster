import { useCallback, useEffect, useState } from "react";
import { api } from "./client";
import type { Worker } from "../types";

// Shared by the Dashboard (compact per-machine chips) and the Workers page
// (full cards) so both read the same machine list the same way. One
// GET /api/workers read carries everything -- hardware, installed builds,
// derived status, and (while a job is active) its run id / live progress --
// no more separate per-worker fetch (MULTIUSER_PLAN.md §1.16).
export function useWorkerStatuses() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    const list = await api.listWorkers();
    setWorkers(list);
    setLoaded(true);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await api.listWorkers();
      if (cancelled) return;
      setWorkers(list);
      setLoaded(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const order = workers.map((w) => w.id);
  const status = Object.fromEntries(workers.map((w) => [w.id, w]));

  return { order, status, workers, loaded, refresh };
}
