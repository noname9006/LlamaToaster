import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type WorkerListEntry } from "./client";
import type { WorkerLlamaCppInfo } from "../types";

export interface WorkerStatus {
  worker: WorkerListEntry;
  loading: boolean;
  info?: WorkerLlamaCppInfo;
  error?: string;
  inaccessible: boolean;
}

// Shared by the Dashboard (compact per-worker chips) and the Workers page
// (full cards) so both read the same reachability signal the same way,
// rather than each page reimplementing its own fetch-and-classify logic.
export function useWorkerStatuses() {
  const [status, setStatus] = useState<Record<string, WorkerStatus>>({});
  const [order, setOrder] = useState<string[]>([]);
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async (worker: WorkerListEntry) => {
    setStatus((s) => ({
      ...s,
      [worker.name]: { ...s[worker.name], worker, loading: true, inaccessible: false },
    }));
    try {
      const info = await api.getWorkerLlamaCpp(worker.name);
      setStatus((s) => ({ ...s, [worker.name]: { worker, loading: false, info, inaccessible: false } }));
    } catch (err) {
      const inaccessible = err instanceof ApiError && err.inaccessible;
      const message = err instanceof Error ? err.message : String(err);
      setStatus((s) => ({ ...s, [worker.name]: { worker, loading: false, error: message, inaccessible } }));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const workers = await api.listWorkers();
      if (cancelled) return;
      setOrder(workers.map((w) => w.name));
      setStatus(
        Object.fromEntries(workers.map((w) => [w.name, { worker: w, loading: true, inaccessible: false }]))
      );
      setLoaded(true);
      await Promise.all(workers.map((w) => refresh(w)));
    })();
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  return { order, status, loaded, refresh };
}
