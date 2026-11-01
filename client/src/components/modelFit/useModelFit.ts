import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { ModelFitResult, ModelFitTarget } from "../../types";

// "Will it fit?" answers per machine (docs/plans/MODEL_FIT_CHECK_PLAN.md): read
// once when a machine appears, polled only while something is being checked.
// Answers arrive one by one, so a whole group fills in as it goes.

const POLL_MS = 3000;
const MAX_TARGETS_PER_REQUEST = 64;

export type FitMarkState =
  | { kind: "unknown" }
  | { kind: "pending" }
  /** Asked for, but the job ended without an answer (worker restarted, say). */
  | { kind: "lost" }
  | { kind: "answered"; result: ModelFitResult };

interface MachineFit {
  results: Map<string, ModelFitResult>;
  pending: Set<string>;
  /** Key -> the answer's created_at when asked, to tell a new answer from an old one. */
  asked: Map<string, number | null>;
  lost: Set<string>;
  installing: string | null;
}

const emptyMachine = (): MachineFit => ({ results: new Map(), pending: new Set(), asked: new Map(), lost: new Set(), installing: null });

export function useModelFit(workerIds: string[]) {
  const [machines, setMachines] = useState<Record<string, MachineFit>>({});
  const machinesRef = useRef(machines);
  machinesRef.current = machines;
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const mounted = useRef(true);

  const refresh = useCallback(async (workerId: string): Promise<void> => {
    clearTimeout(timers.current[workerId]);
    let res;
    try {
      res = await api.listModelFit(workerId);
    } catch {
      // Network hiccup or an older server: what is on screen stands, and the
      // next tick retries if something was being checked.
      if (mounted.current && machineBusy(machinesRef.current[workerId])) {
        timers.current[workerId] = setTimeout(() => void refresh(workerId), POLL_MS);
      }
      return;
    }
    if (!mounted.current) return;
    const prev = machinesRef.current[workerId] ?? emptyMachine();
    const results = new Map(res.results.map((r) => [r.target_key, r]));
    const pending = new Set(res.pending_keys);
    const asked = new Map(prev.asked);
    const lost = new Set(prev.lost);
    for (const [key, createdAt] of asked) {
      const answer = results.get(key);
      if (answer && answer.created_at !== createdAt) {
        asked.delete(key);
        lost.delete(key);
      } else if (!pending.has(key)) {
        // Its job is gone and no new answer came with it.
        asked.delete(key);
        lost.add(key);
      }
    }
    const next: MachineFit = { results, pending, asked, lost, installing: res.installing };
    machinesRef.current = { ...machinesRef.current, [workerId]: next };
    setMachines(machinesRef.current);
    if (machineBusy(next)) timers.current[workerId] = setTimeout(() => void refresh(workerId), POLL_MS);
  }, []);

  const idsKey = workerIds.join("|");
  useEffect(() => {
    mounted.current = true;
    for (const id of idsKey ? idsKey.split("|") : []) {
      const known = machinesRef.current[id];
      if (!known || machineBusy(known)) void refresh(id);
    }
    return () => {
      mounted.current = false;
      for (const t of Object.values(timers.current)) clearTimeout(t);
    };
  }, [idsKey, refresh]);

  const stateOf = useCallback(
    (workerId: string | undefined, key: string | null): FitMarkState => {
      const m = workerId ? machines[workerId] : undefined;
      if (!m || !key) return { kind: "unknown" };
      if (m.pending.has(key) || m.asked.has(key)) return { kind: "pending" };
      if (m.lost.has(key)) return { kind: "lost" };
      const result = m.results.get(key);
      return result ? { kind: "answered", result } : { kind: "unknown" };
    },
    [machines]
  );

  const request = useCallback(
    async (workerId: string, targets: { key: string; target: ModelFitTarget }[], opts: { autoInstall: boolean; force: boolean }) => {
      // A repo can list more quants than one request carries.
      for (let i = 0; i < targets.length; i += MAX_TARGETS_PER_REQUEST) {
        const res = await api.requestModelFit(workerId, {
          targets: targets.slice(i, i + MAX_TARGETS_PER_REQUEST).map((t) => t.target),
          auto_install: opts.autoInstall,
          force: opts.force,
        });
        const prev = machinesRef.current[workerId] ?? emptyMachine();
        const asked = new Map(prev.asked);
        const lost = new Set(prev.lost);
        // The server names what it queued: an answer still fresh, or a target
        // already in a job, must not read as lost when no new answer comes.
        for (const key of res.queued_keys ?? []) {
          asked.set(key, prev.results.get(key)?.created_at ?? null);
          lost.delete(key);
        }
        machinesRef.current = { ...machinesRef.current, [workerId]: { ...prev, asked, lost } };
        setMachines(machinesRef.current);
      }
      await refresh(workerId);
    },
    [refresh]
  );

  return { stateOf, request, installing: (workerId: string | undefined) => (workerId ? machines[workerId]?.installing ?? null : null) };
}

function machineBusy(m: MachineFit | undefined): boolean {
  return !!m && (m.pending.size > 0 || m.asked.size > 0 || m.installing != null);
}

export type ModelFitStore = ReturnType<typeof useModelFit>;
