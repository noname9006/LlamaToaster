import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api/client";
import type { FlowStep, ResultRow, Test, TestItem, TriggerPayload } from "../../types";
import type { FitPoint } from "../../../../shared/fitParams.js";
import type { KvSupportRow } from "../../../../shared/kvSupport.js";
import type { FitMapSpec } from "../../../../shared/optimizeFlow.js";

// One optimization session = the runs of one (machine, model) tied together
// by a flow id stored on each run. The id lives in localStorage per pair and
// in the URL (?flow=), so the page restores from any device.

export interface FlowRun {
  test: Test;
  items: TestItem[];
  results: ResultRow[];
}

export interface FitData {
  testId: string;
  points: FitPoint[];
  kv: KvSupportRow[];
  spec: FitMapSpec | null;
}

const KEY_PREFIX = "llamatoaster.flow.";

function storageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}
function storageSet(key: string, value: string | null): void {
  try {
    if (value == null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* private window -- the URL still carries the flow */
  }
}

export function newFlowId(): string {
  const raw =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
  return raw.replace(/[^A-Za-z0-9]/g, "").slice(0, 20);
}

export function isActive(t: Test): boolean {
  return t.status === "running" || t.status === "scheduled";
}

export function stepOf(t: Test): FlowStep | null {
  return (t.config as { flow_step?: FlowStep }).flow_step ?? null;
}

export function useOptimizeFlow(workerId: string, modelId: string, urlFlowId: string | null) {
  const pairKey = workerId && modelId ? `${KEY_PREFIX}${workerId}.${modelId}` : null;
  const [flowId, setFlowIdState] = useState<string | null>(null);
  const [runs, setRuns] = useState<FlowRun[]>([]);
  const [fit, setFit] = useState<FitData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const tick = useRef(0);

  // Restore: URL wins (another device's link), then the pair's stored id.
  useEffect(() => {
    if (urlFlowId) {
      setFlowIdState(urlFlowId);
      return;
    }
    setFlowIdState(pairKey ? storageGet(pairKey) : null);
    setRuns([]);
    setFit(null);
  }, [pairKey, urlFlowId]);

  const setFlowId = useCallback(
    (id: string | null) => {
      setFlowIdState(id);
      if (pairKey) storageSet(pairKey, id);
    },
    [pairKey]
  );

  const load = useCallback(async () => {
    if (!flowId) return;
    const my = ++tick.current;
    try {
      const { tests } = await api.getFlow(flowId);
      const detailed = await Promise.all(
        tests.map(async (t) => {
          const d = await api.getTest(t.id);
          return { test: d.run, items: d.items, results: d.results } as FlowRun;
        })
      );
      if (my !== tick.current) return;
      setRuns(detailed);
      const fitRun = [...detailed].reverse().find((r) => r.test.kind === "fit");
      if (fitRun) {
        const f = await api.getFitPoints(fitRun.test.id);
        if (my !== tick.current) return;
        setFit({ testId: fitRun.test.id, ...f });
      } else {
        setFit(null);
      }
      setError(null);
    } catch (err) {
      if (my === tick.current) setError(err instanceof Error ? err.message : String(err));
    }
  }, [flowId]);

  // Fast while anything runs, slow otherwise; immediately on focus.
  const anyActive = runs.some((r) => isActive(r.test));
  useEffect(() => {
    if (!flowId) return;
    let cancelled = false;
    let timer: number | undefined;
    const loop = async () => {
      setLoading(true);
      await load();
      setLoading(false);
      if (!cancelled) timer = window.setTimeout(loop, anyActive ? 3000 : 20000);
    };
    void loop();
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        if (timer) window.clearTimeout(timer);
        void loop();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [flowId, load, anyActive]);

  const trigger = useCallback(
    async (step: FlowStep, payload: TriggerPayload): Promise<Test> => {
      const id = flowId ?? newFlowId();
      if (!flowId) setFlowId(id);
      const test = await api.triggerTest({ ...payload, flow_id: id, flow_step: step });
      // Show the new run right away; the poll fills in items/results.
      setRuns((prev) => [...prev, { test, items: [], results: [] }]);
      window.setTimeout(() => void load(), 400);
      return test;
    },
    [flowId, setFlowId, load]
  );

  const latest = useCallback(
    (step: FlowStep): FlowRun | null => [...runs].reverse().find((r) => stepOf(r.test) === step) ?? null,
    [runs]
  );

  const reset = useCallback(() => {
    setFlowId(null);
    setRuns([]);
    setFit(null);
  }, [setFlowId]);

  return useMemo(
    () => ({ flowId, runs, fit, loading, error, trigger, latest, reset, refresh: load, anyActive }),
    [flowId, runs, fit, loading, error, trigger, latest, reset, load, anyActive]
  );
}
