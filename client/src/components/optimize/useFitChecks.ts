import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { FitCheckResult, Model, Worker } from "../../types";
import { machineKind, type FitAnswer } from "./machine";

// docs/plans/HEADER_FIT_PLAN.md: the model list's verdicts from llama-fit-params
// on the machine itself. Asks for every listed model at its verdict context
// (the server skips answers already on file), then polls until each has one.
// Models without an answer yet -- or whose check failed -- keep the header
// estimate (machine.ts estimateFitAnswer).

const POLL_MS = 4000;

export function useFitChecks(
  worker: Worker | undefined,
  models: Model[],
  ctxFor: (m: Model) => number
): { answer: (modelId: string, ctx: number) => FitAnswer | null; pending: boolean } {
  const [results, setResults] = useState<Record<string, FitCheckResult>>({});
  const [pending, setPending] = useState(false);
  // Only a discrete GPU's verdict reads a fit answer (machine.ts modelVerdict).
  const kind = worker ? machineKind(worker) : null;
  const enabled =
    !!worker && !!kind?.hasGpu && !kind.unified && worker.status !== "offline" && (worker.capabilities ?? []).includes("fit-check-v1");
  const workerId = worker?.id;

  // ctx -> model ids, stable across renders that do not change the set.
  const groupsKey = models.map((m) => `${m.id}@${ctxFor(m)}`).join(",");
  const groups = useMemo(() => {
    const g = new Map<number, string[]>();
    for (const part of groupsKey ? groupsKey.split(",") : []) {
      const at = part.lastIndexOf("@");
      const ctx = Number(part.slice(at + 1));
      g.set(ctx, [...(g.get(ctx) ?? []), part.slice(0, at)]);
    }
    return g;
  }, [groupsKey]);

  useEffect(() => {
    setResults({});
    if (!enabled || !workerId || groups.size === 0) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const asked = new Set<number>();
    const tick = async () => {
      let anyPending = false;
      let anyMissing = false;
      for (const [ctx, ids] of groups) {
        try {
          const res = await api.fitChecks(workerId, ctx);
          if (cancelled) return;
          anyPending ||= res.pending;
          const got = new Map(res.results.map((r) => [r.model_id, r]));
          setResults((prev) => {
            const next = { ...prev };
            for (const id of ids) {
              const r = got.get(id);
              if (r) next[`${id}@${ctx}`] = r;
            }
            return next;
          });
          const missing = ids.filter((id) => !got.has(id));
          if (missing.length > 0) anyMissing = true;
          // Ask once per context while this view lives; the server refuses a
          // second job while one is queued, and keeps fresh answers.
          if (!asked.has(ctx) && !res.pending) {
            asked.add(ctx);
            const q = await api.requestFitChecks(workerId, { model_ids: ids, ctx }).catch(() => null);
            if (q && (q.queued > 0 || q.pending)) anyPending = true;
          }
        } catch {
          // Network hiccup or an older server: the estimates stand.
        }
      }
      if (cancelled) return;
      setPending(anyPending);
      const askedAll = [...groups.keys()].every((c) => asked.has(c));
      if (anyPending || (anyMissing && !askedAll)) timer = setTimeout(() => void tick(), POLL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [enabled, workerId, groups]);

  const answer = (modelId: string, ctx: number): FitAnswer | null => {
    const r = results[`${modelId}@${ctx}`];
    const p = r?.point;
    if (!p || p.layers_gpu == null || p.layers_total == null || p.dev_used_mib == null || p.host_used_mib == null) return null;
    return {
      layers_gpu: p.layers_gpu,
      layers_total: p.layers_total,
      overflow_layers: p.overflow_layers,
      dev_used_mib: p.dev_used_mib,
      host_used_mib: p.host_used_mib,
      source: "fit",
    };
  };
  return { answer, pending };
}
