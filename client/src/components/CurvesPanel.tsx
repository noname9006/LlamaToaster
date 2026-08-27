// BENCHMARKING_PLAN_V8.md N1 -- context curves as a first-class deliverable,
// and N5 -- the concurrency knee.
//
// A curve is NOT a new table: it is a deterministic grouping over `results`,
// computed on read. Choreographed server-path points stamp METHOD_VERSION 2,
// which is what keeps ordinary runtime rows' warm-biased TTFT out of curves.
//
// Accessibility: the concurrency chart pairs its bars with an sr-only summary
// sentence, and "Measure missing points" renders aria-disabled with its reason
// when no ladder cell is uncovered.

import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { CurveResponse, KneeResponse } from "../types";

function formatCtx(tokens: number): string {
  return tokens.toLocaleString();
}

export interface CurvesPanelProps {
  modelId: string;
  workerId?: string | null;
  build?: string | null;
  /** Drawn as a vertical marker on the curve. */
  targetCtx?: number | null;
  /** Enqueues a runtime-kind run for the uncovered ladder cells. */
  onMeasureMissing?: (contexts: number[]) => void;
}

export function CurvesPanel({ modelId, workerId, build, targetCtx, onMeasureMissing }: CurvesPanelProps) {
  const [curve, setCurve] = useState<CurveResponse | null>(null);
  const [engine, setEngine] = useState<"server" | "bench">("server");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setCurve(null);
    api
      .getCurve(modelId, { worker: workerId ?? undefined, build: build ?? undefined, engine })
      .then((res) => {
        if (!cancelled) {
          setCurve(res);
          setError("");
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [modelId, workerId, build, engine]);

  const uncovered = useMemo(
    () => (curve?.ladder ?? []).filter((cell) => cell.available && !cell.measured).map((cell) => cell.effectiveCtx),
    [curve]
  );

  if (error) return <p className="text-sm text-danger">Could not build a curve: {error}</p>;
  if (!curve) return <p className="text-sm text-muted">Loading curve…</p>;

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">
          Context curve — this model · this machine{build ? ` · ${build}` : ""} · {engine}
          {curve.method_version != null && (
            <span className="ml-2 rounded-full bg-accent/10 px-2 py-0.5 font-mono text-[10px] font-normal text-accent">
              method_version {curve.method_version}
            </span>
          )}
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex overflow-hidden rounded-lg border border-border" role="radiogroup" aria-label="Engine">
            {(["server", "bench"] as const).map((value) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={engine === value}
                onClick={() => setEngine(value)}
                className={
                  engine === value
                    ? "bg-accent px-3 py-1 text-[11px] font-semibold text-bg"
                    : "bg-raised px-3 py-1 text-[11px] text-muted"
                }
              >
                {value === "server" ? "llama-server" : "llama-bench"}
              </button>
            ))}
          </div>
          <button
            type="button"
            aria-disabled={uncovered.length === 0}
            title={
              uncovered.length === 0
                ? "Every ladder cell this placement can reach is already measured — nothing left to enqueue"
                : `Prices and enqueues ${uncovered.length} uncovered ladder cell(s) as ordinary runtime runs`
            }
            onClick={() => uncovered.length > 0 && onMeasureMissing?.(uncovered)}
            className={
              uncovered.length === 0
                ? "cursor-not-allowed rounded-lg border border-border px-3 py-1 text-[11px] font-semibold text-muted opacity-50"
                : "rounded-lg border border-border px-3 py-1 text-[11px] font-semibold text-fg hover:border-accent/40 hover:text-accent"
            }
          >
            Measure missing points
          </button>
        </div>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-[10px] uppercase tracking-wide text-muted">
              <th className="px-2 py-1.5">effective ctx</th>
              <th className="px-2 py-1.5 text-right">PP tok/s</th>
              <th className="px-2 py-1.5 text-right">TG tok/s</th>
              <th className="px-2 py-1.5 text-right">TTFT (cold)</th>
              <th className="px-2 py-1.5 text-right">VRAM peak</th>
              <th className="px-2 py-1.5">status</th>
            </tr>
          </thead>
          <tbody>
            {curve.points.map((point) => (
              <tr
                key={`${point.runId}:${point.idx}:${point.effectiveCtx}`}
                className={point.superseded || point.excluded ? "border-b border-border/40 opacity-60" : "border-b border-border/40"}
              >
                <td className="px-2 py-1.5 font-mono text-fg">{formatCtx(point.effectiveCtx)}</td>
                <td className="px-2 py-1.5 text-right font-mono text-muted">
                  {point.pp != null ? point.pp.toFixed(1) : "—"}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-muted">
                  {point.tg != null ? point.tg.toFixed(1) : "—"}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-muted">
                  {point.ttftMs != null ? `${(point.ttftMs / 1000).toFixed(1)} s` : "—"}
                  {point.ttftN === 1 && (
                    <span className="ml-1 text-[9px] uppercase tracking-wide" title="single-shot cold reading, deliberately not a p50/p95">
                      1-shot
                    </span>
                  )}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-muted">
                  {point.vramPeakMib != null ? `${point.vramPeakMib} MiB` : "—"}
                </td>
                <td className="px-2 py-1.5 text-muted">
                  {point.excluded ? (
                    <span title={point.excludedReason ?? undefined} className="rounded-full bg-warning-bg px-2 py-0.5 text-[10px] font-bold text-warning">
                      dropped — {point.caveatFlags.join(", ")}
                    </span>
                  ) : point.superseded ? (
                    <span className="rounded-full border border-dashed border-border px-2 py-0.5 text-[10px]">
                      superseded — re-measured later
                    </span>
                  ) : (
                    <>
                      <span className="mr-1 inline-block h-2 w-2 rounded-full bg-success align-middle" /> measured
                      {targetCtx != null && point.effectiveCtx === targetCtx && (
                        <span className="ml-2 rounded-full bg-raised px-2 py-0.5 text-[10px]">your target ▸</span>
                      )}
                    </>
                  )}
                </td>
              </tr>
            ))}
            {curve.ladder
              .filter((cell) => !cell.available)
              .map((cell) => (
                <tr key={`unavailable-${cell.effectiveCtx}`} className="border-b border-border/40 opacity-60">
                  <td className="px-2 py-1.5 font-mono text-muted">{formatCtx(cell.effectiveCtx)}</td>
                  <td className="px-2 py-1.5 text-right text-muted">—</td>
                  <td className="px-2 py-1.5 text-right text-muted">—</td>
                  <td className="px-2 py-1.5 text-right text-muted">—</td>
                  <td className="px-2 py-1.5 text-right text-muted">—</td>
                  <td className="px-2 py-1.5 text-muted">
                    <span className="rounded-full border border-dashed border-border px-2 py-0.5 text-[10px]">
                      {cell.unavailableReason}
                    </span>
                  </td>
                </tr>
              ))}
            {curve.points.length === 0 && (
              <tr>
                <td colSpan={6} className="px-2 py-3 text-muted">
                  No measured points for this pairing yet. “Measure missing points” prices the uncovered ladder
                  cells and enqueues them as ordinary runtime runs — nothing auto-runs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-3 rounded-lg border border-accent/25 bg-accent/[0.06] px-3 py-2 text-[11px] leading-relaxed text-muted">
        Server-path points measure prefill cost once per point with a timed streamed request (that reading{" "}
        <b className="text-fg">is</b> the TTFT column), then run the remaining repeats against the warm cache for
        clean generation numbers — the two clocks never average together. If the server evicts mid-run the row flags{" "}
        <span className="font-mono">cache_evicted</span> and drops out of the curve; if a context shift appears in
        the logs and the binary lacks <span className="font-mono">--no-context-shift</span>, the row flags too. No
        new table: a curve is a deterministic grouping over <span className="font-mono">results</span>.
      </p>
    </div>
  );
}

// N5 -- the knee: the smallest slot count whose TTFT p95 exceeds 2× its
// slots=1 value. Derived on read, never a stored verdict.
export function KneeChart({ runId }: { runId: string }) {
  const [knee, setKnee] = useState<KneeResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getKnee(runId)
      .then((res) => {
        if (!cancelled) setKnee(res);
      })
      .catch(() => {
        /* advisory */
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (!knee || knee.samples.length === 0) return null;
  const max = Math.max(...knee.samples.map((s) => s.ttftP95Ms ?? 0), knee.thresholdMs ?? 0, 1);

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">
          Concurrency — where batching stops being free
        </span>
        <span className="font-mono text-[11px] text-muted">
          slots {knee.samples.map((s) => s.slots).join(" · ")}
        </span>
      </div>

      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <div>
          <div className="flex h-32 items-end gap-3" role="img" aria-label={knee.summary}>
            {knee.samples.map((sample) => {
              const past = knee.knee != null && sample.slots >= knee.knee;
              const height = sample.ttftP95Ms != null ? Math.max(4, (sample.ttftP95Ms / max) * 100) : 4;
              return (
                <div key={sample.slots} className="flex flex-1 flex-col items-center justify-end">
                  <div
                    style={{ height: `${height}%` }}
                    className={
                      past
                        ? "flex w-full justify-center rounded-t-md bg-danger/20 pt-1 text-[10px] text-danger outline outline-1 outline-danger/40"
                        : "flex w-full justify-center rounded-t-md bg-raised pt-1 text-[10px] text-muted outline outline-1 outline-border"
                    }
                  >
                    {sample.ttftP95Ms != null ? `${Math.round(sample.ttftP95Ms / 1000)} s` : "—"}
                  </div>
                  <span className={past ? "mt-1 text-[10px] font-bold text-danger" : "mt-1 text-[10px] text-muted"}>
                    {sample.slots}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-1 text-[10px] text-muted">TTFT p95 · slots</div>
          {/* Paired with the bars so the chart is never colour-only. */}
          <p className="sr-only">{knee.summary}</p>
        </div>

        <div className="text-[12px] leading-relaxed text-muted">
          <b className="text-fg">{knee.summary}</b>
          {knee.thresholdMs != null && (
            <>
              {" "}
              The threshold is 2× the slots=1 reading ({Math.round(knee.thresholdMs / 1000)} s).
            </>
          )}{" "}
          Aggregate throughput{" "}
          {knee.samples
            .map((s) => (s.aggregateTps != null ? `${s.aggregateTps.toFixed(0)}` : "—"))
            .join(" → ")}{" "}
          tok/s — it can keep rising past the knee while per-user latency has already collapsed, which is exactly the
          trade this chart exists to show. Stored as ordinary rows with{" "}
          <span className="font-mono">concurrency</span> set; the knee is derived on read, not a stored verdict.
        </div>
      </div>
    </div>
  );
}
