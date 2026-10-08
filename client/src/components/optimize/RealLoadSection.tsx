import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import { Btn, Kicker, Notice, fmtDuration } from "./ui";
import type { FlowContext } from "./flowContext";
import { templateSweep } from "./flowData";
import { RunProgress } from "./SpeedSections";
import { isActive, type FlowRun } from "./useOptimizeFlow";
import type { ProbeAttemptDto, ProbeTriggerSpec } from "../../types";
import { toLadderAttempts } from "../ProbeFrontier";
import { probeOutcome, type ProbeOutcome } from "../../../../shared/probeLadder.js";
import { describePlacement, fromFit, ladderIndex, placementLadder, toArgs, type PlacementState } from "../../../../shared/moePlacement.js";
import { candidateFromPoint, ctxLabel, fits, type FitSeries } from "../../../../shared/optimizeFlow.js";
import type { SpeedCandidate } from "../../../../shared/speedRun.js";

// Plan step 2 (optional, discrete GPUs only): real loads that confirm fit's
// prediction -- the most layers that keep every byte in VRAM (no spill) and
// the most whose claim stays below free VRAM. The existing context-test
// search, seeded at fit's answer per context and walking fit's own placement
// order (experts back front-first on MoE).

function seedFor(series: FitSeries, ctx: number, ladder: PlacementState[] | null, moe: FlowContext["moe"]): number | null {
  const p = series.points.find((x) => x.ctx === ctx);
  if (!p || !fits(p)) return null;
  if (ladder && moe?.isMoe) {
    const st = fromFit({ ngl: p.ngl, ot: p.ot }, moe);
    const i = st ? ladderIndex(ladder, st) : -1;
    return i >= 0 ? i : null;
  }
  if (p.ngl == null) return null;
  return p.ngl < 0 ? p.layers_total ?? null : p.ngl;
}

export function RealLoadSection({
  ctx,
  series,
  run,
  onAddCandidate,
}: {
  ctx: FlowContext;
  series: FitSeries[];
  run: FlowRun | null;
  onAddCandidate: (c: SpeedCandidate) => void;
}) {
  const [seriesId, setSeriesId] = useState<string>(series[0]?.id ?? "f16-on");
  const setup = series.find((s) => s.id === seriesId) ?? series[0] ?? null;
  const fittingStops = setup ? setup.points.filter(fits).map((p) => p.ctx) : [];
  const target = fittingStops.find((c) => c >= ctx.targetCtx) ?? null;
  const maxFit = fittingStops.length ? Math.max(...fittingStops) : null;
  const [picked, setPicked] = useState<Set<number>>(new Set());
  useEffect(() => {
    setPicked(new Set([target, maxFit].filter((x): x is number => x != null)));
  }, [seriesId, target, maxFit]);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const active = !!run && isActive(run.test);
  const ladder = useMemo(() => (ctx.moe?.isMoe ? placementLadder(ctx.moe) : null), [ctx.moe]);

  async function start() {
    if (!setup) return;
    setError(null);
    setStarting(true);
    try {
      const stops = [...picked].sort((a, b) => a - b);
      const seed: Record<string, number> = {};
      for (const c of stops) {
        const s = seedFor(setup, c, ladder, ctx.moe);
        if (s != null) seed[String(c)] = s;
      }
      const first = seed[String(stops[0])] ?? 0;
      const probe: ProbeTriggerSpec = {
        candidate_ctx: stops[0],
        placement: { ngl: ladder ? Math.min(first, 4096) : first, slots: 1 },
        kv_pair: [setup.ctk, setup.ctv],
        mode: "frontier",
        ctx_stops: stops,
        seed,
        fa: setup.fa,
        ...(ladder ? { ladder } : {}),
      };
      await ctx.trigger("real_load", {
        model_id: ctx.model.id,
        worker_id: ctx.workerId,
        main_gpu: ctx.mainGpu,
        kind: "probe",
        sweep: templateSweep(1),
        probe,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  if (ctx.machine !== "dgpu" || series.length === 0) return null;
  return (
    <div className="mt-5 border-t border-border pt-4">
      <Kicker className="mb-1">Optional · real-load check</Kicker>
      <p className="text-sm text-fg-2">
        Fit predicts from the driver&apos;s free-memory figure, which ignores other apps. Real loads find the most layers that keep every byte in
        VRAM (no spill) and the most whose claim stays below free VRAM — sometimes 1–2 layers more, sometimes fewer. Several model loads per
        context, so this takes longer.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          Setup
          <select
            value={seriesId}
            onChange={(e) => setSeriesId(e.target.value)}
            disabled={active}
            className="h-10 border border-border-strong bg-well px-2 text-sm text-fg"
          >
            {series.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Contexts to confirm">
          {setup?.points.map((p) => {
            const ok = fits(p);
            const on = picked.has(p.ctx);
            return (
              <button
                key={p.ctx}
                type="button"
                aria-pressed={on}
                disabled={!ok || active}
                onClick={() => {
                  const next = new Set(picked);
                  if (on) next.delete(p.ctx);
                  else next.add(p.ctx);
                  setPicked(next);
                }}
                className={`min-h-9 border px-2.5 font-mono text-xs disabled:opacity-40 ${on ? "border-accent bg-accent-tint text-accent" : "border-border-strong text-fg-2"}`}
              >
                {ctxLabel(p.ctx)}
                {p.ctx === target ? " · target" : ""}
              </button>
            );
          })}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Btn onClick={start} disabled={active || starting || picked.size === 0 || ctx.machineOffline}>
          {active ? "Loading…" : `Confirm ${picked.size} context${picked.size === 1 ? "" : "s"} with real loads`}
        </Btn>
        <span className="font-mono text-xs text-muted">{fmtDuration(picked.size * 6 * 40 + 120)} · up to 40 loads</span>
      </div>
      {error && <div className="mt-3"><Notice tone="fail">{error}</Notice></div>}
      {run && active && <div className="mt-3"><RunProgress run={run} /></div>}
      {run && !active && (run.test.status === "failed" || run.test.status === "cancelled") && (
        <div className="mt-3">
          <Notice tone="fail">{realLoadFailure(run)}</Notice>
        </div>
      )}
      {run && <RealLoadResult ctx={ctx} run={run} active={active} onAddCandidate={onAddCandidate} series={series} />}
    </div>
  );
}

/** Why a real-load run stopped, in the user's terms. */
export function realLoadFailure(run: FlowRun): string {
  if (run.test.status === "cancelled") return "The real-load check was cancelled.";
  const raw = run.items.find((i) => i.error)?.error ?? run.test.error ?? "";
  // Measured placements are only taken from enrolled machines (they feed
  // verified results); a machine running on the deployment's shared secret
  // does the loads, then has its answer refused.
  if (/enrolled worker session/.test(raw)) {
    return "The loads ran, but this machine connects with the shared deployment secret, and real-load results are only accepted from enrolled machines. Enrol it (Machines → Add a machine) and run the check again.";
  }
  return raw ? `The real-load check failed: ${raw}` : "The real-load check did not finish.";
}

function RealLoadResult({
  ctx,
  run,
  active,
  onAddCandidate,
  series,
}: {
  ctx: FlowContext;
  run: FlowRun;
  active: boolean;
  onAddCandidate: (c: SpeedCandidate) => void;
  series: FitSeries[];
}) {
  const [rows, setRows] = useState<ProbeAttemptDto[]>([]);
  const spec = (run.test.config as { probe?: ProbeTriggerSpec }).probe ?? null;
  useEffect(() => {
    let cancelled = false;
    api
      .getProbeAttempts(run.test.id)
      .then((r) => !cancelled && setRows(r.attempts))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [run.test.id, run.items[0]?.detail, run.test.status]);

  const outcome: ProbeOutcome | null = useMemo(() => {
    const history = toLadderAttempts(rows);
    if (!spec || history.length === 0) return null;
    return probeOutcome({
      anchored: true,
      mode: "frontier",
      candidateCtx: spec.candidate_ctx,
      candidateNgl: spec.placement.ngl,
      nglMax: rows.find((r) => r.ladder_ngl_max != null)?.ladder_ngl_max ?? Math.max(...history.map((h) => h.ngl)),
      maxCtx: rows.find((r) => r.ladder_max_ctx != null)?.ladder_max_ctx ?? Math.max(...history.map((h) => h.ctx)),
      history,
      freeVramMib: rows.find((r) => r.list_devices_free_mib != null)?.list_devices_free_mib ?? null,
      stops: spec.ctx_stops,
    });
  }, [rows, spec]);

  if (!spec) return null;
  const ladder = spec.ladder ?? null;
  const describe = (index: number | null) => {
    if (index == null) return "none";
    if (ladder && ctx.moe) return describePlacement(ladder[index], ctx.moe);
    return `${index} of ${ctx.layersTotal ?? "?"} layers on GPU`;
  };
  const argsOf = (index: number) => (ladder && ctx.moe ? toArgs(ladder[index], ctx.moe) : { ngl: index, ot: null });
  const setupSeries = series.find((s) => s.ctk === spec.kv_pair[0] && s.ctv === spec.kv_pair[1] && s.fa === (spec.fa ?? "on"));

  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full border-collapse border border-border text-left">
        <thead>
          <tr className="border-b border-border-strong font-mono text-xs text-muted">
            <th scope="col" className="px-3 py-2 font-normal">Context</th>
            <th scope="col" className="px-2 py-2 font-normal">No spill (measured)</th>
            <th scope="col" className="px-2 py-2 font-normal">Claim below free VRAM</th>
            <th scope="col" className="px-2 py-2 font-normal">Fit predicted</th>
            <th scope="col" className="px-3 py-2 font-normal" />
          </tr>
        </thead>
        <tbody>
          {(outcome?.curve ?? spec.ctx_stops?.map((c) => ({ ctx: c, fit: null, clean: { value: null, resolved: false, control: "none" } })) ?? []).map((stop) => {
            const seed = spec.seed?.[String(stop.ctx)];
            const clean = stop.clean.resolved ? stop.clean.value : null;
            const point = setupSeries?.points.find((p) => p.ctx === stop.ctx);
            return (
              <tr key={stop.ctx} className="border-b border-border text-sm">
                <td className="px-3 py-2 font-mono text-xs">{ctxLabel(stop.ctx)}</td>
                <td className="px-2 py-2">{stop.clean.resolved ? describe(clean) : active ? "searching…" : "not settled"}</td>
                <td className="px-2 py-2">{stop.fit ? (stop.fit.resolved ? describe(stop.fit.value) : "—") : "no free-VRAM reading"}</td>
                <td className="px-2 py-2 text-fg-2">{seed != null ? describe(seed) : "—"}</td>
                <td className="px-3 py-2 text-right">
                  {clean != null && point && stop.ctx >= ctx.targetCtx && (
                    <Btn
                      kind="ghost"
                      onClick={() => {
                        const a = argsOf(clean);
                        onAddCandidate({
                          ...candidateFromPoint(point, `No spill · ${setupSeries!.label}`, "probe", ctx.threads),
                          ngl: a.ngl,
                          ot: a.ot,
                        });
                      }}
                    >
                      Add to speed tests
                    </Btn>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {outcome?.curve?.some((s) => s.clean.resolved && s.clean.value != null && spec.seed?.[String(s.ctx)] != null && s.clean.value < spec.seed[String(s.ctx)]) && (
        <div className="mt-3">
          <Notice tone="warn">Real loads kept fewer layers than fit predicted — the 1536 MiB margin is the safer choice on this machine.</Notice>
        </div>
      )}
    </div>
  );
}
