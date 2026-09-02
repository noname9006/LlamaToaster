// BENCHMARKING_PLAN_V8.md M6 (telemetry) + N6 (policy). M6 MEASURES; N6
// DECIDES -- and both policies are offered, never automatic: trimming repeats
// outside the stored configuration would make results non-reproducible.
//
// A thermally throttled row is KEPT, not failed. Burst-versus-sustained is
// exactly the distinction these columns exist to make visible.

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SustainedResponse } from "../types";

export function SustainedState({ testId, refreshKey }: { testId: string; refreshKey?: unknown }) {
  const [data, setData] = useState<SustainedResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getSustained(testId)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        /* advisory panel: a failed read simply renders nothing */
      });
    return () => {
      cancelled = true;
    };
  }, [testId, refreshKey]);

  if (!data) return null;
  const flagged = data.flagged_items.length;

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div className="rounded-xl border border-border bg-surface p-3.5">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">Steady-state option</span>
        <p className="mt-2 text-xs leading-relaxed text-muted">
          Discard the first{" "}
          <b className="text-fg">{Math.max(1, data.steady_state.discard_first_repeats)}</b> repeat on thermally
          volatile machines — implemented as a scoring-side exclusion recorded in this run’s configuration, never
          post-hoc trimming outside it.
          {data.steady_state.available ? (
            <> Available for this run.</>
          ) : (
            <>
              {" "}
              <b className="text-warning">Unavailable here:</b> {data.steady_state.reason}
            </>
          )}
        </p>
      </div>

      <div className="rounded-xl border border-border bg-surface p-3.5">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">Throttle-aware follow-up</span>
        <p className="mt-2 text-xs leading-relaxed text-muted">
          {flagged} of {data.denominator} item{data.denominator === 1 ? "" : "s"} flagged{" "}
          <span className="font-mono">thermally_throttled</span> ({Math.round(data.ratio * 100)} %). Flagged rows are{" "}
          <b className="text-fg">kept, not failed</b> — a throttled run is information about sustained performance.
        </p>
        {data.offer_rerun ? (
          <>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              More than a third of this run flagged, so a re-run of just those items after cooldown is offered
              below. Never scheduled automatically.
            </p>
            <button
              type="button"
              disabled
              title="Re-triggering the flagged items is done from the Benchmark page with the same grid"
              className="mt-2 rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg opacity-60"
            >
              Re-run flagged items · {data.rerun_estimate}
            </button>
            <p className="mt-1 text-[11px] text-muted">Offered with a price estimate — never scheduled automatically.</p>
          </>
        ) : (
          <p className="mt-2 text-[11px] text-muted">
            Below the one-third threshold, so no re-run is offered. Nothing here changes the stored rows either way.
          </p>
        )}
      </div>
    </div>
  );
}
