import { fmtGb, fmtTps } from "./ui";
import { isFailed, spilledMib, type SpeedRunView } from "./flowData";
import type { SpeedCandidate } from "../../../../shared/speedRun.js";

// Small charts for the flow's steps. Plain HTML bars: one measure per chart
// (never two scales on one axis), values as text beside every bar, colour
// only for identity (series-a = dedicated/kept, series-b = shared/other).

/** Where a speed test's GPU memory lived: dedicated VRAM vs the driver's
 * shared (system RAM) pool. */
export function MemorySplitBar({ dedicatedMib, sharedMib }: { dedicatedMib: number | null; sharedMib: number | null }) {
  if (dedicatedMib == null) return null;
  const shared = sharedMib ?? 0;
  const total = dedicatedMib + shared;
  const label = `Dedicated GPU memory ${fmtGb(dedicatedMib)}${shared ? `, shared GPU memory ${fmtGb(shared)}` : ""}`;
  return (
    <figure className="m-0 flex flex-col gap-2">
      <figcaption className="flex flex-wrap gap-x-5 gap-y-1 text-[13px] text-fg-2">
        <span className="inline-flex items-center gap-2">
          <span aria-hidden="true" className="h-3 w-3 bg-series-a" />
          Dedicated GPU memory · {fmtGb(dedicatedMib)}
        </span>
        {shared > 0 && (
          <span className="inline-flex items-center gap-2">
            <span aria-hidden="true" className="h-3 w-3 bg-series-b" />
            Shared GPU memory (system RAM through the driver) · {fmtGb(shared)}
          </span>
        )}
      </figcaption>
      <div role="img" aria-label={label} className="flex h-7 gap-0.5">
        <div className="bg-series-a" style={{ flex: `${dedicatedMib} 1 0` }} title={`Dedicated: ${fmtGb(dedicatedMib)}`} />
        {shared > 0 && <div className="bg-series-b" style={{ flex: `${shared} 1 0` }} title={`Shared: ${fmtGb(shared)}`} />}
      </div>
      {total > 0 && shared > 0 && (
        <p className="m-0 text-[13px] text-muted">{Math.round((shared / total) * 100)}% of the model&apos;s GPU memory was paged into system RAM.</p>
      )}
    </figure>
  );
}

interface Bar {
  key: string;
  label: string;
  value: number | null;
  text: string;
  highlight: boolean;
  note?: string;
}

function Bars({ title, bars }: { title: string; bars: Bar[] }) {
  const max = Math.max(1, ...bars.map((b) => b.value ?? 0));
  return (
    <figure className="m-0 flex min-w-0 flex-col gap-3">
      <figcaption className="text-sm text-fg-2">{title}</figcaption>
      {bars.map((b) => (
        <div key={b.key} className="flex flex-col gap-1">
          <div className="flex justify-between gap-3 text-[13px]">
            <span className="min-w-0 truncate text-fg-2" title={b.label}>
              {b.label}
              {b.note && <span className="text-muted"> · {b.note}</span>}
            </span>
            <span className="font-mono text-fg">{b.text}</span>
          </div>
          <div className="h-2.5 bg-surface-raised" title={`${b.label}: ${b.text}`}>
            <div className={`h-2.5 ${b.highlight ? "bg-series-a" : "bg-border-strong"}`} style={{ width: `${b.value ? Math.max(1, (b.value / max) * 100) : 0}%` }} />
          </div>
        </div>
      ))}
    </figure>
  );
}

/** Generation and prompt speed per config of a speed run, side by side as
 * two charts (different scales), the chosen config highlighted. */
export function SpeedBars({ view, chosen, labelOf }: { view: SpeedRunView; chosen: SpeedCandidate | null; labelOf?: (c: SpeedCandidate) => string }) {
  const done = view.rows.filter((r) => r.outcome.status === "done" && (r.outcome.tg || r.outcome.pp));
  if (done.length < 2) return null;
  const isChosen = (c: SpeedCandidate) => !!chosen && c === chosen;
  const mk = (pick: "tg" | "pp"): Bar[] =>
    view.rows.map(({ candidate, outcome }) => {
      const r = outcome[pick];
      const failed = isFailed(outcome.status);
      return {
        key: `${outcome.idx}`,
        label: labelOf ? labelOf(candidate) : candidate.label,
        value: r?.mean ?? null,
        text: failed ? "failed" : r ? fmtTps(r.mean) : "—",
        highlight: isChosen(candidate),
        note: spilledMib(outcome) != null ? `${fmtGb(spilledMib(outcome))} spilled` : undefined,
      };
    });
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,280px),1fr))] gap-x-10 gap-y-6">
      <Bars title="Generation, tok/s" bars={mk("tg")} />
      <Bars title="Prompt, tok/s" bars={mk("pp")} />
    </div>
  );
}

/** KV cache size per type at one context. */
export function KvSizeBars({ rows, ctxLabel }: { rows: { key: string; label: string; mib: number | null; note?: string }[]; ctxLabel: string }) {
  return (
    <Bars
      title={`Cache size at ${ctxLabel}`}
      bars={rows.map((r) => ({ key: r.key, label: r.label, value: r.mib, text: fmtGb(r.mib, 2), highlight: true, note: r.note }))}
    />
  );
}
