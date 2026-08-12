import { useEffect, useState } from "react";
import type { RunStatus, RunItemStatus, RunItem } from "../types";
import { formatElapsed } from "../utils";

export type PillTone = "accent" | "success" | "danger" | "warning" | "muted";

const TONE_CLASSES: Record<PillTone, string> = {
  accent: "bg-accent/15 text-accent",
  success: "bg-success-bg text-success",
  danger: "bg-danger-bg text-danger",
  warning: "bg-warning-bg text-warning",
  muted: "bg-white/5 text-muted",
};

export function StatusPill({ label, tone }: { label: string; tone: PillTone }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium uppercase tracking-wide ${TONE_CLASSES[tone]}`}
    >
      {label}
    </span>
  );
}

const RUN_STATUS_TONE: Record<RunStatus, PillTone> = {
  running: "accent",
  scheduled: "muted",
  done: "success",
  partial: "warning",
  failed: "danger",
  cancelled: "muted",
};

export function RunStatusPill({ status }: { status: RunStatus }) {
  return <StatusPill label={status} tone={RUN_STATUS_TONE[status]} />;
}

const RUN_ITEM_STATUS_TONE: Record<RunItemStatus, PillTone> = {
  queued: "muted",
  loading: "accent",
  processing: "accent",
  generating: "accent",
  benchmarking: "accent",
  done: "success",
  failed: "danger",
  failed_oom: "danger",
  cancelled: "muted",
};

const RUN_ITEM_STATUS_LABEL: Record<RunItemStatus, string> = {
  queued: "queued",
  loading: "loading",
  processing: "processing",
  generating: "generating",
  benchmarking: "benchmarking",
  done: "done",
  failed: "failed",
  failed_oom: "failed (oom)",
  cancelled: "cancelled",
};

export function RunItemStatusPill({ status }: { status: RunItemStatus }) {
  return <StatusPill label={RUN_ITEM_STATUS_LABEL[status]} tone={RUN_ITEM_STATUS_TONE[status]} />;
}

// Human-readable phase text for a non-terminal item. Terminal items
// (done/failed/cancelled) aren't meaningfully described this way -- callers
// should show their error/result instead.
export function describeItemPhase(item: RunItem): string {
  switch (item.status) {
    case "queued":
      return "queued";
    case "loading":
      return item.detail === "warmup" ? "warming up" : item.detail || "loading model";
    case "processing":
      // detail here is always just the "run X/Y" repeat marker (see
      // worker/src/index.ts's runSweepItem) -- redundant now that every row
      // shows its own repeat-progress dot strip, so it's dropped rather
      // than appended.
      return "processing prompt";
    case "generating":
      return "generating tokens";
    case "benchmarking":
      return item.detail ? `benchmarking — ${item.detail}` : "benchmarking";
    default:
      return item.detail || "";
  }
}

type DotState = "grey" | "blink" | "solid" | "red" | "cancelled";

const RUN_ITEM_DOT_STATE: Record<RunItemStatus, DotState> = {
  queued: "grey",
  loading: "blink",
  processing: "blink",
  generating: "blink",
  benchmarking: "blink",
  done: "solid",
  failed: "red",
  failed_oom: "red",
  cancelled: "cancelled",
};

// Compact status indicator for a dense per-row table (the full RunItemStatusPill
// above is more legible standalone but too wide to repeat once per row of a
// full test table): grey = not started yet, slowly blinking green = the
// currently running test, solid green = done, red = failed/failed_oom (an
// error or a test that never finished).
export function StatusDot({ status }: { status: RunItemStatus }) {
  const state = RUN_ITEM_DOT_STATE[status];
  return (
    <span
      className={`status-dot status-dot--${state}`}
      title={RUN_ITEM_STATUS_LABEL[status]}
      aria-label={RUN_ITEM_STATUS_LABEL[status]}
    />
  );
}

// Status for a *batched* circle that may cover several run_items at once (used
// when a run has many combinations and the strip collapses them). Precedence
// is "running beats all": a running combo wins over done/failed, then any
// failure is red, then a mixed done+failed batch is yellow (static), then all
// done is green, and a queued-only batch is grey.
export type CircleTone = "grey" | "running" | "solid" | "red" | "warn" | "cancelled";

export function StatusCircle({ tone, title }: { tone: CircleTone; title?: string }) {
  const cls =
    tone === "running"
      ? "status-dot status-dot--running"
      : tone === "red"
        ? "status-dot status-dot--red"
        : tone === "warn"
          ? "status-dot status-dot--warn"
          : tone === "cancelled"
            ? "status-dot status-dot--cancelled"
            : tone === "grey"
              ? "status-dot status-dot--grey"
              : "status-dot status-dot--solid";
  return <span className={cls} title={title} aria-label={title} />;
}

const RUNNING_ITEM_STATUSES = new Set<RunItemStatus>(["loading", "processing", "generating", "benchmarking"]);

function comboIsRunning(status: RunItemStatus): boolean {
  return RUNNING_ITEM_STATUSES.has(status);
}
export function comboTone(status: RunItemStatus): CircleTone {
  if (comboIsRunning(status)) return "running";
  if (status === "failed" || status === "failed_oom") return "red";
  if (status === "done") return "solid";
  if (status === "cancelled") return "cancelled";
  return "grey";
}
function bucketToneOf(tones: CircleTone[]): CircleTone {
  if (tones.some((t) => t === "running")) return "running";
  if (tones.some((t) => t === "red")) {
    // mixed done + failed -> yellow (static); all failed -> red
    if (tones.some((t) => t === "solid")) return "warn";
    return "red";
  }
  if (tones.every((t) => t === "solid")) return "solid";
  return "grey";
}

export interface StatusCircleUnit {
  tone: CircleTone;
  label: string;
}

// Collapses many progress units into a compact circle strip. <=49 units get
// one circle each; larger counts bucket sequentially so the strip stays
// readable (x5 at 50-99 units, x10 at 100+). Each rendered circle's tone is
// derived from the units it covers via the "running beats all" precedence.
// What a "unit" represents is up to the caller -- see buildProgressUnits.
export function StatusCircleStrip({ units }: { units: StatusCircleUnit[] }) {
  const count = units.length;
  const size = count <= 49 ? 1 : count <= 99 ? 5 : 10;
  const buckets: StatusCircleUnit[][] = [];
  for (let i = 0; i < count; i += size) buckets.push(units.slice(i, i + size));
  return (
    <div className="mt-1.5 flex max-w-2xl flex-wrap gap-1.5">
      {buckets.map((bucket, bi) => {
        const tone = size === 1 ? bucket[0].tone : bucketToneOf(bucket.map((u) => u.tone));
        const label = size === 1 ? bucket[0].label : `${bucket[0].label} .. ${bucket[bucket.length - 1].label}`;
        return <StatusCircle key={bi} tone={tone} title={label} />;
      })}
    </div>
  );
}

export const REPEAT_PROGRESS_RE = /run (\d+)\/(\d+)/;

// Builds the unit list a StatusCircleStrip renders for a whole run's
// progress (used by the Runs list page, one strip per Run row, summarizing
// every sweep item in that run). Every finished/queued sweep item
// contributes exactly one unit (its overall combo outcome) -- but the
// single currently-running item (at most one at a time, since items
// process sequentially, see worker/src/index.ts's per-item loop) expands
// into one unit per repeat (-r) as soon as it starts running, so a sweep
// with one combo repeated 14 times shows 14 dots from the start (warmup
// included), not just 1 -- "runs" in this app's own language means repeats
// ("Runs averaged per combination" on the New Run page), not sweep combos.
// Before the worker's live "run X/Y" marker is parseable (during warmup, or
// on older llama-bench builds without --progress support), repeat 1 is
// assumed current. Falls back to a single "running" unit only when
// repeats <= 1.
export function buildProgressUnits(items: RunItem[], repeats: number): StatusCircleUnit[] {
  const units: StatusCircleUnit[] = [];
  for (const it of items) {
    const repeatUnits = buildRepeatUnits(it, repeats);
    if (repeatUnits) {
      units.push(...repeatUnits);
      continue;
    }
    units.push({ tone: comboTone(it.status), label: `combo ${it.idx + 1}: ${it.status}` });
  }
  return units;
}

// Builds a single sweep item's repeat-progress strip (used by RunDetail's
// per-item table, rendered under that specific row's live detail text --
// not as a whole-run summary). Returns null when there's nothing
// repeat-level to show: the item isn't currently running, or repeats <= 1.
// Once running, the full set of repeat dots shows immediately -- repeat 1
// is assumed current until the worker's live "run X/Y" marker is parseable
// (see buildProgressUnits' comment for why that can lag, e.g. warmup).
export function buildRepeatUnits(item: RunItem, repeats: number): StatusCircleUnit[] | null {
  if (!comboIsRunning(item.status) || repeats <= 1) return null;
  const match = REPEAT_PROGRESS_RE.exec(item.detail ?? "");
  const currentRep = match ? Number(match[1]) : 1;
  const units: StatusCircleUnit[] = [];
  for (let r = 1; r <= repeats; r++) {
    units.push({
      tone: r < currentRep ? "solid" : r === currentRep ? "running" : "grey",
      label: `repeat ${r}/${repeats}`,
    });
  }
  return units;
}

// Builds a single sweep item's repeat-progress strip for every status, not
// just while running -- unlike buildRepeatUnits (which returns null outside
// the running state, so RunDetail's per-item table always has *a* strip to
// show: queued items render every dot grey, done renders every dot solid,
// failed/failed_oom render every dot red, cancelled render every dot in its
// own tone, and a running item shows the same live per-repeat progress as
// buildRepeatUnits. No per-repeat failure detail is tracked, so a failed
// item's dots are uniform rather than pinpointing which repeat broke.
export function buildItemRepeatUnits(item: RunItem, repeats: number): StatusCircleUnit[] {
  const n = Math.max(repeats, 1);
  if (comboIsRunning(item.status)) {
    const match = REPEAT_PROGRESS_RE.exec(item.detail ?? "");
    const currentRep = match ? Number(match[1]) : 1;
    return Array.from({ length: n }, (_, i) => {
      const r = i + 1;
      return {
        tone: r < currentRep ? "solid" : r === currentRep ? "running" : "grey",
        label: `repeat ${r}/${n}`,
      } as StatusCircleUnit;
    });
  }
  const tone = comboTone(item.status);
  return Array.from({ length: n }, (_, i) => ({ tone, label: `repeat ${i + 1}/${n}` }));
}

export function WorkerStatusPill({ inaccessible, loading }: { inaccessible: boolean; loading?: boolean }) {
  if (loading) return <StatusPill label="checking…" tone="muted" />;
  return inaccessible ? <StatusPill label="Inaccessible" tone="danger" /> : <StatusPill label="Online" tone="success" />;
}

// Ticks on its own from a fixed started_at timestamp, so callers get a live
// elapsed-time display without needing to re-poll the worker/run data itself
// just to keep a clock moving.
export function ElapsedSince({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return <>{formatElapsed(now - startedAt)}</>;
}
