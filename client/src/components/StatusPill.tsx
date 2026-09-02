import { useEffect, useState } from "react";
import type { Test, TestConfig, TestStatus, TestItemStatus, TestItem } from "../types";
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

export const TEST_STATUS_TONE: Record<TestStatus, PillTone> = {
  running: "accent",
  scheduled: "muted",
  done: "success",
  partial: "warning",
  failed: "danger",
  cancelled: "muted",
};

export function TestStatusPill({ status }: { status: TestStatus }) {
  return <StatusPill label={status} tone={TEST_STATUS_TONE[status]} />;
}

// §0.5's chain stages, abbreviated for a compact per-row/per-section chip
// (Tests.tsx's row chip strip, TestDetail.tsx's per-batch-member header). Any
// other kind (runtime/probe/quality/null) falls back to its own kind name
// (or "run" for a legacy/standalone row) rather than a letter, since those
// aren't part of the A/B/C tuning->refine->sweep sequence.
const STAGE_CHIP_LABEL: Partial<Record<string, string>> = { tuning: "A", refine: "B", sweep: "C" };
export function chipLabel(kind: string | null | undefined): string {
  return (kind && STAGE_CHIP_LABEL[kind]) || kind || "run";
}

// A probe batch's chip names the search mode it actually ran (e.g.
// "max_gpu") rather than the generic chipLabel(kind) fallback -- every
// member of a probe batch shares kind "probe", so the plain kind name can't
// tell them apart the way A/B/C already distinguishes a tuning chain's
// stages. Shared between Tests.tsx's row chip strip and TestDetail.tsx's
// per-batch-member Context-tests sections so both name a scenario the same way.
export function memberChipLabel(run: Test): string {
  if (run.kind === "probe") return (run.config as TestConfig)?.probe?.mode ?? "probe";
  return chipLabel(run.kind);
}

// N2 batching -- yellow-blink-while-running / yellow-static-while-queued,
// the same CircleTone StatusCircle already uses for a batched progress strip
// elsewhere, applied here per member chip instead of per sweep-item.
export const MEMBER_CIRCLE_TONE: Record<TestStatus, CircleTone> = {
  running: "running",
  scheduled: "warn",
  done: "solid",
  partial: "warn",
  failed: "red",
  cancelled: "cancelled",
};

const TEST_ITEM_STATUS_TONE: Record<TestItemStatus, PillTone> = {
  queued: "muted",
  loading: "accent",
  processing: "accent",
  generating: "accent",
  benchmarking: "accent",
  done: "success",
  failed: "danger",
  failed_oom: "danger",
  cancelled: "muted",
  // §0.7 -- never measured (an unsupported flag disabled its axis), which is
  // a different outcome from "failed" and reads as its own muted tone.
  skipped: "muted",
};

const TEST_ITEM_STATUS_LABEL: Record<TestItemStatus, string> = {
  queued: "queued",
  loading: "loading",
  processing: "processing",
  generating: "generating",
  benchmarking: "benchmarking",
  done: "done",
  failed: "failed",
  failed_oom: "failed (oom)",
  cancelled: "cancelled",
  skipped: "skipped",
};

export function TestItemStatusPill({ status }: { status: TestItemStatus }) {
  return <StatusPill label={TEST_ITEM_STATUS_LABEL[status]} tone={TEST_ITEM_STATUS_TONE[status]} />;
}

// Short, constant label for the dense per-row "detail" column (TestDetail.tsx)
// when an item has something in `error` -- shown instead of the actual error
// text, which stays available via that cell's title/tooltip. At the
// column's narrow fixed width ("11ch", sized for the longest normal phase
// word -- see MERGED_COLUMN_DEFS there), even a single-sentence error wraps
// across several lines and pushes every other column in that row down with
// it; a failure's real detail belongs in the worker's own log (now with a
// "debug log: <path>" line at the end of every test pointing at its full
// raw dump -- see worker/src/index.ts) or a hover, not baked into the table
// layout.
export function shortItemErrorLabel(status: TestItemStatus): string {
  switch (status) {
    case "failed":
      return "stopped with an error";
    case "failed_oom":
      return "stopped — out of memory";
    case "cancelled":
      return "stopped by user";
    default:
      // "done" with a non-empty `error` is a warning on an otherwise-
      // successful item (e.g. a suspect MTP reading), not a failure -- see
      // finalizeSweepItemResult in worker/src/index.ts, which stores
      // bench.warning as `error` regardless of status.
      return "done — warning";
  }
}

// Human-readable phase text for a non-terminal item. Terminal items
// (done/failed/cancelled) aren't meaningfully described this way -- callers
// should show their error/result instead.
export function describeItemPhase(item: TestItem): string {
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

type DotState = "grey" | "blink" | "solid" | "red" | "cancelled" | "skipped";

const TEST_ITEM_DOT_STATE: Record<TestItemStatus, DotState> = {
  queued: "grey",
  loading: "blink",
  processing: "blink",
  generating: "blink",
  benchmarking: "blink",
  done: "solid",
  failed: "red",
  failed_oom: "red",
  cancelled: "cancelled",
  skipped: "skipped",
};

// Compact status indicator for a dense per-row table (the full TestItemStatusPill
// above is more legible standalone but too wide to repeat once per row of a
// full test table): grey = not started yet, slowly blinking green = the
// currently running test, solid green = done, red = failed/failed_oom (an
// error or a test that never finished).
export function StatusDot({ status }: { status: TestItemStatus }) {
  const state = TEST_ITEM_DOT_STATE[status];
  return (
    <span
      className={`status-dot status-dot--${state}`}
      title={TEST_ITEM_STATUS_LABEL[status]}
      aria-label={TEST_ITEM_STATUS_LABEL[status]}
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

// Background for one half of a StatusSplitCircle, matching StatusCircle's
// per-tone coloring above (including --grey's dimmed opacity, replicated via
// color-mix since a CSS gradient can't apply per-half opacity on its own).
const SPLIT_TONE_BG: Record<CircleTone, string> = {
  running: "var(--color-warning)",
  red: "var(--color-danger)",
  warn: "var(--color-warning)",
  cancelled: "var(--color-muted)",
  grey: "color-mix(in srgb, var(--color-muted) 45%, transparent)",
  solid: "var(--color-success)",
};

// A repeat dot split down the middle -- left half tracks a pg (pp+tg) sweep
// item's prompt-processing progress for that repeat, right half tracks text-
// generation. Only meaningful while genuinely split (see buildItemRepeatUnits);
// used because llama-bench runs every repeat's pp phase to completion before
// starting any repeat's tg phase, so the two halves can be at very different
// points at once and a single dot's tone can't represent both.
export function StatusSplitCircle({
  ppTone,
  tgTone,
  title,
}: {
  ppTone: CircleTone;
  tgTone: CircleTone;
  title?: string;
}) {
  const animated = ppTone === "running" || tgTone === "running";
  return (
    <span
      className={`status-split-dot${animated ? " status-split-dot--blink" : ""}`}
      style={{ background: `linear-gradient(90deg, ${SPLIT_TONE_BG[ppTone]} 50%, ${SPLIT_TONE_BG[tgTone]} 50%)` }}
      title={title}
      aria-label={title}
    />
  );
}

const RUNNING_ITEM_STATUSES = new Set<TestItemStatus>(["loading", "processing", "generating", "benchmarking"]);

function comboIsRunning(status: TestItemStatus): boolean {
  return RUNNING_ITEM_STATUSES.has(status);
}
export function comboTone(status: TestItemStatus): CircleTone {
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

// A repeat unit split into independent pp/tg halves -- see buildItemRepeatUnits.
export interface SplitCircleUnit {
  ppTone: CircleTone;
  tgTone: CircleTone;
  label: string;
}

export type RepeatUnit = StatusCircleUnit | SplitCircleUnit;

function isSplitUnit(u: RepeatUnit): u is SplitCircleUnit {
  return "ppTone" in u;
}

// A split unit's overall tone for bucketing/precedence purposes (see
// bucketToneOf) -- its two halves collapsed via the same "running beats all"
// rule used across halves of a single unit as across whole units.
function unitTone(u: RepeatUnit): CircleTone {
  return isSplitUnit(u) ? bucketToneOf([u.ppTone, u.tgTone]) : u.tone;
}

// Collapses many progress units into a compact circle strip. <=49 units get
// one circle each; larger counts bucket sequentially so the strip stays
// readable (x5 at 50-99 units, x10 at 100+). Each rendered circle's tone is
// derived from the units it covers via the "running beats all" precedence.
// A split unit only renders as an actual split circle when shown on its own
// (bucket size 1) -- once bucketed with neighbors it collapses to one tone
// like any other unit, since a 5-or-10-wide bucket has no room to show two
// phases per neighbor anyway. What a unit represents is up to the caller --
// see buildProgressUnits and buildItemRepeatUnits.
export function StatusCircleStrip({ units }: { units: RepeatUnit[] }) {
  const count = units.length;
  const size = count <= 49 ? 1 : count <= 99 ? 5 : 10;
  const buckets: RepeatUnit[][] = [];
  for (let i = 0; i < count; i += size) buckets.push(units.slice(i, i + size));
  return (
    <div className="mt-1.5 flex max-w-2xl flex-wrap gap-1.5">
      {buckets.map((bucket, bi) => {
        if (size === 1) {
          const u = bucket[0];
          return isSplitUnit(u) ? (
            <StatusSplitCircle key={bi} ppTone={u.ppTone} tgTone={u.tgTone} title={u.label} />
          ) : (
            <StatusCircle key={bi} tone={u.tone} title={u.label} />
          );
        }
        const tone = bucketToneOf(bucket.map(unitTone));
        const label = `${bucket[0].label} .. ${bucket[bucket.length - 1].label}`;
        return <StatusCircle key={bi} tone={tone} title={label} />;
      })}
    </div>
  );
}

export const REPEAT_PROGRESS_RE = /run (\d+)\/(\d+)/;

// Builds the unit list a StatusCircleStrip renders for a whole run's
// progress (used by the Runs list page, one strip per Test row, summarizing
// every sweep item in that run). Every finished/queued sweep item
// contributes exactly one unit (its overall combo outcome) -- but the
// single currently-running item (at most one at a time, since items
// process sequentially, see worker/src/index.ts's per-item loop) expands
// into one unit per repeat (-r) as soon as it starts running, so a sweep
// with one combo repeated 14 times shows 14 dots from the start (warmup
// included), not just 1 -- "runs" in this app's own language means repeats
// ("Runs averaged per combination" on the Custom Test page), not sweep combos.
// Before the worker's live "run X/Y" marker is parseable (during warmup, or
// on older llama-bench builds without --progress support), repeat 1 is
// assumed current. Falls back to a single "running" unit only when
// repeats <= 1.
export function buildProgressUnits(items: TestItem[], repeats: number): StatusCircleUnit[] {
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
export function buildRepeatUnits(item: TestItem, repeats: number): StatusCircleUnit[] | null {
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

// Mirrors shared/sweep.ts's deriveTestType (also inlined in TestDetail.tsx's
// testTypeLabel) -- kept local per the boundary note there.
function deriveTestType(nPrompt: number, nGen: number): "pp" | "tg" | "pg" {
  if (nGen === 0) return "pp";
  if (nPrompt === 0) return "tg";
  return "pg";
}

// Builds a single sweep item's repeat-progress strip for every status, not
// just while running -- unlike buildRepeatUnits (which returns null outside
// the running state, so RunDetail's per-item table always has *a* strip to
// show: queued items render every dot grey, done renders every dot solid,
// failed/failed_oom render every dot red, cancelled render every dot in its
// own tone, and a running item shows the same live per-repeat progress as
// buildRepeatUnits. No per-repeat failure detail is tracked, so a failed
// item's dots are uniform rather than pinpointing which repeat broke.
//
// A pg item (both n_prompt and n_gen set) actually runs as two sequential
// sub-benchmarks under llama-bench -- every repeat's pp phase completes
// before any repeat's tg phase starts (confirmed against the "benchmark N/2:
// prompt run x/y" / "generation run x/y" stderr markers this parses, see
// worker/src/index.ts's PP_RUN_RE/TG_RUN_RE). So while a pg item is live and
// its current phase is known (status "processing" or "generating"), each
// repeat's dot is split into independent pp/tg halves instead of one tone --
// a single tone can't represent "pp done, tg 2/5" for the same repeat.
// Outside that live+attributable window (queued/done/failed/cancelled, or
// the older-llama-bench fallback "benchmarking" status with no per-phase
// marker to key off), falls back to the old single-tone-per-repeat dots.
export function buildItemRepeatUnits(item: TestItem, repeats: number): RepeatUnit[] {
  const n = Math.max(repeats, 1);
  const testType = deriveTestType(item.n_prompt, item.n_gen);
  if (testType === "pg" && (item.status === "processing" || item.status === "generating")) {
    const match = REPEAT_PROGRESS_RE.exec(item.detail ?? "");
    const currentRep = match ? Number(match[1]) : 1;
    const inTgPhase = item.status === "generating";
    return Array.from({ length: n }, (_, i) => {
      const r = i + 1;
      const ppTone: CircleTone = inTgPhase || r < currentRep ? "solid" : r === currentRep ? "running" : "grey";
      const tgTone: CircleTone = !inTgPhase ? "grey" : r < currentRep ? "solid" : r === currentRep ? "running" : "grey";
      const phaseWord = (t: CircleTone) => (t === "solid" ? "done" : t === "running" ? "running" : "queued");
      return {
        ppTone,
        tgTone,
        label: `repeat ${r}/${n} -- pp ${phaseWord(ppTone)}, tg ${phaseWord(tgTone)}`,
      } as SplitCircleUnit;
    });
  }
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
