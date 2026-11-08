import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Test, Worker } from "../types";
import { jobPercent } from "../jobProgress";

// System status: which machine is busy with what, how far along, what is
// queued and how many machines are online. In the sidebar footer on wide
// layouts, one line under the top bar on narrow ones. Purely a read of data
// the app already polls.

function modelLabel(t: Test): string {
  return (t.model_filename ?? t.model_id).replace(/\.gguf$/i, "").split(/[\\/]/).pop() ?? t.model_id;
}

const KIND_LABEL: Record<string, string> = {
  fit: "fit map",
  runtime: "speed test",
  probe: "context test",
  quality: "quality",
};

/** `sidebar`: a quiet block in the sidebar footer (wide layouts). `strip`:
 * one line under the narrow layout's top bar. */
export function StatusBar({ workers, variant }: { workers: Worker[]; variant: "sidebar" | "strip" }) {
  const [tests, setTests] = useState<Test[]>([]);
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const list = await api.listTests();
        if (!cancelled) setTests(list);
      } catch {
        /* transient -- keep the last reading */
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, 8000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  const online = workers.filter((w) => w.status !== "offline");
  const busy = workers.find((w) => w.status === "busy");
  const running = tests.find((t) => t.status === "running");
  const queued = tests.filter((t) => t.status === "scheduled");
  const head = busy ?? online[0] ?? null;
  const pct = busy?.activeJobProgress
    ? jobPercent(busy.activeJobProgress.detail, busy.activeJobProgress.item_idx, busy.activeJobProgress.items_total)
    : running && running.items_total
      ? Math.round((((running.items_done ?? 0) + (running.items_failed ?? 0)) / running.items_total) * 100)
      : 0;

  const side = variant === "sidebar";
  return (
    <section
      aria-label="System status"
      className={
        side
          ? "flex flex-col gap-1.5 px-[18px] text-[13px] text-fg-2"
          : "flex flex-wrap items-center gap-x-5 gap-y-1 border-b border-border px-4 py-2 text-[13px] text-fg-2"
      }
    >
      <span className="inline-flex items-center gap-2">
        <span
          aria-hidden="true"
          className={`h-2 w-2 rounded-full border ${
            !head ? "border-muted bg-transparent" : busy ? "lt-pulse border-accent bg-accent" : "border-online bg-online"
          }`}
        />
        <strong className="font-medium text-fg">{head ? head.displayName : workers.length ? "No machines online" : "No machines yet"}</strong>
        {!side && <span>{head ? (busy ? "testing" : "online") : workers.length ? "waiting for a check-in" : "add one on the Machines page"}</span>}
      </span>
      {running && (
        <span aria-live="polite" className="inline-flex flex-wrap items-center gap-2.5">
          <Link to={`/tests/${running.id}`} className="text-fg-2 hover:text-accent">
            {KIND_LABEL[running.kind ?? ""] ? `${KIND_LABEL[running.kind ?? ""]} · ` : "Testing "}
            {modelLabel(running)}
            {running.items_total ? ` · ${(running.items_done ?? 0) + (running.items_failed ?? 0)}/${running.items_total}` : ""}
          </Link>
          <span
            role="progressbar"
            aria-label="Test progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
            className={`block h-1.5 bg-surface-raised ${side ? "w-full" : "w-24"}`}
          >
            <span className="block h-full bg-accent" style={{ width: `${pct}%` }} />
          </span>
          {busy?.activeJobProgress?.detail && <span className="max-w-[28ch] truncate text-muted">{busy.activeJobProgress.detail}</span>}
        </span>
      )}
      <span className={side ? "inline-flex gap-3 text-muted" : "ml-auto inline-flex gap-4 text-muted"}>
        <span>{queued.length ? `Queue ${queued.length}` : "Queue empty"}</span>
        <span>
          {online.length} of {workers.length} online
        </span>
      </span>
    </section>
  );
}
