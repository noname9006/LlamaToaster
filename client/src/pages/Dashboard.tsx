import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { TokSpeedDemo } from "../components/TokSpeedDemo";
import { platformLabel } from "../components/WorkerCard";
import { LtIcon } from "../components/ltIcons";
import { Btn, PageHeader } from "../components/optimize/ui";
import type { AdminStats, Test, Worker } from "../types";
import { jobPercent } from "../jobProgress";
import { kindLabel, testStatusView } from "../testLabels";

// The v2 dashboard (docs/plans/app-v2.dc.html): your machines as cards --
// what each one is doing right now, its memory, one action -- then recent
// tests. Platform-wide totals (GET /api/stats, an operator request) stay as
// one quiet line at the bottom; the machine and test lists are the caller's own.

const HIDDEN_WORKERS_STORAGE_KEY = "llamatoaster:dashboard:hidden-workers";

function readHiddenWorkers(): string[] {
  try {
    const raw = localStorage.getItem(HIDDEN_WORKERS_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function writeHiddenWorkers(ids: string[]): void {
  try {
    localStorage.setItem(HIDDEN_WORKERS_STORAGE_KEY, JSON.stringify(ids));
  } catch {
    /* localStorage unavailable -- hiding just won't survive a reload */
  }
}

// worker_id is the real FK; worker_name is only a point-in-time snapshot, so
// a run predating that column falls back to matching the name.
function testMatchesWorker(r: Test, worker: Worker): boolean {
  return r.worker_id ? r.worker_id === worker.id : r.worker_name === worker.displayName;
}

function modelLabel(t: Test): string {
  return (t.model_filename ?? t.model_id).replace(/\.gguf$/i, "").split(/[\\/]/).pop() ?? t.model_id;
}

function MachineCard({
  worker,
  runs,
  hidden,
  onHide,
  onUnhide,
}: {
  worker: Worker;
  runs: Test[];
  hidden: boolean;
  onHide: () => void;
  onUnhide: () => void;
}) {
  const navigate = useNavigate();
  const testing = worker.status === "busy";
  const offline = worker.status === "offline";
  const mine = runs.filter((r) => testMatchesWorker(r, worker));
  const active = mine.find((r) => r.status === "running");
  const queued = mine.filter((r) => r.status === "scheduled").length;
  const last = mine.find((r) => r.status !== "running" && r.status !== "scheduled");
  const gpu = worker.hardware?.gpu[0];
  const vramTotal = worker.vram?.ok ? worker.vram.gpu_memory_total_mib : gpu?.vram_listed_total_mb ?? gpu?.vram_mb ?? null;
  const vramFree = worker.vram?.ok ? worker.vram.vram_free_before_mib : null;
  const ramTotal = worker.hardware?.mem_total_bytes ? worker.hardware.mem_total_bytes / 1048576 : null;
  const ramFree = worker.vram?.ram_free_before_mib ?? null;
  const memLabel = vramTotal ? "VRAM" : "RAM";
  const total = vramTotal ?? ramTotal;
  const free = vramTotal ? vramFree : ramFree;
  const used = total != null && free != null ? Math.max(0, total - free) : null;
  const pct = total && used != null ? Math.round((used / total) * 100) : 0;
  const progress = worker.activeJobProgress;
  const progressPct = jobPercent(progress?.detail, progress?.item_idx, progress?.items_total ?? active?.items_total);
  const stateLabel = testing ? "Testing" : offline ? "Offline" : queued ? `Online · ${queued} queued` : "Online · idle";

  return (
    <article className={`relative flex flex-col gap-3 border bg-surface p-4 ${testing ? "border-accent" : "border-border"} ${hidden ? "border-dashed opacity-60" : ""}`}>
      {testing && (
        <span aria-hidden="true" className="absolute -top-3.5 right-6 flex gap-1.5">
          <span className="lt-steam block h-3 w-0.5 bg-muted" />
          <span className="lt-steam block h-3 w-0.5 bg-muted [animation-delay:1.1s]" />
        </span>
      )}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 truncate font-mono text-sm font-medium">{worker.displayName}</h3>
        <span className="inline-flex items-center gap-1.5 font-mono text-xs text-fg-2">
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full border ${offline ? "border-muted bg-transparent" : testing ? "lt-pulse border-accent bg-accent" : "border-online bg-online"}`}
          />
          {stateLabel}
        </span>
      </div>
      <p className="m-0 font-mono text-xs leading-relaxed text-muted">
        {gpu ? `${gpu.model}${vramTotal ? ` · ${(vramTotal / 1024).toFixed(0)} GB` : ""} · ${worker.backend ?? "?"}` : `No GPU · ${worker.backend ?? "cpu"} backend`}
        <br />
        {worker.hardware?.os?.name ?? platformLabel(worker.platform)}
        {worker.hardware?.cpu.brand ? ` · ${worker.hardware.cpu.brand.replace(/\s+\d+-Core Processor$/, "")}` : ""}
        {ramTotal ? ` · ${Math.round(ramTotal / 1024)} GB` : ""}
      </p>
      <div>
        <div className="flex justify-between font-mono text-xs text-muted">
          <span>{memLabel}</span>
          <span>{offline || used == null || total == null ? `— / ${total ? (total / 1024).toFixed(1) : "?"} GB` : `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GB`}</span>
        </div>
        <div aria-hidden="true" className="mt-1 h-3 border border-border-strong bg-well p-0.5 shadow-[inset_0_2px_0_var(--color-bg)]">
          <span className="block h-full bg-muted" style={{ width: `${offline ? 0 : pct}%` }} />
        </div>
      </div>
      {testing ? (
        <div>
          <p className="m-0 text-sm text-fg">{active ? `${modelLabel(active)} · ${kindLabel(active)}` : progress?.phase ?? "working"}</p>
          <div className="mt-1.5 h-2 border border-border-strong bg-well">
            <span className="lt-progress-fill block h-full" style={{ width: `${Math.max(5, progressPct)}%` }} />
          </div>
          {progress?.detail && <p className="mt-1.5 truncate font-mono text-xs text-fg-2">{progress.detail}</p>}
        </div>
      ) : (
        <p className="m-0 text-sm text-fg-2">
          {offline
            ? "Start the worker on the machine; queued tests wait for it."
            : last
              ? `Last: ${modelLabel(last)} · ${kindLabel(last)} · ${testStatusView(last).label.toLowerCase()}`
              : "No tests on this machine yet."}
        </p>
      )}
      <div className="mt-auto flex flex-wrap items-center gap-2">
        <Btn
          onClick={() => {
            if (testing && active) navigate(`/tests/${active.id}`);
            else if (offline) navigate("/workers");
            else navigate("/benchmark");
          }}
        >
          {testing ? "View test" : offline ? "Reconnect steps" : "Start test here"}
        </Btn>
        <Btn kind="ghost" onClick={hidden ? onUnhide : onHide}>
          {hidden ? "Unhide" : "Hide"}
        </Btn>
      </div>
    </article>
  );
}

export function Dashboard() {
  const [runs, setRuns] = useState<Test[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [hiddenWorkerIds, setHiddenWorkerIds] = useState<string[]>(() => readHiddenWorkers());
  const [showHidden, setShowHidden] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const [r, w, s] = await Promise.all([api.listTests(), api.listWorkers(), api.getStats().catch(() => null)]);
        if (cancelled) return;
        setRuns(r);
        setWorkers(w);
        setStats(s);
        setLoaded(true);
      } catch {
        // The finally reschedules either way; a failed tick keeps the last good data.
      } finally {
        if (!cancelled) timerRef.current = window.setTimeout(poll, 5000);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  function setHidden(next: string[]) {
    setHiddenWorkerIds(next);
    writeHiddenWorkers(next);
  }

  const visibleWorkers = showHidden ? workers : workers.filter((w) => !hiddenWorkerIds.includes(w.id));
  const counts = useMemo(() => {
    const testing = workers.filter((w) => w.status === "busy").length;
    const online = workers.filter((w) => w.status === "idle").length;
    const offline = workers.filter((w) => w.status === "offline").length;
    return { testing, online, offline };
  }, [workers]);
  const allOffline = workers.length > 0 && counts.offline === workers.length;
  const queuedTotal = runs.filter((r) => r.status === "scheduled").length;
  const recent = runs.slice(0, 8);

  return (
    <div>
      <PageHeader
        title="Dashboard"
        sub={
          workers.length === 0
            ? "No machines yet."
            : allOffline
              ? "No machine online right now."
              : [counts.testing && `${counts.testing} testing`, counts.online && `${counts.online} online`, counts.offline && `${counts.offline} offline`].filter(Boolean).join(" · ")
        }
        actions={
          <Link
            to="/benchmark"
            className="inline-flex min-h-11 items-center gap-2 border border-accent bg-accent px-5 font-display text-lg font-semibold text-accent-fg hover:bg-accent-hover"
          >
            <LtIcon name="plus" />
            New test
          </Link>
        }
      />

      {allOffline && (
        <div role="alert" className="mb-5 flex flex-wrap items-center gap-x-5 gap-y-3 border border-border-strong bg-surface px-[18px] py-4">
          <LtIcon name="offline" size={22} className="text-fg-2" />
          <div className="flex-[1_1_280px]">
            <h2 className="m-0 font-display text-lg font-semibold">Every bay is cold</h2>
            <p className="m-0 mt-0.5 text-fg-2">
              No machine is checking in.{queuedTotal ? ` Your ${queuedTotal} queued test${queuedTotal === 1 ? "" : "s"} start as soon as one comes back online.` : ""}
            </p>
          </div>
          <Link to="/workers" className="border border-border-strong px-3.5 py-2 text-sm font-medium hover:border-accent hover:text-accent">
            Reconnect a machine
          </Link>
        </div>
      )}

      <section aria-labelledby="h-machines">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="h-machines" className="m-0 font-display text-lg font-semibold">My machines</h2>
          {hiddenWorkerIds.length > 0 && (
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} className="h-3.5 w-3.5 accent-[var(--color-accent)]" />
              Show hidden ({hiddenWorkerIds.length})
            </label>
          )}
        </div>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,260px),1fr))] gap-4">
          {visibleWorkers.map((w) => (
            <MachineCard
              key={w.id}
              worker={w}
              runs={runs}
              hidden={hiddenWorkerIds.includes(w.id)}
              onHide={() => setHidden([...new Set([...hiddenWorkerIds, w.id])])}
              onUnhide={() => setHidden(hiddenWorkerIds.filter((x) => x !== w.id))}
            />
          ))}
          <article className="flex flex-col justify-center gap-2 border border-dashed border-border-strong p-4">
            <h3 className="m-0 font-display text-lg font-semibold">{loaded && workers.length === 0 ? "No machines yet" : "Empty bay"}</h3>
            <p className="m-0 text-fg-2">
              {loaded && workers.length === 0 ? "LlamaToaster runs benchmarks on your own hardware — connecting one takes a minute." : "Add another machine with a one-time enrolment code."}
            </p>
            <Link to="/device" className="self-start border border-border-strong px-3.5 py-2 text-sm font-medium hover:border-accent hover:text-accent">
              Add machine
            </Link>
          </article>
        </div>
      </section>

      <section aria-labelledby="h-recent" className="mt-7 border border-border bg-surface">
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <h2 id="h-recent" className="m-0 font-display text-lg font-semibold">Recent tests</h2>
          <Link to="/tests" className="text-sm font-medium text-accent">
            All tests
          </Link>
        </div>
        {loaded && recent.length === 0 && <p className="px-4 py-3 text-sm text-muted">No tests yet.</p>}
        <ul className="m-0 list-none p-0">
          {recent.map((t) => {
            const st = testStatusView(t);
            return (
              <li key={t.id} className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border px-4 py-2.5 last:border-b-0">
                <div className="min-w-0 flex-[1_1_200px]">
                  <Link to={`/tests/${t.id}`} className="text-sm font-medium text-fg hover:text-accent">
                    {modelLabel(t)}
                  </Link>
                  <div className="font-mono text-xs text-muted">
                    {t.worker_name} · {kindLabel(t)}
                  </div>
                </div>
                <span className={`inline-flex items-center gap-1.5 font-mono text-xs tracking-[0.04em] ${st.tone}`}>
                  {st.icon && <LtIcon name={st.icon} size={14} />}
                  {st.label}
                </span>
                {t.items_total ? (
                  <span className="flex-[1_0_100%] font-mono text-xs text-fg-2">
                    {(t.items_done ?? 0)} of {t.items_total} done{t.items_failed ? ` · ${t.items_failed} failed` : ""}
                  </span>
                ) : null}
              </li>
            );
          })}
        </ul>
      </section>

      <section className="mt-7">
        <TokSpeedDemo />
      </section>

      {stats && (
        <p className="mt-6 font-mono text-xs text-muted">
          Platform: {stats.users} users · {stats.machines} machines · {stats.modelsTested} models tested · {stats.quants} quants · {stats.tests} tests ·{" "}
          {stats.runs} runs
        </p>
      )}
    </div>
  );
}
