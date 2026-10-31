import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import type { ModelFitPlan, ModelFitResult, ModelFitTarget, Worker } from "../../types";
import { formatBytes } from "../../utils";
import { ACCURACY_NOTE, FIT_MARKS, fitDetails, fitHeadline, headerDownloadMib } from "./fitCopy";
import type { ModelFitStore } from "./useModelFit";

export interface FitItem {
  key: string;
  target: ModelFitTarget;
  label: string;
  shards: number;
  /** Already on the machine: nothing to download for this one. */
  isLocalOn: (workerId: string) => boolean;
}

export interface FitDialogRequest {
  title: string;
  /** Asked from one quant, or from a whole group's header. */
  scope: "one" | "all";
  items: FitItem[];
  /** The machine it was asked for; the dialog offers a choice when there is none. */
  workerId?: string;
  /** Opened from a mark that already has an answer: show it first. */
  showResult?: boolean;
}

// The prompt before anything is downloaded, and the details of an answer.
export function FitCheckDialog({
  request,
  workers,
  fit,
  onClose,
  onMachineChosen,
}: {
  request: FitDialogRequest;
  workers: Worker[];
  fit: ModelFitStore;
  onClose: () => void;
  onMachineChosen?: (workerId: string) => void;
}) {
  // With no machine named, the only one that could answer is the obvious choice.
  const online = workers.filter((w) => w.status !== "offline");
  const [workerId, setWorkerId] = useState(request.workerId ?? (online.length === 1 ? online[0].id : workers.length === 1 ? workers[0].id : ""));
  const [plan, setPlan] = useState<ModelFitPlan | null>(null);
  const [planError, setPlanError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [recheck, setRecheck] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    setPlan(null);
    setPlanError("");
    if (!workerId) return;
    let cancelled = false;
    api
      .getModelFitPlan(workerId)
      .then((p) => {
        if (!cancelled) setPlan(p);
      })
      .catch((err) => {
        if (!cancelled) setPlanError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [workerId]);

  const answer = (item: FitItem): ModelFitResult | null => {
    const s = fit.stateOf(workerId, item.key);
    return s.kind === "answered" ? s.result : null;
  };
  const first = request.items[0];
  const firstAnswer = first && request.scope === "one" ? answer(first) : null;
  const showingResult = !!request.showResult && !recheck && !!firstAnswer;

  // A group's button only asks for what has no answer yet; "Check again" asks for all.
  const todo = recheck || request.scope === "one" ? request.items : request.items.filter((i) => !answer(i) || answer(i)?.error);
  const toDownload = workerId ? todo.filter((i) => !i.isLocalOn(workerId)) : todo;
  const perFile = headerDownloadMib(1);
  const totalMib = toDownload.reduce((sum, i) => sum + headerDownloadMib(i.shards), 0);
  const allAnswered = request.scope === "all" && todo.length === 0;
  const machine = workers.find((w) => w.id === workerId);
  const needsInstall = plan?.build != null && !plan.build.installed;
  const canStart = !!plan && plan.online && plan.capable && !!plan.build && todo.length > 0 && !submitting;

  async function start() {
    setSubmitting(true);
    setSubmitError("");
    try {
      onMachineChosen?.(workerId);
      await fit.request(workerId, todo.map((i) => ({ key: i.key, target: i.target })), { autoInstall: needsInstall, force: recheck });
      onClose();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  const buildsLink = (label: string) => (
    <Link to={`/workers#machine-${encodeURIComponent(workerId)}`} className="text-accent hover:underline" onClick={onClose}>
      {label}
    </Link>
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="fit-dialog-title"
        tabIndex={-1}
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl border border-border bg-surface p-5 text-sm text-fg outline-none"
      >
        <h2 id="fit-dialog-title" className="m-0 font-display text-lg font-semibold">
          Will it fit? <span className="font-mono text-sm font-normal text-muted">{request.title}</span>
        </h2>

        {showingResult && firstAnswer ? (
          <ResultDetails result={firstAnswer} machineName={machine?.displayName} />
        ) : (
          <>
            {!request.workerId && workers.length > 1 && (
              <label className="mt-3 flex flex-col gap-1.5">
                <span className="text-muted">Machine</span>
                <select
                  value={workerId}
                  onChange={(e) => setWorkerId(e.target.value)}
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                >
                  <option value="">select…</option>
                  {workers.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.displayName}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {workers.length === 0 && (
              <p className="mt-3 text-danger">
                No machine connected yet. <Link to="/workers" className="text-accent hover:underline" onClick={onClose}>Add a machine</Link> to check this model.
              </p>
            )}

            {allAnswered ? (
              <p className="mt-3">Every quant here already has an answer for {machine?.displayName ?? "this machine"}.</p>
            ) : toDownload.length === 0 && workerId ? (
              <p className="mt-3">
                {request.scope === "one" ? "This file is" : "These files are"} already on {machine?.displayName ?? "this machine"}, so nothing needs to be downloaded.
              </p>
            ) : request.scope === "one" ? (
              <p className="mt-3">
                To detect whether this model can run on the machine, several MBs (about {Math.round(totalMib)} MB) will be downloaded. The whole file isn&apos;t needed.
              </p>
            ) : (
              <p className="mt-3">
                About {perFile} MB for every model ({toDownload.length} of them, about {Math.round(totalMib)} MB in total) will be downloaded and analyzed. The whole files aren&apos;t needed.
                {toDownload.length < todo.length && ` ${todo.length - toDownload.length} already on the machine need no download.`} Each takes from a few seconds to about half a minute.
              </p>
            )}
            {toDownload.length > 0 && !allAnswered && (
              <p className="mt-2 text-muted">Files go to a temp folder on the machine and are deleted right after the check.</p>
            )}

            {planError && <p className="mt-3 text-danger">{planError}</p>}
            {workerId && !plan && !planError && <p className="mt-3 text-muted">Looking at the machine…</p>}
            {plan && (
              <div className="mt-3 rounded-lg border border-border bg-surface-raised px-3 py-2">
                {!plan.online ? (
                  <p className="m-0 text-danger">This machine is offline.</p>
                ) : !plan.capable ? (
                  <p className="m-0 text-danger">This machine runs an older worker. Update it to check whether a model fits.</p>
                ) : !plan.build ? (
                  <p className="m-0 text-danger">
                    {plan.build_error ?? "No llama.cpp build available."} You can {buildsLink("install one in the machine's settings")}.
                  </p>
                ) : needsInstall ? (
                  <>
                    <p className="m-0">
                      llama.cpp is needed and no suitable build was detected. The most recent one, <b>{plan.build.tag}</b>
                      {plan.build.size_bytes ? ` (${formatBytes(plan.build.size_bytes)})` : ""}, will be downloaded automatically and used.
                    </p>
                    <p className="m-0 mt-1">
                      {plan.build.backend === plan.detected_backend ? "Auto-detected backend" : "Backend chosen for this machine"}: <b>{plan.build.backend}</b>
                    </p>
                    <p className="m-0 mt-1 text-muted">You can download another build or backend manually in the {buildsLink("machine's settings")}.</p>
                  </>
                ) : (
                  <p className="m-0">
                    Using llama.cpp <b>{plan.build.tag}</b> ({plan.build.backend}).{" "}
                    <span className="text-muted">{buildsLink("Change build or backend")}</span>
                  </p>
                )}
                {plan.online && plan.busy && <p className="m-0 mt-1 text-muted">The machine is busy: this waits in its queue until the current job ends.</p>}
              </div>
            )}
          </>
        )}

        <p className="mt-3 text-xs text-muted">{ACCURACY_NOTE}</p>
        {submitError && <p className="mt-2 text-danger">{submitError}</p>}

        <div className="mt-4 flex flex-wrap justify-end gap-2">
          <button type="button" onClick={onClose} className="rounded-lg border border-border px-3 py-1.5 text-muted hover:border-accent/40 hover:text-accent">
            {showingResult || allAnswered ? "Close" : "Cancel"}
          </button>
          {showingResult || allAnswered ? (
            <button type="button" onClick={() => setRecheck(true)} className="rounded-lg bg-accent px-4 py-1.5 font-semibold text-accent-fg">
              Check again
            </button>
          ) : (
            <button type="button" disabled={!canStart} onClick={() => void start()} className="rounded-lg bg-accent px-4 py-1.5 font-semibold text-accent-fg disabled:opacity-50">
              {submitting ? "Starting…" : toDownload.length > 0 ? "Download and check" : "Check"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultDetails({ result, machineName }: { result: ModelFitResult; machineName?: string }) {
  return (
    <div className="mt-3">
      <p className="m-0 font-semibold">
        {result.level ? `${FIT_MARKS[result.level]} ` : "⚠️ "}
        {fitHeadline(result)}
      </p>
      {result.error && <p className="mt-2 break-words text-danger">{result.error}</p>}
      <ul className="m-0 mt-2 list-none p-0 text-muted">
        {machineName && <li>Machine: {machineName}</li>}
        {fitDetails(result).map((line) => (
          <li key={line}>{line}</li>
        ))}
        <li>Checked {new Date(result.created_at).toLocaleString()}</li>
      </ul>
    </div>
  );
}
