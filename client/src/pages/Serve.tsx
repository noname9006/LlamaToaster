import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { LtIcon } from "../components/ltIcons";
import { Btn, PageHeader, Panel } from "../components/optimize/ui";
import { StatusPill } from "../components/StatusPill";
import { ConnectionPanel } from "../components/serve/ConnectionPanel";
import { MemoryPanel } from "../components/serve/MemoryPanel";
import { LoadSettingsPanel, SamplingPanel, type LoadExtras } from "../components/serve/SettingsPanels";
import { canUnload, formatTokens, isLive, modelFilename, modelLabel, modelsOnMachine, PHASE_VIEW, projectorFiles, settingsDiffer } from "../components/serve/serveData";
import { UsagePanel } from "../components/serve/UsagePanel";
import { defaultAlias, EMPTY_SAMPLING, type SamplingSettings, type SettingValues } from "../../../shared/daemonSettings";
import type { DaemonUsageView, DaemonView } from "../../../shared/daemonTypes";
import type { Model } from "../types";

// The Serve page (docs/plans/MODEL_DAEMON_PLAN.md): load a model on a machine from
// here, and point external agents at the OpenAI-compatible endpoint on that machine.

const POLL_MS = 3000;
const USAGE_POLL_MS = 10_000;
// With no report for this long, what is on screen is history, not the current state.
const STALE_AFTER_MS = 30_000;
const MACHINE_KEY = "llamatoaster:serve:machine";

function readMachine(): string | null {
  try {
    return localStorage.getItem(MACHINE_KEY);
  } catch {
    return null;
  }
}

function writeMachine(id: string): void {
  try {
    localStorage.setItem(MACHINE_KEY, id);
  } catch {
    // The choice just won't survive a reload.
  }
}

function describeError(err: unknown): string {
  return err instanceof ApiError ? err.message : err instanceof Error ? err.message : String(err);
}

export function Serve() {
  const { workers, loaded: workersLoaded } = useWorkerStatuses();
  const [machineId, setMachineId] = useState<string | null>(readMachine);
  const usable = useMemo(() => workers.filter((w) => w.capabilities?.includes("daemon-v1")), [workers]);
  const worker = useMemo(() => usable.find((w) => w.id === machineId) ?? usable.find((w) => w.status !== "offline") ?? usable[0], [usable, machineId]);
  const workerId = worker?.id ?? null;

  const [models, setModels] = useState<Model[]>([]);
  useEffect(() => {
    api.listModels().then(setModels, () => undefined);
  }, []);
  const onMachine = useMemo(() => modelsOnMachine(models, worker), [models, worker]);

  const [view, setView] = useState<DaemonView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    if (!workerId) return;
    setView(await api.getDaemon(workerId));
  }, [workerId]);

  useEffect(() => {
    setView(null);
    if (!workerId) return;
    let cancelled = false;
    let timer: number | undefined;
    const tick = async () => {
      window.clearTimeout(timer);
      // Each poll tells the machine someone is watching, which starts GPU memory reads
      // that disturb inference timing. A tab nobody can see must not keep that going.
      if (document.visibilityState === "visible") {
        try {
          const next = await api.getDaemon(workerId);
          if (!cancelled) setView(next);
        } catch {
          // The next tick tries again; a dropped session is handled app-wide.
        }
      }
      if (!cancelled) timer = window.setTimeout(() => void tick(), POLL_MS);
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") void tick();
    };
    document.addEventListener("visibilitychange", onVisibility);
    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [workerId]);

  const status = view?.status ?? null;
  const stale = view?.reported_at != null && Date.now() - view.reported_at > STALE_AFTER_MS;
  const offline = worker?.status === "offline";
  const phase = status?.phase;
  const loadedModelId = isLive(phase) || phase === "suspended" ? status?.model_id ?? null : null;

  // The selected model follows what is running until someone picks another, and stays
  // put when the model goes away. An unload or a failed load reports no model, and
  // falling back to the first on the machine would swap the form for another model's
  // settings and make Load load the wrong one.
  const [pickedModelId, setPickedModelId] = useState<string | null>(null);
  const [lastLoadedModelId, setLastLoadedModelId] = useState<string | null>(null);
  useEffect(() => {
    if (status?.model_id) setLastLoadedModelId(status.model_id);
  }, [status?.model_id]);
  useEffect(() => {
    setPickedModelId(null);
    setLastLoadedModelId(null);
  }, [workerId]);
  const modelId = pickedModelId ?? loadedModelId ?? status?.model_id ?? lastLoadedModelId ?? onMachine[0]?.id ?? null;
  const model = onMachine.find((m) => m.id === modelId) ?? models.find((m) => m.id === modelId) ?? null;

  // The form. Fetched once per (machine, model); polling must not overwrite edits.
  const [settings, setSettings] = useState<SettingValues>({});
  const [sampling, setSampling] = useState<SamplingSettings>(EMPTY_SAMPLING);
  const [savedSampling, setSavedSampling] = useState<SamplingSettings>(EMPTY_SAMPLING);
  const [extras, setExtras] = useState<LoadExtras>({ alias: "", mmproj: "", draftModelId: "" });
  useEffect(() => {
    if (!workerId || !modelId) return;
    let cancelled = false;
    api.getDaemonModelSettings(workerId, modelId).then(
      ({ settings: saved }) => {
        if (cancelled) return;
        setSettings(saved?.settings ?? {});
        setSampling(saved?.sampling ?? EMPTY_SAMPLING);
        setSavedSampling(saved?.sampling ?? EMPTY_SAMPLING);
        setExtras({ alias: saved?.alias ?? "", mmproj: saved?.mmproj_filename ?? "", draftModelId: saved?.draft_model_id ?? "" });
      },
      () => undefined
    );
    return () => {
      cancelled = true;
    };
  }, [workerId, modelId]);

  const fallbackAlias = model ? defaultAlias(modelFilename(model)) : "";
  const aliasForLoad = extras.alias.trim() || fallbackAlias;
  const isThisModelLoaded = loadedModelId != null && loadedModelId === modelId;
  const needsReload =
    isThisModelLoaded &&
    phase !== "loading" &&
    (settingsDiffer(settings, status?.settings) ||
      (status?.alias ?? "") !== aliasForLoad ||
      (status?.mmproj_filename ?? "") !== extras.mmproj ||
      (status?.draft_model_id ?? "") !== extras.draftModelId);
  const samplingDirty = JSON.stringify(sampling) !== JSON.stringify(savedSampling);
  const pending = view?.pending_job ?? null;

  const act = (fn: () => Promise<unknown>) => {
    setError(null);
    fn().then(refresh).catch((err: unknown) => setError(describeError(err)));
  };

  const load = () => {
    if (!workerId || !modelId) return;
    // Pinned from here, so a failed load leaves this model on screen to try again.
    setPickedModelId(modelId);
    act(async () => {
      await api.loadDaemonModel(workerId, {
        model_id: modelId,
        alias: extras.alias.trim() || undefined,
        settings,
        sampling,
        mmproj_filename: extras.mmproj || null,
        draft_model_id: extras.draftModelId || null,
      });
      setSavedSampling(sampling);
    });
  };

  const applySampling = () => {
    if (!workerId || !modelId) return;
    act(async () => {
      const res = await api.saveDaemonSampling(workerId, modelId, sampling);
      setSampling(res.sampling);
      setSavedSampling(res.sampling);
    });
  };

  // Usage for the selected model. Only refetched while the page is visible.
  const [usage, setUsage] = useState<DaemonUsageView | null>(null);
  // Different placements have different speeds, so the view starts on the load that
  // is running. "" is the explicit choice to see them all.
  const [pickedLoadHash, setPickedLoadHash] = useState<string | null>(null);
  const loadHash = pickedLoadHash ?? (isThisModelLoaded ? status?.load_hash ?? "" : "");
  const [includeConcurrent, setIncludeConcurrent] = useState(false);
  const usageGeneration = useRef(0);
  const fetchUsage = useCallback(async () => {
    if (!workerId || !modelId) return;
    const generation = ++usageGeneration.current;
    const next = await api.getDaemonUsage(workerId, modelId, { loadHash: loadHash || null, includeConcurrent });
    // A slow answer for a previous selection must not replace a newer one.
    if (generation === usageGeneration.current) setUsage(next);
  }, [workerId, modelId, loadHash, includeConcurrent]);

  useEffect(() => {
    setUsage(null);
    setPickedLoadHash(null);
  }, [workerId, modelId]);

  useEffect(() => {
    if (!workerId || !modelId) return;
    let cancelled = false;
    let timer: number | undefined;
    const tick = async () => {
      if (document.visibilityState === "visible") await fetchUsage().catch(() => undefined);
      if (!cancelled) timer = window.setTimeout(() => void tick(), USAGE_POLL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [workerId, modelId, fetchUsage]);

  const phaseView = status ? PHASE_VIEW[status.phase] : null;

  if (workersLoaded && usable.length === 0) {
    return (
      <div>
        <PageHeader title="Serve" sub="Run a model on one of your machines and point agents at it." />
        <Panel>
          <p className="m-0 text-fg-2">
            {workers.length > 0 ? "Your machines run a worker too old to serve models. Update the worker, then come back." : "No machines yet."}{" "}
            <Link className="text-accent underline" to="/workers">
              Go to Machines
            </Link>
          </p>
        </Panel>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Serve"
        sub="Run a model on one of your machines and point agents at it."
        actions={
          usable.length > 1 ? (
            <label className="flex items-center gap-2 text-sm text-fg-2">
              Machine
              <select
                value={workerId ?? ""}
                onChange={(e) => {
                  writeMachine(e.target.value);
                  setMachineId(e.target.value);
                  setPickedModelId(null);
                }}
                className="min-h-10 border border-border-strong bg-well px-2 text-sm text-fg"
              >
                {usable.map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.displayName}
                    {w.status === "offline" ? " (offline)" : ""}
                  </option>
                ))}
              </select>
            </label>
          ) : undefined
        }
      />

      <div className="flex flex-col gap-4">
        <Panel>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
              {offline ? <StatusPill label="Machine offline" tone="muted" /> : phaseView && <StatusPill label={phaseView.label} tone={phaseView.tone} />}
              <label className="flex min-w-0 items-center gap-2 text-sm text-fg-2">
                Model
                <select
                  value={modelId ?? ""}
                  onChange={(e) => setPickedModelId(e.target.value)}
                  disabled={onMachine.length === 0}
                  className="min-h-10 min-w-0 max-w-full border border-border-strong bg-well px-2 font-mono text-sm text-fg"
                >
                  {onMachine.length === 0 && <option value="">None on this machine</option>}
                  {onMachine.map((m) => (
                    <option key={m.id} value={m.id}>
                      {modelLabel(m)}
                      {m.id === loadedModelId ? " (loaded)" : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              {canUnload(phase) && (
                <Btn onClick={() => workerId && act(() => api.unloadDaemonModel(workerId))} disabled={offline}>
                  <LtIcon name="stop" />
                  {phase === "loading" ? "Cancel load" : "Unload"}
                </Btn>
              )}
              {!isThisModelLoaded || needsReload || phase === "failed" || phase === "crashed" ? (
                <Btn kind="primary" onClick={load} disabled={offline || !modelId || pending?.type === "daemon_load"}>
                  <LtIcon name="play" />
                  {needsReload ? "Reload with these settings" : "Load model"}
                </Btn>
              ) : null}
            </div>
          </div>

          {status && isLive(phase) && (
            <dl className="m-0 mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm">
              {status.actual?.gpu_layers != null && <Fact label="GPU layers" value={`${status.actual.gpu_layers}${status.actual.total_layers != null ? ` of ${status.actual.total_layers}` : ""}`} />}
              {status.actual?.n_ctx != null && <Fact label="Context per request" value={formatTokens(status.actual.n_ctx)} />}
              {status.actual?.n_slots != null && <Fact label="Slots" value={String(status.actual.n_slots)} />}
              <Fact label="In flight" value={String(status.in_flight)} />
              {status.build_tag && <Fact label="Build" value={status.build_tag} />}
              {status.restarts > 0 && <Fact label="Restarts" value={String(status.restarts)} />}
            </dl>
          )}
          {pending && (
            <p className="m-0 mt-2 text-sm text-muted">
              {pending.status === "pending"
                ? `${pending.type === "daemon_load" ? "Load" : "Unload"} is queued behind what this machine is running.`
                : pending.type === "daemon_load"
                  ? "Starting llama-server."
                  : "Unloading."}
            </p>
          )}
          {status?.message && <p className={`m-0 mt-2 whitespace-pre-wrap text-sm ${phase === "failed" || phase === "crashed" ? "text-danger" : "text-fg-2"}`}>{status.message}</p>}
          {view?.last_error && view.last_error !== status?.message && view.last_error !== "load cancelled" && phase !== "ready" && <p className="m-0 mt-2 whitespace-pre-wrap text-sm text-danger">{view.last_error}</p>}
          {stale && !offline && <p className="m-0 mt-2 text-sm text-warning">This machine has not reported for a while, so this may be out of date.</p>}
          {error && <p className="m-0 mt-2 text-sm text-danger">{error}</p>}
          {needsReload && <p className="m-0 mt-2 text-sm text-warning">The load settings below differ from what is running.</p>}
          {status && status.usage_dropped > 0 && <p className="m-0 mt-2 text-xs text-muted">{status.usage_dropped} usage rows were lost while this machine could not reach the server.</p>}
          {onMachine.length === 0 && (
            <p className="m-0 mt-2 text-sm text-fg-2">
              No models on this machine yet.{" "}
              <Link className="text-accent underline" to="/models">
                Get one from Models
              </Link>
            </p>
          )}
        </Panel>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <MemoryPanel status={status} machineBusy={worker?.status === "busy"} />
          {workerId && <ConnectionPanel key={workerId} workerId={workerId} status={status} gateway={view?.gateway ?? { host: "127.0.0.1", port: 8642, cors_origins: [], autostart: false }} keys={view?.keys ?? []} alias={isThisModelLoaded ? status?.alias ?? null : aliasForLoad || null} onChanged={() => void refresh()} />}
        </div>

        <UsagePanel
          view={usage}
          modelName={model ? modelLabel(model) : null}
          loadHash={loadHash}
          onLoadHash={setPickedLoadHash}
          includeConcurrent={includeConcurrent}
          onIncludeConcurrent={setIncludeConcurrent}
          onClear={() => {
            if (!workerId || !modelId) return;
            if (!window.confirm("Delete this model's usage history on this machine? This cannot be undone.")) return;
            act(async () => {
              await api.clearDaemonUsage(workerId, modelId);
              setPickedLoadHash(null);
              await fetchUsage();
            });
          }}
        />

        <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
          <LoadSettingsPanel
            values={settings}
            onChange={setSettings}
            extras={extras}
            onExtras={setExtras}
            defaultAlias={fallbackAlias}
            supported={status?.supported_settings ?? []}
            buildTag={status?.active_build_tag ?? null}
            projectors={projectorFiles(worker)}
            draftModels={onMachine.filter((m) => m.id !== modelId).map((m) => ({ id: m.id, label: modelLabel(m) }))}
            cores={worker?.hardware?.cpu.cores ?? null}
            footer={
              <Btn kind="primary" onClick={load} disabled={offline || !modelId || pending?.type === "daemon_load" || (isThisModelLoaded && !needsReload && phase !== "failed" && phase !== "crashed")}>
                <LtIcon name="play" />
                {isThisModelLoaded ? "Reload with these settings" : "Load model"}
              </Btn>
            }
          />
          <SamplingPanel
            value={sampling}
            onChange={setSampling}
            defaults={status?.generation_defaults ?? null}
            footer={
              <div className="flex flex-wrap items-center gap-3">
                <Btn onClick={applySampling} disabled={!samplingDirty || !modelId}>
                  Apply
                </Btn>
                <span className="text-xs text-muted">{samplingDirty ? "Not applied yet." : "Saved for this model on this machine."}</span>
              </div>
            }
          />
        </div>
      </div>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <dt className="text-muted">{label}</dt>
      <dd className="m-0 font-mono text-fg">{value}</dd>
    </div>
  );
}
