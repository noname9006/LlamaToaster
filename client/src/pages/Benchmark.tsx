// The New test page: the optimization flow of
// docs/plans/OPTIMIZATION_FLOW_REDESIGN.md laid out as the v2 design's
// "New test" screen (docs/plans/app-v2.dc.html) -- pick a machine and a model,
// state the context you need, then:
//   1  Baseline        llama-server at that context, every layer on the GPU
//   2  Find the best fit  llama-fit-params map of context x layers, FA on/off,
//                      every supported K/V cache, optional real-load check
//   3  CPU threads     topology defaults, optional -t / -tb sweep
//   4  Confirm         the chosen configs measured like the baseline
// Each step is its own run, tied together by a flow id so the page restores
// from any device (?flow=). Custom Test stays beside it for exact flags.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import type { Model, Worker } from "../types";
import { isMtpDraftModel } from "../types";
import { LtIcon } from "../components/ltIcons";
import { Btn, Kicker, Notice, PageHeader, Seg, fmtGb } from "../components/optimize/ui";
import { useOptimizeFlow, isActive, type FlowRun } from "../components/optimize/useOptimizeFlow";
import type { FlowContext } from "../components/optimize/flowContext";
import { BaselineSection, ConfirmSection } from "../components/optimize/SpeedSections";
import { FitMapSection } from "../components/optimize/FitMapSection";
import { RealLoadSection } from "../components/optimize/RealLoadSection";
import { ThreadsSection, threadArgsFromPlan } from "../components/optimize/ThreadsSection";
import { speedRunView } from "../components/optimize/flowData";
import { machineKind, modelVerdict, stateLabel } from "../components/optimize/machine";
import { fitCtxStops } from "../../../shared/fitParams.js";
import { detectMoe } from "../../../shared/moePlacement.js";
import { buildFitSeries, ctxLabel, machineClassOf, proposeCandidates } from "../../../shared/optimizeFlow.js";
import {
  SPEED_DEFAULT_REPEATS,
  SPEED_MAX_REPEATS,
  SPEED_MIN_REPEATS,
  baselineCandidate,
  candidateKey,
  dedupeCandidates,
  type SpeedCandidate,
  type ThreadArgs,
} from "../../../shared/speedRun.js";
import { fallbackTopology } from "../../../shared/threadPlan.js";

// Shared with CustomTest.tsx: which machine/model you last looked at is one fact.
const LAST_WORKER_KEY = "llamatoaster:custom-test:last-worker";
const lastModelKey = (workerId: string) => `llamatoaster:custom-test:last-model:${workerId}`;
const repeatsKey = "llamatoaster:flow:repeats";
const ctxKey = (modelId: string) => `llamatoaster:flow:target-ctx:${modelId}`;
const threadsKey = (workerId: string) => `llamatoaster:flow:threads:${workerId}`;
const candidatesKey = (flowId: string) => `llamatoaster:flow:candidates:${flowId}`;

function readStore<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw == null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}
function writeStore(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private window -- nothing persists, the page still works */
  }
}

function modelLabelOf(m: Model): string {
  return (m.hf_file ?? m.filename ?? m.id).replace(/\.gguf$/i, "").split(/[\\/]/).pop() ?? m.id;
}

export function Benchmark() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlFlow = searchParams.get("flow");
  const { workers, order, loaded } = useWorkerStatuses();
  const [models, setModels] = useState<Model[]>([]);
  const [locations, setLocations] = useState<Record<string, string[]> | null>(null);
  // ?worker=&model= (the Models page's "map it" links) preselect the pairing.
  const [workerId, setWorkerId] = useState<string>(() => searchParams.get("worker") || readStore<string>(LAST_WORKER_KEY, "") || "");
  const [modelId, setModelId] = useState<string>(() => searchParams.get("model") ?? "");
  const [repeats, setRepeats] = useState<number>(() => readStore(repeatsKey, SPEED_DEFAULT_REPEATS));
  const [targetCtx, setTargetCtx] = useState<number | null>(null);
  const [mainGpu, setMainGpu] = useState<number | undefined>(undefined);

  useEffect(() => {
    void api.listModels().then(setModels).catch(() => undefined);
    void api
      .getModelLocations()
      .then((r) => setLocations(r.locations))
      .catch(() => setLocations({}));
  }, []);

  // URL edits read window.location, not the hook's searchParams: two edits in
  // one commit would otherwise each start from the same stale copy.
  useEffect(() => {
    if (!searchParams.has("worker") && !searchParams.has("model")) return;
    const next = new URLSearchParams(window.location.search);
    next.delete("worker");
    next.delete("model");
    setSearchParams(next, { replace: true });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // A link with ?flow= names its machine and model through the runs; adopt them.
  useEffect(() => {
    if (!urlFlow) return;
    void api
      .getFlow(urlFlow)
      .then(({ tests }) => {
        const t = tests[0];
        if (t?.worker_id) setWorkerId(t.worker_id);
        if (t?.model_id) setModelId(t.model_id);
      })
      .catch(() => undefined);
  }, [urlFlow]);

  // Default machine: the remembered one, else the first online.
  useEffect(() => {
    if (!loaded || workers.length === 0) return;
    if (workerId && workers.some((w) => w.id === workerId)) return;
    const first = workers.find((w) => w.status !== "offline") ?? workers[0];
    setWorkerId(first.id);
  }, [loaded, workers, workerId]);

  useEffect(() => {
    if (workerId) writeStore(LAST_WORKER_KEY, workerId);
  }, [workerId]);

  const worker = workers.find((w) => w.id === workerId);
  const presentModels = useMemo(() => {
    if (!locations) return [];
    return models.filter((m) => !isMtpDraftModel(m) && (workerId ? locations[m.id]?.includes(workerId) : (locations[m.id]?.length ?? 0) > 0));
  }, [models, locations, workerId]);

  useEffect(() => {
    if (!workerId || presentModels.length === 0) return;
    if (modelId && presentModels.some((m) => m.id === modelId)) return;
    if (urlFlow && modelId) return;
    const remembered = readStore<string>(lastModelKey(workerId), "");
    setModelId(presentModels.some((m) => m.id === remembered) ? remembered : presentModels[0].id);
  }, [workerId, presentModels, modelId, urlFlow]);

  useEffect(() => {
    if (workerId && modelId) writeStore(lastModelKey(workerId), modelId);
  }, [workerId, modelId]);
  useEffect(() => writeStore(repeatsKey, repeats), [repeats]);

  const model = models.find((m) => m.id === modelId);
  const trainedCtx = typeof model?.metadata.trained_ctx === "number" ? model.metadata.trained_ctx : null;
  const stops = useMemo(() => fitCtxStops(trainedCtx), [trainedCtx]);
  useEffect(() => {
    if (!modelId) return;
    const stored = readStore<number | null>(ctxKey(modelId), null);
    setTargetCtx(stored && stops.includes(stored) ? stored : stops[stops.length - 1]);
  }, [modelId, stops]);
  useEffect(() => {
    if (modelId && targetCtx) writeStore(ctxKey(modelId), targetCtx);
  }, [modelId, targetCtx]);

  const flow = useOptimizeFlow(workerId, modelId, urlFlow);

  // Picking another machine or model leaves the linked session behind.
  const dropUrlFlow = () => {
    if (!searchParams.has("flow")) return;
    const next = new URLSearchParams(window.location.search);
    next.delete("flow");
    setSearchParams(next, { replace: true });
  };

  // Keep ?flow= in the address bar once a session exists, so the link restores it.
  useEffect(() => {
    if (flow.flowId && flow.flowId !== urlFlow) {
      const next = new URLSearchParams(window.location.search);
      next.set("flow", flow.flowId);
      setSearchParams(next, { replace: true });
    }
  }, [flow.flowId]); // eslint-disable-line react-hooks/exhaustive-deps

  const topology = useMemo(() => worker?.hardware?.cpu_topology ?? fallbackTopology(worker?.hardware?.cpu.physical_cores ?? worker?.hardware?.cpu.cores ?? 4), [worker]);
  const [threads, setThreadsState] = useState<ThreadArgs | null>(null);
  useEffect(() => {
    if (!workerId) return;
    setThreadsState(readStore<ThreadArgs | null>(threadsKey(workerId), null) ?? threadArgsFromPlan({ topology }));
  }, [workerId, topology]);
  const setThreads = useCallback(
    (t: ThreadArgs) => {
      setThreadsState(t);
      if (workerId) writeStore(threadsKey(workerId), t);
    },
    [workerId]
  );

  const [candidates, setCandidatesState] = useState<SpeedCandidate[]>([]);
  useEffect(() => {
    setCandidatesState(flow.flowId ? readStore<SpeedCandidate[]>(candidatesKey(flow.flowId), []) : []);
  }, [flow.flowId]);
  const setCandidates = useCallback(
    (list: SpeedCandidate[]) => {
      const next = dedupeCandidates(list);
      setCandidatesState(next);
      if (flow.flowId) writeStore(candidatesKey(flow.flowId), next);
    },
    [flow.flowId]
  );

  const kind = worker ? machineKind(worker) : null;
  const machine = kind ? machineClassOf(kind) : "dgpu";
  const moe = useMemo(() => (model ? detectMoe(model.metadata.tensor_layer_bytes ?? null) : null), [model]);
  const fitPoints = flow.fit?.points ?? [];
  const layersTotal = fitPoints.find((p) => p.layers_total != null)?.layers_total ?? (typeof model?.metadata.n_layer === "number" ? model.metadata.n_layer + 1 : null);
  const activeBuild = worker?.installedBuilds.find((b) => b.active) ?? null;
  const caps = new Set(worker?.capabilities ?? []);
  const canFit = !!worker && caps.has("fit-map-v1") && !!activeBuild?.fit_params_path;
  const canFitReason = !worker
    ? "Pick a machine."
    : !caps.has("fit-map-v1")
      ? "This machine runs an older worker — update it to map context against layers."
      : !activeBuild
        ? "No llama.cpp build is installed on this machine yet — install one on the Machines page."
        : `llama.cpp ${activeBuild.tag} has no llama-fit-params — install a newer build on the Machines page.`;
  const gpuName = kind?.gpus[mainGpu ?? 0]?.model ?? null;
  const freeVramMib =
    worker?.vram?.vram_free_before_mib ?? kind?.gpus[mainGpu ?? 0]?.vram_usable_mb ?? kind?.gpus[mainGpu ?? 0]?.vram_mb ?? null;

  const ctx: FlowContext | null =
    worker && model && targetCtx
      ? {
          workerId,
          worker,
          model,
          modelLabel: modelLabelOf(model),
          modelFile: (model.hf_file ?? model.filename ?? `${model.id}.gguf`).split(/[\\/]/).pop()!,
          machine,
          moe,
          trainedCtx,
          targetCtx,
          repeats,
          mainGpu,
          gpuName,
          freeVramMib,
          ramTotalMib: worker.hardware?.mem_total_bytes ? Math.round(worker.hardware.mem_total_bytes / 1048576) : null,
          layersTotal,
          build: activeBuild?.tag ?? null,
          canFit,
          canFitReason,
          machineOffline: worker.status === "offline",
          topology,
          threads,
          trigger: flow.trigger,
        }
      : null;

  const baselineRun = flow.latest("baseline");
  const fitRun = flow.latest("fit");
  const realRun = flow.latest("real_load");
  const threadRun = flow.latest("threads");
  const confirmRun = flow.latest("confirm");
  const series = useMemo(() => buildFitSeries(fitPoints, flow.fit?.kv ?? []), [fitPoints, flow.fit?.kv]);

  // Once a fit map finishes, propose up to three candidates if none are picked yet.
  const fitDone = fitRun && !isActive(fitRun.test) && fitPoints.length > 0;
  useEffect(() => {
    if (!fitDone || !targetCtx || candidates.length > 0) return;
    const proposed = proposeCandidates({ series, targetCtx, moe, threads });
    if (proposed.length) setCandidates(proposed);
  }, [fitDone, fitRun?.test.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const candidateKeys = useMemo(() => new Set(candidates.map(candidateKey)), [candidates]);
  const addCandidate = (c: SpeedCandidate) => setCandidates([...candidates, { ...c, threads }]);
  // Threads chosen later apply to every candidate measured after.
  const confirmList = candidates.map((c) => ({ ...c, threads }));
  const threadPlacement = confirmList[0] ?? { ...baselineCandidate(), fa: "on" as const, threads };

  if (loaded && workers.length === 0) {
    return (
      <div>
        <PageHeader title="New test" />
        <Notice tone="warn">
          No machine is enrolled yet. <Link to="/workers" className="text-accent">Add one on the Machines page</Link> — the installer takes a minute.
        </Notice>
      </div>
    );
  }

  const steps: { label: string; state: string; done: boolean; active: boolean }[] = [
    stepState("Baseline", baselineRun),
    stepState("Fit map", fitRun),
    { label: "Threads", state: threadRun ? runState(threadRun) : threads && JSON.stringify(threads) !== JSON.stringify(threadArgsFromPlan({ topology })) ? "custom" : "defaults", done: true, active: !!threadRun && isActive(threadRun.test) },
    stepState("Confirm", confirmRun),
  ];
  const best = speedRunView(confirmRun);

  return (
    <div>
      <PageHeader
        title="New test"
        sub="Pick a machine and model, say how much context you need — the flow measures the default, maps what fits, and confirms the best config."
      />
      <div className="flex flex-wrap items-start gap-6">
        <div className="flex min-w-0 flex-[2_1_520px] flex-col gap-7">
          <fieldset className="m-0 border-0 p-0">
            <legend className="mb-2.5 p-0 font-display text-lg font-semibold">Machine</legend>
            <div role="radiogroup" aria-label="Machine" className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,210px),1fr))] gap-2">
              {order.map((id) => workers.find((w) => w.id === id)).filter((w): w is Worker => !!w).map((w) => {
                const on = w.id === workerId;
                const k = machineKind(w);
                return (
                  <button
                    key={w.id}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    onClick={() => {
                      if (w.id !== workerId) dropUrlFlow();
                      setWorkerId(w.id);
                      setMainGpu(undefined);
                    }}
                    className={`flex flex-col gap-0.5 border px-3 py-2.5 text-left ${on ? "border-accent bg-accent-tint" : "border-border bg-surface hover:border-accent"}`}
                  >
                    <span className="font-mono text-sm">{w.displayName}</span>
                    <span className="text-xs text-fg-2">
                      {k.hasGpu ? `${k.gpus[0]?.model ?? "GPU"}${k.gpus[0]?.vram_mb ? ` ${Math.round(k.gpus[0].vram_mb / 1024)} GB` : ""}` : w.hardware?.gpu.length ? "CPU backend · GPU not used" : "No GPU · CPU only"}
                      {k.unified ? " · unified" : ""}
                    </span>
                    <span className="font-mono text-xs text-muted">{stateLabel(w)}</span>
                  </button>
                );
              })}
            </div>
            {kind && kind.gpus.length > 1 && (
              <label className="mt-2 flex items-center gap-2 text-sm text-fg-2">
                GPU
                <select
                  value={mainGpu ?? ""}
                  onChange={(e) => setMainGpu(e.target.value === "" ? undefined : Number(e.target.value))}
                  className="h-10 border border-border-strong bg-well px-2 text-sm text-fg"
                >
                  <option value="">llama.cpp default (all GPUs)</option>
                  {kind.gpus.map((g, i) => (
                    <option key={i} value={i}>
                      {g.model}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </fieldset>

          <fieldset className="m-0 border-0 p-0">
            <legend className="mb-2.5 p-0 font-display text-lg font-semibold">Model</legend>
            {presentModels.length === 0 ? (
              <Notice>
                {locations == null ? "Loading the models on this machine…" : "No models on this machine yet."}{" "}
                <Link to="/models" className="text-accent">Manage models</Link>
              </Notice>
            ) : (
              <div role="radiogroup" aria-label="Model" className="flex flex-col border border-border">
                {presentModels.map((m) => {
                  const on = m.id === modelId;
                  const mTrained = typeof m.metadata.trained_ctx === "number" ? m.metadata.trained_ctx : null;
                  const v = modelVerdict(m, worker, Math.min(targetCtx ?? 32768, mTrained ?? Infinity));
                  const mo = detectMoe(m.metadata.tensor_layer_bytes ?? null);
                  return (
                    <button
                      key={m.id}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      onClick={() => {
                        if (m.id !== modelId) dropUrlFlow();
                        setModelId(m.id);
                      }}
                      className={`flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-l-2 border-b-border px-3.5 py-2.5 text-left last:border-b-0 ${
                        on ? "border-l-accent bg-surface-raised" : "border-l-transparent bg-surface hover:bg-surface-raised"
                      }`}
                    >
                      <span className="min-w-0 flex-[1_1_200px]">
                        <span className="block truncate text-sm font-medium">{modelLabelOf(m)}</span>
                        <span className="block font-mono text-xs text-muted">
                          {(m.size_bytes / 1073741824).toFixed(1)} GB
                          {typeof m.metadata.n_layer === "number" ? ` · ${m.metadata.n_layer} layers` : ""}
                          {typeof m.metadata.trained_ctx === "number" ? ` · ${ctxLabel(m.metadata.trained_ctx)} trained` : ""}
                          {mo.isMoe ? ` · MoE, ${mo.moeLayers.length} expert layers` : ""}
                        </span>
                      </span>
                      {v && (
                        <span className={`inline-flex items-center gap-1.5 text-sm ${v.tone}`}>
                          <LtIcon name={v.icon} size={16} />
                          {v.text}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
            <p className="mt-2 text-xs text-muted">
              Verdicts are a quick estimate at your target context; the fit map below replaces them with llama.cpp&apos;s own sizing.{" "}
              <Link to="/models" className="text-accent">Manage models</Link>
            </p>
          </fieldset>

          {model && targetCtx && (
            <fieldset className="m-0 border-0 p-0">
              <legend className="mb-2.5 p-0 font-display text-lg font-semibold">Goal</legend>
              <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,260px),1fr))] gap-x-5 gap-y-4">
                <div>
                  <Kicker className="mb-1.5">Target context</Kicker>
                  <Seg
                    label="Target context"
                    value={targetCtx}
                    onChange={setTargetCtx}
                    options={stops.map((s) => ({ value: s, label: ctxLabel(s) }))}
                  />
                  <div className="mt-1 text-xs text-muted">
                    {trainedCtx ? `Trained for ${ctxLabel(trainedCtx)}. ` : ""}Every speed test runs at this context.
                  </div>
                </div>
                <div>
                  <Kicker className="mb-1.5">Repeats per speed test</Kicker>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={SPEED_MIN_REPEATS}
                      max={SPEED_MAX_REPEATS}
                      value={repeats}
                      onChange={(e) => setRepeats(Number(e.target.value))}
                      aria-label="Repeats per speed test"
                      className="w-40"
                    />
                    <span className="font-mono text-sm">{repeats}</span>
                  </div>
                  <div className="mt-1 text-xs text-muted">
                    Fewer = faster, more = more accurate.{repeats === 1 ? " One run measures no spread." : ""}
                  </div>
                </div>
              </div>
              {moe?.isMoe && (
                <p className="mt-3 inline-flex items-center gap-2 border border-border px-2.5 py-1 font-mono text-xs text-fg-2">
                  <LtIcon name="box" size={14} />
                  Mixture of Experts
                  {fitPoints[0]?.n_expert ? ` · ${fitPoints[0].n_expert} experts, ${fitPoints[0].n_expert_used} active` : ""} · {moe.moeLayers.length} MoE layers
                </p>
              )}
            </fieldset>
          )}

          {ctx && (
            <>
              <BaselineSection ctx={ctx} run={baselineRun} />
              <div>
                <FitMapSection
                  ctx={ctx}
                  fitRun={fitRun}
                  fit={flow.fit}
                  onTrigger={async (payload) => {
                    await ctx.trigger("fit", payload);
                  }}
                  onAddCandidate={addCandidate}
                  candidateKeys={candidateKeys}
                />
                {series.length > 0 && fitRun && !isActive(fitRun.test) && machine === "dgpu" && (
                  <div className="border border-t-0 border-border bg-surface px-4 pb-4 sm:px-[18px]">
                    <RealLoadSection ctx={ctx} series={series} run={realRun} onAddCandidate={addCandidate} />
                  </div>
                )}
              </div>
              {threads && <ThreadsSection ctx={ctx} value={threads} onChange={setThreads} placement={threadPlacement} run={threadRun} />}
              <ConfirmSection
                ctx={ctx}
                candidates={confirmList}
                onRemove={(key) => setCandidates(candidates.filter((c) => candidateKey({ ...c, threads }) !== key))}
                run={confirmRun}
                baselineRun={baselineRun}
              />
            </>
          )}
        </div>

        <aside aria-labelledby="h-summary" className="sticky top-14 flex-[1_1_280px] border border-border bg-surface px-[18px] py-4">
          <h2 id="h-summary" className="mb-3 font-display text-lg font-semibold">Summary</h2>
          <dl className="m-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3.5 gap-y-1.5 text-sm">
            <dt className="text-muted">Machine</dt>
            <dd className="m-0 truncate font-mono">{worker?.displayName ?? "—"}</dd>
            <dt className="text-muted">Model</dt>
            <dd className="m-0 truncate">{model ? modelLabelOf(model) : "—"}</dd>
            <dt className="text-muted">Context</dt>
            <dd className="m-0 font-mono">{targetCtx ? ctxLabel(targetCtx) : "—"}</dd>
            <dt className="text-muted">Memory</dt>
            <dd className="m-0">{machine === "dgpu" ? `discrete GPU${freeVramMib ? ` · ${fmtGb(freeVramMib)} free` : ""}` : machine === "unified" ? "unified pool" : "CPU only"}</dd>
            <dt className="text-muted">Repeats</dt>
            <dd className="m-0 font-mono">{repeats}</dd>
          </dl>
          <ol className="mt-4 flex list-none flex-col border-t border-border p-0">
            {steps.map((s, i) => (
              <li key={s.label} className="grid grid-cols-[24px_minmax(0,1fr)_auto] items-baseline gap-2.5 border-b border-border py-2">
                <span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
                <span className="font-medium">{s.label}</span>
                <span className={`font-mono text-xs ${s.active ? "text-accent" : s.done ? "text-fg-2" : "text-muted"}`}>{s.state}</span>
              </li>
            ))}
          </ol>
          {best && <p className="mt-3 text-sm text-fg-2">Results are in step 4 — the recommended config has a copyable command line.</p>}
          {flow.error && <p className="mt-3 text-xs text-danger">{flow.error}</p>}
          <div className="mt-4 flex flex-wrap gap-2">
            {flow.flowId && (
              <Btn
                kind="ghost"
                onClick={() => {
                  flow.reset();
                  const next = new URLSearchParams(window.location.search);
                  next.delete("flow");
                  setSearchParams(next, { replace: true });
                }}
                disabled={flow.anyActive}
              >
                Start over
              </Btn>
            )}
          </div>
          <p className="mt-3 border-t border-border pt-3 text-sm text-fg-2">
            Need exact flags? <Link to="/custom-test" className="text-accent">Open custom test</Link>
          </p>
        </aside>
      </div>
    </div>
  );
}

function runState(run: FlowRun): string {
  switch (run.test.status) {
    case "scheduled":
      return "queued";
    case "running":
      return "running";
    case "done":
      return "done";
    case "partial":
      return "partly done";
    case "failed":
      return "failed";
    case "cancelled":
      return "stopped";
  }
}

function stepState(label: string, run: FlowRun | null) {
  return { label, state: run ? runState(run) : "not run", done: !!run && run.test.status === "done", active: !!run && isActive(run.test) };
}
