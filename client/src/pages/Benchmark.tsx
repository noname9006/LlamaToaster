// The New test page: the optimization flow as seven steps, one on screen at
// a time (the v3 design on the UX review canvas):
//   1 Machine   2 Model   3 Fit (llama-fit-params at the target context)
//   4 Baseline  5 Context and offload (FA on/off map, real loads, threads)
//   6 KV cache  (KV map + speed of the best sizes)   7 Batch sizes (-b/-ub)
// Each measuring step is its own run, tied together by a flow id so the page
// restores from any device (?flow=); ?step= names the open step. Each step's
// winner is read back from its finished run and carried into the next.
// Custom Test stays beside it for exact flags.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import type { Model, Worker } from "../types";
import { isMtpDraftModel } from "../types";
import { Btn, Notice, PageHeader } from "../components/optimize/ui";
import { useOptimizeFlow, isActive } from "../components/optimize/useOptimizeFlow";
import type { FlowContext } from "../components/optimize/flowContext";
import { carryForward } from "../components/optimize/SpeedSections";
import { threadArgsFromPlan } from "../components/optimize/ThreadsSection";
import { speedRunView } from "../components/optimize/flowData";
import { estimateFitAnswer, machineKind, modelVerdict } from "../components/optimize/machine";
import { useFitChecks } from "../components/optimize/useFitChecks";
import { FlowStepper } from "../components/optimize/FlowStepper";
import {
  BaselineStep,
  BatchStep,
  FitStep,
  KvStep,
  MachineStep,
  ModelStep,
  OffloadStep,
  StepNav,
  machineReady,
  modelLabelOf,
} from "../components/optimize/FlowSteps";
import { defaultStep, isStepId, stepStates, type StepId } from "../components/optimize/steps";
import { fitCtxStops } from "../../../shared/fitParams.js";
import { detectMoe } from "../../../shared/moePlacement.js";
import { buildFitSeries, ctxLabel, machineClassOf, proposeCandidates } from "../../../shared/optimizeFlow.js";
import {
  SPEED_DEFAULT_REPEATS,
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
const DEFAULT_TARGET_CTX = 32768;

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

  // The list's verdict context: the target, never past the model's maximum context.
  const verdictCtx = useCallback(
    (m: Model) => Math.min(targetCtx ?? 32768, typeof m.metadata.trained_ctx === "number" ? m.metadata.trained_ctx : Infinity),
    [targetCtx]
  );
  const fitChecks = useFitChecks(worker, presentModels, verdictCtx);

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
    // No choice yet: 32k, a common working size, rather than the model's
    // maximum (which on most GPUs reads as "doesn't fit" for every model).
    setTargetCtx(stored && stops.includes(stored) ? stored : stops.includes(DEFAULT_TARGET_CTX) ? DEFAULT_TARGET_CTX : stops[stops.length - 1]);
  }, [modelId, stops]);
  useEffect(() => {
    if (modelId && targetCtx) writeStore(ctxKey(modelId), targetCtx);
  }, [modelId, targetCtx]);

  const flow = useOptimizeFlow(workerId, modelId, urlFlow);

  // A restored session keeps the context its runs measured at.
  const sessionCtx = useMemo(() => {
    for (const r of [...flow.runs].reverse()) {
      const c = r.test.config as { speed_run?: { target_ctx?: number }; fit_map?: { target_ctx?: number | null } };
      const t = c.speed_run?.target_ctx ?? c.fit_map?.target_ctx;
      if (t) return t;
    }
    return null;
  }, [flow.runs]);
  useEffect(() => {
    if (sessionCtx && stops.includes(sessionCtx)) setTargetCtx(sessionCtx);
  }, [sessionCtx, stops]);

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

  const kvFitRun = flow.latest("kv_fit");
  const kvRun = flow.latest("kv_confirm");
  const batchRun = flow.latest("batch");
  // An older session mapped KV inside step 5's fit map; step 6 shows that.
  const kvFitData = flow.kvFit ?? (flow.fit && flow.fit.kv.length > 0 ? flow.fit : null);
  const kvFitRunShown = kvFitRun ?? (flow.kvFit == null && flow.fit && flow.fit.kv.length > 0 ? fitRun : null);

  // Winners, read back from finished runs and carried forward.
  const confirmView = speedRunView(confirmRun);
  const placementPick = confirmView && !isActive(confirmRun!.test) ? carryForward(confirmView) : null;
  const placementWinner = placementPick?.candidate ?? null;
  const baselineDone = speedRunView(baselineRun)?.rows[0]?.outcome.status === "done";
  const placement: SpeedCandidate | null = placementWinner ? { ...placementWinner } : baselineDone ? baselineCandidate() : null;
  const kvView = speedRunView(kvRun);
  const kvWinner = kvView && !isActive(kvRun!.test) ? carryForward(kvView, "the step 5 config")?.candidate ?? null : null;
  const batchView = speedRunView(batchRun);
  const batchWinner = batchView && !isActive(batchRun!.test) ? carryForward(batchView, "the default batch sizes")?.candidate ?? null : null;
  const kvBase = placement;
  const batchBase = kvWinner ?? placement;

  const verdictFor = (m: Model) => {
    const c = verdictCtx(m);
    return modelVerdict(m, worker, c, fitChecks.answer(m.id, c));
  };
  // llama-fit-params' answer, else the header estimate the verdict also uses.
  const fitAnswer =
    model && targetCtx
      ? fitChecks.answer(model.id, verdictCtx(model)) ?? (freeVramMib != null ? estimateFitAnswer(model, freeVramMib, verdictCtx(model)) : null)
      : null;

  const states = stepStates({
    hasMachine: !!worker,
    machineReady: machineReady(worker),
    hasModel: !!model,
    hasTargetCtx: !!targetCtx,
    fitAnswered: !!(model && verdictFor(model)),
    baselineDone,
    hasPlacement: !!placement,
    placementDone: !!placementWinner,
    kvDone: !!kvWinner,
    batchDone: !!batchWinner,
    hasRuns: { kv: !!kvRun || !!kvFitRun, batch: !!batchRun },
  });
  const urlStep = searchParams.get("step");
  const requested: StepId | null = isStepId(urlStep) ? urlStep : null;
  const step: StepId = requested && !states.find((s) => s.id === requested)?.locked ? requested : defaultStep(states);
  const goTo = (id: StepId) => {
    const next = new URLSearchParams(window.location.search);
    next.set("step", id);
    setSearchParams(next);
    window.scrollTo?.({ top: 0 });
  };

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

  const using = [
    worker?.displayName,
    model ? modelLabelOf(model) : null,
    targetCtx ? ctxLabel(targetCtx) : null,
    placementWinner ? `${placementWinner.label}${placementPick?.spilled ? " (spilled)" : ""}` : null,
    kvWinner && (kvWinner.ctk !== "f16" || kvWinner.ctv !== "f16") ? `${kvWinner.ctk} / ${kvWinner.ctv} cache` : null,
    batchWinner && batchWinner.ub != null ? `-b ${batchWinner.b} -ub ${batchWinner.ub}` : null,
  ].filter(Boolean);
  const orderedWorkers = order.map((id) => workers.find((w) => w.id === id)).filter((w): w is Worker => !!w);
  const needCtx = <Notice>Pick a machine, a model and a target context first.</Notice>;

  return (
    <div>
      <PageHeader
        title="New test"
        sub={using.length ? <span className="text-sm">Using: {using.join(" · ")}</span> : "Find the fastest settings for one model on one machine."}
        actions={
          <>
            {flow.flowId && (
              <Btn
                kind="ghost"
                onClick={() => {
                  flow.reset();
                  const next = new URLSearchParams(window.location.search);
                  next.delete("flow");
                  next.delete("step");
                  setSearchParams(next, { replace: true });
                }}
                disabled={flow.anyActive}
              >
                Start over
              </Btn>
            )}
            <Link to="/custom-test" className="inline-flex min-h-10 items-center px-2 text-sm text-accent hover:text-accent-hover">
              Custom test
            </Link>
          </>
        }
      />
      <FlowStepper states={states} current={step} onPick={goTo} />
      {flow.error && (
        <div className="mb-4">
          <Notice tone="fail">{flow.error}</Notice>
        </div>
      )}

      {step === "machine" && (
        <MachineStep
          workers={orderedWorkers}
          workerId={workerId}
          onPick={(id) => {
            if (id !== workerId) dropUrlFlow();
            setWorkerId(id);
            setMainGpu(undefined);
          }}
          mainGpu={mainGpu}
          onMainGpu={setMainGpu}
        />
      )}
      {step === "model" && (
        <ModelStep
          models={presentModels}
          modelId={modelId}
          onPick={(id) => {
            if (id !== modelId) dropUrlFlow();
            setModelId(id);
          }}
          loading={locations == null}
          verdictOf={verdictFor}
          workerName={worker?.displayName ?? "this machine"}
        />
      )}
      {step === "fit" &&
        (model && targetCtx ? (
          <FitStep
            model={model}
            stops={stops}
            targetCtx={targetCtx}
            onCtx={setTargetCtx}
            trainedCtx={trainedCtx}
            answer={fitAnswer}
            verdict={verdictFor(model)}
            pending={fitChecks.pending}
            freeVramMib={freeVramMib}
            ramTotalMib={ctx?.ramTotalMib ?? null}
            repeats={repeats}
            onRepeats={setRepeats}
            machine={machine}
          />
        ) : (
          needCtx
        ))}
      {step === "baseline" && (ctx ? <BaselineStep ctx={ctx} run={baselineRun} /> : needCtx)}
      {step === "offload" &&
        (ctx ? (
          <OffloadStep
            ctx={ctx}
            fit={flow.fit}
            fitRun={fitRun}
            realRun={realRun}
            confirmRun={confirmRun}
            baselineRun={baselineRun}
            threadRun={threadRun}
            threads={threads}
            onThreads={setThreads}
            candidates={confirmList}
            candidateKeys={candidateKeys}
            onAddCandidate={addCandidate}
            onRemoveCandidate={(key) => setCandidates(candidates.filter((c) => candidateKey({ ...c, threads }) !== key))}
            series={series}
          />
        ) : (
          needCtx
        ))}
      {step === "kv" &&
        (ctx && kvBase ? (
          <KvStep ctx={ctx} kvFit={kvFitData} kvFitRun={kvFitRunShown} kvRun={kvRun} base={kvBase} candidateKeys={candidateKeys} />
        ) : (
          needCtx
        ))}
      {step === "batch" && (ctx && batchBase ? <BatchStep ctx={ctx} run={batchRun} base={batchBase} /> : needCtx)}

      <StepNav current={step} states={states} onPick={goTo} />
    </div>
  );
}
