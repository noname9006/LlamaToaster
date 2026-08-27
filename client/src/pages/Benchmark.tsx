// BENCHMARKING_PLAN_V8.md's benchmark console -- the page
// docs/benchmark-page-mockup-v8.html renders as screen 1 ("Test A -- the goal
// is captured before the grid is built") and screen 4's "Start Test B".
//
// The difference from NewRun.tsx (which stays, unchanged, at /new-run) is
// WHO builds the grid. There, the user builds one flat cross-product by
// hand and every trigger is a standalone root. Here the user states intent
// -- model, machine, the M2 questionnaire, a repeat count -- and the page
// derives the three-stage chain §0.5 defines (tuning -> refine -> sweep),
// each stage a real run linked to the previous one by `parent_run_id`, which
// is what makes the server resolve them to one `root_run_id` and score them
// as one universe (repo.ts's listChainScoringRuns).
//
// Two properties are load-bearing here and are why nothing on this page
// shows a hardcoded number the way the mockup's static HTML does:
//
//  * M4's exit criterion -- the count a stage advertises is
//    expandSweep(thatStage).length, the same function the server pre-creates
//    run_items with and the worker iterates. Tolerance pruning happens at
//    grid-BUILD time, so the count, the priced ETA and what actually runs
//    can never disagree.
//  * §0.3's scoring universe is `kind = 'sweep'` (or NULL) only. Tuning and
//    refine feed the sweep its batch/ubatch values; they never produce cards
//    of their own. That is exactly why the chain strip renders "cards per
//    goal" as a fourth, dashed tile after the third stage rather than beside
//    it.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { ModelPicker } from "../components/ModelPicker";
import { GoalQuestionnaire } from "../components/GoalQuestionnaire";
import { backendVisibleGpus } from "../types";
import type { Model, Run, ResultRow, RunKind, SweepConfig } from "../types";
import { formatBytes, formatGpuLabel } from "../utils";
import { estimateSafeNgl, estimateVramNeededMib, maxAffordableContext } from "../vramEstimate";
import {
  defaultGoals,
  goalsEqualDefaults,
  normalizeGoals,
  pruneCacheTypes,
  ensureUnquantizedPairSurvives,
  recommendedKvGrid,
  DEFAULT_RECOMMENDED_KV_PAIRS,
  WORKLOAD_WEIGHTS,
  type GoalsConfig,
} from "../goals";
import { expandSweep } from "../../../shared/sweep";
import { priceMatrix, ETA_UNAVAILABLE } from "../../../shared/pricing";
import type { ModelRatesResponse } from "../types";

type Sweep = Omit<SweepConfig, "model_id">;

// The three chain stages, in the order §0.5 links them. "runtime", "probe"
// and "quality" are other kinds entirely (N1/N2/N4) and are triggered from
// their own surfaces, not from this strip.
const STAGES = ["tuning", "refine", "sweep"] as const;
type StageKind = (typeof STAGES)[number];

const STAGE_TITLE: Record<StageKind, string> = {
  tuning: "Tune coarse",
  refine: "Tune refine",
  sweep: "Sweep",
};

const STAGE_START_LABEL: Record<StageKind, string> = {
  tuning: "Start Test A",
  refine: "Start Test B",
  sweep: "Start Sweep",
};

// P0.5 -- below 3 repeats llama-bench reports a standard deviation of
// exactly 0, so §0.3's stability gate would pass on a number that was never
// measured. 1 and 2 stay VISIBLE and disabled (with the reason) rather than
// being dropped from the row: the mockup's own posture, and the same one
// GoalQuestionnaire's pruned KV chips take.
const REPEAT_CHOICES = [1, 2, 3, 5, 10] as const;
const MIN_SCORING_REPEATS = 3;

const NGL_FALLBACK_MAX = 999;
const OUTPUT_LAYER = 1;

// Terminal in the sense that matters here: the stage will produce no further
// results, so the next stage may be derived from what it did produce. A
// "partial" stage still tuned something; a failed/cancelled one hasn't.
const TERMINAL: ReadonlySet<string> = new Set(["done", "partial", "failed", "cancelled"]);

// --- persistence -----------------------------------------------------------
//
// The pairing keys are deliberately the SAME ones NewRun.tsx writes: which
// machine and model you were last looking at is one fact about the session,
// not one per page, and a user moving between the two surfaces should not
// have to re-pick. The goals key is shared for the same reason plus M5's:
// goals describe the workload, not the page that captured them.

const LAST_WORKER_STORAGE_KEY = "llamatoaster:new-run:last-worker";

function lastModelStorageKey(workerId: string): string {
  return `llamatoaster:new-run:last-model:${workerId}`;
}

function goalsStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:goals:${modelId}:${workerId}`;
}

// The chain, on the other hand, is this page's own: which three run ids the
// current pairing's stages resolved to, so a reload (or coming back an hour
// later while the sweep is still queued) picks the strip back up instead of
// offering to start a chain that already exists.
function chainStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:benchmark:chain:${modelId}:${workerId}`;
}

const PRESETS_STORAGE_KEY = "llamatoaster:benchmark:presets";
const REPEATS_STORAGE_KEY = "llamatoaster:benchmark:repeats";

type ChainState = Partial<Record<StageKind, string>>;

// M5 -- a preset here carries INTENT (goals + repeats) and nothing machine-
// specific: no worker id, no ngl, no layer counts. That is the whole reason
// it can load verbatim on any machine, and it is also why this page has no
// "preset grid" concept at all -- the grid is derived from the intent.
interface BenchPreset {
  goals: GoalsConfig;
  repeats: number;
}

function readJson<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* localStorage unavailable (private browsing, quota) -- never worth failing a run over */
  }
}

// --- grid derivation -------------------------------------------------------

interface StageInputs {
  /** Prompt size every stage prefills; anchored to Q2's target when stated. */
  ppTokens: number;
  nGen: number;
  threads: number;
  ngl: number;
  cpuMoe: number;
  repeats: number;
  goals: GoalsConfig;
  /** Winning (batch, ubatch) from the previous stage; null until it lands. */
  tuned: { batch: number; ubatch: number } | null;
  noGpu: boolean;
}

// Every stage is a full SweepConfig, because that is what the server
// validates and the worker executes -- there is no second "stage" notion
// anywhere below this page. What differs per stage is which axes carry more
// than one value.
function baseSweep(input: StageInputs): Sweep {
  return {
    n_prompt: [input.ppTokens],
    n_gen: [input.nGen],
    n_depth: [0],
    threads: [input.threads],
    n_gpu_layers: [input.ngl],
    batch_size: [2048],
    ubatch_size: [512],
    cache_type_k: ["f16"],
    cache_type_v: ["f16"],
    flash_attn: ["on"],
    mtp: ["off"],
    n_gpu_layers_draft: [0],
    n_cpu_moe: [input.cpuMoe],
    repeats: input.repeats,
  };
}

// Test A. PP only (n_gen 0), batch x ubatch, everything else held. Only
// ubatch values that are actually <= every batch value are offered: a
// cross-product cannot express the triangular b >= ub constraint, and an
// item that violates it is a whole process spawned to measure a silently
// clamped configuration -- the same reasoning shared/sweep.ts's isValidCombo
// applies to quantized-KV-without-flash-attention.
const COARSE_BATCHES = [1024, 2048, 4096];
const COARSE_UBATCHES = [128, 256, 512, 1024];

function tuningSweep(input: StageInputs): Sweep {
  return {
    ...baseSweep(input),
    n_gen: [0],
    batch_size: COARSE_BATCHES,
    ubatch_size: COARSE_UBATCHES.filter((u) => u <= Math.min(...COARSE_BATCHES)),
  };
}

// Test B. The mockup's "rules, not values yet": the neighbourhood is defined
// here as one octave either side of whatever Test A actually won, so this
// stage has no values at all until that result exists.
function refineSweep(input: StageInputs): Sweep {
  const coarse = tuningSweep(input);
  if (!input.tuned) return { ...coarse, batch_size: [], ubatch_size: [] };
  const batches = [...new Set([input.tuned.batch / 2, input.tuned.batch, input.tuned.batch * 2])]
    .filter((b) => Number.isInteger(b) && b >= 32 && b <= 8192)
    .sort((a, b) => a - b);
  const smallestBatch = batches.length > 0 ? batches[0] : input.tuned.batch;
  const ubatches = [...new Set([input.tuned.ubatch / 2, input.tuned.ubatch, input.tuned.ubatch * 2])]
    .filter((u) => Number.isInteger(u) && u >= 32 && u <= smallestBatch)
    .sort((a, b) => a - b);
  return {
    ...coarse,
    batch_size: batches.length > 0 ? batches : [input.tuned.batch],
    // Every candidate can exceed the smallest batch (a winner whose ubatch
    // already equals its batch) -- pin to that batch rather than emitting a
    // clamped item.
    ubatch_size: ubatches.length > 0 ? ubatches : [smallestBatch],
  };
}

// Test C. The scored stage: KV x FA x ngl x depth at the tuned batch/ubatch,
// with n_prompt AND n_gen both non-zero so llama-bench emits a pp row and a
// tg row per configuration -- §0.3 rejects a tuple that has only one of the
// two under `missing_pp_or_tg`.
function sweepStageSweep(input: StageInputs): Sweep {
  const base = baseSweep(input);
  const tolerance = input.goals.kv_tolerance ?? "q4_0_ok";
  const recommended = recommendedKvGrid(tolerance);
  // Same primitives NewRun.tsx prunes with, in the same order: prune to the
  // tolerance, then re-add an unquantized pair if the tolerance took the
  // last one (M4's inviolable rule -- flash-attention-off needs something to
  // vary against).
  const prunedK = pruneCacheTypes([...new Set(recommended.map(([k]) => k))], tolerance);
  const prunedV = pruneCacheTypes([...new Set(recommended.map(([, v]) => v))], tolerance);
  const { cache_type_k, cache_type_v } = ensureUnquantizedPairSurvives(
    prunedK,
    prunedV,
    DEFAULT_RECOMMENDED_KV_PAIRS,
    true
  );

  // Two offload points, not one: -ngl is an axis in this stage per the
  // mockup, and the second point is what a Low Memory card is scored from.
  // A GPU-less worker keeps the single 0 the backend forces anyway.
  const nglAxis = input.noGpu
    ? [0]
    : [...new Set([input.ngl, Math.floor((input.ngl * 2) / 3)])].filter((v) => v > 0).sort((a, b) => b - a);

  // §0.2/M7 -- depth is anchored to Q2's stated target, never to a house
  // number. Without a target there is nothing to take a percentage OF, so
  // the axis stays at 0 rather than inventing a depth.
  const target = input.goals.target_ctx;
  const depthAxis = target != null && target > 0 ? [0, Math.round(target * 0.5)] : [0];

  return {
    ...base,
    batch_size: [input.tuned?.batch ?? 2048],
    ubatch_size: [input.tuned?.ubatch ?? 512],
    cache_type_k,
    cache_type_v,
    flash_attn: ["on", "off"],
    n_gpu_layers: nglAxis.length > 0 ? nglAxis : [0],
    n_depth: depthAxis,
  };
}

function sweepForStage(stage: StageKind, input: StageInputs): Sweep {
  if (stage === "tuning") return tuningSweep(input);
  if (stage === "refine") return refineSweep(input);
  return sweepStageSweep(input);
}

// The pp winner of a finished stage -- the only thing a tuning stage hands
// downstream. Rows with a zero rate never won anything; they are failures
// that still ingested a row.
function bestPpPlacement(results: ResultRow[]): { batch: number; ubatch: number } | null {
  const pp = results.filter((r) => r.test_type === "pp" && r.avg_tps > 0);
  if (pp.length === 0) return null;
  const best = pp.reduce((a, b) => (b.avg_tps > a.avg_tps ? b : a));
  return { batch: best.batch_size, ubatch: best.ubatch_size };
}

// The KV pairs the sweep stage will ACTUALLY expand to. Rendered instead of
// the axis values because the cross-product of two pruned axes is a superset
// of the recommended pair list, and M4's whole point is that the page shows
// what runs rather than something the count has to be reconciled against.
function expandedKvPairs(sweep: Sweep): string[] {
  const pairs = new Set<string>();
  for (const item of expandSweep(sweep)) pairs.add(`${item.cache_type_k} / ${item.cache_type_v}`);
  return [...pairs];
}

export function Benchmark() {
  const [models, setModels] = useState<Model[]>([]);
  const [locations, setLocations] = useState<Record<string, string[]> | null>(null);
  const [unreachableLocationWorkers, setUnreachableLocationWorkers] = useState<string[]>([]);
  const [modelId, setModelId] = useState("");
  const [workerId, setWorkerId] = useState("");
  const [selectedGpuRawIndex, setSelectedGpuRawIndex] = useState<number | undefined>(undefined);
  const { order: workerOrder, status: workerStatus } = useWorkerStatuses();

  const [goals, setGoals] = useState<GoalsConfig>(defaultGoals());
  const [goalsUnset, setGoalsUnset] = useState(true);
  const [repeats, setRepeats] = useState<number>(() => readJson<number>(REPEATS_STORAGE_KEY) ?? MIN_SCORING_REPEATS);
  const [presets, setPresets] = useState<Record<string, BenchPreset>>(
    () => readJson<Record<string, BenchPreset>>(PRESETS_STORAGE_KEY) ?? {}
  );
  const [presetsOpen, setPresetsOpen] = useState(false);

  const [chain, setChain] = useState<ChainState>({});
  const [stageData, setStageData] = useState<Record<string, { run: Run; results: ResultRow[] }>>({});
  const [busyStage, setBusyStage] = useState<StageKind | null>(null);
  const [msg, setMsg] = useState("");

  const [rates, setRates] = useState<ModelRatesResponse | null>(null);

  useEffect(() => {
    (async () => {
      setModels(await api.listModels());
    })();
    (async () => {
      try {
        const res = await api.getModelLocations();
        setLocations(res.locations);
        setUnreachableLocationWorkers(res.unreachable);
      } catch {
        /* best-effort -- the picker just offers nothing until this resolves */
      }
    })();
  }, []);

  // Same strictness NewRun.tsx settled on: only models a worker has
  // CONFIRMED on disk right now are offered. An unreachable worker offers
  // nothing rather than a remembered guess.
  const presentModels = useMemo(() => {
    if (!locations) return [];
    if (workerId) return models.filter((m) => locations[m.id]?.includes(workerId));
    return models.filter((m) => (locations[m.id]?.length ?? 0) > 0);
  }, [models, locations, workerId]);

  useEffect(() => {
    if (workerId || workerOrder.length === 0) return;
    const remembered = (() => {
      try {
        return localStorage.getItem(LAST_WORKER_STORAGE_KEY);
      } catch {
        return null;
      }
    })();
    if (remembered && workerOrder.includes(remembered)) setWorkerId(remembered);
    else if (workerOrder.length === 1) setWorkerId(workerOrder[0]);
  }, [workerOrder, workerId]);

  useEffect(() => {
    if (!workerId) return;
    try {
      localStorage.setItem(LAST_WORKER_STORAGE_KEY, workerId);
    } catch {
      /* see writeJson */
    }
  }, [workerId]);

  // A GPU index means nothing across machines -- see NewRun.tsx's own reset.
  useEffect(() => {
    setSelectedGpuRawIndex(undefined);
  }, [workerId]);

  const restoredModelForRef = useRef<string | null>(null);
  useEffect(() => {
    if (!workerId || presentModels.length === 0) return;
    if (restoredModelForRef.current === workerId) return;
    restoredModelForRef.current = workerId;
    if (modelId) return;
    const remembered = readJson<{ modelId: string }>(lastModelStorageKey(workerId));
    if (remembered && presentModels.some((m) => m.id === remembered.modelId)) setModelId(remembered.modelId);
  }, [workerId, presentModels, modelId]);

  useEffect(() => {
    if (!workerId || !modelId) return;
    writeJson(lastModelStorageKey(workerId), { modelId, mtpModelId: "" });
  }, [workerId, modelId]);

  // Goals and the chain both belong to the PAIRING, so both reload when it
  // changes. M5's "unset" marker survives: a pairing with no stored goals
  // shows defaults, marked, and fabricates no answers.
  useEffect(() => {
    if (!modelId || !workerId) {
      setGoals(defaultGoals());
      setGoalsUnset(true);
      setChain({});
      return;
    }
    const stored = readJson<unknown>(goalsStorageKey(modelId, workerId));
    const restored = stored ? normalizeGoals(stored) : null;
    setGoals(restored ?? defaultGoals());
    setGoalsUnset(restored == null);
    setChain(readJson<ChainState>(chainStorageKey(modelId, workerId)) ?? {});
  }, [modelId, workerId]);

  useEffect(() => {
    writeJson(REPEATS_STORAGE_KEY, repeats);
  }, [repeats]);

  const persistChain = useCallback(
    (next: ChainState) => {
      setChain(next);
      if (modelId && workerId) writeJson(chainStorageKey(modelId, workerId), next);
    },
    [modelId, workerId]
  );

  // §0.6 -- ETAs price from a measured rate or not at all; the label travels
  // with the number, which is why nothing here formats a duration itself.
  useEffect(() => {
    let cancelled = false;
    setRates(null);
    if (!modelId || !workerId) return;
    api
      .getModelRates(modelId, workerId)
      .then((r) => {
        if (!cancelled) setRates(r);
      })
      .catch(() => {
        /* advisory -- an absent rate renders "ETA unavailable", never a guess */
      });
    return () => {
      cancelled = true;
    };
  }, [modelId, workerId]);

  // Poll every stage run this pairing knows about. Self-rescheduling on the
  // same 5s cadence as useWorkerStatuses/Runs.tsx, and it stops asking once
  // every known stage is terminal -- a finished chain is not a reason to
  // keep a timer alive.
  const chainIds = useMemo(() => STAGES.map((s) => chain[s]).filter((id): id is string => Boolean(id)), [chain]);
  useEffect(() => {
    if (chainIds.length === 0) return;
    let cancelled = false;
    let timer: number | undefined;
    async function poll() {
      const next: Record<string, { run: Run; results: ResultRow[] }> = {};
      let anyLive = false;
      for (const id of chainIds) {
        try {
          const res = await api.getRun(id);
          next[id] = { run: res.run, results: res.results };
          if (!TERMINAL.has(res.run.status)) anyLive = true;
        } catch {
          /* a deleted/unknown run simply stops contributing to the strip */
        }
      }
      if (cancelled) return;
      setStageData(next);
      if (anyLive) timer = window.setTimeout(poll, 5000);
    }
    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [chainIds]);

  // --- machine + model context ---------------------------------------------

  const selectedModel = models.find((m) => m.id === modelId);
  const selectedWorker = workerId ? workerStatus[workerId] : undefined;
  const workerHardware = selectedWorker?.hardware ?? undefined;
  const workerDefaultBackend = selectedWorker?.backend ?? undefined;
  const noGpu = workerHardware ? workerHardware.gpu.length === 0 : workerDefaultBackend === "cpu";
  const gpuList = workerHardware?.gpu ?? [];
  // Unlike NewRun.tsx, this page offers only GPUs the worker's CURRENT
  // backend can actually see. Switching a machine's backend for one run is a
  // deliberate expert action with its own confirmation flow over there; a
  // console whose whole job is deriving a chain should not be where that
  // decision gets made.
  const visibleGpus = workerDefaultBackend ? backendVisibleGpus(gpuList, workerDefaultBackend) : gpuList;
  const activeBuild = selectedWorker?.installedBuilds?.find((b) => b.active)?.tag ?? null;
  const sensors = selectedWorker?.sensors ?? null;

  const modelLayerCount = typeof selectedModel?.metadata.n_layer === "number" ? selectedModel.metadata.n_layer : null;
  const baseLayerCount = modelLayerCount != null ? modelLayerCount + OUTPUT_LAYER : null;
  const trainedCtx = typeof selectedModel?.metadata.trained_ctx === "number" ? selectedModel.metadata.trained_ctx : null;
  const expertCount = typeof selectedModel?.metadata.expert_count === "number" ? selectedModel.metadata.expert_count : null;

  const liveVramTotalMib = selectedWorker?.vram?.gpu_memory_total_mib ?? null;
  const liveVramFreeMib = selectedWorker?.vram?.vram_free_before_mib ?? null;

  const threads = Math.max(1, (workerHardware?.cpu.cores ?? 8) - 1);

  // The offload point every stage runs at: everything, unless the live free-
  // VRAM reading says everything will not fit, in which case the same
  // estimator NewRun.tsx's banner uses caps it. Advisory in both directions
  // -- shared/vramEstimate.ts's flat per-layer average is weakest exactly
  // where it matters most (MoE), which is why the cap is surfaced in the
  // chain note rather than applied silently.
  const fullNgl = noGpu ? 0 : baseLayerCount ?? NGL_FALLBACK_MAX;
  const vramCap = useMemo(() => {
    if (!selectedModel || baseLayerCount == null || liveVramFreeMib == null || fullNgl <= 0) return null;
    const needed = estimateVramNeededMib({
      modelSizeBytes: selectedModel.size_bytes,
      totalModelLayers: baseLayerCount,
      requestedNgl: fullNgl,
    });
    if (needed == null || needed <= liveVramFreeMib) return null;
    return {
      neededMib: needed,
      freeMib: liveVramFreeMib,
      safeNgl: estimateSafeNgl(selectedModel.size_bytes, baseLayerCount, liveVramFreeMib),
    };
  }, [selectedModel, baseLayerCount, liveVramFreeMib, fullNgl]);
  const stageNgl = vramCap ? vramCap.safeNgl : fullNgl;

  // M1's inverse estimate, for the two KV pairs the questionnaire itself
  // talks about. Never gates anything: it ranks and annotates.
  const affordability = useMemo(() => {
    if (liveVramTotalMib == null || !selectedModel || baseLayerCount == null) return null;
    const weightsMib = estimateVramNeededMib({
      modelSizeBytes: selectedModel.size_bytes,
      totalModelLayers: baseLayerCount,
      requestedNgl: stageNgl,
    });
    const shared = {
      totalMib: liveVramTotalMib,
      weightsMib,
      nLayer: modelLayerCount ?? 0,
      nHeadKv: selectedModel.metadata.n_head_kv ?? 0,
      headDimK: selectedModel.metadata.head_dim_k,
      headDimV: selectedModel.metadata.head_dim_v,
      nEmbd: selectedModel.metadata.n_embd,
      nHead: selectedModel.metadata.n_head,
      slidingWindow: selectedModel.metadata.sliding_window,
      trainedCtx: trainedCtx ?? undefined,
    };
    return {
      weightsMib,
      f16: maxAffordableContext({ ...shared, cacheTypeK: "f16", cacheTypeV: "f16" }),
      q8: maxAffordableContext({ ...shared, cacheTypeK: "q8_0", cacheTypeV: "q8_0" }),
    };
  }, [liveVramTotalMib, selectedModel, baseLayerCount, modelLayerCount, stageNgl, trainedCtx]);

  // --- the chain ------------------------------------------------------------

  const ppTokens = useMemo(() => {
    const target = goals.target_ctx;
    if (target != null && target > 0) return Math.max(128, Math.min(4096, target));
    return 2048;
  }, [goals.target_ctx]);

  const tunedFrom = useCallback(
    (stage: StageKind): { batch: number; ubatch: number } | null => {
      const id = chain[stage];
      if (!id) return null;
      const data = stageData[id];
      if (!data || !TERMINAL.has(data.run.status)) return null;
      return bestPpPlacement(data.results);
    },
    [chain, stageData]
  );

  const stageInputs = useCallback(
    (stage: StageKind): StageInputs => ({
      ppTokens,
      // Only the scored stage measures generation: the tuning stages exist
      // to find batch/ubatch, which is a prompt-processing question, and
      // paying for a tg phase in each of them buys nothing.
      nGen: stage === "sweep" ? 128 : 0,
      threads,
      ngl: stageNgl,
      cpuMoe: 0,
      repeats,
      goals,
      tuned: stage === "sweep" ? tunedFrom("refine") ?? tunedFrom("tuning") : stage === "refine" ? tunedFrom("tuning") : null,
      noGpu,
    }),
    [ppTokens, threads, stageNgl, repeats, goals, tunedFrom, noGpu]
  );

  const stagePlans = useMemo(
    () =>
      STAGES.map((stage) => {
        const sweep = sweepForStage(stage, stageInputs(stage));
        const items = expandSweep(sweep);
        const runId = chain[stage];
        const data = runId ? stageData[runId] : undefined;
        return { stage, sweep, itemCount: items.length, runId, run: data?.run, results: data?.results ?? [] };
      }),
    [stageInputs, chain, stageData]
  );

  const priceStage = useCallback(
    (sweep: Sweep) => {
      if (!rates) return `${ETA_UNAVAILABLE} — no measured rates for this pairing yet`;
      return priceMatrix(
        expandSweep(sweep).map((item) => ({
          nPrompt: item.n_prompt,
          nGen: item.n_gen,
          repeats: sweep.repeats,
          ppRate: rates.pp,
          tgRate: rates.tg,
        }))
      ).display;
    },
    [rates]
  );

  const totalItems = stagePlans.reduce((acc, p) => acc + p.itemCount, 0);
  const totalPriced = useMemo(() => {
    if (!rates) return `${ETA_UNAVAILABLE} — no measured rates for this pairing yet`;
    return priceMatrix(
      stagePlans.flatMap((p) =>
        expandSweep(p.sweep).map((item) => ({
          nPrompt: item.n_prompt,
          nGen: item.n_gen,
          repeats: p.sweep.repeats,
          ppRate: rates.pp,
          tgRate: rates.tg,
        }))
      )
    ).display;
  }, [rates, stagePlans]);

  // Which stage the console is actually offering to start. A stage is
  // startable when it has no run yet, its grid has values, and its parent
  // (if any) has finished and handed down a winner.
  function stageState(stage: StageKind): "done" | "live" | "startable" | "blocked" {
    const plan = stagePlans.find((p) => p.stage === stage)!;
    if (plan.run) return TERMINAL.has(plan.run.status) ? "done" : "live";
    if (plan.itemCount === 0) return "blocked";
    const parent = stage === "refine" ? "tuning" : stage === "sweep" ? "refine" : null;
    if (parent && !chain[parent]) return "blocked";
    if (parent) {
      const parentRun = stageData[chain[parent] as string]?.run;
      if (!parentRun || !TERMINAL.has(parentRun.status)) return "blocked";
    }
    return "startable";
  }

  async function startStage(stage: StageKind, parentOverride?: StageKind): Promise<void> {
    if (!modelId || !workerId) {
      setMsg("Pick a machine and a model first.");
      return;
    }
    const plan = stagePlans.find((p) => p.stage === stage)!;
    const parentStage = parentOverride ?? (stage === "refine" ? "tuning" : stage === "sweep" ? "refine" : null);
    const parentRunId = parentStage ? chain[parentStage] : undefined;
    setBusyStage(stage);
    setMsg(`Queuing ${STAGE_TITLE[stage]}…`);
    try {
      const selectedGpu = selectedGpuRawIndex != null ? visibleGpus[selectedGpuRawIndex] : undefined;
      const run = await api.triggerRun({
        model_id: modelId,
        worker_id: workerId,
        kind: stage as RunKind,
        parent_run_id: parentRunId,
        main_gpu: selectedGpu ? selectedGpuRawIndex : undefined,
        sweep: plan.sweep,
        // M2's "skippable by construction" -- an untouched questionnaire
        // sends no goals key at all, so the payload stays byte-identical to
        // what a chain with no stated intent would have sent.
        goals: goalsEqualDefaults(goals) && goalsUnset ? undefined : goals,
      });
      if (modelId && workerId) writeJson(goalsStorageKey(modelId, workerId), goals);
      persistChain({ ...chain, [stage]: run.id });
      setMsg(`${STAGE_TITLE[stage]} scheduled ✓`);
    } catch (err) {
      setMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusyStage(null);
    }
  }

  function resetAll(): void {
    setGoals(defaultGoals());
    setGoalsUnset(true);
    setRepeats(MIN_SCORING_REPEATS);
    persistChain({});
    setStageData({});
    setMsg("");
  }

  function savePreset(): void {
    const name = window.prompt("Save this intent (goal, target context, workload shape, KV tolerance, repeats) as:");
    if (!name) return;
    const next = { ...presets, [name]: { goals, repeats } };
    setPresets(next);
    writeJson(PRESETS_STORAGE_KEY, next);
    setMsg(`Preset “${name}” saved.`);
  }

  function loadPreset(name: string): void {
    const preset = presets[name];
    if (!preset) return;
    setGoals(normalizeGoals(preset.goals) ?? defaultGoals());
    setGoalsUnset(false);
    setRepeats(preset.repeats);
    setPresetsOpen(false);
    setMsg(`Preset “${name}” loaded — grids below are rebuilt from it, nothing is re-measured.`);
  }

  const sweepPlan = stagePlans.find((p) => p.stage === "sweep")!;
  const kvPairs = useMemo(() => expandedKvPairs(sweepPlan.sweep), [sweepPlan.sweep]);
  const prunedPairs = useMemo(() => {
    const kept = new Set(recommendedKvGrid(goals.kv_tolerance ?? "q4_0_ok").map(([k, v]) => `${k} / ${v}`));
    return DEFAULT_RECOMMENDED_KV_PAIRS.map(([k, v]) => `${k} / ${v}`).filter((p) => !kept.has(p));
  }, [goals.kv_tolerance]);

  const cardsRunId = chain.sweep ?? null;
  const ready = Boolean(modelId && workerId);

  const goalLabel =
    goals.goal === "max_context"
      ? `Max context · ≥ ${Math.round((goals.speed_floor_frac ?? 0.5) * 100)} % TG`
      : goals.goal === "max_speed"
        ? "Max tok/s"
        : "Balanced";

  const inputCls = "rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent";

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-semibold text-fg">Benchmark</h1>
        <p className="mt-2 max-w-3xl text-sm text-muted">
          State the goal; the console derives the chain. Three stages run in order — two tune prompt
          processing, the third is the one that produces scored cards — and each is a real run linked to
          the last, so all three score as one universe. Building a grid by hand instead lives on{" "}
          <Link to="/new-run" className="text-accent hover:underline">
            New Run
          </Link>
          .
        </p>
      </div>

      {/* Pairing ------------------------------------------------------------ */}
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="text-muted">Machine</span>
          <select value={workerId} onChange={(e) => setWorkerId(e.target.value)} className={`${inputCls} w-56`}>
            <option value="" disabled>
              select…
            </option>
            {workerOrder.map((id) => (
              <option key={id} value={id}>
                {workerStatus[id]?.displayName ?? id}
              </option>
            ))}
          </select>
        </label>
        {visibleGpus.length > 1 && (
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted">GPU</span>
            <select
              value={selectedGpuRawIndex ?? ""}
              onChange={(e) => setSelectedGpuRawIndex(e.target.value === "" ? undefined : Number(e.target.value))}
              className={`${inputCls} w-64`}
            >
              <option value="">Auto (split across all GPUs)</option>
              {visibleGpus.map((g, i) => (
                <option key={i} value={i}>
                  {i}: {formatGpuLabel(g)}
                </option>
              ))}
            </select>
          </label>
        )}
        <ModelPicker
          models={presentModels}
          value={modelId}
          mtpValue=""
          onSelect={(id) => setModelId(id)}
          hfUpdates={{}}
          className="w-[26rem]"
        />
      </div>

      {workerId && unreachableLocationWorkers.includes(workerId) && (
        <p className="-mt-2 text-xs text-warning">
          Couldn't check what's on {selectedWorker?.displayName ?? "this machine"} — its models aren't offered
          until that's confirmed, even if they're actually still there.
        </p>
      )}

      {/* Model + Machine ---------------------------------------------------- */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-border bg-surface p-5">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted">Model</span>
          {selectedModel ? (
            <>
              <div className="mt-1.5 break-all font-mono text-[12.5px] text-fg">{selectedModel.filename}</div>
              <dl className="mt-3 flex flex-col gap-1.5 text-[12.5px]">
                <Kv label="Size on disk" value={formatBytes(selectedModel.size_bytes)} />
                <Kv
                  label="Layers · experts"
                  value={
                    modelLayerCount != null
                      ? `${modelLayerCount}${expertCount != null && expertCount > 0 ? ` · ${expertCount} (MoE)` : " · dense"}`
                      : "not read from this file's header"
                  }
                  read={modelLayerCount != null}
                />
                {/* Trained context sits right under size/layers because it's
                    the model's own hard ceiling -- every context sizing path
                    (M2's target clamp, N1's ladder) clamps to it, so it can
                    never be raised by config no matter how much VRAM exists. */}
                <Kv
                  label="Context (trained)"
                  value={trainedCtx != null ? trainedCtx.toLocaleString() : "not read"}
                  read={trainedCtx != null}
                />
                <Kv
                  label="KV heads · head dim"
                  value={
                    selectedModel.metadata.n_head_kv != null
                      ? `${selectedModel.metadata.n_head_kv} · ${selectedModel.metadata.head_dim_k ?? "—"}`
                      : "not read"
                  }
                  read={selectedModel.metadata.n_head_kv != null}
                />
                <Kv
                  label="Sliding window"
                  value={
                    selectedModel.metadata.sliding_window != null && selectedModel.metadata.sliding_window > 0
                      ? selectedModel.metadata.sliding_window.toLocaleString()
                      : "none — plain GQA"
                  }
                  read={selectedModel.metadata.sliding_window != null}
                />
              </dl>
            </>
          ) : (
            <p className="mt-2 text-sm text-muted">Pick a model to see what its header actually reported.</p>
          )}
        </div>

        <div className="rounded-xl border border-border bg-surface p-5">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted">Machine</span>
          {selectedWorker ? (
            <>
              <div className="mt-1.5 font-mono text-[12.5px] text-fg">{selectedWorker.displayName}</div>
              <dl className="mt-3 flex flex-col gap-1.5 text-[12.5px]">
                <Kv
                  label="GPU"
                  value={gpuList.length > 0 ? gpuList.map((g) => formatGpuLabel(g)).join(" · ") : "none detected"}
                />
                <Kv
                  label="Backend · build"
                  value={`${selectedWorker.backend ?? "unknown"} · ${activeBuild ?? "no build activated"}`}
                />
                <Kv
                  label="Sensors"
                  value={
                    sensors
                      ? [sensors.clock ? "clock" : null, sensors.temp ? "temp" : null].filter(Boolean).join(" · ") ||
                        "neither available on this platform"
                      : "not reported by this worker"
                  }
                  // M6/N6 -- declared UP FRONT so a missing thermally_throttled
                  // flag later reads as "this machine can't produce one",
                  // never as "nothing throttled".
                  hint="Whether this machine can report clock and temperature at all — a run that can't sample them never raises a thermal flag."
                />
                <Kv label="Status" value={selectedWorker.status} />
              </dl>
            </>
          ) : (
            <p className="mt-2 text-sm text-muted">Pick a machine to see its backend, sensors and current state.</p>
          )}
        </div>
      </div>

      {/* M2 -- captured BEFORE the grid is built ---------------------------- */}
      <GoalQuestionnaire
        goals={goals}
        onChange={(next) => {
          setGoals(next);
          setGoalsUnset(false);
        }}
        trainedCtx={trainedCtx}
        unset={goalsUnset}
        affordability={{
          totalMib: liveVramTotalMib,
          weightsMib: affordability?.weightsMib ?? null,
          nLayer: modelLayerCount ?? undefined,
          nHeadKv: selectedModel?.metadata.n_head_kv,
          headDimK: selectedModel?.metadata.head_dim_k,
          headDimV: selectedModel?.metadata.head_dim_v,
          nEmbd: selectedModel?.metadata.n_embd,
          nHead: selectedModel?.metadata.n_head,
          slidingWindow: selectedModel?.metadata.sliding_window,
        }}
      />

      {/* M1 -- the inverse estimate ----------------------------------------- */}
      <div className="rounded-xl border border-border bg-raised p-4">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted">
          Affordability — the inverse estimate
        </span>
        {affordability && affordability.f16.confidence !== "unknown" && affordability.f16.tokens > 0 ? (
          <>
            <div className="mt-2 flex flex-wrap items-baseline gap-x-5 gap-y-1.5 text-[12.5px]">
              <span className="text-fg">
                Given this card, this placement, this KV type — <b>the largest context that still fits</b>:
              </span>
              <span className="font-mono text-fg">
                f16/f16 ≈ <b>{affordability.f16.tokens.toLocaleString()} tok</b>
              </span>
              <span className="font-mono text-fg">
                q8_0/q8_0 ≈ <b>{affordability.q8.tokens.toLocaleString()} tok</b>
              </span>
              <span className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[11px] text-accent">
                confidence: {affordability.f16.confidence}
              </span>
            </div>
            <p className="mt-2 text-[11.5px] leading-relaxed text-muted">
              From this placement's resident weights
              {affordability.weightsMib != null ? ` (~${affordability.weightsMib.toLocaleString()} MiB)` : ""}, a 10 %
              activation headroom and scratch. Binding constraint:{" "}
              <span className="font-mono">{affordability.f16.binding ?? "—"}</span>. Advisory only — it ranks and
              annotates, it never removes a configuration from the sweep.
            </p>
          </>
        ) : affordability && affordability.f16.confidence !== "unknown" ? (
          <p className="mt-2 text-[12px] leading-relaxed text-muted">
            <b className="text-fg">This placement leaves no room for a KV cache at all.</b> The weights alone
            {affordability.weightsMib != null ? ` (~${affordability.weightsMib.toLocaleString()} MiB)` : ""} fill the
            card once a 10 % activation headroom and scratch are reserved, so the affordable context at every KV type
            is zero — binding constraint <span className="font-mono">weights-placement</span>, not KV. The lever is
            offload, not cache quality: the sweep stage already carries a lower{" "}
            <span className="font-mono">-ngl</span> point, and <span className="font-mono">--n-cpu-moe</span> on New Run moves expert weights off the card
            entirely. Advisory, as always — nothing is removed from the grid over it.
          </p>
        ) : (
          <p className="mt-2 text-[12px] leading-relaxed text-muted">
            Unavailable for this pairing — the estimate needs this model's KV geometry from its header and a live
            VRAM total from the machine. It never fabricates a number in their absence.
            {affordability?.f16.conservativeFloorTokens != null && (
              <>
                {" "}
                A conservative floor does exist: this model's full trained context of{" "}
                <b className="text-fg">{affordability.f16.conservativeFloorTokens.toLocaleString()}</b> tokens, offered
                as a floor, not as an estimate.
              </>
            )}
          </p>
        )}
      </div>

      {/* The chain ----------------------------------------------------------- */}
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <span className="text-sm font-medium text-fg">Chain — 3 stages, all llama-bench</span>
          <span className="font-mono text-[11.5px] text-muted">
            {totalItems} test{totalItems === 1 ? "" : "s"} · est. {totalPriced}
          </span>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {stagePlans.map((plan) => {
            const state = stageState(plan.stage);
            return (
              <div key={plan.stage} className="flex flex-col rounded-lg border border-border bg-raised p-3">
                <div className="flex items-baseline justify-between gap-2">
                  <b className="text-[12.5px] text-fg">{STAGE_TITLE[plan.stage]}</b>
                  {plan.run && <StatusDot status={plan.run.status} />}
                </div>
                <div className="mt-1 font-mono text-[11px] leading-relaxed text-muted">
                  {plan.itemCount > 0 ? (
                    <>
                      {plan.itemCount} test{plan.itemCount === 1 ? "" : "s"} ·{" "}
                      {plan.stage === "sweep" ? "KV × FA × ngl × depth" : "PP only · batch/ubatch"}
                    </>
                  ) : (
                    <>rules, not values yet</>
                  )}
                </div>
                {plan.itemCount > 0 && <div className="mt-1 text-[11px] text-muted">est. {priceStage(plan.sweep)}</div>}
                {plan.itemCount === 0 && (
                  <div className="mt-1 text-[11px] leading-relaxed text-muted">
                    One octave either side of whatever {STAGE_TITLE.tuning} wins — there is nothing to centre on
                    until it has.
                  </div>
                )}
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  {plan.runId ? (
                    <Link to={`/runs/${plan.runId}`} className="text-[11.5px] font-semibold text-accent hover:underline">
                      View run →
                    </Link>
                  ) : (
                    <button
                      type="button"
                      disabled={state !== "startable" || busyStage != null || !ready}
                      onClick={() => void startStage(plan.stage)}
                      className="rounded-lg bg-accent px-3 py-1 text-[11.5px] font-semibold text-accent-fg disabled:opacity-40"
                    >
                      {STAGE_START_LABEL[plan.stage]}
                    </button>
                  )}
                  {/* Refine is a convenience, not a requirement: a chain may
                      go straight from the coarse winner to the scored sweep,
                      which is depth 2 of the 3 §0.5 allows. */}
                  {plan.stage === "sweep" && !plan.runId && !chain.refine && chain.tuning && (
                    <button
                      type="button"
                      disabled={busyStage != null || stageState("refine") === "blocked"}
                      onClick={() => void startStage("sweep", "tuning")}
                      className="rounded-lg border border-border px-2.5 py-1 text-[11.5px] font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-40"
                    >
                      Skip refine
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          <div className="flex flex-col justify-center rounded-lg border border-dashed border-border p-3 text-center">
            <b className="text-[12.5px] text-accent">cards per goal</b>
            <div className="mt-1 font-mono text-[11px] text-muted">
              {goals.goal === "max_context"
                ? "Max Context + Low Memory"
                : goals.goal === "max_speed"
                  ? "Max Speed + Low Memory"
                  : "Balanced + Low Memory"}
            </div>
            {cardsRunId ? (
              <Link to={`/runs/${cardsRunId}`} className="mt-2 text-[11.5px] font-semibold text-accent hover:underline">
                Open scored cards →
              </Link>
            ) : (
              <div className="mt-2 text-[11px] leading-relaxed text-muted">
                Only the sweep stage scores — tuning stages feed it values, never cards.
              </div>
            )}
          </div>
        </div>

        {/* M4 -- the axis, as it will actually expand ---------------------- */}
        <p className="mt-3 text-xs leading-relaxed text-muted">
          <b className="text-fg">Sweep axis after your tolerance:</b>{" "}
          {kvPairs.map((pair) => (
            <span
              key={pair}
              className="mr-1.5 inline-block rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 font-mono text-[11px] text-accent"
            >
              {pair}
            </span>
          ))}
          {prunedPairs.map((pair) => (
            <span
              key={pair}
              className="mr-1.5 inline-block rounded-full border border-border bg-raised px-2 py-0.5 font-mono text-[11px] text-muted line-through opacity-60"
            >
              {pair}
            </span>
          ))}
          {prunedPairs.length > 0 ? (
            <>
              — removed by your KV tolerance before expansion, so the {sweepPlan.itemCount}-test count above already
              reflects it. The stage says the axis shrank rather than letting the count imply it.
            </>
          ) : (
            <>— nothing is pruned at this tolerance.</>
          )}
        </p>

        {/* M7/M2 -- held fields keep their reasons -------------------------- */}
        <p className="mt-2 text-xs leading-relaxed text-muted">
          <b className="text-fg">Held fields keep their reasons.</b> Threads are held at {threads} (one less than this
          machine's cores): these stages target fully offloaded configurations where <span className="font-mono">-t</span>{" "}
          barely bites. On CPU-bound rows it is the dominant variable and stays an axis on the New Run page, with the
          running build's own ISA provenance recorded alongside those rows.{" "}
          {goals.target_ctx != null ? (
            <>
              Depth is anchored to your target:{" "}
              <span className="font-mono">{Math.round(goals.target_ctx * 0.5).toLocaleString()} = 50 % of your{" "}
              {goals.target_ctx.toLocaleString()}</span>.
            </>
          ) : (
            <>
              Depth stays at 0 — without a target context in Q2 there is nothing for a percentage to be a percentage
              of, and a house number would not be your workload.
            </>
          )}
        </p>

        {vramCap && (
          <p className="mt-2 rounded-lg border border-warning/30 bg-warning-bg px-3 py-2 text-xs leading-relaxed text-warning">
            <b>Offload capped at {vramCap.safeNgl} layers.</b> Full offload of this model is estimated at ~
            {formatBytes(vramCap.neededMib * 1024 * 1024)} against ~{formatBytes(vramCap.freeMib * 1024 * 1024)} free on
            this machine right now. Every stage above runs at the capped value. The estimate is a flat per-layer
            average — weakest exactly for Mixture-of-Experts models — so it is shown rather than applied silently.
          </p>
        )}
      </div>

      {/* Repeats + presets ---------------------------------------------------- */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-border bg-surface p-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted">Repeats per test</span>
          <div className="mt-2 flex flex-wrap gap-1.5" role="radiogroup" aria-label="Repeats per test">
            {REPEAT_CHOICES.map((n) => {
              const disabled = n < MIN_SCORING_REPEATS;
              return (
                <button
                  key={n}
                  type="button"
                  role="radio"
                  aria-checked={repeats === n}
                  aria-disabled={disabled}
                  disabled={disabled}
                  onClick={() => setRepeats(n)}
                  className={
                    repeats === n
                      ? "rounded-full border border-accent bg-accent/10 px-3 py-0.5 text-[12px] font-semibold text-accent"
                      : disabled
                        ? "rounded-full border border-border bg-raised px-3 py-0.5 text-[12px] text-muted line-through opacity-50"
                        : "rounded-full border border-border bg-raised px-3 py-0.5 text-[12px] text-muted hover:text-fg"
                  }
                >
                  {n}
                </button>
              );
            })}
          </div>
          <p className="mt-2 text-[11.5px] leading-relaxed text-muted">
            Fewer than {MIN_SCORING_REPEATS} reports a standard deviation of exactly 0 — §0.3's stability gate would
            pass on a number that was never measured, so those two stay visible and unselectable rather than quietly
            absent.
          </p>
        </div>

        <div className="rounded-xl border border-border bg-surface p-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted">Presets carry intent too</span>
          <p className="mt-1.5 text-[12px] leading-relaxed text-muted">
            Saving stores the goals block and the repeat count — goal, target context, workload shape, KV tolerance.
            They describe your workload, not a machine, so they load verbatim everywhere and the grids above are
            rebuilt from them.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={resetAll}
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              Reset all
            </button>
            <div className="relative">
              <button
                type="button"
                onClick={() => setPresetsOpen((open) => !open)}
                aria-expanded={presetsOpen}
                disabled={Object.keys(presets).length === 0}
                className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-40"
              >
                Load preset ▾
              </button>
              {presetsOpen && (
                <div className="absolute z-10 mt-1 flex min-w-40 flex-col rounded-lg border border-border bg-surface p-1 shadow-lg">
                  {Object.keys(presets).map((name) => (
                    <button
                      key={name}
                      type="button"
                      onClick={() => loadPreset(name)}
                      className="rounded px-2 py-1 text-left text-xs text-fg hover:bg-white/5"
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={savePreset}
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              Save preset…
            </button>
            <button
              type="button"
              disabled={stageState("tuning") !== "startable" || busyStage != null || !ready}
              onClick={() => void startStage("tuning")}
              className="rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-accent-fg disabled:opacity-40"
            >
              {STAGE_START_LABEL.tuning}
            </button>
          </div>
          {msg && <p className="mt-2 text-xs text-muted">{msg}</p>}
        </div>
      </div>

      {/* Once this finishes ---------------------------------------------------- */}
      <div className="rounded-xl border border-accent/35 bg-accent/[0.06] p-4" role="status">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[12.5px] font-semibold text-fg">Once this finishes</div>
            <div className="mt-1 text-[12px] leading-relaxed text-muted">
              Cards ranked for{" "}
              <b className="text-fg">
                {goalLabel}
                {goals.target_ctx != null ? ` · ${goals.target_ctx.toLocaleString()}` : ""} · {goals.workload} (wPP{" "}
                {WORKLOAD_WEIGHTS[goals.workload].wPP.toFixed(2)} · wTG{" "}
                {WORKLOAD_WEIGHTS[goals.workload].wTG.toFixed(2)})
              </b>{" "}
              — change the goal later and re-scoring is instant post-processing over stored results, never
              re-measurement.
            </div>
          </div>
          {cardsRunId ? (
            <Link
              to={`/runs/${cardsRunId}`}
              className="rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-accent-fg"
            >
              Scored cards →
            </Link>
          ) : (
            <span className="text-[11.5px] text-muted">available once the sweep stage has run</span>
          )}
        </div>
      </div>
    </div>
  );
}

function Kv({ label, value, read, hint }: { label: string; value: string; read?: boolean; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3" title={hint}>
      <dt className="shrink-0 text-muted">{label}</dt>
      {/* Fail-soft display: an explicitly-unread value (`read === false`)
          dims instead of looking authoritative -- the app never fabricates a
          number, so "not read" must read as absence, not as data. Rows that
          always have a real value (size, sliding-window "none") leave `read`
          unset and render normally. */}
      <dd className={`text-right ${read === false ? "text-muted" : "text-fg"}`}>
        {value}
        {read && (
          <span className="ml-1.5 rounded-full border border-accent/40 bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">
            read ✓
          </span>
        )}
      </dd>
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  const cls =
    status === "done"
      ? "border-accent/40 bg-accent/10 text-accent"
      : status === "failed" || status === "cancelled"
        ? "border-danger/40 bg-danger/10 text-danger"
        : status === "partial"
          ? "border-warning/40 bg-warning-bg text-warning"
          : "border-border bg-raised text-muted";
  return <span className={`rounded-full border px-2 py-0.5 text-[10px] ${cls}`}>{status}</span>;
}
