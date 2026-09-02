// BENCHMARKING_PLAN_V8.md's benchmark console -- the page laid out in
// docs/Benchmark - Proposed Layout (standalone).html: a numbered 1-2-3
// wizard (pairing -> goal -> chain) beside a sticky "Your test" summary,
// replacing the stacked-sections layout screen 1 of
// docs/benchmark-page-mockup-v8.html originally sketched.
//
// The difference from CustomTest.tsx (which stays, unchanged, at /custom-test) is
// WHO builds the grid. There, the user builds one flat cross-product by
// hand and every trigger is a standalone root. Here the user states intent
// -- model, machine, the M2 questionnaire, a repeat count -- and the page
// derives the three-stage chain §0.5 defines (tuning -> refine -> sweep),
// each stage a real run linked to the previous one by `parent_run_id`, which
// is what makes the server resolve them to one `root_run_id` and score them
// as one universe (repo.ts's listChainScoringTests).
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

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { ModelPicker } from "../components/ModelPicker";
import { GoalQuestionnaire, KV_PRESET_LABEL } from "../components/GoalQuestionnaire";
import { TestStatusPill } from "../components/StatusPill";
import { IconArrowRight, IconChevronDown, IconInfo } from "../components/icons";
import { backendVisibleGpus } from "../types";
import type { Model, Test, ResultRow, TestKind, SweepConfig } from "../types";
import { formatBytes, formatGpuLabel } from "../utils";
import { estimateSafeNgl, estimateVramNeededMib, maxAffordableContext } from "../vramEstimate";
import {
  defaultGoals,
  goalsEqualDefaults,
  normalizeGoals,
  kvPresetPairs,
  WORKLOAD_WEIGHTS,
  type GoalsConfig,
} from "../goals";
import { expandSweep } from "../../../shared/sweep";
import type { ProbeGranularity, ProbeMode } from "../../../shared/probeLadder";
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
// being dropped from the row: the mockup's own posture.
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
// The pairing keys are deliberately the SAME ones CustomTest.tsx writes: which
// machine and model you were last looking at is one fact about the session,
// not one per page, and a user moving between the two surfaces should not
// have to re-pick. The goals key is shared for the same reason plus M5's:
// goals describe the workload, not the page that captured them.

const LAST_WORKER_STORAGE_KEY = "llamatoaster:custom-test:last-worker";

function lastModelStorageKey(workerId: string): string {
  return `llamatoaster:custom-test:last-model:${workerId}`;
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

// Placement (the offload slider) is machine-specific, not workload intent --
// M5's own principle for why goals travel with presets and placement never
// does. Page-local, keyed to the pairing, same as chainStorageKey above.
function placementStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:benchmark:placement:${modelId}:${workerId}`;
}

// The Tested-configurations cards' own in-flight/finished probe results --
// same reasoning as chainStorageKey/placementStorageKey above, and the same
// reason it matters MORE here: without this, navigating away from Benchmark
// and back (or a reload) while a card's probe is still running loses the
// only record that it was ever started. The card shows no spinner, "Test
// all" looks idle, and the poll loop that would have picked up the eventual
// verified/failed result never gets re-armed -- the run keeps executing
// server-side, but this page forgets it exists.
function verifyStatesStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:benchmark:verify:${modelId}:${workerId}`;
}

// N2 batching -- which probe root new "Run test" triggers attach to, so a
// batch survives the same remounts verifyStatesStorageKey's own comment
// describes (a reload, or selectedModel/baseLayerCount going briefly null)
// while its scenarios are still running/queued.
function batchRootIdStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:benchmark:batch-root:${modelId}:${workerId}`;
}

const PRESETS_STORAGE_KEY = "llamatoaster:benchmark:presets";
const REPEATS_STORAGE_KEY = "llamatoaster:benchmark:repeats";
const AUTO_ADVANCE_STORAGE_KEY = "llamatoaster:benchmark:auto-advance";

type ChainState = Partial<Record<StageKind, string>>;

// The Step-2 placement fit check's own verify lifecycle -- ngl/ctx are
// SNAPSHOTTED at the moment Verify was clicked (not a live reference to the
// sliders), so the eventual result banner always names the placement it
// actually checked even if the user keeps dragging while it's in flight.
interface PlacementVerifyState {
  ngl: number;
  ctx: number;
  testId: string;
  // "cancelled" -- a user stop mid-ladder (worker/src/index.ts's
  // stopRequested check), distinct from failed/failed_oom: it means nobody
  // let the search finish, not that the placement doesn't fit.
  status: "pending" | "verified" | "failed" | "failed_oom" | "cancelled" | "error";
  detail?: string;
  verifiedCtxTokens?: number | null;
  /** Which Tested-configurations card fired this probe. */
  mode?: ProbeMode;
  /** The winning rung's own placement and real measured usage -- pulled from
   * probe_attempts once verified, since the ladder may have moved ngl away
   * from what this card started at, and the estimate ("needed") can be
   * meaningfully wrong (see the gemma4 SWA case) where the measured peak
   * cannot. */
  measuredNgl?: number | null;
  measuredVramPeakMib?: number | null;
  measuredRamPeakMib?: number | null;
  /** When this card's Test was clicked -- captured once at trigger time (not
   * re-stamped by the poll's terminal/verified updates), so it reads as
   * "when the test ran" even for a still-pending or since-reset card. */
  testedAt?: number;
  /** The run_item's own live `detail` text, refreshed on every non-terminal
   * poll tick (see startPolling below) -- worker/src/index.ts's probe loop
   * now calls sendTick once per candidate load, the same mechanism a sweep
   * item's live phase text already used. Cleared implicitly once the run
   * goes terminal (nothing writes it after that point). */
  liveDetail?: string;
  /** Scheduled (queued, not yet claimed by the worker) vs actually running --
   * only meaningful while status is "pending". N2 batching can leave a card
   * queued behind an earlier batch member for a while, so this is what lets
   * the card distinguish "waiting its turn" from "loading right now" (see
   * GoalQuestionnaire.tsx's ModeCard). Undefined once terminal. */
  runStatus?: "scheduled" | "running";
}

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

// Each side of a winner's neighbourhood decides independently whether Test
// A's own grid already has a value past it. A winner sitting at that grid's
// floor or ceiling extends past it (the +/-1-octave step the old scheme used
// everywhere) since nothing out there has been measured; a winner elsewhere
// refines within it (+/-25%) since Test A's own neighbour in that direction
// is already a known, already-measured value.
function refineNeighbors(winner: number, floor: number, ceiling: number): [number, number, number] {
  const lower = winner === floor ? winner / 2 : winner * 0.75;
  const upper = winner === ceiling ? winner * 2 : winner * 1.25;
  return [lower, winner, upper];
}

// Test B. The mockup's "rules, not values yet": the neighbourhood is defined
// here as refineNeighbors() of whatever Test A actually won, so an interior
// winner gets genuinely new candidates instead of Test A's own values
// measured a second time, while an edge winner still probes past Test A's
// own floor/ceiling. It also widens n_prompt to half and double the anchored
// size, since a winner measured at one prompt length was never checked
// against shorter or longer workloads. This stage has no values at all until
// Test A's result exists.
function refineSweep(input: StageInputs): Sweep {
  const coarse = tuningSweep(input);
  if (!input.tuned) return { ...coarse, batch_size: [], ubatch_size: [] };
  const batches = [
    ...new Set(refineNeighbors(input.tuned.batch, Math.min(...COARSE_BATCHES), Math.max(...COARSE_BATCHES))),
  ]
    .filter((b) => Number.isInteger(b) && b >= 32 && b <= 8192)
    .sort((a, b) => a - b);
  const smallestBatch = batches.length > 0 ? batches[0] : input.tuned.batch;
  const ubatches = [
    ...new Set(refineNeighbors(input.tuned.ubatch, Math.min(...COARSE_UBATCHES), Math.max(...COARSE_UBATCHES))),
  ]
    .filter((u) => Number.isInteger(u) && u >= 32 && u <= smallestBatch)
    .sort((a, b) => a - b);
  const nPrompts = [...new Set([Math.round(input.ppTokens * 0.5), input.ppTokens, input.ppTokens * 2])]
    .filter((p) => Number.isInteger(p) && p >= 32)
    .sort((a, b) => a - b);
  return {
    ...coarse,
    n_prompt: nPrompts.length > 0 ? nPrompts : [input.ppTokens],
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
  // The preset IS the exact set of (K,V) pairs this stage runs -- no
  // tolerance-pruning step needed, since these curated lists never contain a
  // pair the grid has to filter back out. cache_type_pairs carries the
  // coupling through expandSweep (shared/sweep.ts); cache_type_k/v stay
  // populated with the pairs' own unique values for any caller that only
  // reads the plain axis arrays (e.g. display labels).
  const pairs = kvPresetPairs(input.goals.kv_preset);
  const cache_type_k = [...new Set(pairs.map(([k]) => k))];
  const cache_type_v = [...new Set(pairs.map(([, v]) => v))];

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
    cache_type_pairs: [...pairs],
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
// that still ingested a row. anchorPrompt pins the comparison to the one
// n_prompt every stage is anchored on: refine's own n_prompt axis now spans
// 0.5x/1x/2x that value (see refineSweep), and comparing across all three
// would just hand the next stage whichever prompt length happens to score
// the highest tok/s, not the batch/ubatch that actually won at the anchor.
function bestPpPlacement(results: ResultRow[], anchorPrompt: number): { batch: number; ubatch: number } | null {
  const pp = results.filter((r) => r.test_type === "pp" && r.avg_tps > 0 && r.n_prompt === anchorPrompt);
  if (pp.length === 0) return null;
  const best = pp.reduce((a, b) => (b.avg_tps > a.avg_tps ? b : a));
  return { batch: best.batch_size, ubatch: best.ubatch_size };
}

// The KV pairs the sweep stage will ACTUALLY expand to -- exactly the chosen
// preset's curated list now that cache_type_pairs carries the coupling
// through expandSweep, rather than a cross-product superset of it.
function expandedKvPairs(sweep: Sweep): string[] {
  const pairs = new Set<string>();
  for (const item of expandSweep(sweep)) pairs.add(`${item.cache_type_k} / ${item.cache_type_v}`);
  return [...pairs];
}

// Compact facts-strip status word: offline reads as an error, busy as
// "something is happening right now", idle as the normal ready state --
// same three-way split WorkerCard.tsx's own status handling already uses.
function workerStatusTone(status: string): string {
  if (status === "offline") return "text-danger";
  if (status === "busy") return "text-accent";
  return "text-success";
}

export function Benchmark() {
  const [models, setModels] = useState<Model[]>([]);
  const [locations, setLocations] = useState<Record<string, string[]> | null>(null);
  const [, setUnreachableLocationWorkers] = useState<string[]>([]);
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

  // Step 1's compact facts strip hides everything else behind these two
  // disclosures by default -- the same "collapsed, one click away" posture
  // as GoalQuestionnaire's own KV-preset section.
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [scoringDetailsOpen, setScoringDetailsOpen] = useState(false);

  const [chain, setChain] = useState<ChainState>({});
  const [stageData, setStageData] = useState<Record<string, { run: Test; results: ResultRow[] }>>({});
  const [busyStage, setBusyStage] = useState<StageKind | null>(null);
  const [msg, setMsg] = useState("");
  const [autoAdvance, setAutoAdvance] = useState<boolean>(() => readJson<boolean>(AUTO_ADVANCE_STORAGE_KEY) ?? false);
  // Guards each parent->child hand-off to at most one attempt per parent run:
  // keyed on the parent's own run id, so a re-run of the parent (new id) is
  // free to retry, but a failed trigger is not retried every 5s poll tick.
  const autoAdvanceAttempted = useRef<Set<string>>(new Set());

  const [rates, setRates] = useState<ModelRatesResponse | null>(null);

  // Step 2's placement (offload) slider -- null means "use the auto-derived
  // default" (today's silent auto-cap), a number once the user has actually
  // dragged it. Never part of `goals`/presets -- see placementStorageKey.
  const [nglOverride, setNglOverride] = useState<number | null>(null);
  // Keyed by mode, not a single shared slot -- every Tested-configurations
  // card keeps its own independent result, in flight or finished, so Test
  // All can run every mode at once without one overwriting another.
  const [verifyStates, setVerifyStates] = useState<Partial<Record<ProbeMode, PlacementVerifyState>>>({});
  // N2 batching -- the probe root every "Run test" trigger for this pairing
  // currently attaches to, once one has been established; null means the
  // next trigger starts a fresh batch. Lifted up so it survives a remount
  // the same way verifyStates does (see batchRootIdStorageKey).
  const [batchRootId, setBatchRootId] = useState<string | null>(null);
  // A real failed/failed_oom verify result means the estimate was wrong at
  // that point -- bumped so the NEXT round of suggestions doesn't just
  // recompute the same number that already failed, rather than trusting the
  // live reading at full confidence forever.
  const [poolHaircutFrac, setPoolHaircutFrac] = useState(0);

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
      setNglOverride(null);
      setVerifyStates({});
      setBatchRootId(null);
      setPoolHaircutFrac(0);
      return;
    }
    const stored = readJson<unknown>(goalsStorageKey(modelId, workerId));
    const restored = stored ? normalizeGoals(stored) : null;
    setGoals(restored ?? defaultGoals());
    setGoalsUnset(restored == null);
    setChain(readJson<ChainState>(chainStorageKey(modelId, workerId)) ?? {});
    setNglOverride(readJson<number>(placementStorageKey(modelId, workerId)));
    const restoredVerify = readJson<Partial<Record<ProbeMode, PlacementVerifyState>>>(
      verifyStatesStorageKey(modelId, workerId)
    );
    setVerifyStates(restoredVerify ?? {});
    setBatchRootId(readJson<string>(batchRootIdStorageKey(modelId, workerId)) ?? null);
    // A card left "pending" (its probe still running server-side) when this
    // page was last torn down has no live poll loop anymore -- the one
    // verifyPlacement started died with the old mount. Re-arm one per
    // restored pending card so the eventual verified/failed/failed_oom
    // result still lands instead of the card being stuck "Testing…" forever.
    for (const [mode, state] of Object.entries(restoredVerify ?? {}) as [ProbeMode, PlacementVerifyState][]) {
      if (state?.status === "pending" && state.testId) startPolling(mode, state.testId);
    }
    setPoolHaircutFrac(0);
  }, [modelId, workerId]);

  useEffect(() => {
    if (!modelId || !workerId) return;
    writeJson(verifyStatesStorageKey(modelId, workerId), verifyStates);
  }, [modelId, workerId, verifyStates]);

  useEffect(() => {
    if (!modelId || !workerId) return;
    writeJson(batchRootIdStorageKey(modelId, workerId), batchRootId);
  }, [modelId, workerId, batchRootId]);

  useEffect(() => {
    if (!modelId || !workerId || nglOverride == null) return;
    writeJson(placementStorageKey(modelId, workerId), nglOverride);
  }, [modelId, workerId, nglOverride]);

  useEffect(() => {
    writeJson(REPEATS_STORAGE_KEY, repeats);
  }, [repeats]);

  useEffect(() => {
    writeJson(AUTO_ADVANCE_STORAGE_KEY, autoAdvance);
  }, [autoAdvance]);

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
      const next: Record<string, { run: Test; results: ResultRow[] }> = {};
      let anyLive = false;
      for (const id of chainIds) {
        try {
          const res = await api.getTest(id);
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
    // A backgrounded tab gets its setTimeout cadence throttled (or paused
    // outright) by the browser, so a stage that finishes while the user is
    // away from the tab can sit "done" without this ever re-polling to
    // notice -- which is exactly what stalls autoAdvance below until the
    // user does something that happens to trigger a fresh fetch. Re-poll
    // immediately the instant the tab regains visibility so a finished
    // stage is picked up right away rather than waiting on a timer that may
    // not fire again for minutes.
    function onVisible() {
      if (document.visibilityState !== "visible") return;
      if (timer) window.clearTimeout(timer);
      void poll();
    }
    document.addEventListener("visibilitychange", onVisible);
    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
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

  const liveVramTotalMib = selectedWorker?.vram?.gpu_memory_total_mib ?? null;
  const liveVramFreeMib = selectedWorker?.vram?.vram_free_before_mib ?? null;
  const liveRamTotalMib = selectedWorker?.vram?.system_memory_total_mib ?? null;
  const liveRamFreeMib = selectedWorker?.vram?.ram_free_before_mib ?? null;

  // Metal, or any visible GPU with vram_dynamic:true (a shared-memory iGPU)
  // -- VRAM and RAM describe the same physical bytes, so the offload slider
  // has nothing real to trade and stays locked full-GPU (see effectiveNgl).
  const unifiedPool = selectedWorker?.vram?.backend === "metal" || visibleGpus.some((g) => g.vram_dynamic === true);

  const threads = Math.max(1, (workerHardware?.cpu.cores ?? 8) - 1);

  // The offload point every stage runs at: everything, unless the live free-
  // VRAM reading says everything will not fit, in which case the same
  // estimator NewRun.tsx's banner uses caps it. Advisory in both directions
  // -- shared/vramEstimate.ts's real per-tensor placement (see
  // placeWeightBytes) is used whenever this model carries a tensor_layer_bytes
  // breakdown; a model registered before that existed falls back to a flat
  // per-layer average, weakest exactly where it matters most (MoE). Either
  // way the cap is surfaced in the chain note rather than applied silently.
  const fullNgl = noGpu ? 0 : baseLayerCount ?? NGL_FALLBACK_MAX;
  const vramCap = useMemo(() => {
    if (!selectedModel || baseLayerCount == null || liveVramFreeMib == null || fullNgl <= 0) return null;
    const tensorBreakdown = selectedModel.metadata.tensor_layer_bytes ?? null;
    const needed = estimateVramNeededMib({
      modelSizeBytes: selectedModel.size_bytes,
      totalModelLayers: baseLayerCount,
      requestedNgl: fullNgl,
      tensorBreakdown,
    });
    if (needed == null || needed <= liveVramFreeMib) return null;
    return {
      neededMib: needed,
      freeMib: liveVramFreeMib,
      safeNgl: estimateSafeNgl(selectedModel.size_bytes, baseLayerCount, liveVramFreeMib, tensorBreakdown),
    };
  }, [selectedModel, baseLayerCount, liveVramFreeMib, fullNgl]);
  const stageNgl = vramCap ? vramCap.safeNgl : fullNgl;

  // The placement that ACTUALLY runs every stage: stageNgl's auto-cap is
  // only ever the slider's initial default now -- once the user drags it
  // (nglOverride set), their choice wins. Locked machines (CPU-only, unified
  // memory) ignore both and use their own fixed point, matching the
  // slider-lock rules the fit matrix enforces in GoalQuestionnaire.
  const effectiveNgl = noGpu ? 0 : unifiedPool ? baseLayerCount ?? NGL_FALLBACK_MAX : nglOverride ?? stageNgl;

  // M1's inverse estimate, for the two KV pairs the questionnaire itself
  // talks about. Never gates anything: it ranks and annotates.
  const affordability = useMemo(() => {
    if (liveVramTotalMib == null || !selectedModel || baseLayerCount == null) return null;
    const weightsMib = estimateVramNeededMib({
      modelSizeBytes: selectedModel.size_bytes,
      totalModelLayers: baseLayerCount,
      requestedNgl: effectiveNgl,
      tensorBreakdown: selectedModel.metadata.tensor_layer_bytes ?? null,
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
  }, [liveVramTotalMib, selectedModel, baseLayerCount, modelLayerCount, effectiveNgl, trainedCtx]);

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
      return bestPpPlacement(data.results, ppTokens);
    },
    [chain, stageData, ppTokens]
  );

  const stageInputs = useCallback(
    (stage: StageKind): StageInputs => ({
      ppTokens,
      // Only the scored stage measures generation: the tuning stages exist
      // to find batch/ubatch, which is a prompt-processing question, and
      // paying for a tg phase in each of them buys nothing.
      nGen: stage === "sweep" ? 128 : 0,
      threads,
      ngl: effectiveNgl,
      cpuMoe: 0,
      repeats,
      goals,
      tuned: stage === "sweep" ? tunedFrom("refine") ?? tunedFrom("tuning") : stage === "refine" ? tunedFrom("tuning") : null,
      noGpu,
    }),
    [ppTokens, threads, effectiveNgl, repeats, goals, tunedFrom, noGpu]
  );

  const stagePlans = useMemo(
    () =>
      STAGES.map((stage) => {
        const sweep = sweepForStage(stage, stageInputs(stage));
        const items = expandSweep(sweep);
        const testId = chain[stage];
        const data = testId ? stageData[testId] : undefined;
        return { stage, sweep, itemCount: items.length, testId, run: data?.run, results: data?.results ?? [] };
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
      const run = await api.triggerTest({
        model_id: modelId,
        worker_id: workerId,
        kind: stage as TestKind,
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

  // Step 2's fit-check "Verify" -- rides the ordinary N2 probe trigger,
  // exactly the way ProfileCards.tsx's own verifyWithProbe already does for
  // a scored card, just sourced from the placement sliders instead of a
  // ScoredConfig, and callable before any chain run exists (a probe root is
  // standalone, never part of the chain). f16/f16 anchors the check, same
  // convention GoalQuestionnaire's own feasibility readout already uses.
  // Disables synchronously (setVerifyStates before the await) so a
  // double-click on the SAME card can never fire two triggers -- a
  // different card's own mode is untouched, so several modes can be fired
  // one after another in the same "Run test" click.
  //
  // batchRootId, when passed, attaches this trigger as a sibling under an
  // existing probe root (N2 batching -- see shared/types.ts's
  // TriggerPayload.probe_batch_root_id) instead of starting a fresh one, so
  // several modes fired from one Run test click collapse into one Tests-list
  // row. Returns the created run's id so the caller (GoalQuestionnaire's Run
  // test handler) can thread the FIRST call's id into the rest of the same
  // click's triggers; undefined on failure.
  async function verifyPlacement(
    ngl: number,
    ctx: number,
    mode: ProbeMode,
    granularity: ProbeGranularity,
    batchRootIdArg?: string
  ): Promise<string | undefined> {
    if (!modelId || !workerId || verifyStates[mode]?.status === "pending") return undefined;
    const testedAt = Date.now();
    setVerifyStates((prev) => ({ ...prev, [mode]: { ngl, ctx, testId: "", status: "pending", mode, testedAt } }));
    try {
      const selectedGpu = selectedGpuRawIndex != null ? visibleGpus[selectedGpuRawIndex] : undefined;
      const run = await api.triggerTest({
        model_id: modelId,
        worker_id: workerId,
        kind: "probe",
        main_gpu: selectedGpu ? selectedGpuRawIndex : undefined,
        probe_batch_root_id: batchRootIdArg,
        probe: {
          candidate_ctx: ctx,
          placement: { ngl, slots: 1 },
          kv_pair: ["f16", "f16"],
          mode,
          granularity,
        },
        // Vestigial -- the worker derives its own n_prompt/n_gen from
        // candidate_ctx for a probe load (see worker/src/index.ts's
        // runOneProbeLoad); this block only needs to satisfy the trigger
        // route's own validation and create the tracked run_item.
        sweep: {
          n_prompt: [512],
          n_gen: [128],
          threads: [threads],
          n_gpu_layers: [ngl],
          batch_size: [2048],
          ubatch_size: [512],
          cache_type_k: ["f16"],
          cache_type_v: ["f16"],
          flash_attn: ["on"],
          mtp: ["off"],
          n_gpu_layers_draft: [0],
          n_cpu_moe: [0],
          repeats: 1,
        },
      });
      setVerifyStates((prev) => ({ ...prev, [mode]: { ngl, ctx, testId: run.id, status: "pending", mode, testedAt } }));
      startPolling(mode, run.id);
      return run.id;
    } catch (err) {
      setVerifyStates((prev) => ({
        ...prev,
        [mode]: {
          ngl,
          ctx,
          testId: "",
          status: "error",
          mode,
          testedAt,
          detail: err instanceof Error ? err.message : String(err),
        },
      }));
      return undefined;
    }
  }

  // N2 batching -- fires every requested mode's verify in order (awaited
  // sequentially so batch membership is deterministic; NOT waiting for each
  // run to finish -- verifyPlacement's own startPolling tracks each mode
  // independently, so every selected card shows scheduled/running in place
  // immediately while the worker_jobs FIFO queue serializes actual
  // execution). The first trigger of a fresh batch establishes the root;
  // every later one in this call, and any later "Run test" click while that
  // batch is still open, attaches to it instead of 409ing or starting a new
  // row on the Runs list.
  async function runModes(
    modes: ProbeMode[],
    modeStarts: Record<ProbeMode, { ngl: number; ctx: number }>,
    granularity: ProbeGranularity
  ): Promise<void> {
    const toRun = modes.filter((m) => verifyStates[m]?.status !== "pending");
    if (toRun.length === 0) return;
    const batchStillOpen = Object.values(verifyStates).some((s) => s?.status === "pending");
    let rootId = batchStillOpen ? (batchRootId ?? undefined) : undefined;
    for (const mode of toRun) {
      const start = modeStarts[mode];
      const newId = await verifyPlacement(start.ngl, start.ctx, mode, granularity, rootId);
      if (!rootId && newId) {
        rootId = newId;
        setBatchRootId(newId);
      }
    }
  }

  // A card's own "reset" -- purely local, no server call. The probe run
  // itself still exists in Runs/RunDetail regardless; this only clears the
  // card back to its untested Apply/Test state so it can be re-run cleanly.
  function resetVerify(mode: ProbeMode): void {
    setVerifyStates((prev) => {
      if (!(mode in prev)) return prev;
      const next = { ...prev };
      delete next[mode];
      return next;
    });
  }

  // Polls one card's probe root independently of chainIds -- a probe is
  // never part of the tuning->refine->sweep chain. Coarse pass/fail/OOM comes
  // straight off items[0] (TerminalTestItemStatus distinguishes failed_oom
  // from failed even though the aggregate Test.status collapses both to
  // "failed"); a verified pass then looks up the precise ceiling AND (for the
  // card's own measured-needs line) the winning rung's real peak usage.
  //
  // Started IMPERATIVELY from verifyPlacement below, once per triggered run,
  // rather than from a useEffect reactively watching verifyStates. That
  // shape was tried first and had a real bug: an effect whose own dependency
  // array includes the state IT ALSO WRITES gets torn down and restarted by
  // React on every write -- including the write this very poll makes to
  // record "verified". The teardown's `cancelled = true` fired on THIS
  // closure before its own subsequent getVerifiedLimits/getProbeAttempts
  // lookup could apply, so the coarse verdict always landed but the
  // measured-needs upgrade was silently dropped every single time. Calling
  // this directly sidesteps the whole effect-restart lifecycle; only a real
  // component unmount (below) should ever cancel it.
  const pollingRunIds = useRef<Set<string>>(new Set());
  const unmountedRef = useRef(false);
  const visibilityListeners = useRef<(() => void)[]>([]);
  useEffect(() => {
    // StrictMode (main.tsx) double-invokes every effect in dev: mount, a
    // SIMULATED unmount (runs the cleanup below), then a real remount --
    // synchronously, before any poll ever runs. Without resetting the flag
    // here on that remount, the simulated unmount's cleanup would leave
    // unmountedRef permanently true, and every poll() call's first line
    // would bail out before ever calling api.getTest -- exactly the silent
    // "never actually polls" bug this comment is here to prevent regressing.
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      visibilityListeners.current.forEach((remove) => remove());
      visibilityListeners.current = [];
    };
  }, []);

  function startPolling(mode: ProbeMode, testId: string): void {
    if (pollingRunIds.current.has(testId)) return;
    pollingRunIds.current.add(testId);
    let timer: number | undefined;

    async function poll() {
      if (unmountedRef.current) return;
      try {
        const res = await api.getTest(testId);
        if (unmountedRef.current) return;
        if (TERMINAL.has(res.run.status)) {
          const item = res.items[0];
          // "cancelled" (a user stop, see worker/src/index.ts's stopRequested
          // check inside executeRunProbeJob) is its own outcome, not a
          // failure -- it means nobody let the ladder finish, not that the
          // estimate was wrong or the placement doesn't fit.
          const status: PlacementVerifyState["status"] =
            item?.status === "failed_oom"
              ? "failed_oom"
              : item?.status === "done"
                ? "verified"
                : item?.status === "cancelled"
                  ? "cancelled"
                  : "failed";
          setVerifyStates((prev) => {
            const cur = prev[mode];
            return cur && cur.testId === testId
              ? { ...prev, [mode]: { ...cur, status, detail: item?.detail, runStatus: undefined } }
              : prev;
          });
          if (status === "failed" || status === "failed_oom") {
            setPoolHaircutFrac((f) => Math.min(0.3, f === 0 ? 0.15 : 0.3));
          }
          if (status === "verified" && modelId && workerId) {
            try {
              const [limits, attemptsRes] = await Promise.all([
                api.getVerifiedLimits(modelId, workerId),
                api.getProbeAttempts(testId),
              ]);
              const match = limits.limits
                .filter((l) => l.kv_type === "f16/f16")
                .sort((a, b) => b.created_at - a.created_at)[0];
              // The largest passing context wins, ties broken by ngl -- the
              // same rule bestLadderResult (shared/probeLadder.ts) applies
              // server-side, so this always names the SAME rung the stored
              // ceiling actually came from.
              const winner = attemptsRes.attempts
                .filter((a) => a.ok)
                .sort((a, b) => b.candidate_ctx - a.candidate_ctx || (b.ngl ?? 0) - (a.ngl ?? 0))[0];
              if (!unmountedRef.current && (match || winner)) {
                setVerifyStates((prev) => {
                  const cur = prev[mode];
                  if (!cur || cur.testId !== testId) return prev;
                  return {
                    ...prev,
                    [mode]: {
                      ...cur,
                      verifiedCtxTokens: match?.verified_ctx_tokens,
                      measuredNgl: winner?.ngl ?? null,
                      measuredVramPeakMib: winner?.vram_peak_mib ?? null,
                      measuredRamPeakMib: winner?.ram_peak_mib ?? null,
                    },
                  };
                });
              }
            } catch {
              /* advisory upgrade only -- the coarse verdict above already landed */
            }
          }
          document.removeEventListener("visibilitychange", onVisible);
          pollingRunIds.current.delete(testId);
          return;
        }
        // Still running -- surface the run_item's own live detail text (the
        // worker ticks this once per candidate load, see
        // worker/src/index.ts's sendTick call in executeRunProbeJob) so the
        // card shows progress instead of sitting on a bare "Testing…" until
        // the whole ladder finishes. Also track scheduled-vs-running (N2
        // batching can leave a card queued behind an earlier batch member
        // for a while) so the card's dot can distinguish "waiting its turn"
        // from "actually loading right now" instead of showing the same
        // indicator for both.
        const liveDetail = res.items[0]?.detail;
        const runStatus = res.run.status === "running" ? "running" : "scheduled";
        setVerifyStates((prev) => {
          const cur = prev[mode];
          if (!cur || cur.testId !== testId) return prev;
          // liveDetail only ever UPGRADES (never reverts to undefined on a
          // poll that briefly has none -- see the original single-field
          // version of this check above) -- runStatus, unlike liveDetail,
          // has a real "not yet" value ("scheduled") so it's fine to set
          // unconditionally.
          if ((!liveDetail || cur.liveDetail === liveDetail) && cur.runStatus === runStatus) return prev;
          return { ...prev, [mode]: { ...cur, liveDetail: liveDetail || cur.liveDetail, runStatus } };
        });
      } catch {
        /* transient -- keep polling */
      }
      if (!unmountedRef.current) timer = window.setTimeout(poll, 5000);
    }
    // Same background-tab throttling gotcha as the chain poll above -- catch
    // up immediately on regained visibility instead of waiting on a timer
    // the browser may have paused.
    function onVisible() {
      if (document.visibilityState !== "visible") return;
      if (timer) window.clearTimeout(timer);
      void poll();
    }
    document.addEventListener("visibilitychange", onVisible);
    visibilityListeners.current.push(() => document.removeEventListener("visibilitychange", onVisible));
    void poll();
  }

  // Fires the same transition the "Start Test B"/"Start Sweep" buttons do,
  // the moment stageState says a stage is startable -- so autoAdvance is
  // never a second code path, just this page clicking its own button.
  useEffect(() => {
    if (!autoAdvance) return;
    for (const stage of STAGES) {
      const parent = stage === "refine" ? "tuning" : stage === "sweep" ? "refine" : null;
      if (!parent || chain[stage]) continue;
      if (stageState(stage) !== "startable") continue;
      const key = `${stage}:${chain[parent]}`;
      if (autoAdvanceAttempted.current.has(key)) continue;
      autoAdvanceAttempted.current.add(key);
      void startStage(stage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stageState/startStage close over chain+stageData, already deps below
  }, [autoAdvance, chain, stageData]);

  function resetAll(): void {
    setGoals(defaultGoals());
    setGoalsUnset(true);
    setRepeats(MIN_SCORING_REPEATS);
    persistChain({});
    setStageData({});
    setMsg("");
  }

  function savePreset(): void {
    const name = window.prompt("Save this intent (goal, target context, workload shape, KV cache preset, repeats) as:");
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
          State the goal; the console derives the chain — two tuning passes, then the sweep that produces scored
          cards, each a real run linked to the last. Building a grid by hand instead lives on{" "}
          <Link to="/custom-test" className="text-accent hover:underline">
            Custom Test
          </Link>
          .
        </p>
      </div>

      <div className="flex flex-wrap items-start gap-8">
        {/* Left column -- the numbered wizard ------------------------------- */}
        <div className="mx-auto flex min-w-0 flex-1 flex-col" style={{ maxWidth: 820 }}>
          {/* Step 1 -- pairing --------------------------------------------- */}
          <Step n={1} title="Pick your pairing" desc="Machine and model — everything below derives from this pair.">
            {/* Pickers: the same 2-col grid the detail cards below use, so
                Machine and Model line up pixel-perfectly with the cards
                each one derives -- Machine left, Model right. */}
            <div className="grid grid-cols-2 items-start gap-x-4 gap-y-3.5">
              <div className="flex flex-col gap-3">
                <label className="flex flex-col gap-1.5 text-sm">
                  <span className="text-muted">Machine</span>
                  <select
                    value={workerId}
                    onChange={(e) => setWorkerId(e.target.value)}
                    className={`${inputCls} min-h-[52px] w-full`}
                  >
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
                      className={`${inputCls} w-full`}
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
              </div>
              <ModelPicker
                models={presentModels}
                value={modelId}
                mtpValue=""
                onSelect={(id) => setModelId(id)}
                hfUpdates={{}}
                buttonClassName="min-h-[52px]"
              />
            </div>

            {selectedModel && selectedWorker ? (
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 rounded-lg border border-border bg-surface-raised px-3.5 py-2.5 text-[11.5px] text-muted">
                <span className="font-mono text-fg">{formatBytes(selectedModel.size_bytes)}</span>
                <span aria-hidden="true">·</span>
                <span>{modelLayerCount != null ? `${modelLayerCount} layers` : "layers not read"}</span>
                <span aria-hidden="true">·</span>
                <span>{trainedCtx != null ? `${trainedCtx.toLocaleString()} ctx` : "ctx not read"}</span>
                <span className="mx-1 hidden h-4 w-px self-stretch bg-border sm:block" aria-hidden="true" />
                <span>{gpuList.length > 0 ? gpuList.map((g) => formatGpuLabel(g)).join(" · ") : "no GPU detected"}</span>
                <span aria-hidden="true">·</span>
                <span>{selectedWorker.backend ?? "unknown"}</span>
                <span aria-hidden="true">·</span>
                <span className={workerStatusTone(selectedWorker.status)}>{selectedWorker.status}</span>
                <button
                  type="button"
                  onClick={() => setDetailsOpen((open) => !open)}
                  aria-expanded={detailsOpen}
                  className="ml-auto flex items-center gap-1 whitespace-nowrap text-[11.5px] font-semibold text-accent"
                >
                  {detailsOpen ? "Hide details" : "Show details"}
                  <IconChevronDown width={13} height={13} className={`transition-transform ${detailsOpen ? "rotate-180" : ""}`} />
                </button>
              </div>
            ) : (
              <p className="text-sm text-muted">Pick a machine and a model to see what their headers actually reported.</p>
            )}

            {detailsOpen && selectedModel && selectedWorker && (
              <div className="flex flex-col gap-3">
                {/* Same 2-col grid as the pickers above, strict 16px gutter,
                    both cards stretched to equal height -- Machine left,
                    Model right, matching the pickers' own left/right. */}
                <div className="grid grid-cols-2 items-stretch gap-4">
                  <div className="rounded-lg border border-border bg-surface-raised p-3.5">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">Machine</span>
                    <dl className="mt-2 flex flex-col gap-1.5 text-[12.5px]">
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
                        // M6/N6 -- declared UP FRONT so a missing
                        // thermally_throttled flag later reads as "this
                        // machine can't produce one", never as "nothing
                        // throttled".
                        hint="Whether this machine can report clock and temperature at all — a run that can't sample them never raises a thermal flag."
                      />
                    </dl>
                  </div>

                  <div className="rounded-lg border border-border bg-surface-raised p-3.5">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">Model</span>
                    <dl className="mt-2 flex flex-col gap-1.5 text-[12.5px]">
                      <Kv
                        label="Layers"
                        value={modelLayerCount != null ? `${modelLayerCount}` : "not read from this file's header"}
                        read={modelLayerCount != null}
                      />
                      {/* Trained context sits right under layers because it's
                          the model's own hard ceiling -- every context sizing
                          path (M2's target clamp, N1's ladder) clamps to it,
                          so it can never be raised by config no matter how
                          much VRAM exists. */}
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
                  </div>
                </div>

                {/* M1 -- the inverse estimate, one click away rather than
                    always-on: Q2 below already inlines its headline numbers
                    at the point they matter. --------------------------- */}
                <div className="rounded-lg border border-border bg-surface-raised p-3.5">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">
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
                        {affordability.weightsMib != null ? ` (~${affordability.weightsMib.toLocaleString()} MiB)` : ""}, a
                        10 % activation headroom and scratch. Binding constraint:{" "}
                        <span className="font-mono">{affordability.f16.binding ?? "—"}</span>. Advisory only — it ranks
                        and annotates, it never removes a configuration from the sweep.
                      </p>
                    </>
                  ) : affordability && affordability.f16.confidence !== "unknown" ? (
                    <p className="mt-2 text-[12px] leading-relaxed text-muted">
                      <b className="text-fg">This placement leaves no room for a KV cache at all.</b> The weights alone
                      {affordability.weightsMib != null ? ` (~${affordability.weightsMib.toLocaleString()} MiB)` : ""} fill
                      the card once a 10 % activation headroom and scratch are reserved, so the affordable context at
                      every KV type is zero — binding constraint <span className="font-mono">weights-placement</span>,
                      not KV. The lever is offload, not cache quality: the sweep stage already carries a lower{" "}
                      <span className="font-mono">-ngl</span> point, and <span className="font-mono">--n-cpu-moe</span>{" "}
                      on Custom Test moves expert weights off the card entirely. Advisory, as always — nothing is removed
                      from the grid over it.
                    </p>
                  ) : (
                    <p className="mt-2 text-[12px] leading-relaxed text-muted">
                      Unavailable for this pairing — the estimate needs this model's KV geometry from its header and a
                      live VRAM total from the machine. It never fabricates a number in their absence.
                      {affordability?.f16.conservativeFloorTokens != null && (
                        <>
                          {" "}
                          A conservative floor does exist: this model's full trained context of{" "}
                          <b className="text-fg">{affordability.f16.conservativeFloorTokens.toLocaleString()}</b> tokens,
                          offered as a floor, not as an estimate.
                        </>
                      )}
                    </p>
                  )}
                </div>
              </div>
            )}
          </Step>

          {/* Step 2 -- goal --------------------------------------------------- */}
          <Step n={2} title="State your goal" desc="Three quick calls — skip them and the console just optimizes.">
            <GoalQuestionnaire
              goals={goals}
              onChange={(next) => {
                setGoals(next);
                setGoalsUnset(false);
              }}
              trainedCtx={trainedCtx}
              unset={goalsUnset}
              // This console runs the chain over slider-expressible contexts
              // only, so the free-form number entry would offer a value
              // nothing downstream here can act on. Custom Test keeps it.
              showCtxNumberInput={false}
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
              placement={
                selectedModel && baseLayerCount != null
                  ? {
                      ngl: effectiveNgl,
                      onNglChange: setNglOverride,
                      nglMax: baseLayerCount,
                      kvLayerCount: modelLayerCount ?? 0,
                      modelSizeBytes: selectedModel.size_bytes,
                      tensorBreakdown: selectedModel.metadata.tensor_layer_bytes ?? null,
                      locked: noGpu ? "cpu" : unifiedPool ? "unified" : null,
                      vram: { freeMib: liveVramFreeMib, totalMib: liveVramTotalMib },
                      ram: { freeMib: liveRamFreeMib, totalMib: liveRamTotalMib },
                      unifiedPool,
                      noGpu,
                      poolHaircutFrac,
                      onRunModes: runModes,
                      onReset: resetVerify,
                      verifyResults: verifyStates,
                    }
                  : undefined
              }
            />
          </Step>

          {/* Step 3 -- the chain ----------------------------------------------- */}
          <Step n={3} title="Review the chain & run" desc="Two tuning passes feed the sweep — only the sweep scores cards." last>
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <span className="text-sm font-medium text-fg">Chain — 3 stages, all llama-bench</span>
              <span className="font-mono text-[11.5px] text-muted">
                {totalItems} test{totalItems === 1 ? "" : "s"} · est. {totalPriced}
              </span>
            </div>

            {/* Same control, same state as the sticky "Your test" card below --
                it governs the whole chain's behaviour, so it belongs where
                the chain is being reviewed too, not only where it's started. */}
            <label className="flex items-center gap-2 text-[11.5px] text-muted">
              <input
                type="checkbox"
                checked={autoAdvance}
                onChange={(e) => setAutoAdvance(e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              Auto-start the next stage as soon as the previous one finishes
            </label>

            <div className="mt-2 flex flex-wrap items-stretch gap-y-2">
              {stagePlans.map((plan) => {
                const state = stageState(plan.stage);
                const highlight = state === "startable" || state === "live";
                const dim = state === "blocked" && plan.itemCount === 0;
                return (
                  <div key={plan.stage} className="contents">
                    <div
                      className={`flex w-[180px] shrink-0 flex-col gap-1.5 rounded-lg bg-surface-raised p-3 ${
                        highlight ? "border-2 border-accent" : "border border-border"
                      } ${dim ? "opacity-65" : ""}`}
                    >
                      <div className="flex items-center gap-1.5">
                        {!plan.run && (
                          <span
                            className={
                              state === "blocked"
                                ? "h-[7px] w-[7px] shrink-0 rounded-full shadow-[inset_0_0_0_1.5px_var(--color-muted)]"
                                : "h-[7px] w-[7px] shrink-0 rounded-full bg-accent"
                            }
                          />
                        )}
                        <b className="text-[12.5px] text-fg">{STAGE_TITLE[plan.stage]}</b>
                      </div>
                      {plan.run && <TestStatusPill status={plan.run.status} />}
                      <div className="font-mono text-[10.5px] leading-relaxed text-muted">
                        {plan.itemCount > 0 ? (
                          <>
                            {plan.itemCount} test{plan.itemCount === 1 ? "" : "s"} ·{" "}
                            {plan.stage === "sweep" ? "KV × FA × ngl × depth" : "PP only · batch/ubatch"}
                          </>
                        ) : (
                          <>rules, not values yet</>
                        )}
                      </div>
                      {plan.itemCount > 0 && <div className="text-[10.5px] text-muted">est. {priceStage(plan.sweep)}</div>}
                      {plan.itemCount === 0 && (
                        <div className="text-[10.5px] leading-relaxed text-muted">
                          One octave either side of whatever {STAGE_TITLE.tuning} wins — there is nothing to centre on
                          until it has.
                        </div>
                      )}

                      <div className="mt-auto flex flex-col gap-1 pt-1">
                        {plan.testId ? (
                          <Link
                            to={`/tests/${plan.testId}`}
                            className="text-center text-[11px] font-semibold text-accent hover:underline"
                          >
                            View run →
                          </Link>
                        ) : dim ? (
                          <span className="rounded-md border border-border py-1 text-center text-[10.5px] text-muted">
                            Waiting
                          </span>
                        ) : (
                          <button
                            type="button"
                            disabled={state !== "startable" || busyStage != null || !ready}
                            onClick={() => void startStage(plan.stage)}
                            className="rounded-md bg-accent py-1 text-[11px] font-bold text-accent-fg disabled:opacity-40"
                          >
                            {STAGE_START_LABEL[plan.stage]}
                          </button>
                        )}
                        {/* Refine is a convenience, not a requirement: a chain
                            may go straight from the coarse winner to the
                            scored sweep, which is depth 2 of the 3 §0.5
                            allows. */}
                        {plan.stage === "sweep" && !plan.testId && !chain.refine && chain.tuning && (
                          <button
                            type="button"
                            disabled={busyStage != null || stageState("refine") === "blocked"}
                            onClick={() => void startStage("sweep", "tuning")}
                            className="rounded-md border border-border py-1 text-[10.5px] font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-40"
                          >
                            Skip refine
                          </button>
                        )}
                      </div>
                    </div>
                    <div className="flex w-6 shrink-0 items-center justify-center text-border">
                      <IconArrowRight width={16} height={16} />
                    </div>
                  </div>
                );
              })}

              <div className="flex w-[180px] shrink-0 flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-border p-3 text-center">
                <b className="text-[12.5px] text-accent">Cards per goal</b>
                <div className="font-mono text-[10.5px] text-muted">
                  {goals.goal === "max_context"
                    ? "Max Context + Low Memory"
                    : goals.goal === "max_speed"
                      ? "Max Speed + Low Memory"
                      : "Balanced + Low Memory"}
                </div>
                {cardsRunId ? (
                  <Link to={`/tests/${cardsRunId}`} className="mt-1 text-[11px] font-semibold text-accent hover:underline">
                    Open scored cards →
                  </Link>
                ) : (
                  <div className="mt-1 text-[10.5px] leading-relaxed text-muted">
                    Only the sweep stage scores — tuning stages feed it values, never cards.
                  </div>
                )}
              </div>
            </div>

            <div>
              <button
                type="button"
                onClick={() => setScoringDetailsOpen((open) => !open)}
                aria-expanded={scoringDetailsOpen}
                className="flex items-center gap-1 text-xs text-muted hover:text-fg"
              >
                {scoringDetailsOpen ? "Hide scoring details ▴" : "Show scoring details ▾"}
              </button>
              {scoringDetailsOpen && (
                <div className="mt-2.5 rounded-lg border border-border bg-surface-raised p-3.5">
                  {/* M4 -- the axis, as it will actually expand ------------ */}
                  <p className="text-[11.5px] leading-relaxed text-muted">
                    <b className="text-fg">
                      Sweep axis ({KV_PRESET_LABEL[goals.kv_preset ?? "extended"]} preset, {kvPairs.length} pair
                      {kvPairs.length === 1 ? "" : "s"}):
                    </b>{" "}
                    {kvPairs.map((pair) => (
                      <span
                        key={pair}
                        className="mr-1.5 inline-block rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 font-mono text-[11px] text-accent"
                      >
                        {pair}
                      </span>
                    ))}
                    — the {sweepPlan.itemCount}-test count above already reflects it.
                  </p>

                  {/* M7/M2 -- held fields keep their reasons ---------------- */}
                  <p className="mt-2.5 text-[11.5px] leading-relaxed text-muted">
                    <b className="text-fg">Held fields keep their reasons.</b> Threads are held at {threads} (one less
                    than this machine's cores): these stages target fully offloaded configurations where{" "}
                    <span className="font-mono">-t</span> barely bites. On CPU-bound rows it is the dominant variable
                    and stays an axis on the Custom Test page, with the running build's own ISA provenance recorded
                    alongside those rows.{" "}
                    {goals.target_ctx != null ? (
                      <>
                        Depth is anchored to your target:{" "}
                        <span className="font-mono">
                          {Math.round(goals.target_ctx * 0.5).toLocaleString()} = 50 % of your{" "}
                          {goals.target_ctx.toLocaleString()}
                        </span>
                        .
                      </>
                    ) : (
                      <>
                        Depth stays at 0 — without a target context in Q2 there is nothing for a percentage to be a
                        percentage of, and a house number would not be your workload.
                      </>
                    )}
                  </p>
                </div>
              )}
            </div>

            {/* Repeats --------------------------------------------------------- */}
            <div className="flex flex-wrap items-center gap-3.5 border-t border-border pt-5">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted">Repeats per test</span>
              <div className="flex gap-1.5" role="radiogroup" aria-label="Repeats per test">
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
                            ? "rounded-full border border-border bg-surface-raised px-3 py-0.5 text-[12px] text-muted line-through opacity-50"
                            : "rounded-full border border-border bg-surface-raised px-3 py-0.5 text-[12px] text-muted hover:text-fg"
                      }
                    >
                      {n}
                    </button>
                  );
                })}
              </div>
              <span
                className="inline-flex items-center gap-1 text-[11px] text-muted"
                title="Fewer than 3 reports a standard deviation of exactly 0 — the stability gate would pass on a number that was never measured."
              >
                below {MIN_SCORING_REPEATS}, results aren't stable enough to score <IconInfo width={12} height={12} />
              </span>
            </div>

            {msg && <p className="text-xs text-muted">{msg}</p>}
          </Step>
        </div>

        {/* Right column -- the sticky run summary ---------------------------- */}
        <aside className="sticky top-6 flex w-[300px] shrink-0 flex-col gap-3.5 rounded-xl border border-border bg-surface p-5">
          <span className="text-sm font-bold text-fg">Your test</span>

          {ready ? (
            <div className="flex flex-col gap-0.5">
              <span className="break-all font-mono text-[11.5px] text-fg">{selectedModel?.filename ?? modelId}</span>
              <span className="text-[11.5px] text-muted">on {selectedWorker?.displayName ?? workerId}</span>
            </div>
          ) : (
            <p className="text-[11.5px] text-muted">Pick a machine and a model above to build a chain.</p>
          )}

          <div className="h-px bg-border" />

          <div>
            <div className="text-[12.5px] font-semibold text-fg">
              {goalLabel}
              {goals.target_ctx != null ? ` · ${goals.target_ctx.toLocaleString()}` : ""} · {goals.workload}
            </div>
            <div className="mt-0.5 font-mono text-[11px] text-muted">
              wPP {WORKLOAD_WEIGHTS[goals.workload].wPP.toFixed(2)} · wTG {WORKLOAD_WEIGHTS[goals.workload].wTG.toFixed(2)}
            </div>
            <p className="mt-1.5 text-[10.5px] leading-relaxed text-muted">
              Change the goal later — re-scoring is instant post-processing over stored results, never re-measurement.
            </p>
          </div>

          <div className="h-px bg-border" />

          <div className="flex flex-col gap-1.5 text-[11.5px]">
            {stagePlans.map((plan) => (
              <div key={plan.stage} className="flex justify-between gap-2">
                <span className="text-muted">{STAGE_TITLE[plan.stage]}</span>
                <span className={plan.itemCount > 0 ? "text-fg" : "text-muted"}>
                  {plan.itemCount > 0 ? `${plan.itemCount} tests` : "—"}
                </span>
              </div>
            ))}
            <div className="flex justify-between gap-2 border-t border-border pt-1.5 font-bold">
              <span className="text-fg">Total</span>
              <span className="text-fg">{totalItems} tests</span>
            </div>
            <div className="leading-relaxed text-muted">est. {totalPriced}</div>
          </div>

          {/* Sits with the start button rather than inside the chain strip:
              it governs what happens AFTER you press start, so this is where
              the decision is actually being made. */}
          <label className="flex items-center gap-2 text-[11.5px] text-muted">
            <input
              type="checkbox"
              checked={autoAdvance}
              onChange={(e) => setAutoAdvance(e.target.checked)}
              className="h-3.5 w-3.5 accent-accent"
            />
            Auto-start the next stage as soon as the previous one finishes
          </label>

          <button
            type="button"
            disabled={stageState("tuning") !== "startable" || busyStage != null || !ready}
            onClick={() => void startStage("tuning")}
            className="mt-0.5 w-full rounded-lg bg-accent py-2.5 text-sm font-bold text-accent-fg disabled:opacity-40"
          >
            {STAGE_START_LABEL.tuning}
          </button>
          {!chain.tuning && <div className="-mt-1.5 text-center text-[10.5px] text-muted">Unlocks Tune refine when it finishes</div>}

          {msg && <p className="text-center text-[11px] leading-relaxed text-muted">{msg}</p>}

          <div className="h-px bg-border" />

          <div className="text-[11px] leading-relaxed text-muted">
            {cardsRunId ? (
              <Link to={`/tests/${cardsRunId}`} className="font-semibold text-accent hover:underline">
                Open scored cards →
              </Link>
            ) : (
              <>Scored cards will appear here once the sweep stage runs.</>
            )}
          </div>

          <div className="relative flex flex-wrap justify-center gap-x-3.5 gap-y-1 text-[11px]">
            <button
              type="button"
              onClick={savePreset}
              title="Saves the goals block and repeat count — goal, target context, workload shape, KV cache preset. Describes your workload, not a machine, so it loads verbatim on any pairing."
              className="text-muted hover:text-fg"
            >
              Save preset
            </button>
            <button
              type="button"
              onClick={() => setPresetsOpen((open) => !open)}
              aria-expanded={presetsOpen}
              disabled={Object.keys(presets).length === 0}
              className="text-muted hover:text-fg disabled:opacity-40"
            >
              Load preset
            </button>
            <button type="button" onClick={resetAll} className="text-muted hover:text-fg">
              Reset all
            </button>
            {presetsOpen && (
              <div className="absolute bottom-full z-10 mb-1 flex max-h-56 min-w-40 flex-col overflow-y-auto rounded-lg border border-border bg-surface-raised p-1 shadow-lg">
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
        </aside>
      </div>
    </div>
  );
}

// One numbered step of the wizard: a circular badge connected to the next
// step by a vertical rule (omitted on the last step), a click-to-collapse
// title+description header, then whatever the step needs in a
// loosely-gapped column. Kept local -- this shape (badge, connector,
// heading) is specific to this page's own pairing -> goal -> chain
// narrative, not a general-purpose disclosure.
//
// Collapsing hides the body with `hidden` rather than unmounting it, so a
// step's own state (Step 1's detailsOpen, GoalQuestionnaire's KV-preset
// disclosure) survives being closed and reopened.
function Step({
  n,
  title,
  desc,
  last,
  children,
}: {
  n: number;
  title: string;
  desc: string;
  last?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="flex gap-4">
      <div className="flex w-[26px] shrink-0 flex-col items-center">
        <div className="flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-full bg-accent text-[12.5px] font-bold text-accent-fg">
          {n}
        </div>
        {!last && <div className="mt-1.5 w-0.5 flex-1 bg-border" />}
      </div>
      <div className={`min-w-0 flex-1 ${last ? "" : "pb-7"}`}>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex w-full items-center gap-2 text-left"
        >
          <span className="min-w-0 flex-1">
            <span className="block text-base font-bold text-fg">{title}</span>
            <span className="mt-0.5 block text-xs text-muted">{desc}</span>
          </span>
          <IconChevronDown
            width={16}
            height={16}
            className={`shrink-0 text-muted transition-transform ${open ? "" : "-rotate-90"}`}
          />
        </button>
        <div className={open ? "mt-3.5 flex flex-col gap-3" : "hidden"}>{children}</div>
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
