import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { ChipInput } from "../components/ChipInput";
import { ModelPicker } from "../components/ModelPicker";
import { SliderChipInput } from "../components/SliderChipInput";
import { ToggleChipGroup } from "../components/ToggleChipGroup";
import { IconChevronDown } from "../components/icons";
import { isMtpDraftModel, backendVisibleGpus, detectBackend } from "../types";
import type { Model, SweepConfig, Backend, WorkerVramInfo } from "../types";
import { formatGpuLabel, formatBytes } from "../utils";
import { estimateVramNeededMib, estimateSafeNgl } from "../vramEstimate";
import { GoalQuestionnaire } from "../components/GoalQuestionnaire";
import { defaultGoals, goalsEqualDefaults, normalizeGoals, type GoalsConfig } from "../goals";
import { expandSweep } from "../../../shared/sweep";
import { priceMatrix, ETA_UNAVAILABLE } from "../../../shared/pricing";
import type { ModelRatesResponse } from "../types";

type Sweep = Omit<SweepConfig, "model_id">;

// Shown only until a model+worker pairing is picked (fields are disabled
// until then -- see the fieldset below) or once one is picked with no
// remembered sweep for it. Deliberately empty rather than pre-filled with
// opinionated numbers: the user asked not to see default values that look
// chosen but aren't, especially while the controls are still disabled.
const EMPTY_SWEEP: Sweep = {
  n_prompt: [],
  n_gen: [],
  threads: [],
  n_gpu_layers: [],
  batch_size: [],
  ubatch_size: [],
  cache_type_k: [],
  cache_type_v: [],
  flash_attn: [],
  mtp: [],
  n_gpu_layers_draft: [],
  n_cpu_moe: [],
  repeats: 1,
};

// Keyed by workerId (a server-assigned UUID) since MULTIUSER_PLAN.md's pull
// queue -- old entries keyed by the previous Tailscale-derived worker name
// simply go unused and expire naturally; not worth an explicit migration for
// a device-local convenience cache.
function sweepStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:sweep:${modelId}:${workerId}`;
}

function loadRememberedSweep(modelId: string, workerId: string): Sweep | null {
  try {
    const raw = localStorage.getItem(sweepStorageKey(modelId, workerId));
    return raw ? sanitizeSweep(JSON.parse(raw) as Sweep) : null;
  } catch {
    return null;
  }
}

// M5 -- the goals block is saved beside the grid, under its own key so a
// preset written before this amendment simply has no entry and loads as
// "unset" instead of silently gaining fabricated answers.
function goalsStorageKey(modelId: string, workerId: string): string {
  return `llamatoaster:goals:${modelId}:${workerId}`;
}

function loadRememberedGoals(modelId: string, workerId: string): GoalsConfig | null {
  try {
    const raw = localStorage.getItem(goalsStorageKey(modelId, workerId));
    return raw ? (normalizeGoals(JSON.parse(raw)) ?? null) : null;
  } catch {
    return null;
  }
}

function rememberGoals(modelId: string, workerId: string, goals: GoalsConfig): void {
  try {
    localStorage.setItem(goalsStorageKey(modelId, workerId), JSON.stringify(goals));
  } catch {
    /* localStorage unavailable -- not worth failing the run over */
  }
}

function rememberSweep(modelId: string, workerId: string, sweep: Sweep): void {
  try {
    localStorage.setItem(sweepStorageKey(modelId, workerId), JSON.stringify(sweep));
  } catch {
    /* localStorage unavailable (private browsing, quota) -- not worth failing the run over */
  }
}

// Same "remember across visits" idea as the sweep above, but for the
// worker+model pairing itself -- last worker picked overall, and (since a
// worker's registered models differ) last model picked *for that worker*.
const LAST_WORKER_STORAGE_KEY = "llamatoaster:new-run:last-worker";

function lastModelStorageKey(workerId: string): string {
  return `llamatoaster:new-run:last-model:${workerId}`;
}

interface LastModelSelection {
  modelId: string;
  mtpModelId: string;
}

function loadLastWorker(): string | null {
  try {
    return localStorage.getItem(LAST_WORKER_STORAGE_KEY);
  } catch {
    return null;
  }
}

function rememberLastWorker(workerId: string): void {
  try {
    localStorage.setItem(LAST_WORKER_STORAGE_KEY, workerId);
  } catch {
    /* localStorage unavailable -- not worth failing over */
  }
}

function loadLastModel(workerId: string): LastModelSelection | null {
  try {
    const raw = localStorage.getItem(lastModelStorageKey(workerId));
    return raw ? (JSON.parse(raw) as LastModelSelection) : null;
  } catch {
    return null;
  }
}

function rememberLastModel(workerId: string, selection: LastModelSelection): void {
  try {
    localStorage.setItem(lastModelStorageKey(workerId), JSON.stringify(selection));
  } catch {
    /* localStorage unavailable -- not worth failing over */
  }
}

// Sorted biggest to smallest (bits per weight): f32 (32) -> bf16/f16 (16) ->
// q8_0 (8) -> q5_1/q5_0 (~5) -> q4_1/q4_0/iq4_nl (~4) -- llama.cpp's full
// accepted set for --cache-type-k/-ctk and --cache-type-v/-ctv.
const CACHE_TYPES = ["f32", "bf16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"];
const FLASH_ATTN_OPTIONS = ["on", "off"];
// "off" first (not "on" like FLASH_ATTN_OPTIONS above) -- sanitizeSweep below
// falls back to options[0] for an empty/invalid value, and unlike flash
// attention, "on" isn't a safe universal default (most models have no MTP
// head at all), so the fallback must land on "off".
const MTP_OPTIONS = ["off", "on"];

// Toggle-group option sets for the sweep's string-enum fields, used to
// sanitize a remembered sweep on load below.
const STRING_FIELD_OPTIONS = {
  cache_type_k: CACHE_TYPES,
  cache_type_v: CACHE_TYPES,
  flash_attn: FLASH_ATTN_OPTIONS,
  mtp: MTP_OPTIONS,
} as const;

// Drops values that no longer match a field's valid options -- guards against
// a stray/invalid value (e.g. from a past seeding bug, or an option set that's
// since changed) getting permanently stuck in a remembered sweep, since
// ToggleChipGroup can only ever toggle off a value it renders a button for.
// `cleaned[field] ?? []` matters here specifically for "mtp": a sweep
// remembered in localStorage before this field existed has no "mtp" key at
// all, so without the fallback this would throw calling .filter on undefined.
function sanitizeSweep(sweep: Sweep): Sweep {
  const cleaned = { ...sweep };
  for (const field of Object.keys(STRING_FIELD_OPTIONS) as (keyof typeof STRING_FIELD_OPTIONS)[]) {
    const validOptions: readonly string[] = STRING_FIELD_OPTIONS[field];
    const filtered = (cleaned[field] ?? []).filter((v) => validOptions.includes(v));
    cleaned[field] = filtered.length > 0 ? filtered : [validOptions[0]];
  }
  // Same missing-key issue as "mtp" above, but for a numeric field a remembered
  // sweep predating it -- falls back to the same [0] "not applicable yet"
  // placeholder the initial-seed effect below uses, which the draftLayerCount
  // effect then upgrades once a draft model is actually picked.
  if (!cleaned.n_gpu_layers_draft || cleaned.n_gpu_layers_draft.length === 0) {
    cleaned.n_gpu_layers_draft = [0];
  }
  // Same "not applicable yet" placeholder story as n_gpu_layers_draft above,
  // for a remembered sweep predating this field.
  if (!cleaned.n_cpu_moe || cleaned.n_cpu_moe.length === 0) {
    cleaned.n_cpu_moe = [0];
  }
  return cleaned;
}

// llama-bench's -b/-ub both take a plain integer with no enforced enum --
// these are the conventional power-of-two steps its own docs/examples sweep
// over (e.g. "-b 128,256,512,1024"), spanning from well below the -ub
// default (512) to well above the -b default (2048). uBatch is capped lower
// since it must stay ≤ batch size (its own default is already 4x smaller).
const BATCH_SIZE_PRESETS = [32, 64, 128, 256, 512, 1024, 2048, 4096];
const UBATCH_SIZE_PRESETS = [32, 64, 128, 256, 512, 1024, 2048];

// Ceilings used until the real number is known -- unreachable/predates-hardware-
// reporting worker, or a model registered before GGUF layer-count parsing
// existed. Matches the old static "999 offloads everything" convention this
// page used before it could read a model's actual layer count.
const NGL_FALLBACK_MAX = 999;
const THREADS_FALLBACK_MAX = 64;

const NUMERIC_FIELDS = [
  "n_prompt",
  "n_gen",
  "threads",
  "n_gpu_layers",
  "batch_size",
  "ubatch_size",
  "n_gpu_layers_draft",
  "n_cpu_moe",
] as const;
const STRING_FIELDS = ["cache_type_k", "cache_type_v", "flash_attn", "mtp"] as const;

// Quick client-side shape check so a bad "Apply JSON" fails fast with a
// specific message -- the server (routes/runs.ts's validateSweep) remains
// the actual source of truth and re-validates on submit regardless.
function validateSweepShape(s: unknown): string | null {
  if (!s || typeof s !== "object") return "sweep must be an object";
  const obj = s as Record<string, unknown>;
  for (const f of NUMERIC_FIELDS) {
    const v = obj[f];
    if (!Array.isArray(v) || v.length === 0 || !v.every((x) => typeof x === "number" && Number.isFinite(x))) {
      return `${f} must be a non-empty array of numbers`;
    }
  }
  for (const f of STRING_FIELDS) {
    const v = obj[f];
    if (!Array.isArray(v) || v.length === 0 || !v.every((x) => typeof x === "string" && x.length > 0)) {
      return `${f} must be a non-empty array of strings`;
    }
  }
  if (typeof obj.repeats !== "number" || !Number.isInteger(obj.repeats) || obj.repeats < 1) {
    return "repeats must be a positive integer";
  }
  return null;
}

function field<K extends keyof Sweep>(
  sweep: Sweep,
  setSweep: (updater: (s: Sweep) => Sweep) => void,
  key: K
) {
  return {
    value: sweep[key],
    onChange: (value: Sweep[K]) => setSweep((s) => ({ ...s, [key]: value })),
  };
}

export function NewRun() {
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  // Companion --model-draft model for a base model whose MTP head isn't
  // baked into its own file (Gemma-4-style) -- unused/irrelevant for a
  // Qwen-style model with its own metadata.mtp_layers. See the mtp toggle
  // section below for how the two cases are distinguished in the UI.
  const [mtpModelId, setMtpModelId] = useState("");
  const [workerId, setWorkerId] = useState("");
  const { order: workerOrder, status: workerStatus } = useWorkerStatuses();
  // Raw index into the selected worker's hardware.gpu array (the picker's
  // own display order) -- undefined means "Auto" (let llama.cpp use its own
  // default, split across every visible GPU). Distinct from the mainGpu
  // value actually sent to the server: that one must be relative to
  // whichever backend will run this request (backendVisibleGpus), which can
  // differ from this raw index whenever gpuOverrideBackend below is set --
  // see mainGpuForSubmit/mainGpuBackendForSubmit further down. Only ever
  // meaningful when that worker's own GPU list has more than one entry, see
  // showGpuPicker below.
  const [selectedGpuRawIndex, setSelectedGpuRawIndex] = useState<number | undefined>(undefined);
  // Set once the user confirms switching backends for a GPU the worker's own
  // default backend can't see at all (see the GPU <select>'s onChange below)
  // -- the specific backend confirmed, forwarded as RunConfig.main_gpu_backend
  // so the server installs (without disturbing the worker's own default) and
  // the worker runs this one sweep against it. Cleared whenever the
  // selection changes back to Auto or a GPU the default backend already
  // covers.
  const [gpuOverrideBackend, setGpuOverrideBackend] = useState<Backend | undefined>(undefined);
  const [sweep, setSweep] = useState<Sweep>(EMPTY_SWEEP);
  // M2 -- the questionnaire's answers. Kept beside the sweep and saved with
  // it (M5: goals are intent about the WORKLOAD, not a property of a machine,
  // so they travel with presets exactly like grids do). `goalsUnset` marks a
  // preset saved before this existed: the controls show defaults, MARKED as
  // unset, and no answers are fabricated.
  const [goals, setGoals] = useState<GoalsConfig>(defaultGoals());
  const [goalsUnset, setGoalsUnset] = useState(true);
  const [jsonText, setJsonText] = useState("");
  const [jsonError, setJsonError] = useState("");
  const [msg, setMsg] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [lastRunId, setLastRunId] = useState("");
  const [lookingUpLayers, setLookingUpLayers] = useState(false);
  const [layerLookupMsg, setLayerLookupMsg] = useState("");
  // Live per-model HF "last modified" check, keyed by model id -- not stored
  // anywhere, just fetched once for the picker's "Updated X ago" / "update?"
  // hints. See server/src/routes/models.ts's /api/models/hf-updates.
  const [hfUpdates, setHfUpdates] = useState<Record<string, string | null>>({});
  // Live "which worker(s) have this model's file" map, same endpoint Models.tsx
  // uses to hide deleted-from-disk models. null until the fetch resolves --
  // see presentModels below, which treats "no data yet" the same as "nothing
  // confirmed present" rather than guessing.
  const [locations, setLocations] = useState<Record<string, string[]> | null>(null);
  // Workers the locations scan couldn't reach right now -- purely informational
  // here (presentModels below already treats an unreachable worker as having
  // nothing confirmed, same as Models.tsx's per-worker grouping does), used
  // only to explain *why* the picker looks empty for that worker.
  const [unreachableLocationWorkers, setUnreachableLocationWorkers] = useState<string[]>([]);

  useEffect(() => {
    (async () => {
      setModels(await api.listModels());
    })();
    (async () => {
      try {
        setHfUpdates(await api.getModelHfUpdates());
      } catch {
        /* best-effort -- the picker just shows no "Updated"/"update?" hints */
      }
    })();
    (async () => {
      try {
        const res = await api.getModelLocations();
        setLocations(res.locations);
        setUnreachableLocationWorkers(res.unreachable);
      } catch {
        /* best-effort -- picker falls back to the unfiltered registry, see above */
      }
    })();
  }, []);

  // Only offer models actually confirmed present -- on the currently
  // selected worker once one's picked, or on any worker before that.
  // Deliberately strict, per the user's explicit call: no remembered model
  // for an offline worker, only ones actually there right now. So unlike an
  // earlier version of this filter, there's no "unreachable/not-loaded-yet
  // -> fall back to the unfiltered registry" escape hatch -- no confirmed
  // data means nothing is offered, exactly mirroring Models.tsx's own
  // per-worker `locations[id].includes(worker.name)` gating (which likewise
  // shows nothing for a worker it couldn't reach).
  const presentModels = useMemo(() => {
    if (!locations) return [];
    if (workerId) return models.filter((m) => locations[m.id]?.includes(workerId));
    return models.filter((m) => (locations[m.id]?.length ?? 0) > 0);
  }, [models, locations, workerId]);

  // Prefer whichever worker was picked last time (if it's still configured);
  // only fall back to auto-picking the sole worker when there's nothing
  // remembered yet.
  useEffect(() => {
    if (workerId || workerOrder.length === 0) return;
    const remembered = loadLastWorker();
    if (remembered && workerOrder.includes(remembered)) {
      setWorkerId(remembered);
    } else if (workerOrder.length === 1) {
      setWorkerId(workerOrder[0]);
    }
  }, [workerOrder, workerId]);

  useEffect(() => {
    if (workerId) rememberLastWorker(workerId);
  }, [workerId]);

  // A companion model picked for one base model doesn't carry over to a
  // different one -- reset whenever the base model selection changes.
  // Skipped once when handleModelSelect below sets both modelId and
  // mtpModelId together (picking an MTP/draft row pairs them in one click) --
  // without the flag this effect would fire right after and immediately wipe
  // out the mtpModelId it just set.
  const skipMtpResetRef = useRef(false);
  useEffect(() => {
    if (skipMtpResetRef.current) {
      skipMtpResetRef.current = false;
      return;
    }
    setMtpModelId("");
  }, [modelId]);

  function handleModelSelect(newModelId: string, draftId?: string) {
    if (draftId) skipMtpResetRef.current = true;
    setModelId(newModelId);
    setMtpModelId(draftId ?? "");
  }

  // Once a worker is picked (restored or manual) and this worker's models
  // have loaded, restore whichever model was last picked *for that worker* --
  // once per worker change (restoredForRef guards against re-firing after
  // the restore itself sets modelId), and never if the user has already
  // picked something for it in this session.
  const restoredModelForRef = useRef<string | null>(null);
  useEffect(() => {
    if (!workerId || presentModels.length === 0) return;
    if (restoredModelForRef.current === workerId) return;
    restoredModelForRef.current = workerId;
    if (modelId) return;
    const remembered = loadLastModel(workerId);
    if (!remembered || !presentModels.some((m) => m.id === remembered.modelId)) return;
    const draftId =
      remembered.mtpModelId && presentModels.some((m) => m.id === remembered.mtpModelId)
        ? remembered.mtpModelId
        : undefined;
    handleModelSelect(remembered.modelId, draftId);
  }, [workerId, presentModels, modelId]);

  useEffect(() => {
    if (!workerId || !modelId) return;
    rememberLastModel(workerId, { modelId, mtpModelId });
  }, [workerId, modelId, mtpModelId]);

  // Hardware/layer-count context for the selected pairing -- drives the
  // GPU-layers and threads sliders below. Best-effort: hardware is only
  // known once that worker's own /llama-cpp+/hardware fetch resolves (see
  // useWorkerStatuses), and a model's layer count is only known if it was
  // downloaded through this app after GGUF parsing existed (worker/src/gguf.ts).
  const selectedModel = models.find((m) => m.id === modelId);
  const modelLayerCount = typeof selectedModel?.metadata.n_layer === "number" ? selectedModel.metadata.n_layer : null;

  // Live free-VRAM reading for the pre-flight VRAM-fit banner below (see
  // shared/vramEstimate.ts) -- fetched once per worker pick, not polled, so
  // it stays cheap and non-intrusive. null while unfetched, while no worker
  // is picked yet, or if the fetch failed (e.g. a cpu-backend worker with
  // nothing meaningful to report) -- vramFitEstimate below treats null the
  // same "don't guess" way every other unknown-context value in this file
  // does (see nglMax's NGL_FALLBACK_MAX fallback).
  const [liveVram, setLiveVram] = useState<WorkerVramInfo | null>(null);
  useEffect(() => {
    setLiveVram(null);
    if (!workerId) return;
    let cancelled = false;
    api
      .getWorkerVram(workerId)
      .then((info) => {
        if (!cancelled) setLiveVram(info);
      })
      .catch(() => {
        if (!cancelled) setLiveVram(null);
      });
    return () => {
      cancelled = true;
    };
  }, [workerId]);

  // MTP context -- see shared/types.ts's ModelMetadata for what these mean.
  // A model with its own mtp_layers is self-sufficient (Qwen-style); one
  // without needs an explicit --model-draft companion picked from a draft
  // model registered from the same hf_repo (Gemma-4-style, see
  // isMtpDraftModel). The base-model dropdown above excludes draft models
  // entirely -- they're not standalone-benchmarkable. Both the companion
  // picker and the on/off toggle below are shown unconditionally once a
  // model is picked (rather than only for models auto-detected as
  // MTP-capable) since that detection is best-effort -- see the warning
  // notice next to the toggle.
  const modelMtpCapable = typeof selectedModel?.metadata.mtp_layers === "number" && selectedModel.metadata.mtp_layers > 0;
  // Every registered draft is listed (not hidden) so the user can see the
  // full set that exists, but only ones sharing the base model's own
  // hf_repo -- the real HF folder-structure signal a drafter file is
  // published under (e.g. unsloth's MTP/ subfolder lives inside the same
  // repo as its base quants) -- are selectable; the rest render disabled
  // with a reason via `fitReason`. Architecture "family" alone was tried
  // and rejected: Gemma-4-E2B and Gemma-4-E4B are both "Gemma" but ship
  // unrelated drafter files from separate repos (see modelGrouping.ts /
  // ModelPicker.tsx, which already pair this same way).
  function draftFitReason(draft: Model): string | null {
    if (!selectedModel?.hf_repo) {
      return "Selected model has no Hugging Face repo on record, so no draft can be confirmed to match it.";
    }
    if (!draft.hf_repo) {
      return "This draft has no Hugging Face repo on record, so it can't be confirmed to match the selected model.";
    }
    if (draft.hf_repo !== selectedModel.hf_repo) {
      return `From a different Hugging Face repo (${draft.hf_repo}) than the selected model (${selectedModel.hf_repo}) -- won't work as its --model-draft companion.`;
    }
    return null;
  }
  const mtpDraftCandidates = presentModels.filter((m) => isMtpDraftModel(m));
  const showMtpModelPicker = Boolean(modelId);

  const selectedWorker = workerId ? workerStatus[workerId] : undefined;
  const workerHardware = selectedWorker?.hardware ?? undefined;
  const workerPlatform = selectedWorker?.platform ?? undefined;
  const workerDefaultBackend = selectedWorker?.backend ?? undefined;
  // Prefer the directly-detected GPU list once hardware has loaded; until
  // then fall back to the worker's configured backend (already inline on
  // every Worker, no separate fetch needed) since a "cpu" backend build
  // ignores -ngl regardless of what's physically in the box.
  const noGpu = workerHardware ? workerHardware.gpu.length === 0 : workerDefaultBackend === "cpu";
  // The full, unfiltered picker -- every physically detected GPU is offered,
  // even ones the worker's current default backend can't see at all (e.g.
  // an Intel iGPU under a cuda build). Picking one of those triggers a
  // confirm prompt in the <select>'s onChange below rather than being
  // silently hidden or silently mis-routed -- see gpuOverrideBackend above.
  const gpuList = workerHardware?.gpu ?? [];
  const showGpuPicker = gpuList.length > 1;

  // The backend this run will actually execute under: the confirmed
  // override if the user picked a GPU the worker's own default backend
  // can't see, otherwise that default.
  const effectiveBackend = gpuOverrideBackend ?? workerDefaultBackend;
  const selectedGpu = selectedGpuRawIndex != null ? gpuList[selectedGpuRawIndex] : undefined;
  // mainGpu sent to the server must be relative to whichever backend will
  // actually run this request (see shared/types.ts's backendVisibleGpus'
  // doc comment on RunConfig.main_gpu) -- not selectedGpuRawIndex above,
  // which is just this picker's own display order.
  const mainGpuForSubmitRaw =
    selectedGpu && effectiveBackend ? backendVisibleGpus(gpuList, effectiveBackend).indexOf(selectedGpu) : -1;
  const mainGpuForSubmit = mainGpuForSubmitRaw >= 0 ? mainGpuForSubmitRaw : undefined;

  // A GPU picked for one worker doesn't carry over to a different one --
  // its hardware.gpu array (and therefore what index N even refers to) is
  // specific to that worker, and any pending backend-switch confirmation
  // was about that worker's own build, not the new one.
  useEffect(() => {
    setSelectedGpuRawIndex(undefined);
    setGpuOverrideBackend(undefined);
  }, [workerId]);

  // llama.cpp's own runtime layer accounting is n_layer + 1 -- it counts the
  // output layer separately from the transformer stack (confirmed live, see
  // shared/types.ts's ResultRow.total_model_layers comment) -- so the
  // slider's max has to match that, or a fully-offloaded run's own "X/43"
  // result can look like it silently undershot a request that was actually
  // correct (see -- the exact confusion this fixes).
  const OUTPUT_LAYER = 1;
  const baseLayerCount = modelLayerCount != null ? modelLayerCount + OUTPUT_LAYER : null;

  // The draft/MTP model's own layers ride alongside -ngl but are offloaded
  // independently of it, via llama-server's own -ngld/--n-gpu-layers-draft
  // (confirmed live against the installed build's --help -- see
  // worker/src/serverBench.ts's buildArgs) -- hence its own slider below
  // rather than folding it into the base model's -ngl range. Qwen-style
  // self-sufficient models (modelMtpCapable) fold their MTP head into the
  // same file's own n_layer already, so there's no separate draft model or
  // slider for them at all.
  const selectedMtpDraftModel = mtpModelId ? presentModels.find((m) => m.id === mtpModelId) : undefined;
  const draftLayerCount =
    typeof selectedMtpDraftModel?.metadata.n_layer === "number"
      ? selectedMtpDraftModel.metadata.n_layer + OUTPUT_LAYER
      : null;
  const mtpSelected = sweep.mtp.includes("on") && (modelMtpCapable || Boolean(mtpModelId));
  const showDraftNglSlider = mtpSelected && !modelMtpCapable && Boolean(mtpModelId);
  const draftNglMax = noGpu ? 0 : draftLayerCount ?? NGL_FALLBACK_MAX;
  const draftNglSuggested = noGpu ? 0 : draftLayerCount ?? NGL_FALLBACK_MAX;
  const draftNglSuggestedLabel = noGpu
    ? "Suggested: 0 -- this worker has no GPU backend"
    : draftLayerCount != null
      ? `Suggested: ${draftLayerCount} -- offload every layer (this draft model has ${draftLayerCount}, including the output layer)`
      : `Suggested: ${NGL_FALLBACK_MAX} -- draft model's layer count isn't known yet, showing the "offload everything" default`;

  const nglMax = noGpu ? 0 : baseLayerCount ?? NGL_FALLBACK_MAX;
  const nglSuggested = noGpu ? 0 : baseLayerCount ?? NGL_FALLBACK_MAX;
  const nglSuggestedLabel = noGpu
    ? "Suggested: 0 -- this worker has no GPU backend"
    : baseLayerCount != null
      ? `Suggested: ${baseLayerCount} -- offload every layer (this model has ${baseLayerCount}, including the output layer)`
      : `Suggested: ${NGL_FALLBACK_MAX} -- model's layer count isn't known yet, showing the "offload everything" default`;

  // --n-cpu-moe's own N counts transformer blocks the same way llama.cpp's
  // -ngl does at the low end (0 = off), but its ceiling is modelLayerCount
  // itself, NOT baseLayerCount (+1) above -- the output layer has no MoE
  // experts of its own to keep on CPU, so it isn't part of what --n-cpu-moe
  // counts. Suggested starts at 0 regardless of context (inert by default,
  // per the feature's own "greyed out/disabled until deliberately raised"
  // intent) rather than mirroring nglSuggested's "default to max" posture.
  const cpuMoeMax = noGpu ? 0 : modelLayerCount ?? NGL_FALLBACK_MAX;
  const cpuMoeSuggested = 0;
  const cpuMoeSuggestedLabel = noGpu
    ? "Suggested: 0 -- this worker has no GPU backend"
    : modelLayerCount != null
      ? `Suggested: 0 -- raise to keep some of this model's ${modelLayerCount} MoE layers' experts on CPU RAM instead of GPU VRAM`
      : `Suggested: 0 -- model's layer count isn't known yet`;

  // Pre-flight VRAM-fit estimate -- null (no banner) whenever any input is
  // unknown (model size, layer count, or a live VRAM reading), same "don't
  // guess" posture as nglSuggestedLabel's own NGL_FALLBACK_MAX fallback
  // above. Uses the highest -ngl value currently in the sweep, since that's
  // the one that would actually risk overcommitting VRAM. See
  // shared/vramEstimate.ts for the formula and its known limitations
  // (a flat per-layer average, weakest for MoE models).
  const vramFitEstimate = useMemo(() => {
    if (!selectedModel || baseLayerCount == null || liveVram?.vram_free_before_mib == null) return null;
    const requestedNgl = sweep.n_gpu_layers.length > 0 ? Math.max(...sweep.n_gpu_layers) : 0;
    if (requestedNgl <= 0) return null;
    const neededMib = estimateVramNeededMib({
      modelSizeBytes: selectedModel.size_bytes,
      totalModelLayers: baseLayerCount,
      requestedNgl,
    });
    if (neededMib == null) return null;
    const freeMib = liveVram.vram_free_before_mib;
    return {
      modelSizeBytes: selectedModel.size_bytes,
      neededMib,
      freeMib,
      safeNgl: estimateSafeNgl(selectedModel.size_bytes, baseLayerCount, freeMib),
      fits: neededMib <= freeMib,
    };
  }, [selectedModel, baseLayerCount, sweep.n_gpu_layers, liveVram]);

  const threadsMax = workerHardware?.cpu.cores || THREADS_FALLBACK_MAX;
  const threadsSuggested = Math.max(1, threadsMax - 1);
  const threadsSuggestedLabel = workerHardware?.cpu.cores
    ? `Suggested: ${threadsSuggested} -- one less than this worker's ${threadsMax} threads`
    : `Suggested: ${threadsSuggested} -- worker thread count isn't known yet, showing a default range`;

  // Once a specific model+worker pairing is picked, swap in whatever sweep
  // was last triggered for that exact pairing -- or a blank sweep if this
  // pairing has never been run, rather than leaving the previous selection's
  // values in place unnoticed or showing opinionated numbers nobody chose.
  // The GPU-layers/threads sliders still surface a contextual recommendation
  // via their "Suggested: X" label/handle position, but nothing is added to
  // the sweep's actual value chips until the user picks a preset or hits Add.
  useEffect(() => {
    if (!modelId || !workerId) {
      setSweep(EMPTY_SWEEP);
      return;
    }
    const remembered = loadRememberedSweep(modelId, workerId);
    if (remembered) {
      setSweep(remembered);
      // M5 -- a legacy preset lacks the goals key: treat goals as UNSET
      // (defaults, marked) rather than inventing answers the user never gave.
      const rememberedGoals = loadRememberedGoals(modelId, workerId);
      setGoals(rememberedGoals ?? defaultGoals());
      setGoalsUnset(rememberedGoals == null);
      return;
    }
    setGoals(defaultGoals());
    setGoalsUnset(true);
    // No remembered sweep: seed sensible defaults so the form is submittable.
    // CPU-only workers: n_gpu_layers must be 0 (input is disabled).
    // GPU workers: offer 0 and max (offload everything) as starting points.
    const baseSweep = { ...EMPTY_SWEEP };
    baseSweep.n_prompt = [512];
    baseSweep.n_gen = [128];
    baseSweep.threads = [4];
    baseSweep.batch_size = [512];
    baseSweep.ubatch_size = [512];
    baseSweep.cache_type_k = ["q8_0"];
    baseSweep.cache_type_v = ["q8_0"];
    baseSweep.flash_attn = ["on"];
    baseSweep.mtp = ["off"];
    if (noGpu) {
      baseSweep.n_gpu_layers = [0];
    } else {
      baseSweep.n_gpu_layers = [0, nglMax];
    }
    // Always seeded inert (0), never a "try both" [0, max] pair like
    // n_gpu_layers above -- unlike full GPU offload, there's no sensible
    // universal "on" default for MoE-CPU-offload to suggest trying, so the
    // control just starts inert as a manual knob the user can raise for a
    // MoE model that won't fit in VRAM.
    baseSweep.n_cpu_moe = [0];
    // Placeholder until a draft model is actually picked (mtpModelId isn't
    // known yet at this point -- it's reset separately, see the effect
    // above) -- the draftLayerCount effect below upgrades this to a real
    // "offload everything" default once one is. [0] rather than [] so the
    // form stays submittable even if MTP/draft never gets touched at all.
    baseSweep.n_gpu_layers_draft = [0];
    setSweep(baseSweep);
  }, [modelId, workerId, noGpu, nglMax]);

  // Mirrors lookupLayerCount's "only fill in if the user hasn't customized
  // it yet" posture, but automatic rather than button-triggered: fires the
  // moment a draft model's layer count becomes known (picked just now, or
  // already known when a remembered sweep restored mtpModelId). Only
  // replaces the untouched [0] placeholder from baseSweep/sanitizeSweep
  // above -- never overwrites a value the user (or a remembered sweep) has
  // actually set.
  useEffect(() => {
    if (draftLayerCount == null) return;
    setSweep((s) =>
      s.n_gpu_layers_draft.length === 1 && s.n_gpu_layers_draft[0] === 0
        ? { ...s, n_gpu_layers_draft: [draftLayerCount] }
        : s
    );
  }, [draftLayerCount]);

  function setField<K extends keyof Sweep>(key: K, value: Sweep[K]) {
    setSweep((s) => ({ ...s, [key]: value }));
  }

  // Backfills n_layer for a model registered before GGUF parsing existed --
  // asks every configured worker for whichever of them already has the file
  // (see server/src/routes/models.ts) rather than re-downloading anything.
  async function lookupLayerCount() {
    if (!modelId) return;
    setLookingUpLayers(true);
    setLayerLookupMsg("");
    try {
      const res = await api.backfillModelLayerCount(modelId);
      setModels(await api.listModels());
      if (res.n_layer == null) {
        setLayerLookupMsg("Not found on any configured worker yet.");
      } else {
        setLayerLookupMsg("");
        // Only fill in the sweep's GPU-layers chips if the user hasn't added
        // any yet -- never overwrite values they've already customized.
        setSweep((s) => (s.n_gpu_layers.length === 0 ? { ...s, n_gpu_layers: [res.n_layer as number] } : s));
      }
    } catch (err) {
      setLayerLookupMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLookingUpLayers(false);
    }
  }

  function openAdvanced() {
    setJsonText(JSON.stringify(sweep, null, 2));
    setJsonError("");
  }

  function applyJson() {
    try {
      const parsed = JSON.parse(jsonText);
      const err = validateSweepShape(parsed);
      if (err) {
        setJsonError(err);
        return;
      }
      setSweep(parsed as Sweep);
      setJsonError("");
    } catch {
      setJsonError("Invalid JSON");
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!modelId || !workerId) {
      setMsg("Pick a model and a worker first");
      return;
    }
    setSubmitting(true);
    setMsg("Queuing…");
    try {
      const run = await api.triggerRun({
        model_id: modelId,
        worker_id: workerId,
        mtp_model_id: mtpModelId || undefined,
        main_gpu: mainGpuForSubmit,
        main_gpu_backend: gpuOverrideBackend,
        sweep,
        // M2's "skippable by construction": an untouched questionnaire sends
        // NO goals key at all, making the payload byte-identical to what this
        // page sent before the amendment existed.
        goals: goalsEqualDefaults(goals) && goalsUnset ? undefined : goals,
      });
      rememberSweep(modelId, workerId, sweep);
      rememberGoals(modelId, workerId, goals);
      setLastRunId(run.id);
      // Every trigger comes back 'scheduled' now -- it only flips to
      // 'running' once the worker actually claims the job off its queue
      // (MULTIUSER_PLAN.md §1.12/§1.13), which can take up to one queue-poll
      // cycle even for an idle machine.
      setMsg("Run scheduled ✓");
    } catch (err) {
      setMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSubmitting(false);
    }
  }

  const [workflowMsg, setWorkflowMsg] = useState("");

  // N6 -- the CPU-focused preset template. Threads are the DOMINANT variable
  // on CPU-bound rows (M7 says so explicitly), so this is where -t stops being
  // held and starts being an axis; ngl is pinned at 0 to keep it that way.
  function applyCpuTemplate(): void {
    const ladder = [...new Set([1, 2, 4, Math.max(1, threadsMax - 1), threadsMax])]
      .filter((n) => n >= 1 && n <= threadsMax)
      .sort((a, b) => a - b);
    setSweep((current) => ({
      ...current,
      threads: ladder,
      n_gpu_layers: [0],
      n_cpu_moe: [0],
      n_gpu_layers_draft: [0],
      mtp: ["off"],
    }));
    setWorkflowMsg(
      `CPU template applied: threads varied across ${ladder.join(", ")} with ngl pinned at 0. The running build's own ISA dispatch is recorded on these rows, so two CPUs on the same build tag are no longer silently comparable.`
    );
  }

  // M2 -- Test B's prompt sizes are PRE-SELECTED around the stated target, so
  // the default matrix is centred on the user's own question rather than a
  // fixed ladder.
  function applyTargetSizes(): void {
    const target = goals.target_ctx;
    if (target == null) {
      setWorkflowMsg("State a target context first — that is what the sizes are anchored to.");
      return;
    }
    const around = [...new Set([Math.round(target / 8), Math.round(target / 2), target])]
      .filter((n) => n >= 128)
      .sort((a, b) => a - b);
    setSweep((current) => ({ ...current, n_prompt: around }));
    setWorkflowMsg(`Prompt sizes pre-selected around your ${target.toLocaleString()} target: ${around.join(", ")}.`);
  }

  // N5 -- the concurrency ladder. Rows land as ordinary results with
  // `concurrency` set; the knee is derived on read, never stored as a verdict.
  async function startKneeLadder(): Promise<void> {
    if (!modelId || !workerId) {
      setWorkflowMsg("Pick a model and a machine first.");
      return;
    }
    const nPrompt = goals.target_ctx ?? sweep.n_prompt[0] ?? 4096;
    const nGen = sweep.n_gen[0] ?? 128;
    setWorkflowMsg("Enqueueing the concurrency ladder…");
    try {
      const run = await api.triggerRun({
        model_id: modelId,
        worker_id: workerId,
        kind: "runtime",
        knee: { n_prompt: nPrompt, n_gen: nGen, slots: [1, 2, 4, 8], repeats: 1 },
        sweep,
      });
      setLastRunId(run.id);
      setWorkflowMsg(
        "Concurrency ladder scheduled at slots 1 · 2 · 4 · 8. The knee is the smallest slot count whose TTFT p95 exceeds 2× its slots = 1 value — derived from the stored rows, never asserted."
      );
    } catch (err) {
      setWorkflowMsg(err instanceof Error ? err.message : String(err));
    }
  }

  // The LIVE count is the actual expansion -- this page builds one flat
  // cross-product by hand (K x V included: there's no curated-preset concept
  // here, that's the Benchmark page's job), so nothing needs pruning before
  // expandSweep sees it. A quantized pair crossed with flash_attn:"off" is
  // still silently skipped per-item by expandSweep itself (shared/sweep.ts's
  // isValidCombo), which is why the count below can be lower than the raw
  // axis product without anything here having to reconcile it.
  const expandedItems = useMemo(() => {
    try {
      return expandSweep(sweep).length;
    } catch {
      return 0;
    }
  }, [sweep]);

  // §0.6 -- the ETA prices from a {server, spec:"off"} rate first, then a
  // labeled llama-bench rate, then not at all. No number renders without its
  // source, so the label travels with the rate.
  const [rates, setRates] = useState<ModelRatesResponse | null>(null);
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
        /* pricing is advisory; a failed lookup renders "ETA unavailable" */
      });
    return () => {
      cancelled = true;
    };
  }, [modelId, workerId]);

  const priced = useMemo(() => {
    if (!rates) return { seconds: null, display: `${ETA_UNAVAILABLE} — no measured rates for this pairing yet` };
    const items = expandSweep(sweep);
    return priceMatrix(
      items.map((item) => ({
        nPrompt: item.n_prompt,
        nGen: item.n_gen,
        repeats: sweep.repeats,
        ppRate: rates.pp,
        tgRate: rates.tg,
      }))
    );
  }, [rates, sweep]);

  const inputCls =
    "rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent";

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">New Run</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Every combination of the values below is expanded into one llama-bench sweep, averaged
        over the repeat count.
      </p>

      <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-6">
        {/* M2 -- captured BEFORE the grid is built: everything downstream
            (scoring weights, the KV axis, depth anchoring, which cards exist
            at completion) derives from these answers. */}
        <GoalQuestionnaire
          goals={goals}
          onChange={(next) => {
            setGoals(next);
            setGoalsUnset(false);
          }}
          trainedCtx={typeof selectedModel?.metadata.trained_ctx === "number" ? selectedModel.metadata.trained_ctx : null}
          unset={goalsUnset}
          affordability={{
            totalMib: liveVram?.gpu_memory_total_mib ?? null,
            weightsMib:
              selectedModel && baseLayerCount != null
                ? estimateVramNeededMib({
                    modelSizeBytes: selectedModel.size_bytes,
                    totalModelLayers: baseLayerCount,
                    requestedNgl: Math.max(...(sweep.n_gpu_layers.length > 0 ? sweep.n_gpu_layers : [0])),
                  })
                : null,
            nLayer: selectedModel?.metadata.n_layer,
            nHeadKv: selectedModel?.metadata.n_head_kv,
            headDimK: selectedModel?.metadata.head_dim_k,
            headDimV: selectedModel?.metadata.head_dim_v,
            nEmbd: selectedModel?.metadata.n_embd,
            nHead: selectedModel?.metadata.n_head,
            slidingWindow: selectedModel?.metadata.sliding_window,
          }}
        />

        <div className="flex flex-wrap gap-4">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted">Worker</span>
            <select
              value={workerId}
              onChange={(e) => setWorkerId(e.target.value)}
              className={`${inputCls} w-48`}
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
          {showGpuPicker && (
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">GPU</span>
              <select
                value={selectedGpuRawIndex ?? ""}
                onChange={(e) => {
                  const raw = e.target.value === "" ? undefined : Number(e.target.value);
                  if (raw == null) {
                    setSelectedGpuRawIndex(undefined);
                    setGpuOverrideBackend(undefined);
                    return;
                  }
                  const gpu = gpuList[raw];
                  const required = workerPlatform ? detectBackend(workerPlatform, [gpu]) : workerDefaultBackend;
                  // Only a real mismatch (both known, and different) needs
                  // confirming -- an unreachable/unknown worker backend
                  // falls back to "assume it matches," same as noGpu above,
                  // rather than blocking every selection on missing data.
                  if (workerDefaultBackend && required && required !== workerDefaultBackend) {
                    const proceed = window.confirm(
                      `${formatGpuLabel(gpu)} needs the "${required}" llama.cpp backend -- this worker's build is ` +
                        `currently "${workerDefaultBackend}".\n\n` +
                        `Install and activate a ${required} build for just this run? (The worker's own default ` +
                        `stays ${workerDefaultBackend} for every other run.)\n\n` +
                        `Choose Cancel to pick a different GPU instead.`
                    );
                    if (!proceed) {
                      // The native <select> already jumped to the clicked
                      // option before this handler ran; since state (and
                      // therefore the controlled `value` prop) isn't
                      // changing on a cancel, nothing would otherwise force
                      // the DOM back in sync with React's model -- revert it
                      // imperatively instead of leaving the visible
                      // selection out of sync with selectedGpuRawIndex.
                      e.target.value = selectedGpuRawIndex != null ? String(selectedGpuRawIndex) : "";
                      return;
                    }
                    setGpuOverrideBackend(required);
                  } else {
                    setGpuOverrideBackend(undefined);
                  }
                  setSelectedGpuRawIndex(raw);
                }}
                className={`${inputCls} w-56`}
              >
                <option value="">Auto (split across all GPUs)</option>
                {gpuList.map((g, i) => (
                  <option key={i} value={i}>
                    {i}: {formatGpuLabel(g)}
                  </option>
                ))}
              </select>
              {gpuOverrideBackend && (
                <span className="text-xs text-muted">
                  Will install/activate a {gpuOverrideBackend} build for this run only.
                </span>
              )}
            </label>
          )}
          <ModelPicker
            models={presentModels}
            value={modelId}
            mtpValue={mtpModelId}
            onSelect={handleModelSelect}
            hfUpdates={hfUpdates}
            className="w-[26rem]"
          />
          {showMtpModelPicker && (
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">MTP model (draft)</span>
              <select
                value={mtpModelId}
                onChange={(e) => setMtpModelId(e.target.value)}
                className={`${inputCls} w-72`}
              >
                <option value="">none</option>
                {mtpDraftCandidates.map((m) => {
                  const fitReason = draftFitReason(m);
                  return (
                    <option key={m.id} value={m.id} disabled={fitReason !== null} title={fitReason ?? undefined}>
                      {m.filename} ({m.id.slice(0, 12)})
                      {fitReason ? " -- doesn't match selected model" : ""}
                    </option>
                  );
                })}
              </select>
            </label>
          )}
        </div>

        {workerId && unreachableLocationWorkers.includes(workerId) && (
          <p className="-mt-2 text-xs text-warning">
            Couldn't check what's on {selectedWorker?.displayName ?? "this machine"} -- its models aren't shown
            below until that's confirmed, even if they're actually still there.
          </p>
        )}

        {!modelId && (
          <p className="-mt-2 text-sm text-muted">Select a model to configure sweep parameters.</p>
        )}

        <fieldset
          disabled={!modelId}
          className={`flex min-w-0 flex-col gap-6 border-0 p-0 m-0 ${
            modelId ? "" : "pointer-events-none opacity-50"
          }`}
        >
        <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-5 rounded-xl border border-border bg-surface p-5">
          <ChipInput
            label="Prompt length (-p)"
            hint="Prompt tokens processed before generating (the pp test). 0 disables the pp phase."
            presets={[128, 512, 2048, 4096]}
            {...field(sweep, setSweep, "n_prompt")}
          />
          <ChipInput
            label="Generation length (-n)"
            hint="Tokens generated after the prompt (the tg test). 0 disables the tg phase."
            presets={[128, 256]}
            {...field(sweep, setSweep, "n_gen")}
          />
          {/* BENCHMARKING_PLAN_V8.md §0.2 -- the depth axis. -d prefills the
              KV cache before each test, so TG measures a context already
              populated to this depth. llama-bench ONLY: llama-server has no
              KV-prefill flag, so the trigger rejects a non-zero depth on any
              server-engine configuration and names the fix. Depth values are
              anchored to the target context stated in Q2. */}
          <ChipInput
            label="KV depth (-d)"
            hint={
              goals.target_ctx != null
                ? `Tokens already in the KV cache before each test — llama-bench only. Percentages of your ${goals.target_ctx.toLocaleString()} target: ${[0.25, 0.5, 0.75].map((f) => Math.round((goals.target_ctx ?? 0) * f)).join(", ")}.`
                : "Tokens already in the KV cache before each test — llama-bench only, because llama-server has no KV-prefill flag. State a target context in Q2 to anchor the percentages."
            }
            presets={
              goals.target_ctx != null
                ? [0, ...[0.25, 0.5, 0.75].map((f) => Math.round((goals.target_ctx ?? 0) * f))]
                : [0, 4096, 16384]
            }
            value={sweep.n_depth ?? []}
            onChange={(value: number[]) => setSweep((current) => ({ ...current, n_depth: value }))}
          />
          <SliderChipInput
            label="Threads (-t)"
            hint="CPU threads used for compute — matters most on cpu or partially-offloaded runs."
            min={1}
            max={threadsMax}
            suggested={threadsSuggested}
            suggestedLabel={threadsSuggestedLabel}
            {...field(sweep, setSweep, "threads")}
          />
          <SliderChipInput
            label="GPU layers (-ngl)"
            hint="Model layers offloaded to GPU, capped to this model's real layer count once known. 0 keeps it all on CPU."
            min={0}
            max={nglMax}
            suggested={nglSuggested}
            suggestedLabel={nglSuggestedLabel}
            disabled={noGpu}
            disabledNote="This worker's build has no GPU backend, so offload is forced to 0."
            footer={
              modelId &&
              modelLayerCount == null &&
              !noGpu && (
                <div className="mt-1.5 flex items-center gap-2">
                  <button
                    type="button"
                    disabled={lookingUpLayers}
                    onClick={lookupLayerCount}
                    className="text-xs font-semibold text-accent hover:underline disabled:opacity-50"
                  >
                    {lookingUpLayers ? "Looking up…" : "Look up this model's real layer count"}
                  </button>
                  {layerLookupMsg && <span className="text-xs text-muted">{layerLookupMsg}</span>}
                </div>
              )
            }
            {...field(sweep, setSweep, "n_gpu_layers")}
          />
          <SliderChipInput
            label="MoE CPU layers (--n-cpu-moe)"
            hint="Keeps this many of the model's Mixture-of-Experts layers' expert weights on CPU RAM instead of GPU VRAM -- lets a MoE model too big for VRAM still get GPU-accelerated attention. 0 omits the flag entirely."
            min={0}
            max={cpuMoeMax}
            suggested={cpuMoeSuggested}
            suggestedLabel={cpuMoeSuggestedLabel}
            disabled={noGpu}
            disabledNote={
              noGpu ? "This worker's build has no GPU backend, so there's nothing to keep on CPU vs GPU." : undefined
            }
            {...field(sweep, setSweep, "n_cpu_moe")}
          />
          <ChipInput
            label="Batch size (-b)"
            hint="Logical batch size — max tokens grouped per prompt-eval step."
            presets={BATCH_SIZE_PRESETS}
            {...field(sweep, setSweep, "batch_size")}
          />
          <ChipInput
            label="uBatch size (-ub)"
            hint="Physical batch size — must be ≤ batch size; smaller uses less VRAM per step."
            presets={UBATCH_SIZE_PRESETS}
            {...field(sweep, setSweep, "ubatch_size")}
          />
        </div>

        {vramFitEstimate && !vramFitEstimate.fits && (
          <div className="rounded-lg border border-warning/30 bg-warning-bg px-3.5 py-2.5 text-xs text-warning">
            <span className="font-semibold">May not fit in free VRAM.</span> This model
            (~{formatBytes(vramFitEstimate.modelSizeBytes)}) at the GPU-layers value(s) you've set needs an estimated
            ~{formatBytes(vramFitEstimate.neededMib * 1024 * 1024)}, but this worker currently has only
            ~{formatBytes(vramFitEstimate.freeMib * 1024 * 1024)} free. If it doesn't fit, llama.cpp may still
            report success while silently falling back to system RAM -- results would then reflect CPU, not GPU,
            performance (see the run's "Download logs" output for a warning if that happens). This is a rough
            estimate based on average layer size, not an exact calculation, and doesn't account for which GPU a
            multi-GPU worker actually uses.
            <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-warning/30 pt-2">
              <button
                type="button"
                onClick={() =>
                  setSweep((s) => ({
                    ...s,
                    n_gpu_layers: [...new Set(s.n_gpu_layers.map((v) => Math.min(v, vramFitEstimate.safeNgl)))],
                  }))
                }
                className="rounded-md border border-warning/50 px-2.5 py-1 text-xs font-semibold text-warning hover:bg-warning/10"
              >
                Cap GPU layers to ~{vramFitEstimate.safeNgl} (estimated safe)
              </button>
              <span className="text-warning/80">
                Lowers every GPU-layers value in your sweep to fit the estimate above. Still approximate -- your run
                could fail with an out-of-memory error, especially for Mixture-of-Experts models where layer sizes
                vary.
              </span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-5 rounded-xl border border-border bg-surface p-5">
          <ToggleChipGroup
            label="K cache type (-ctk)"
            hint="Quantization of the K cache. Lower precision saves memory at some quality cost."
            options={CACHE_TYPES}
            {...field(sweep, setSweep, "cache_type_k")}
          />
          <ToggleChipGroup
            label="V cache type (-ctv)"
            hint="Quantization of the V cache. Lower precision saves memory at some quality cost."
            options={CACHE_TYPES}
            {...field(sweep, setSweep, "cache_type_v")}
          />
          <ToggleChipGroup
            label="Flash attention (-fa)"
            hint="Fused attention kernel — usually faster and lower-memory when the backend supports it."
            options={FLASH_ATTN_OPTIONS}
            {...field(sweep, setSweep, "flash_attn")}
          />
          <ToggleChipGroup
            label="MTP (multi-token prediction)"
            hint={
              modelMtpCapable
                ? "Speculative decoding using this model's own built-in MTP head. Runs via llama-server instead of llama-bench -- llama-bench itself has no MTP support."
                : mtpModelId
                  ? "Speculative decoding using the MTP model picked above as --model-draft. Runs via llama-server instead of llama-bench."
                  : "Needs either a model with its own built-in MTP head, or an MTP model (draft) picked above -- otherwise triggering with this on will fail."
            }
            options={MTP_OPTIONS}
            {...field(sweep, setSweep, "mtp")}
          />
          {showDraftNglSlider && (
            <SliderChipInput
              label="Draft GPU layers (-ngld)"
              hint="MTP draft model's own layers offloaded to GPU, independent of the base model's -ngl above."
              min={0}
              max={draftNglMax}
              suggested={draftNglSuggested}
              suggestedLabel={draftNglSuggestedLabel}
              disabled={noGpu}
              disabledNote="This worker's build has no GPU backend, so offload is forced to 0."
              {...field(sweep, setSweep, "n_gpu_layers_draft")}
            />
          )}
        </div>

        <div className="rounded-lg border border-warning/30 bg-warning-bg px-3.5 py-2.5 text-xs text-warning">
          <span className="font-semibold">Experimental — MTP (multi-token prediction).</span> Support isn't
          detected reliably for every model. Check the model's page on Hugging Face first to confirm it
          actually has MTP support (or a matching MTP/draft companion file) before turning this on --
          triggering it against an unsupported model or mismatched pairing will fail the run.
        </div>

        <div className="rounded-xl border border-border bg-surface p-5">
          <div className="flex items-baseline justify-between">
            <label className="text-sm font-medium text-fg">Repeats (-r)</label>
            <span className="text-xs text-muted">Runs averaged per combination — higher smooths out noise, takes longer.</span>
          </div>
          <div className="mt-1.5 flex items-center gap-3">
            <button
              type="button"
              onClick={() => setField("repeats", Math.max(1, sweep.repeats - 1))}
              className="h-8 w-8 rounded-lg border border-border text-fg hover:border-accent/40"
            >
              −
            </button>
            <input
              type="range"
              min={1}
              max={20}
              value={sweep.repeats}
              onChange={(e) => setField("repeats", Number(e.target.value))}
              className="flex-1"
            />
            <span className="w-10 text-center text-sm text-fg">{sweep.repeats}×</span>
            <button
              type="button"
              onClick={() => setField("repeats", sweep.repeats + 1)}
              className="h-8 w-8 rounded-lg border border-border text-fg hover:border-accent/40"
            >
              +
            </button>
          </div>
          {sweep.repeats < 3 && (
            <p className="mt-2 text-[11px] leading-relaxed text-warning" role="alert">
              Fewer than 3 reports a standard deviation of exactly 0 — the stability gate would pass on a number
              that was never measured. §0.3 requires n ≥ 3 for a config to score at all; below that, every
              placement here will land in the "stability" rejection tally instead of a profile card.
            </p>
          )}
        </div>

        <details
          className="group rounded-xl border border-border bg-surface"
          onToggle={(e) => {
            if ((e.target as HTMLDetailsElement).open) openAdvanced();
          }}
        >
          <summary
            tabIndex={modelId ? undefined : -1}
            onClick={(e) => {
              if (!modelId) e.preventDefault();
            }}
            className="flex cursor-pointer items-center justify-between px-5 py-3 text-sm font-semibold uppercase tracking-wide text-muted"
          >
            Advanced: raw JSON
            <IconChevronDown width={16} height={16} className="transition-transform group-open:rotate-180" />
          </summary>
          <div className="flex flex-col gap-2 border-t border-border px-5 py-4">
            <textarea
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              rows={14}
              spellCheck={false}
              className="w-full rounded-lg border border-border bg-surface-raised p-3 font-mono text-xs text-fg outline-none focus:border-accent"
            />
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={applyJson}
                className="self-start rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent"
              >
                Apply JSON to form
              </button>
              {jsonError && <span className="text-sm text-danger">{jsonError}</span>}
            </div>
          </div>
        </details>
        </fieldset>

        {/* The workflows v8 adds on top of an ordinary sweep. Each states what
            it does to the grid rather than silently rewriting it. */}
        <div className="rounded-xl border border-border bg-surface p-5">
          <span className="text-sm font-medium text-fg">Workflows</span>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={applyCpuTemplate}
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              CPU-focused template
            </button>
            <button
              type="button"
              onClick={applyTargetSizes}
              aria-disabled={goals.target_ctx == null}
              title={
                goals.target_ctx == null
                  ? "State a target context in Q2 first — that is what the sizes anchor to"
                  : "Pre-selects prompt sizes around your stated target"
              }
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              Sizes around my target
            </button>
            <button
              type="button"
              onClick={() => void startKneeLadder()}
              disabled={!modelId || !workerId}
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
            >
              Start concurrency ladder
            </button>
          </div>
          {workflowMsg && <p className="mt-2 text-xs leading-relaxed text-muted">{workflowMsg}</p>}
        </div>

        {/* The grid as it will actually run. The LIVE COUNT is the real
            expansion, so the number here and the number of tests that
            execute never disagree. */}
        <div className="rounded-xl border border-border bg-surface p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <span className="text-sm font-medium text-fg">Grid as it will run</span>
            <span className="font-mono text-xs text-muted">
              {expandedItems} test{expandedItems === 1 ? "" : "s"} · {priced.display}
            </span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-muted">
            <b className="text-fg">Held fields keep their reasons.</b> Threads are held at the machine default:
            Test A targets fully offloaded configurations where <span className="font-mono">-t</span> barely
            bites. On CPU-bound rows it is the dominant variable and stays an Expert-mode axis, with the running
            build’s own ISA dispatch recorded alongside those rows.
            {goals.target_ctx != null && (
              <>
                {" "}
                Depth percentages are anchored to your target:{" "}
                <span className="font-mono">depth % of your {goals.target_ctx.toLocaleString()} target</span>.
              </>
            )}
          </p>
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Once this finishes, cards are ranked for{" "}
            <b className="text-fg">
              {goals.goal === "max_context"
                ? `Max context · ≥ ${Math.round((goals.speed_floor_frac ?? 0.5) * 100)} % TG`
                : goals.goal === "max_speed"
                  ? "Max tok/s"
                  : "Balanced"}
              {goals.target_ctx != null ? ` · ${goals.target_ctx.toLocaleString()}` : ""} ·{" "}
              {goals.workload}
            </b>
            . Changing the goal later re-scores instantly over stored results — never re-measurement.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <button
            type="submit"
            disabled={submitting}
            className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-accent-fg disabled:opacity-50"
          >
            Trigger run
          </button>
          {msg && <span className="text-sm text-muted">{msg}</span>}
          {lastRunId && (
            <Link to={`/runs/${lastRunId}`} className="text-sm text-accent hover:underline">
              View run →
            </Link>
          )}
        </div>
      </form>
    </div>
  );
}
