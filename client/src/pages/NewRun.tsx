import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { ChipInput } from "../components/ChipInput";
import { ModelPicker } from "../components/ModelPicker";
import { SliderChipInput } from "../components/SliderChipInput";
import { ToggleChipGroup } from "../components/ToggleChipGroup";
import { IconChevronDown } from "../components/icons";
import { isMtpDraftModel } from "../types";
import type { Model, SweepConfig } from "../types";

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
  repeats: 1,
};

function sweepStorageKey(modelId: string, workerName: string): string {
  return `llamatoaster:sweep:${modelId}:${workerName}`;
}

function loadRememberedSweep(modelId: string, workerName: string): Sweep | null {
  try {
    const raw = localStorage.getItem(sweepStorageKey(modelId, workerName));
    return raw ? sanitizeSweep(JSON.parse(raw) as Sweep) : null;
  } catch {
    return null;
  }
}

function rememberSweep(modelId: string, workerName: string, sweep: Sweep): void {
  try {
    localStorage.setItem(sweepStorageKey(modelId, workerName), JSON.stringify(sweep));
  } catch {
    /* localStorage unavailable (private browsing, quota) -- not worth failing the run over */
  }
}

// Same "remember across visits" idea as the sweep above, but for the
// worker+model pairing itself -- last worker picked overall, and (since a
// worker's registered models differ) last model picked *for that worker*.
const LAST_WORKER_STORAGE_KEY = "llamatoaster:new-run:last-worker";

function lastModelStorageKey(workerName: string): string {
  return `llamatoaster:new-run:last-model:${workerName}`;
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

function rememberLastWorker(workerName: string): void {
  try {
    localStorage.setItem(LAST_WORKER_STORAGE_KEY, workerName);
  } catch {
    /* localStorage unavailable -- not worth failing over */
  }
}

function loadLastModel(workerName: string): LastModelSelection | null {
  try {
    const raw = localStorage.getItem(lastModelStorageKey(workerName));
    return raw ? (JSON.parse(raw) as LastModelSelection) : null;
  } catch {
    return null;
  }
}

function rememberLastModel(workerName: string, selection: LastModelSelection): void {
  try {
    localStorage.setItem(lastModelStorageKey(workerName), JSON.stringify(selection));
  } catch {
    /* localStorage unavailable -- not worth failing over */
  }
}

const CACHE_TYPES = ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"];
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

const NUMERIC_FIELDS = ["n_prompt", "n_gen", "threads", "n_gpu_layers", "batch_size", "ubatch_size"] as const;
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
  const [workerName, setWorkerName] = useState("");
  const { order: workerOrder, status: workerStatus } = useWorkerStatuses();
  const [sweep, setSweep] = useState<Sweep>(EMPTY_SWEEP);
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
  }, []);

  // Prefer whichever worker was picked last time (if it's still configured);
  // only fall back to auto-picking the sole worker when there's nothing
  // remembered yet.
  useEffect(() => {
    if (workerName || workerOrder.length === 0) return;
    const remembered = loadLastWorker();
    if (remembered && workerOrder.includes(remembered)) {
      setWorkerName(remembered);
    } else if (workerOrder.length === 1) {
      setWorkerName(workerOrder[0]);
    }
  }, [workerOrder, workerName]);

  useEffect(() => {
    if (workerName) rememberLastWorker(workerName);
  }, [workerName]);

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
    if (!workerName || models.length === 0) return;
    if (restoredModelForRef.current === workerName) return;
    restoredModelForRef.current = workerName;
    if (modelId) return;
    const remembered = loadLastModel(workerName);
    if (!remembered || !models.some((m) => m.id === remembered.modelId)) return;
    const draftId =
      remembered.mtpModelId && models.some((m) => m.id === remembered.mtpModelId) ? remembered.mtpModelId : undefined;
    handleModelSelect(remembered.modelId, draftId);
  }, [workerName, models, modelId]);

  useEffect(() => {
    if (!workerName || !modelId) return;
    rememberLastModel(workerName, { modelId, mtpModelId });
  }, [workerName, modelId, mtpModelId]);

  // Hardware/layer-count context for the selected pairing -- drives the
  // GPU-layers and threads sliders below. Best-effort: hardware is only
  // known once that worker's own /llama-cpp+/hardware fetch resolves (see
  // useWorkerStatuses), and a model's layer count is only known if it was
  // downloaded through this app after GGUF parsing existed (worker/src/gguf.ts).
  const selectedModel = models.find((m) => m.id === modelId);
  const modelLayerCount = typeof selectedModel?.metadata.n_layer === "number" ? selectedModel.metadata.n_layer : null;

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
  const mtpDraftCandidates = models.filter((m) => isMtpDraftModel(m));
  const showMtpModelPicker = Boolean(modelId);

  const selectedWorker = workerName ? workerStatus[workerName] : undefined;
  const workerHardware = selectedWorker?.info?.hardware;
  // Prefer the directly-detected GPU list once hardware has loaded; until
  // then fall back to the worker's configured backend (available instantly --
  // see WorkerListEntry) since a "cpu" backend build ignores -ngl regardless
  // of what's physically in the box.
  const noGpu = workerHardware ? workerHardware.gpu.length === 0 : selectedWorker?.worker.backend === "cpu";

  const nglMax = noGpu ? 0 : modelLayerCount ?? NGL_FALLBACK_MAX;
  const nglSuggested = noGpu ? 0 : modelLayerCount ?? NGL_FALLBACK_MAX;
  const nglSuggestedLabel = noGpu
    ? "Suggested: 0 -- this worker has no GPU backend"
    : modelLayerCount != null
      ? `Suggested: ${modelLayerCount} -- offload every layer (this model has ${modelLayerCount})`
      : `Suggested: ${NGL_FALLBACK_MAX} -- model's layer count isn't known yet, showing the "offload everything" default`;

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
    if (!modelId || !workerName) {
      setSweep(EMPTY_SWEEP);
      return;
    }
    const remembered = loadRememberedSweep(modelId, workerName);
    if (remembered) {
      setSweep(remembered);
      return;
    }
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
    setSweep(baseSweep);
  }, [modelId, workerName, noGpu, nglMax]);

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
    if (!modelId || !workerName) {
      setMsg("Pick a model and a worker first");
      return;
    }
    setSubmitting(true);
    // If the worker has no active llama.cpp build, the server activates the
    // latest installed one or downloads+installs the latest available
    // release before this resolves -- a fresh worker can take a few minutes
    // here rather than the usual instant round trip.
    setMsg("Triggering… (can take a few minutes if a llama.cpp build needs to be installed first)");
    try {
      const run = await api.triggerRun({
        model_id: modelId,
        worker_name: workerName,
        mtp_model_id: mtpModelId || undefined,
        sweep,
      });
      rememberSweep(modelId, workerName, sweep);
      setLastRunId(run.id);
      // The server queues a trigger instead of rejecting it when the target
      // worker already has a run in flight (or already has a queue) -- see
      // server/src/routes/runs.ts's dispatchScheduledRun -- so this can come
      // back "scheduled" rather than "running" even on success.
      setMsg(run.status === "scheduled" ? "Run scheduled ✓" : "Triggered ✓");
    } catch (err) {
      setMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSubmitting(false);
    }
  }

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
        <div className="flex flex-wrap gap-4">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted">Worker</span>
            <select
              value={workerName}
              onChange={(e) => setWorkerName(e.target.value)}
              className={`${inputCls} w-48`}
            >
              <option value="" disabled>
                select…
              </option>
              {workerOrder.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <ModelPicker
            models={models}
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
