// The New test page's seven steps: one guided path from a machine to a tuned
// config. Each step answers one question and hands its winner to the next.
// Kept out of React so the gating can be tested.

export const STEP_IDS = ["machine", "model", "fit", "baseline", "offload", "kv", "batch"] as const;
export type StepId = (typeof STEP_IDS)[number];

export const STEP_LABEL: Record<StepId, string> = {
  machine: "Machine",
  model: "Model",
  fit: "Fit",
  baseline: "Baseline",
  offload: "Context and offload",
  kv: "KV cache",
  batch: "Batch sizes",
};

export function isStepId(v: unknown): v is StepId {
  return typeof v === "string" && (STEP_IDS as readonly string[]).includes(v);
}

export interface StepFacts {
  hasMachine: boolean;
  /** Online, a recent enough worker and a build with llama-fit-params. */
  machineReady: boolean;
  hasModel: boolean;
  hasTargetCtx: boolean;
  /** The fit check for the model at the target context has an answer. */
  fitAnswered: boolean;
  baselineDone: boolean;
  /** Step 5 produced a config to build on (or a baseline exists to fall back to). */
  hasPlacement: boolean;
  placementDone: boolean;
  kvDone: boolean;
  batchDone: boolean;
  /** Runs of a step already exist in this session -- always viewable. */
  hasRuns: Partial<Record<StepId, boolean>>;
}

export interface StepState {
  id: StepId;
  done: boolean;
  /** Why the step can't be opened yet; null = reachable. */
  locked: string | null;
}

export function stepStates(f: StepFacts): StepState[] {
  const need = (ok: boolean, reason: string): string | null => (ok ? null : reason);
  const lock: Record<StepId, string | null> = {
    machine: null,
    model: need(f.hasMachine, "Pick a machine first."),
    fit: need(f.hasMachine && f.hasModel, "Pick a model first."),
    baseline: need(f.hasModel && f.hasTargetCtx, "Choose a target context first."),
    offload: need(f.hasModel && f.hasTargetCtx, "Choose a target context first."),
    kv: need(f.hasPlacement || !!f.hasRuns.kv, "Finish step 5 (or the baseline) first."),
    batch: need(f.hasPlacement || !!f.hasRuns.batch, "Finish step 5 (or the baseline) first."),
  };
  const done: Record<StepId, boolean> = {
    machine: f.machineReady,
    model: f.hasModel,
    fit: f.hasTargetCtx && f.fitAnswered,
    baseline: f.baselineDone,
    offload: f.placementDone,
    kv: f.kvDone,
    batch: f.batchDone,
  };
  return STEP_IDS.map((id) => ({ id, done: done[id], locked: lock[id] }));
}

/** Where the page opens without ?step=: the first unfinished step that can be
 * opened (the last step when everything is done). */
export function defaultStep(states: readonly StepState[]): StepId {
  const open = states.filter((s) => !s.locked);
  return (open.find((s) => !s.done) ?? open[open.length - 1] ?? states[0]).id;
}

export function nextStep(id: StepId): StepId | null {
  const i = STEP_IDS.indexOf(id);
  return i >= 0 && i < STEP_IDS.length - 1 ? STEP_IDS[i + 1] : null;
}

export function prevStep(id: StepId): StepId | null {
  const i = STEP_IDS.indexOf(id);
  return i > 0 ? STEP_IDS[i - 1] : null;
}
