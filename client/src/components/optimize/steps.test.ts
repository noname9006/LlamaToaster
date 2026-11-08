import { describe, expect, it } from "vitest";
import { defaultStep, nextStep, prevStep, stepStates, type StepFacts } from "./steps";

const none: StepFacts = {
  hasMachine: false,
  machineReady: false,
  hasModel: false,
  hasTargetCtx: false,
  fitAnswered: false,
  baselineDone: false,
  hasPlacement: false,
  placementDone: false,
  kvDone: false,
  batchDone: false,
  hasRuns: {},
};

const lockedIds = (f: StepFacts) => stepStates(f).filter((s) => s.locked).map((s) => s.id);

describe("stepStates", () => {
  it("only the machine step is open on an empty page", () => {
    expect(lockedIds(none)).toEqual(["model", "fit", "baseline", "offload", "kv", "batch"]);
    expect(defaultStep(stepStates(none))).toBe("machine");
  });

  it("an offline machine still lets you pick a model, but the machine step stays unfinished", () => {
    const f = { ...none, hasMachine: true };
    expect(lockedIds(f)).toEqual(["fit", "baseline", "offload", "kv", "batch"]);
    expect(defaultStep(stepStates(f))).toBe("machine");
  });

  it("KV and batch open once there is a placement to build on", () => {
    const f = { ...none, hasMachine: true, machineReady: true, hasModel: true, hasTargetCtx: true, fitAnswered: true, baselineDone: true };
    expect(lockedIds(f)).toEqual(["kv", "batch"]);
    expect(defaultStep(stepStates(f))).toBe("offload");
    const placed = { ...f, hasPlacement: true, placementDone: true };
    expect(lockedIds(placed)).toEqual([]);
    expect(defaultStep(stepStates(placed))).toBe("kv");
  });

  it("existing runs stay viewable even when the prerequisite is gone", () => {
    const f = { ...none, hasMachine: true, hasModel: true, hasTargetCtx: true, hasRuns: { batch: true } };
    expect(lockedIds(f)).toEqual(["kv"]);
  });

  it("everything done opens the last step", () => {
    const all = { ...none, hasMachine: true, machineReady: true, hasModel: true, hasTargetCtx: true, fitAnswered: true, baselineDone: true, hasPlacement: true, placementDone: true, kvDone: true, batchDone: true };
    expect(defaultStep(stepStates(all))).toBe("batch");
  });

  it("walks forward and back", () => {
    expect(nextStep("machine")).toBe("model");
    expect(nextStep("batch")).toBeNull();
    expect(prevStep("machine")).toBeNull();
    expect(prevStep("kv")).toBe("offload");
  });
});
