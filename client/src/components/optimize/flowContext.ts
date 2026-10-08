import type { FlowStep, Model, Test, TriggerPayload, Worker } from "../../types";
import type { MoeInfo } from "../../../../shared/moePlacement.js";
import type { MachineClass } from "../../../../shared/optimizeFlow.js";
import type { ThreadArgs } from "../../../../shared/speedRun.js";
import type { CpuTopology } from "../../../../shared/threadPlan.js";

/** Everything the flow's sections need about the current pairing. Built once
 * by the New test page. */
export interface FlowContext {
  workerId: string;
  worker: Worker;
  model: Model;
  modelLabel: string;
  /** The file name llama-server is pointed at in copyable command lines. */
  modelFile: string;
  machine: MachineClass;
  moe: MoeInfo | null;
  trainedCtx: number | null;
  targetCtx: number;
  repeats: number;
  mainGpu?: number;
  gpuName: string | null;
  freeVramMib: number | null;
  ramTotalMib: number | null;
  /** n_layer + 1 (llama.cpp's -ngl ceiling). */
  layersTotal: number | null;
  build: string | null;
  canFit: boolean;
  canFitReason: string;
  machineOffline: boolean;
  topology: CpuTopology;
  /** The thread setting every run from the threads step on uses. */
  threads: ThreadArgs | null;
  trigger: (step: FlowStep, payload: TriggerPayload) => Promise<Test>;
}
