import type { VramDiscrepancyPolicy } from "../../shared/types.js";

// What the worker's post-run VRAM-discrepancy detection translates into for
// one sweep item, given the operator's policy (shared/types.ts's
// VramDiscrepancyPolicy) and this attempt's context:
//   record_done_with_warning -- original behavior: results stand, warning attached
//   retry_once              -- skip the terminal report entirely; the caller
//                              re-runs the item once (heals transient VRAM
//                              contention from another process)
//   fail_item               -- post a failed terminal, no result rows
export type VramDiscrepancyAction = "record_done_with_warning" | "retry_once" | "fail_item";

export interface VramPolicyDecision {
  action: VramDiscrepancyAction;
}

// hardFallback is the unambiguous signature only: llama.cpp's own
// post-allocation buffer report contradicting the offload claim with ~0 bytes
// on the GPU buffer (~0 layers resident). Softer shortfalls never reach the
// hard actions regardless of policy -- see VramDiscrepancyPolicy's doc
// comment in shared/types.ts.
// persistentMemo: a previous item in this same run already reproduced the
// fallback across its own retry, so the cause is confirmed deterministic on
// this machine right now -- later items skip their retry instead of each
// paying another full bench just to re-confirm the same thing.
export function resolveVramDiscrepancyAction(
  policy: VramDiscrepancyPolicy,
  hardFallback: boolean,
  alreadyRetried: boolean,
  persistentMemo: boolean
): VramPolicyDecision {
  if (!hardFallback || policy === "warn") return { action: "record_done_with_warning" };
  if (policy === "fail") return { action: "fail_item" };
  return alreadyRetried || persistentMemo ? { action: "fail_item" } : { action: "retry_once" };
}
