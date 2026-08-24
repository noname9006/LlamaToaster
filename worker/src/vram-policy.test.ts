import { describe, expect, it } from "vitest";
import { resolveVramDiscrepancyAction } from "./vram-policy.js";

describe("resolveVramDiscrepancyAction", () => {
  it("warn policy (the default) never escalates, even on the hard signature", () => {
    expect(resolveVramDiscrepancyAction("warn", true, false, false)).toEqual({ action: "record_done_with_warning" });
    expect(resolveVramDiscrepancyAction("warn", true, true, true)).toEqual({ action: "record_done_with_warning" });
  });

  it("soft shortfalls stay at warn under every policy -- only the hard signature escalates", () => {
    for (const policy of ["warn", "retry_once_then_fail", "fail"] as const) {
      expect(resolveVramDiscrepancyAction(policy, false, false, false)).toEqual({
        action: "record_done_with_warning",
      });
    }
  });

  it("fail policy marks the item failed immediately, retry memo notwithstanding", () => {
    expect(resolveVramDiscrepancyAction("fail", true, false, false)).toEqual({ action: "fail_item" });
    expect(resolveVramDiscrepancyAction("fail", true, true, true)).toEqual({ action: "fail_item" });
  });

  it("retry_once_then_fail retries exactly once, then fails", () => {
    expect(resolveVramDiscrepancyAction("retry_once_then_fail", true, false, false)).toEqual({
      action: "retry_once",
    });
    expect(resolveVramDiscrepancyAction("retry_once_then_fail", true, true, false)).toEqual({ action: "fail_item" });
  });

  it("a confirmed-persistent fallback in this run makes later items fail without their own retry", () => {
    // The whole point of the run-scoped memo: a 20-combo sweep against a
    // deterministic driver condition must not pay 20 retry taxes.
    expect(resolveVramDiscrepancyAction("retry_once_then_fail", true, false, true)).toEqual({ action: "fail_item" });
  });

  it("memo only matters when the current attempt is also hard-fallback (a healed item is not persistent)", () => {
    // Second attempt came back CLEAN (transient contention went away) --
    // record it; the stale memo from an earlier model must not fail this one.
    expect(resolveVramDiscrepancyAction("retry_once_then_fail", false, true, true)).toEqual({
      action: "record_done_with_warning",
    });
  });
});
