import { describe, expect, it } from "vitest";
import type { Test } from "../types";
import { memberChipLabel } from "./StatusPill";

// The removed context-test modes still exist in stored tests. Their configs are
// never re-validated on read, and the chip names whatever mode was stored.
describe("memberChipLabel", () => {
  it("still names a stored probe that ran a removed mode", () => {
    for (const mode of ["max_gpu", "max_context", "balanced"]) {
      const run = { kind: "probe", config: { probe: { mode } } } as unknown as Test;
      expect(memberChipLabel(run)).toBe(mode);
    }
  });

  it("names a current mode", () => {
    const run = { kind: "probe", config: { probe: { mode: "frontier" } } } as unknown as Test;
    expect(memberChipLabel(run)).toBe("frontier");
  });
});
