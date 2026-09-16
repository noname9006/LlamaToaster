import { describe, expect, it } from "vitest";
import { parseTraceLine, windowsTraceScript } from "./memoryTrace.js";

describe("memory trace", () => {
  it("parses a reading, and reads an empty counter as no reading rather than zero", () => {
    expect(parseTraceLine("t=1758000000123 d=338620416")).toEqual({ atMs: 1758000000123, dedicatedMib: 338620416 / 1048576 });
    expect(parseTraceLine("t=1758000000123 d=")).toEqual({ atMs: 1758000000123, dedicatedMib: null });
    expect(parseTraceLine("WARNING: something")).toBeNull();
  });

  it("anchors the counter on the pid's trailing underscore and exits with either process", () => {
    const script = windowsTraceScript(4452, 77);
    expect(script).toContain("pid_4452_*");
    expect(script).toContain("Get-Process -Id 4452");
    expect(script).toContain("Get-Process -Id 77");
  });
});
