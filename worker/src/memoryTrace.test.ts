import { describe, expect, it } from "vitest";
import { parseTraceLine, windowsTraceScript } from "./memoryTrace.js";

describe("memory trace", () => {
  it("parses a paired reading, and reads an empty counter as no reading rather than zero", () => {
    // Recorded live from the script below against dwm.exe.
    expect(parseTraceLine("t=1789638482306 d=1039056896 s=29716480")).toEqual({
      atMs: 1789638482306,
      dedicatedMib: 1039056896 / 1048576,
      sharedMib: 29716480 / 1048576,
    });
    expect(parseTraceLine("t=1758000000123 d=338620416 s=0")).toMatchObject({ sharedMib: 0 });
    expect(parseTraceLine("t=1758000000123 d= s=")).toEqual({ atMs: 1758000000123, dedicatedMib: null, sharedMib: null });
    expect(parseTraceLine("WARNING: something")).toBeNull();
  });

  it("still reads a line without shared as dedicated only", () => {
    expect(parseTraceLine("t=1758000000123 d=338620416")).toEqual({ atMs: 1758000000123, dedicatedMib: 338620416 / 1048576, sharedMib: null });
  });

  it("anchors the counter on the pid's trailing underscore and exits with either process", () => {
    const script = windowsTraceScript(4452, 77);
    expect(script).toContain("pid_4452_*");
    expect(script).toContain("Get-Process -Id 4452");
    expect(script).toContain("Get-Process -Id 77");
    // One Get-Counter call for both, so the pair is one sample.
    expect(script.match(/Get-Counter/g)).toHaveLength(1);
    expect(script).toContain("Shared Usage");
  });
});
