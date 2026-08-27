import { describe, expect, it } from "vitest";
import { parseCpuIsaBanner } from "./binary-probe.js";

describe("N6 ISA provenance (llama.cpp's own startup banner)", () => {
  it("keeps the features the running build actually dispatches on", () => {
    const banner =
      "system_info: n_threads = 8 (n_threads_batch = 8) / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | LLAMAFILE = 1 |";
    expect(parseCpuIsaBanner(banner)).toBe("AVX AVX2 FMA LLAMAFILE");
  });

  it("distinguishes two builds that differ only in AVX512", () => {
    const withAvx512 = parseCpuIsaBanner("system_info: AVX = 1 | AVX2 = 1 | AVX512 = 1 |");
    const without = parseCpuIsaBanner("system_info: AVX = 1 | AVX2 = 1 | AVX512 = 0 |");
    expect(withAvx512).toContain("AVX512");
    expect(without).not.toContain("AVX512");
    expect(withAvx512).not.toBe(without);
  });

  it("returns null rather than a fabricated ISA string when there is no banner", () => {
    expect(parseCpuIsaBanner("version: 1234 (abcdef)\nbuilt with gcc")).toBeNull();
    expect(parseCpuIsaBanner("")).toBeNull();
  });

  it("never reports n_threads as an ISA feature", () => {
    expect(parseCpuIsaBanner("system_info: n_threads = 8 | AVX = 1 |")).toBe("AVX");
  });
});
