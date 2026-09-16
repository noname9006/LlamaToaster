import { describe, expect, it } from "vitest";
import { listDevicesFreeMib, parseCpuIsaBanner, parseListDevices, usedListedDevices } from "./binary-probe.js";

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

describe("--list-devices free VRAM", () => {
  // Verbatim from llama-server b10956 on the RX 6600 XT, CRLF and all.
  const single = "Available devices:\r\n  Vulkan0: AMD Radeon RX 6600 XT (8176 MiB, 7378 MiB free)\r\n";
  const dual = [
    "load_backend: loaded CUDA backend",
    "Available devices:",
    "  CUDA0: NVIDIA GeForce RTX 3080 (10239 MiB, 9001 MiB free)",
    "  CUDA1: NVIDIA GeForce RTX 3060 (12287 MiB, 11800 MiB free)",
  ].join("\n");

  it("parses each device line", () => {
    expect(parseListDevices(single)).toEqual([
      { name: "Vulkan0", description: "AMD Radeon RX 6600 XT", totalMib: 8176, freeMib: 7378 },
    ]);
    expect(parseListDevices(dual).map((d) => d.name)).toEqual(["CUDA0", "CUDA1"]);
  });

  it("sums every device unless one is pinned with -mg", () => {
    expect(listDevicesFreeMib(parseListDevices(dual))).toBe(20801);
    expect(listDevicesFreeMib(parseListDevices(dual), 1)).toBe(11800);
    expect(listDevicesFreeMib(parseListDevices(dual), 2)).toBeNull();
  });

  it("uses only the -mg device when one is pinned, every device otherwise", () => {
    const devices = parseListDevices(dual);
    expect(usedListedDevices(devices, 0).map((d) => d.name)).toEqual(["CUDA0"]);
    expect(usedListedDevices(devices).map((d) => d.name)).toEqual(["CUDA0", "CUDA1"]);
    expect(usedListedDevices(devices, 5)).toEqual([]);
  });

  it("is unknown when the build lists nothing", () => {
    expect(listDevicesFreeMib(parseListDevices("error: unknown argument: --list-devices"))).toBeNull();
  });
});
