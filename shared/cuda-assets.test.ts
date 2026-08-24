import { describe, expect, it } from "vitest";
import { extractCudaVariant, cudaDriverSupports, sortAssetsForWorker } from "./types.js";
import type { LlamaCppAsset } from "./types.js";

// Real asset names from ggml-org/llama.cpp releases (verified live against
// b10612): two CUDA toolkit variants per platform/arch plus the cudart
// redistributables, alongside every other backend.
const asset = (name: string, size = 100): LlamaCppAsset => ({ name, download_url: `https://x/${name}`, size_bytes: size });

describe("extractCudaVariant", () => {
  it("parses major.minor out of real cuda asset names", () => {
    expect(extractCudaVariant("llama-b10612-bin-win-cuda-12.4-x64.zip")).toEqual({ major: 12, minor: 4 });
    expect(extractCudaVariant("llama-b10612-bin-win-cuda-13.3-x64.zip")).toEqual({ major: 13, minor: 3 });
    expect(extractCudaVariant("llama-b10612-bin-win-cuda-13.4-arm64.zip")).toEqual({ major: 13, minor: 4 });
  });

  it("returns null for every non-CUDA backend variant", () => {
    expect(extractCudaVariant("llama-b10612-bin-win-cpu-x64.zip")).toBeNull();
    expect(extractCudaVariant("llama-b10612-bin-win-vulkan-x64.zip")).toBeNull();
    expect(extractCudaVariant("llama-b10612-bin-win-rocm-7.14-x64.zip")).toBeNull();
    expect(extractCudaVariant("llama-b10612-bin-ubuntu-x64.zip")).toBeNull();
  });

  it("returns null for a bare 'cuda' token with no version", () => {
    expect(extractCudaVariant("llama-b1-bin-win-cuda-x64.zip")).toBeNull();
  });
});

describe("cudaDriverSupports", () => {
  it("accepts builds up to the driver's max CUDA version", () => {
    // nvidia-smi "CUDA Version: 12.7" == driver runs any 12.x toolkit <= 12.7
    expect(cudaDriverSupports("12.7", { major: 12, minor: 4 })).toBe(true);
    expect(cudaDriverSupports("12.7", { major: 12, minor: 7 })).toBe(true);
    expect(cudaDriverSupports("12.7", { major: 12, minor: 9 })).toBe(false);
    expect(cudaDriverSupports("12.7", { major: 13, minor: 0 })).toBe(false);
  });

  it("accepts anything once the driver's major is newer", () => {
    expect(cudaDriverSupports("13.0", { major: 12, minor: 4 })).toBe(true);
    expect(cudaDriverSupports("13.4", { major: 13, minor: 4 })).toBe(true);
  });

  it("treats unknown/unparseable driver info as compatible (fail-open)", () => {
    expect(cudaDriverSupports(null, { major: 99, minor: 0 })).toBe(true);
    expect(cudaDriverSupports(undefined, { major: 99, minor: 0 })).toBe(true);
    expect(cudaDriverSupports("n/a", { major: 99, minor: 0 })).toBe(true);
  });
});

describe("sortAssetsForWorker", () => {
  const win = [
    asset("llama-b10612-bin-win-cuda-13.3-x64.zip"),
    asset("llama-b10612-bin-win-cuda-12.4-x64.zip"),
  ];

  it("keeps newest compatible CUDA variant first", () => {
    const sorted = sortAssetsForWorker(win, "13.3");
    expect(sorted[0]!.name).toContain("cuda-13.3");
    expect(sorted[1]!.name).toContain("cuda-12.4");
  });

  it("demotes a CUDA variant newer than the driver below an older one", () => {
    // The exact real-world failure this exists for: a CUDA-12.7-era driver
    // can't load cuda-13.x DLLs -- assets[0] must be the 12.4 build.
    const sorted = sortAssetsForWorker(win, "12.7");
    expect(sorted[0]!.name).toContain("cuda-12.4");
    expect(sorted[1]!.name).toContain("cuda-13.3");
  });

  it("with unknown driver info puts the OLDER variant first (conservative default)", () => {
    const sorted = sortAssetsForWorker(win, null);
    expect(sorted[0]!.name).toContain("cuda-12.4");
  });

  it("leaves non-CUDA assets in their original relative order", () => {
    const mixed = [
      asset("llama-b10612-bin-win-cpu-x64.zip"),
      asset("llama-b10612-bin-win-vulkan-x64.zip"),
      asset("llama-b10612-bin-win-rocm-7.14-x64.zip"),
    ];
    expect(sortAssetsForWorker(mixed, "12.7").map((a) => a.name)).toEqual(mixed.map((a) => a.name));
  });

  it("orders incompatible variants oldest-first among themselves", () => {
    const sorted = sortAssetsForWorker(
      [
        asset("llama-b10612-bin-win-cuda-13.4-x64.zip"),
        asset("llama-b10612-bin-win-cuda-13.3-x64.zip"),
      ],
      "12.7"
    );
    expect(sorted.map((a) => a.name)).toEqual([
      "llama-b10612-bin-win-cuda-13.3-x64.zip",
      "llama-b10612-bin-win-cuda-13.4-x64.zip",
    ]);
  });
});
