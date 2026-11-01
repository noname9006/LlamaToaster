import { describe, expect, it } from "vitest";
import { backendOfAsset } from "./types.js";

describe("backendOfAsset", () => {
  it.each([
    ["llama-b12023-bin-win-vulkan-x64.zip", "vulkan"],
    ["llama-b12023-bin-win-cuda-12.4-x64.zip", "cuda"],
    ["llama-b12023-bin-win-rocm-10.0-x64.zip", "rocm"],
    ["llama-b9503-bin-win-hip-radeon-x64.zip", "rocm"],
    ["llama-b12023-bin-win-sycl-x64.zip", "sycl"],
    ["llama-b12023-bin-ubuntu-openvino-x64.tar.gz", "openvino"],
    ["llama-b12023-bin-win-opencl-adreno-arm64.zip", "opencl-adreno"],
    ["llama-b12023-bin-win-cpu-x64.zip", "cpu"],
    ["llama-b12023-bin-ubuntu-x64.tar.gz", "cpu"],
  ])("%s is %s", (name, backend) => {
    expect(backendOfAsset(name)).toBe(backend);
  });

  it("says nothing about a hand-imported build, so activating one keeps the backend", () => {
    expect(backendOfAsset("(imported)")).toBeNull();
  });
});
