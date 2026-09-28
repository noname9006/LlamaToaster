import { describe, expect, it } from "vitest";
import {
  BUILTIN_ROCM_SUPPORT,
  isRocmSupportList,
  isRocmSupportedGpu,
  normalizeGpuName,
  type RocmSupportList,
} from "./rocmSupport.js";
import { detectBackend, isRocmAssetName, findCommunityRocmGpu } from "./types.js";

describe("normalizeGpuName", () => {
  it("makes AMD's table names and detected names compare equal", () => {
    expect(normalizeGpuName("AMD Radeon RX 7900 XTX")).toBe("rx 7900 xtx");
    expect(normalizeGpuName("AMD Radeon(TM) RX 7900 XTX")).toBe("rx 7900 xtx");
    expect(normalizeGpuName("AMD RX 9070 GRE")).toBe("rx 9070 gre");
    expect(normalizeGpuName("Advanced Micro Devices, Inc. Radeon RX 6600 XT Graphics")).toBe("rx 6600 xt");
    expect(normalizeGpuName("AMD Instinct MI300X")).toBe("mi300x");
    expect(normalizeGpuName("AMD Radeon PRO W7900 Dual Slot")).toBe("pro w7900 dual slot");
  });

  it("keeps model numbers that merely start with r", () => {
    expect(normalizeGpuName("AMD Radeon AI PRO R9700")).toBe("ai pro r9700");
  });
});

// Shaped like the real Aug 2026 tables: Windows marks every RDNA2 card
// unsupported, Linux supports the PRO W6800.
const live: RocmSupportList = {
  win32: {
    supported: ["rx 7900 xtx", "rx 7900 xt", "rx 9070", "pro w7900"],
    unsupported: ["rx 6600 xt", "rx 6800", "pro w6800"],
  },
  linux: {
    supported: ["rx 7900 xtx", "rx 7900 xt", "rx 7900 gre", "rx 9070", "pro w6800", "mi250x", "mi250"],
    unsupported: ["mi50", "mi25", "radeon vii"],
  },
};

describe("isRocmSupportedGpu", () => {
  it("is false for a GPU AMD doesn't list, and for an empty name", () => {
    expect(isRocmSupportedGpu("AMD Radeon RX 6600 XT", "win32", live)).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon RX 5700 XT", "linux", live)).toBe(false);
    expect(isRocmSupportedGpu("", "linux", live)).toBe(false);
  });

  it("matches detected names regardless of vendor prefix / trademark noise", () => {
    expect(isRocmSupportedGpu("AMD Radeon RX 7900 XTX", "win32", live)).toBe(true);
    expect(isRocmSupportedGpu("AMD Radeon(TM) RX 7900 XTX Graphics", "win32", live)).toBe(true);
    expect(isRocmSupportedGpu("Radeon RX 7900 XTX", "linux", live)).toBe(true);
  });

  it("matches a variant that extends a listed name (RX 9070 XT via RX 9070)", () => {
    expect(isRocmSupportedGpu("AMD Radeon RX 9070 XT", "win32", live)).toBe(true);
  });

  it("does not let a shorter model number swallow a different card", () => {
    // "rx 7900 xt" must not match "rx 7900 gre" (its own separate row).
    expect(isRocmSupportedGpu("AMD Radeon RX 7900 GRE", "win32", live)).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon RX 7900 GRE", "linux", live)).toBe(true);
    // "mi25" (unsupported) is a prefix of "mi250x"'s digits but not a token match.
    expect(isRocmSupportedGpu("AMD Instinct MI250X", "linux", live)).toBe(true);
    expect(isRocmSupportedGpu("AMD Instinct MI25", "linux", live)).toBe(false);
  });

  it("uses the platform's own table", () => {
    expect(isRocmSupportedGpu("AMD Radeon PRO W6800", "win32", live)).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon PRO W6800", "linux", live)).toBe(true);
  });

  it("lets the longest matching marker decide", () => {
    const list: RocmSupportList = {
      win32: { supported: ["rx 6600"], unsupported: ["rx 6600 xt"] },
      linux: { supported: [], unsupported: [] },
    };
    expect(isRocmSupportedGpu("AMD Radeon RX 6600", "win32", list)).toBe(true);
    expect(isRocmSupportedGpu("AMD Radeon RX 6600 XT", "win32", list)).toBe(false);
  });

  it("treats a marker listed as both supported and unsupported as supported", () => {
    const list: RocmSupportList = {
      win32: { supported: ["rx 7600"], unsupported: ["rx 7600"] },
      linux: { supported: [], unsupported: [] },
    };
    expect(isRocmSupportedGpu("AMD Radeon RX 7600", "win32", list)).toBe(true);
  });
});

describe("built-in fallback list", () => {
  it("covers everything the old hand-copied list did", () => {
    for (const name of [
      "AMD Radeon RX 7600",
      "AMD Radeon RX 7650 GRE",
      "AMD Radeon RX 7700 XT",
      "AMD Radeon RX 7800 XT",
      "AMD Radeon RX 7900 XT",
      "AMD Radeon RX 7900 XTX",
      "AMD Radeon RX 9060 XT",
      "AMD Radeon RX 9070 XT",
      "AMD Radeon PRO W7700",
      "AMD Radeon PRO W7800",
      "AMD Radeon PRO W7900",
      "AMD Radeon PRO W6800",
      "AMD Radeon PRO V620",
      "AMD Radeon PRO V710",
      "AMD Radeon AI PRO R9700",
      "AMD Instinct MI100",
      "AMD Instinct MI300X",
      "AMD Instinct MI325X",
      "AMD Instinct MI350X",
      "AMD Instinct MI355X",
    ]) {
      expect(isRocmSupportedGpu(name, "win32"), name).toBe(true);
      expect(isRocmSupportedGpu(name, "linux"), name).toBe(true);
    }
  });

  it("does not claim the RX 6600 XT", () => {
    expect(isRocmSupportedGpu("AMD Radeon RX 6600 XT", "win32")).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon RX 6600 XT", "linux")).toBe(false);
  });
});

describe("isRocmSupportList", () => {
  it("accepts a well-formed list and rejects malformed input", () => {
    expect(isRocmSupportList(BUILTIN_ROCM_SUPPORT)).toBe(true);
    expect(isRocmSupportList(null)).toBe(false);
    expect(isRocmSupportList({})).toBe(false);
    expect(isRocmSupportList({ win32: { supported: [], unsupported: [] } })).toBe(false);
    expect(isRocmSupportList({ ...live, linux: { supported: [1], unsupported: [] } })).toBe(false);
  });
});

describe("detectBackend with the ROCm list", () => {
  const amd = (model: string) => [{ vendor: "AMD", model }];

  it("sends an unsupported AMD GPU (RX 6600 XT) to vulkan and a supported one to rocm", () => {
    expect(detectBackend("win32", amd("AMD Radeon RX 6600 XT"))).toBe("vulkan");
    expect(detectBackend("win32", amd("AMD Radeon RX 7900 XTX"))).toBe("rocm");
  });

  it("honors a passed-in (fetched) list over the built-in one", () => {
    // Built-in doesn't know the 7900 GRE; the fetched Linux list does.
    expect(detectBackend("linux", amd("AMD Radeon RX 7900 GRE"))).toBe("vulkan");
    expect(detectBackend("linux", amd("AMD Radeon RX 7900 GRE"), live)).toBe("rocm");
    // ...and the Windows table doesn't.
    expect(detectBackend("win32", amd("AMD Radeon RX 7900 GRE"), live)).toBe("vulkan");
  });

  it("leaves the non-AMD paths alone", () => {
    expect(detectBackend("darwin", amd("AMD Radeon RX 7900 XTX"), live)).toBe("cpu");
    expect(detectBackend("win32", [{ vendor: "NVIDIA", model: "RTX 4090" }], live)).toBe("cuda");
    expect(detectBackend("linux", [{ vendor: "Intel", model: "Arc A770" }], live)).toBe("vulkan");
    expect(detectBackend("linux", [], live)).toBe("cpu");
  });
});

describe("isRocmAssetName", () => {
  it("matches ROCm/HIP release assets and nothing else", () => {
    expect(isRocmAssetName("llama-b10361-bin-win-rocm-7.14-x64.zip")).toBe(true);
    expect(isRocmAssetName("llama-b6000-bin-win-hip-radeon-x64.zip")).toBe(true);
    expect(isRocmAssetName("llama-b5000-bin-win-hipblas-x64.zip")).toBe(true);
    expect(isRocmAssetName("llama-b10361-bin-win-vulkan-x64.zip")).toBe(false);
    expect(isRocmAssetName("llama-b10361-bin-win-cuda-12.4-x64.zip")).toBe(false);
    expect(isRocmAssetName("llama-b10361-bin-ubuntu-x64.tar.gz")).toBe(false);
  });
});

describe("findCommunityRocmGpu", () => {
  const amd = (model: string) => ({ vendor: "AMD", model: `AMD Radeon ${model}` });

  it("matches exactly the curated Windows cards, and no other GPU", () => {
    for (const model of ["RX 6600", "RX 6600 XT", "RX 6650 XT", "RX 6700", "RX 6700 XT", "RX 6750 XT"]) {
      const g = amd(model);
      expect(findCommunityRocmGpu([g], "win32")).toBe(g);
    }
    // Other AMD cards -- including ones only Linux covers -- get no notice.
    for (const model of ["RX 6500 XT", "RX 5700", "RX 5700 XT", "RX 6800 XT", "RX 6900 XT", "RX 7900 XTX", "RX 660"]) {
      expect(findCommunityRocmGpu([amd(model)], "win32")).toBeNull();
    }
    expect(findCommunityRocmGpu([{ vendor: "NVIDIA", model: "RX 6600" }], "win32")).toBeNull();
    expect(findCommunityRocmGpu([], "win32")).toBeNull();
  });

  it("adds the RX 6500 XT and RX 5700 / 5700 XT on Linux", () => {
    for (const model of ["RX 6600", "RX 6750 XT", "RX 6500 XT", "RX 5700", "RX 5700 XT"]) {
      const g = amd(model);
      expect(findCommunityRocmGpu([g], "linux")).toBe(g);
    }
    for (const model of ["RX 5600 XT", "RX 5500 XT", "RX 6800 XT", "RX 7900 XTX"]) {
      expect(findCommunityRocmGpu([amd(model)], "linux")).toBeNull();
    }
  });

  it("stays quiet where there are no ROCm builds", () => {
    expect(findCommunityRocmGpu([amd("RX 6600")], "darwin")).toBeNull();
    expect(findCommunityRocmGpu([amd("RX 6600")], "")).toBeNull();
  });
});
