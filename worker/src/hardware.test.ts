import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock systeminformation so detectHardware's vram_dynamic correction can be
// asserted deterministically without querying real GPUs. readNvidiaDriverInfo
// is only called for NVIDIA boxes -- mocked out so the nvidia-smi spawn path
// never runs in a test.
vi.mock("systeminformation", () => ({
  default: {
    cpu: vi.fn(),
    graphics: vi.fn(),
    mem: vi.fn(),
  },
}));

vi.mock("./vram.js", () => ({
  readNvidiaDriverInfo: vi.fn().mockResolvedValue(null),
}));

import siModule from "systeminformation";
import { detectHardware } from "./hardware.js";

const si = vi.mocked(siModule);

// systeminformation's Windows vramDynamic derivation flags VideoMemoryType
// === '2', which is the enum value for *dedicated* VRAM (graphics.js:1227).
// These cases mirror what the library actually returns for the two adapter
// classes this logic must tell apart.
interface FakeController {
  vendor?: string;
  model?: string;
  vram?: number;
  vramDynamic?: boolean;
}

function setUp(gpuControllers: FakeController[]) {
  si.cpu.mockResolvedValue({
    manufacturer: "AMD",
    brand: "AMD Ryzen 5 5600X",
    cores: 6,
  } as never);
  si.graphics.mockResolvedValue({
    controllers: gpuControllers,
  } as never);
  si.mem.mockResolvedValue({ total: 16 * 1024 * 1024 * 1024 } as never);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("detectHardware gpu vram_dynamic correction", () => {
  it("keeps a discrete GPU with dedicated VRAM (vram>0, VideoMemoryType=2) unlabeled as shared", async () => {
    // RX 6600 XT style: systeminformation reports real VRAM AND (Windows
    // false-positive) vramDynamic true.
    setUp([{ vendor: "Advanced Micro Devices, Inc.", model: "AMD Radeon RX 6600 XT", vram: 8192, vramDynamic: true }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_mb).toBe(8192);
    expect(hw.gpu[0].vram_dynamic).toBe(false);
  });

  it("resets the false-positive flag even when vramDynamic is true but VRAM is real", async () => {
    // Convention: any non-zero dedicated vram_mb is ground truth that the
    // card is not shared/unified -- regardless of what the library says.
    setUp([{ vendor: "Intel Corporation", model: "Intel Arc A770", vram: 16384, vramDynamic: true }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_dynamic).toBe(false);
  });

  it("preserves 'shared' for unified adapters that report no dedicated VRAM", async () => {
    // iGPU with no on-die pool (unified memory): vram null, vramDynamic true.
    setUp([{ vendor: "Intel Corporation", model: "Intel(R) UHD Graphics 620", vram: 0, vramDynamic: true }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_mb).toBeNull();
    expect(hw.gpu[0].vram_dynamic).toBe(true);
  });

  it("non-dynamic discrete GPU stays non-dynamic", async () => {
    setUp([{ vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3060", vram: 12288, vramDynamic: false }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_dynamic).toBe(false);
  });
});