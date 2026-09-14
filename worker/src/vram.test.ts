import { beforeEach, describe, expect, it, vi } from "vitest";

// readGpuMemory("cuda") on Windows: nvidia-smi's per-process used_memory is
// "[N/A]" under WDDM and it has no shared-memory figure at all, so the process
// dedicated/shared readings and the whole-adapter shared reading come from
// WDDM's own performance counters instead. child_process and os are mocked so
// each source can be driven independently; the stdout shapes below are the
// real ones captured on a GeForce MX150 (driver 582.66) running llama-server.

const { mockExecFile, mockPlatform } = vi.hoisted(() => ({
  mockExecFile: vi.fn(),
  mockPlatform: vi.fn(() => "win32"),
}));

vi.mock("node:child_process", () => ({ execFile: mockExecFile }));
vi.mock("node:os", async (importOriginal) => ({
  ...(await importOriginal<typeof import("node:os")>()),
  platform: mockPlatform,
}));
vi.mock("systeminformation", () => ({ default: { graphics: vi.fn() } }));

const { readGpuMemory, parseWindowsCudaMemoryOutput, windowsCudaMemoryCommand } = await import("./vram.js");

const PID = 708;
const MIB = 1024 * 1024;

interface Outputs {
  wholeAdapter?: string;
  computeApps?: string;
  powershell?: string | Error;
}

// promisify(execFile) on a plain mock uses the callback convention, so the
// resolved value is whatever is handed to the callback: { stdout, stderr }.
function driveExecFile(outputs: Outputs): void {
  mockExecFile.mockImplementation((file: string, args: string[], ...rest: unknown[]) => {
    const callback = rest[rest.length - 1] as (err: Error | null, value?: { stdout: string; stderr: string }) => void;
    let out: string | Error | undefined;
    if (file === "nvidia-smi" && args[0]?.startsWith("--query-gpu=")) out = outputs.wholeAdapter;
    else if (file === "nvidia-smi" && args[0]?.startsWith("--query-compute-apps=")) out = outputs.computeApps;
    else if (file === "powershell.exe") out = outputs.powershell;
    if (out === undefined) callback(new Error(`unexpected exec: ${file}`));
    else if (out instanceof Error) callback(out);
    else callback(null, { stdout: out, stderr: "" });
  });
}

function powershellCalls(): string[][] {
  return mockExecFile.mock.calls.filter((c) => c[0] === "powershell.exe").map((c) => c[1] as string[]);
}

describe("parseWindowsCudaMemoryOutput", () => {
  it("reads all three labeled sums in MiB", () => {
    expect(
      parseWindowsCudaMemoryOutput("dedicated=1009205248\r\nshared=75497472\r\nadapter_shared=78180352\r\n")
    ).toEqual({ dedicatedMib: 962, sharedMib: 72, adapterSharedMib: 75 });
  });

  it("treats an empty sum as no reading, not zero", () => {
    expect(parseWindowsCudaMemoryOutput("dedicated=\r\nshared=\r\nadapter_shared=\r\n")).toEqual({
      dedicatedMib: null,
      sharedMib: null,
      adapterSharedMib: null,
    });
  });

  it("keeps a measured zero as zero", () => {
    expect(parseWindowsCudaMemoryOutput("dedicated=0\nshared=9601024\nadapter_shared=0\n")).toEqual({
      dedicatedMib: 0,
      sharedMib: 9,
      adapterSharedMib: 0,
    });
  });

  it("cannot shift one value into another's slot when a line is missing", () => {
    expect(parseWindowsCudaMemoryOutput("shared=190054400\n")).toEqual({
      dedicatedMib: null,
      sharedMib: 181,
      adapterSharedMib: null,
    });
  });

  it("rejects non-numeric output", () => {
    expect(parseWindowsCudaMemoryOutput("dedicated=oops\nshared=-5\nadapter_shared=x\n")).toEqual({
      dedicatedMib: null,
      sharedMib: null,
      adapterSharedMib: null,
    });
  });
});

describe("windowsCudaMemoryCommand", () => {
  const command = windowsCudaMemoryCommand(4);

  it("anchors every per-process path on the trailing underscore so pid 4 never matches pid 4452", () => {
    expect(command).toContain("\\GPU Process Memory(pid_4_*)\\Dedicated Usage");
    expect(command).toContain("\\GPU Process Memory(pid_4_*)\\Shared Usage");
    expect(command).toContain("-like 'pid_4_*'");
    expect(command).not.toMatch(/pid_4\*/);
  });

  it("reads the whole-adapter shared counter in the same Get-Counter call", () => {
    expect(command.match(/Get-Counter/g)).toHaveLength(1);
    expect(command).toContain("\\GPU Adapter Memory(*)\\Shared Usage");
  });

  it("finds CUDA adapters from any process's engine instances, not only this one's", () => {
    expect(command).toContain("\\GPU Engine(*engtype_cuda)\\Utilization Percentage");
    expect(command).toContain("'_phys_*'");
  });

  it("emits labeled lines", () => {
    expect(command).toContain("'dedicated='");
    expect(command).toContain("'shared='");
    expect(command).toContain("'adapter_shared='");
  });
});

describe("readGpuMemory on the cuda backend", () => {
  beforeEach(() => {
    mockExecFile.mockReset();
    mockPlatform.mockReset().mockReturnValue("win32");
  });

  it("falls back to WDDM counters for everything nvidia-smi cannot report under WDDM", async () => {
    driveExecFile({
      wholeAdapter: "2048, 963\n",
      computeApps: `${PID}, [N/A]\n`,
      powershell: `dedicated=${962 * MIB}\r\nshared=${72 * MIB}\r\nadapter_shared=${75 * MIB}\r\n`,
    });
    const r = await readGpuMemory("cuda", PID);
    expect(r.total).toEqual({ mib: 2048, accuracy: "exact", source: "driver_reported_memory" });
    expect(r.used).toEqual({ mib: 963, accuracy: "high", source: "driver_reported_memory" });
    expect(r.usedShared).toEqual({ mib: 75, accuracy: "high", source: "driver_reported_memory" });
    expect(r.process).toEqual({ mib: 962, accuracy: "exact", source: "process_gpu_usage" });
    expect(r.processShared).toEqual({ mib: 72, accuracy: "exact", source: "process_gpu_usage" });
    expect(powershellCalls()).toHaveLength(1);
    expect(powershellCalls()[0]).toContain(windowsCudaMemoryCommand(PID));
  });

  it("still prefers nvidia-smi's per-process figure whenever it has one", async () => {
    driveExecFile({
      wholeAdapter: "24564, 9000\n",
      computeApps: `${PID}, 8700\n`,
      powershell: `dedicated=${8650 * MIB}\r\nshared=${40 * MIB}\r\nadapter_shared=${310 * MIB}\r\n`,
    });
    const r = await readGpuMemory("cuda", PID);
    expect(r.process?.mib).toBe(8700);
    expect(r.processShared?.mib).toBe(40);
    expect(r.usedShared?.mib).toBe(310);
  });

  it("reports no process reading when neither source has attributed the pid yet", async () => {
    driveExecFile({
      wholeAdapter: "2048, 35\n",
      computeApps: "",
      powershell: `dedicated=\r\nshared=\r\nadapter_shared=${1 * MIB}\r\n`,
    });
    const r = await readGpuMemory("cuda", PID);
    expect(r.process).toBeUndefined();
    expect(r.processShared).toBeUndefined();
    expect(r.usedShared?.mib).toBe(1);
    expect(r.used.mib).toBe(35);
  });

  it("omits the whole-adapter shared reading when no CUDA adapter was identified", async () => {
    driveExecFile({
      wholeAdapter: "2048, 963\n",
      computeApps: `${PID}, [N/A]\n`,
      powershell: `dedicated=${962 * MIB}\r\nshared=${72 * MIB}\r\nadapter_shared=\r\n`,
    });
    const r = await readGpuMemory("cuda", PID);
    expect(r.usedShared).toBeUndefined();
    expect(r.process?.mib).toBe(962);
  });

  it("keeps the nvidia-smi readings when the counter query itself fails", async () => {
    driveExecFile({ wholeAdapter: "2048, 1955\n", computeApps: `${PID}, [N/A]\n`, powershell: new Error("timeout") });
    const r = await readGpuMemory("cuda", PID);
    expect(r.used.mib).toBe(1955);
    expect(r.process).toBeUndefined();
    expect(r.processShared).toBeUndefined();
    expect(r.usedShared).toBeUndefined();
  });

  it("never spawns PowerShell on Linux", async () => {
    mockPlatform.mockReturnValue("linux");
    driveExecFile({ wholeAdapter: "24564, 9000\n", computeApps: `${PID}, 8700\n` });
    const r = await readGpuMemory("cuda", PID);
    expect(r.process?.mib).toBe(8700);
    expect(r.processShared).toBeUndefined();
    expect(r.usedShared).toBeUndefined();
    expect(powershellCalls()).toHaveLength(0);
  });

  it("skips every counter query when there is no pid yet", async () => {
    driveExecFile({ wholeAdapter: "2048, 0\n" });
    const r = await readGpuMemory("cuda", undefined);
    expect(r.used.mib).toBe(0);
    expect(r.process).toBeUndefined();
    expect(r.usedShared).toBeUndefined();
    expect(mockExecFile).toHaveBeenCalledTimes(1);
  });
});
