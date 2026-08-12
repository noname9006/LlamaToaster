import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

const BYTES_PER_MIB = 1024 * 1024;

// Windows-only: reads GPU VRAM via the OS's built-in "GPU Process Memory" /
// "GPU Adapter Memory" performance counters. These are tracked by the WDDM
// graphics kernel itself, not the GPU vendor's driver -- so unlike
// systeminformation's graphics() (see sampler.ts's comment: memoryUsed there
// is populated only from nvidia-smi output), this works the same way for
// AMD/Intel/NVIDIA under any backend (vulkan/cuda/rocm). Confirmed live: on
// an AMD RX 6600 XT under Vulkan, si.graphics() never reports memoryUsed at
// all, while these counters match Task Manager's GPU memory graph exactly.
//
// Get-Counter itself costs ~1s per call regardless of what's queried --
// tested against both wildcard and pre-resolved concrete instance paths,
// same cost either way, so this isn't fixable by being cleverer about the
// query. Rather than eating that cost on every 2s sample tick, one
// PowerShell process loops internally for the run's duration and streams a
// fresh reading over stdout roughly every ~1-1.5s; MemorySampler just reads
// whatever's most recently arrived.
function buildLoopScript(pid: number): string {
  return `
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
  try {
    $sum = (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples |
      Where-Object { $_.InstanceName -match '^pid_${pid}_' } |
      Measure-Object -Property CookedValue -Sum |
      Select-Object -ExpandProperty Sum
    if (-not $sum) { $sum = 0 }
    Write-Output $sum
  } catch {
    Write-Output -1
  }
  Start-Sleep -Milliseconds 200
}
`;
}

export class WindowsGpuMemoryReader {
  private proc: ReturnType<typeof spawn> | null = null;
  private latestBytes: number | null = null;

  start(pid: number): void {
    try {
      this.proc = spawn(
        "powershell.exe",
        ["-NoProfile", "-NonInteractive", "-Command", buildLoopScript(pid)],
        { windowsHide: true, stdio: ["ignore", "pipe", "ignore"] }
      );
      const rl = createInterface({ input: this.proc.stdout! });
      rl.on("line", (line) => {
        const n = Number(line.trim());
        if (Number.isFinite(n) && n >= 0) this.latestBytes = n;
      });
      this.proc.on("error", () => {
        this.latestBytes = null;
      });
    } catch {
      /* best-effort -- current stays null, caller falls back */
      this.proc = null;
    }
  }

  // Most recent per-process dedicated VRAM reading, in bytes. Null until the
  // background PowerShell loop has produced its first sample (~1-1.5s after
  // start()), or permanently if the counter category isn't available on
  // this box.
  get currentBytes(): number | null {
    return this.latestBytes;
  }

  stop(): void {
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
    this.latestBytes = null;
  }
}

// One-off snapshot of system-wide dedicated VRAM currently in use (all
// processes, all adapters), for the pre-run free-memory baseline -- see
// captureFreeMemoryBaseline in sampler.ts. A single Get-Counter call, same
// ~1s cost as above, but this only runs once right before a bench starts.
export async function readAdapterDedicatedUsageMib(): Promise<number | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: number | null) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    try {
      const script =
        "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | " +
        "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum";
      const proc = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
        windowsHide: true,
        stdio: ["ignore", "pipe", "ignore"],
      });
      let out = "";
      proc.stdout?.on("data", (d) => (out += d.toString()));
      proc.on("close", () => {
        const n = Number(out.trim());
        finish(Number.isFinite(n) && n >= 0 ? Math.round(n / BYTES_PER_MIB) : null);
      });
      proc.on("error", () => finish(null));
      // Get-Counter has never taken more than ~1.5s in testing; 5s is a
      // generous ceiling so a hung/missing counter provider can't delay a
      // run's start.
      const timer = setTimeout(() => {
        proc.kill();
        finish(null);
      }, 5000);
      proc.on("close", () => clearTimeout(timer));
    } catch {
      finish(null);
    }
  });
}
