// About one dedicated-VRAM reading a second for one llama-server process, for
// the whole life of a context-test load -- what shared/gpuSpill.ts's
// measurePhasedGpuSpill judges the ready hold and the prompt+generation on.
//
// Windows: ONE powershell.exe for the whole load, looping on Get-Counter. A
// fresh powershell.exe per reading (MemorySampler's path) costs ~5.5s for the
// first Get-Counter of a session; inside one session each call takes ~1s,
// because Get-Counter itself samples over its one-second interval. Measured on
// the RX 6600 XT with a llama-server running 10 layers of a 1B model: the
// process's counter instance was visible from the first reading, and readings
// came 1.00-1.02s apart. The loop exits by itself when either the traced
// process or this worker is gone, so a crashed worker cannot leave it running.
//
// Everywhere else: readGpuMemory's per-process figure (amdgpu fdinfo,
// nvidia-smi), polled from this process about once a second, one reading at a
// time.
//
// Windows sums the pid's segments across every adapter. On an Optimus laptop
// that would add any iGPU allocation of the same process -- llama-server makes
// none there that has been seen, but it is not filtered out.

import { spawn, type ChildProcess } from "node:child_process";
import { platform as osPlatform } from "node:os";
import type { Backend } from "../../shared/types.js";
import type { MemoryReading } from "../../shared/gpuSpill.js";
import { readGpuMemory } from "./vram.js";

const INTERVAL_MS = 1000;

export function windowsTraceScript(pid: number, parentPid: number): string {
  return [
    `$ErrorActionPreference = 'SilentlyContinue'`,
    `while ($true) {`,
    `  if (-not (Get-Process -Id ${pid})) { break }`,
    `  if (-not (Get-Process -Id ${parentPid})) { break }`,
    `  $c = (Get-Counter '\\GPU Process Memory(pid_${pid}_*)\\Dedicated Usage').CounterSamples`,
    `  $d = ($c | Measure-Object -Property CookedValue -Sum).Sum`,
    `  [Console]::Out.WriteLine('t=' + [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() + ' d=' + $d)`,
    `}`,
  ].join("\n");
}

/** One `t=<epoch ms> d=<bytes>` line; an empty d means no counter instance yet. */
export function parseTraceLine(line: string): MemoryReading | null {
  const m = /^t=(\d+) d=(\d*(?:\.\d+)?)\s*$/.exec(line.trim());
  if (!m) return null;
  const bytes = m[2] === "" ? null : Number(m[2]);
  return {
    atMs: Number(m[1]),
    dedicatedMib: bytes != null && Number.isFinite(bytes) && bytes > 0 ? bytes / (1024 * 1024) : null,
  };
}

export class MemoryTrace {
  private readonly readings: MemoryReading[] = [];
  private child: ChildProcess | null = null;
  private timer: NodeJS.Timeout | null = null;
  private polling = false;
  private buffered = "";

  start(pid: number | undefined, backend: Backend): void {
    if (pid == null || backend === "cpu") return;
    if (osPlatform() === "win32") {
      // -EncodedCommand, not stdin: a multi-line loop fed through `-Command -`
      // is parsed line by line and never runs as one block.
      const encoded = Buffer.from(windowsTraceScript(pid, process.pid), "utf16le").toString("base64");
      this.child = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], {
        windowsHide: true,
        stdio: ["ignore", "pipe", "ignore"],
      });
      this.child.stdout?.on("data", (chunk: Buffer) => {
        this.buffered += chunk.toString();
        const lines = this.buffered.split(/\r?\n/);
        this.buffered = lines.pop() ?? "";
        for (const line of lines) {
          const reading = parseTraceLine(line);
          if (reading) this.readings.push(reading);
        }
      });
      this.child.on("error", () => undefined);
      return;
    }
    // One reading at a time: the next is scheduled only after the previous one
    // returns, so a slow nvidia-smi never overlaps itself or reorders readings.
    this.polling = true;
    const tick = async () => {
      const startedAt = Date.now();
      const reading = await readGpuMemory(backend, pid).catch(() => null);
      if (!this.polling) return;
      this.readings.push({ atMs: Date.now(), dedicatedMib: reading?.process?.mib ?? null });
      this.timer = setTimeout(() => void tick(), Math.max(0, INTERVAL_MS - (Date.now() - startedAt)));
    };
    this.timer = setTimeout(() => void tick(), INTERVAL_MS);
  }

  /** Readings so far, oldest first. */
  snapshot(): MemoryReading[] {
    return [...this.readings];
  }

  stop(): MemoryReading[] {
    this.polling = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.child && this.child.exitCode == null) this.child.kill();
    this.child = null;
    return this.snapshot();
  }
}
