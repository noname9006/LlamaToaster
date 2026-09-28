// Workers page "Measure memory speed" button -- an on-demand raw
// read/write/copy bandwidth measurement of system RAM and (Windows only,
// via OpenCL) VRAM. Distinct from every other measurement in this app: it
// says nothing about any particular model's tok/s, only the hardware
// ceiling those numbers are bounded by. Result shape: shared/types.ts's
// MemorySpeedResult.
//
// RAM is measured with real OS threads (node:worker_threads) so the
// thread-count sweep reflects genuine parallel memory traffic, not a single
// core's throughput -- see memSpeedWorker.mjs, the per-thread entry point.
// That file is deliberately plain, untranspiled JavaScript (not .ts): a
// worker_threads Worker loads its script fresh, and depending on how this
// process itself was started (tsx's CLI wrapper vs a future bundled build)
// that fresh load may not carry the same TypeScript loader hooks this file
// enjoys. A .mjs sidesteps the question entirely.
//
// VRAM bandwidth has no equivalent portable story: there is no GPU compute
// binding available here (no native module builds on this project -- see
// project memory), so it is measured only on Windows, by shelling out to
// PowerShell for a short OpenCL kernel sweep (memSpeedVram.ps1, works for
// NVIDIA/AMD/Intel alike since it goes through OpenCL rather than a
// vendor-specific tool). Any other platform, or any failure along the way
// (no OpenCL runtime, no GPU, driver error), reports vram: null with a
// human-readable vramUnavailableReason instead of failing the whole job --
// the RAM half of the result is still useful on its own.

import { Worker as ThreadWorker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { platform as osPlatform, cpus as osCpus } from "node:os";
import si from "systeminformation";
import { log } from "./log.js";
import type { MemorySpeedResult, MemorySpeedThreadPoint } from "../../shared/types.js";

const execFileAsync = promisify(execFile);

const WORKER_SCRIPT_URL = new URL("./memSpeedWorker.mjs", import.meta.url);
const VRAM_SCRIPT_PATH = fileURLToPath(new URL("./memSpeedVram.ps1", import.meta.url));

// 256 MiB per array (512 MiB peak with both the read/write source and the
// copy destination alive at once) -- comfortably bigger than any consumer
// CPU's L3 (so the sweep really measures DRAM, not cache), while staying
// light enough not to trouble a modest worker box that's also expected to
// load multi-GB GGUF models.
const RAM_MIB_PER_ARRAY = 256;
const RAM_TRIALS = 3;
const RAM_REPS = 4;
const VRAM_TIMEOUT_MS = 60_000;
const WORKER_READY_TIMEOUT_MS = 15_000;

// Rows to sweep: 1 and 2 threads always (to show the ramp-up), a 4-thread
// midpoint when there's room for it, the physical core count (if known and
// distinct), and the full logical count -- deduped and capped at
// logicalCores. Kept deliberately short (typically 4-5 rows) so the whole
// job finishes in a few seconds rather than sweeping every thread count up
// to a 64-core box's limit.
function candidateThreadCounts(physicalCores: number | null, logicalCores: number): number[] {
  const set = new Set<number>([1]);
  if (logicalCores >= 2) set.add(2);
  if (logicalCores >= 3) set.add(Math.min(4, logicalCores));
  if (physicalCores && physicalCores > 0) set.add(physicalCores);
  set.add(Math.max(1, logicalCores));
  return Array.from(set)
    .filter((t) => t >= 1 && t <= Math.max(1, logicalCores))
    .sort((a, b) => a - b);
}

type Op = "read" | "write" | "copy";

interface ThreadHandle {
  worker: ThreadWorker;
  ready: Promise<void>;
}

function spawnThread(bufA: SharedArrayBuffer, bufB: SharedArrayBuffer, start: number, end: number): ThreadHandle {
  const worker = new ThreadWorker(WORKER_SCRIPT_URL, { workerData: { bufA, bufB, start, end } });
  const ready = new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("memory-speed worker thread did not become ready in time")), WORKER_READY_TIMEOUT_MS);
    worker.once("message", (msg: { ready?: boolean }) => {
      clearTimeout(timer);
      if (msg?.ready) resolve();
      else reject(new Error("memory-speed worker thread sent an unexpected first message"));
    });
    worker.once("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
  return { worker, ready };
}

// Runs one op across every already-ready thread and returns the row's
// GB/s: total bytes moved by all threads divided by the SLOWEST thread's
// elapsed time (the other threads necessarily finished no later, since all
// were sent the same 'run' message at the same wall-clock instant) -- the
// standard way to report a parallel-threads bandwidth figure.
async function runOp(threads: ThreadHandle[], op: Op, reps: number): Promise<number> {
  const results = await Promise.all(
    threads.map(
      (t) =>
        new Promise<{ bytes: number; elapsedMs: number }>((resolve, reject) => {
          t.worker.once("message", (msg: { done?: boolean; bytes: number; elapsedMs: number }) => {
            if (!msg?.done) {
              reject(new Error("memory-speed worker thread sent an unexpected message"));
              return;
            }
            resolve({ bytes: msg.bytes, elapsedMs: msg.elapsedMs });
          });
          t.worker.once("error", reject);
          t.worker.postMessage({ cmd: "run", op, reps });
        })
    )
  );
  const totalBytes = results.reduce((sum, r) => sum + r.bytes, 0);
  const slowestMs = Math.max(...results.map((r) => r.elapsedMs));
  if (slowestMs <= 0) return 0;
  return totalBytes / (slowestMs / 1000) / 1e9;
}

async function measureRamRow(threadCount: number): Promise<MemorySpeedThreadPoint> {
  const elemsPerArray = Math.floor((RAM_MIB_PER_ARRAY * 1024 * 1024) / 8);
  const chunkLen = Math.floor(elemsPerArray / threadCount);
  const usedElems = chunkLen * threadCount;
  const bufA = new SharedArrayBuffer(usedElems * 8);
  const bufB = new SharedArrayBuffer(usedElems * 8);

  const threads: ThreadHandle[] = [];
  for (let i = 0; i < threadCount; i++) {
    threads.push(spawnThread(bufA, bufB, i * chunkLen, (i + 1) * chunkLen));
  }
  try {
    await Promise.all(threads.map((t) => t.ready));
    const best: Record<Op, number> = { read: 0, write: 0, copy: 0 };
    for (const op of ["read", "write", "copy"] as const) {
      for (let trial = 0; trial < RAM_TRIALS; trial++) {
        const gbs = await runOp(threads, op, RAM_REPS);
        if (gbs > best[op]) best[op] = gbs;
      }
    }
    return { threads: threadCount, readGBs: best.read, writeGBs: best.write, copyGBs: best.copy };
  } finally {
    await Promise.all(threads.map((t) => t.worker.terminate().catch(() => undefined)));
  }
}

async function measureRamBandwidth(onProgress?: (detail: string) => void): Promise<MemorySpeedResult["ram"]> {
  const cpu = await si.cpu().catch(() => null);
  const logicalCores = cpu?.cores && cpu.cores > 0 ? cpu.cores : osCpus().length || 1;
  const physicalCores = cpu?.physicalCores && cpu.physicalCores > 0 ? cpu.physicalCores : null;
  const threadCounts = candidateThreadCounts(physicalCores, logicalCores);

  const points: MemorySpeedThreadPoint[] = [];
  for (const threadCount of threadCounts) {
    onProgress?.(`measuring RAM bandwidth (${threadCount} thread${threadCount === 1 ? "" : "s"})…`);
    points.push(await measureRamRow(threadCount));
  }
  return { physicalCores, logicalCores, points };
}

async function measureVramBandwidth(): Promise<{ vram: MemorySpeedResult["vram"]; reason?: string }> {
  if (osPlatform() !== "win32") {
    return { vram: null, reason: "VRAM bandwidth measurement is only implemented for Windows (OpenCL) right now." };
  }
  try {
    const { stdout } = await execFileAsync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", VRAM_SCRIPT_PATH],
      { timeout: VRAM_TIMEOUT_MS, windowsHide: true }
    );
    // The script prints exactly one JSON line, but be defensive about stray
    // PowerShell warning noise ahead of it and take the last non-empty line.
    const lines = stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const lastLine = lines[lines.length - 1] ?? "";
    const parsed = JSON.parse(lastLine) as
      | { ok: true; deviceName: string; globalMemMiB: number; readGBs: number; writeGBs: number; copyGBs: number }
      | { ok: false; reason: string };
    if (!parsed.ok) return { vram: null, reason: parsed.reason };
    return {
      vram: {
        deviceName: parsed.deviceName,
        globalMemMiB: parsed.globalMemMiB,
        readGBs: parsed.readGBs,
        writeGBs: parsed.writeGBs,
        copyGBs: parsed.copyGBs,
      },
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.warn(`VRAM bandwidth measurement failed (non-fatal, RAM result still reported): ${message}`);
    return { vram: null, reason: `VRAM measurement failed: ${message}` };
  }
}

export async function measureMemorySpeed(onProgress?: (detail: string) => void): Promise<MemorySpeedResult> {
  const ram = await measureRamBandwidth(onProgress);
  onProgress?.("measuring VRAM bandwidth…");
  const { vram, reason } = await measureVramBandwidth();
  return { measuredAt: Date.now(), ram, vram, vramUnavailableReason: reason };
}
