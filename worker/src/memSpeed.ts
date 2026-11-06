// Workers page "Measure memory speed" button -- an on-demand raw
// read/write/copy bandwidth measurement of system RAM and (Windows only,
// via OpenCL) VRAM. Distinct from every other measurement in this app: it
// says nothing about any particular model's tok/s, only the hardware
// ceiling those numbers are bounded by. Result shape: shared/types.ts's
// MemorySpeedResult.
//
// RAM is measured with real OS threads (node:worker_threads) so the
// thread-count sweep reflects genuine parallel memory traffic, not a single
// core's throughput. The per-thread code (memSpeedWorker.mjs) and the
// WebAssembly SIMD kernels it runs (memSpeedKernels.mjs) are deliberately
// plain, untranspiled JavaScript (.mjs, not .ts): a worker_threads Worker
// loads its script fresh, and depending on how this process itself was
// started (tsx's CLI wrapper vs a future bundled build) that fresh load may
// not carry the same TypeScript loader hooks this file enjoys.
//
// Two modes (the job payload's `deep`):
//   * quick -- a fixed 512 MiB region (two 256 MiB halves: read/write cover
//     it all, copy goes first half -> second half), swept over 1, 2, 4,
//     physical-cores and logical-cores threads. Every cell is the MEDIAN of
//     several ~150 ms synchronized windows.
//   * deep -- quick, plus a scan of (nearly) all free RAM in 512 MiB chunks
//     (see planDeepChunks for the safety margin), reporting the spread over
//     chunks. This is what exposes uneven memory (some physical ranges slower
//     than others), which a single small buffer samples at random.
//
// VRAM has no equivalent portable story: there is no GPU compute binding
// available here (no native module builds on this project -- see project
// memory), so it is measured only on Windows, by shelling out to PowerShell
// for an OpenCL kernel sweep (memSpeedVram.ps1, works for NVIDIA/AMD/Intel
// alike since it goes through OpenCL rather than a vendor-specific tool).
// Deep mode passes -Deep to that script, which scans all VRAM (minus a
// margin) the same way. Any other platform, or any failure along the way (no
// OpenCL runtime, no GPU, driver error), reports vram: null with a
// human-readable vramUnavailableReason instead of failing the whole job --
// the RAM half of the result is still useful on its own.

import { Worker as ThreadWorker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { platform as osPlatform, cpus as osCpus, freemem as osFreemem } from "node:os";
import v8 from "node:v8";
import vm from "node:vm";
import si from "systeminformation";
import { log } from "./log.js";
import type {
  MemorySpeedResult,
  MemorySpeedThreadPoint,
  MemorySpeedDeep,
  MemorySpeedStat,
} from "../../shared/types.js";
import { allocSharedMemory, type SharedWasmMemory } from "./memSpeedKernels.mjs";

const execFileAsync = promisify(execFile);

const WORKER_SCRIPT_URL = new URL("./memSpeedWorker.mjs", import.meta.url);
const VRAM_SCRIPT_PATH = fileURLToPath(new URL("./memSpeedVram.ps1", import.meta.url));

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

// Quick mode: one region of 512 MiB = two 256 MiB halves -- comfortably
// bigger than any consumer CPU's L3 (so the sweep really measures DRAM, not
// cache), while staying light enough not to trouble a modest worker box
// that's also expected to load multi-GB GGUF models.
const QUICK_REGION_BYTES = 512 * MIB;
// Each timed window targets ~150 ms of work (the number of passes is
// calibrated from a warm-up pass); a cell is the median of QUICK_TRIALS windows.
const TARGET_WINDOW_NS = 150e6;
const MAX_PASSES = 400;
const QUICK_TRIALS = 5;

// Deep mode: chunk size, how many timed passes per chunk and op (best kept,
// like scripts/membench.py), and the safety margin left free.
export const DEEP_CHUNK_BYTES = 512 * MIB;
const DEEP_PASSES_PER_CHUNK = 3;
const DEEP_MAX_CHUNKS = 256; // 128 GiB
export const DEEP_RESERVE_MIN_BYTES = 2 * GIB;
export const DEEP_RESERVE_FRACTION = 0.15;
const DEEP_SLOW_FRACTION = 0.85;
const DEEP_MAP_MAX_ENTRIES = 512;

const VRAM_TIMEOUT_MS = 60_000;
const VRAM_DEEP_TIMEOUT_MS = 5 * 60_000;
const WORKER_READY_TIMEOUT_MS = 15_000;

// ---------------------------------------------------------------------------
// Pure helpers (exported for tests)
// ---------------------------------------------------------------------------

// Rows to sweep: 1 and 2 threads always (to show the ramp-up), a 4-thread
// midpoint when there's room for it, the physical core count (if known and
// distinct), and the full logical count -- deduped and capped at
// logicalCores. Kept deliberately short (typically 4-5 rows) so the whole
// job finishes in a short time rather than sweeping every thread count up
// to a 64-core box's limit.
export function candidateThreadCounts(physicalCores: number | null, logicalCores: number): number[] {
  const set = new Set<number>([1]);
  if (logicalCores >= 2) set.add(2);
  if (logicalCores >= 3) set.add(Math.min(4, logicalCores));
  if (physicalCores && physicalCores > 0) set.add(physicalCores);
  set.add(Math.max(1, logicalCores));
  return Array.from(set)
    .filter((t) => t >= 1 && t <= Math.max(1, logicalCores))
    .sort((a, b) => a - b);
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function summarize(values: number[]): MemorySpeedStat {
  if (values.length === 0) return { medianGBs: 0, minGBs: 0, maxGBs: 0 };
  return { medianGBs: median(values), minGBs: Math.min(...values), maxGBs: Math.max(...values) };
}

// Splits [0, total) into `threads` equal, `align`-aligned slices (the
// remainder, less than one slice per thread, is simply not touched).
export function splitRange(total: number, threads: number, align = 4096): Array<{ start: number; end: number }> {
  const per = Math.floor(total / threads / align) * align;
  if (per <= 0) throw new Error("region too small to split across that many threads");
  return Array.from({ length: threads }, (_, i) => ({ start: i * per, end: (i + 1) * per }));
}

// How many deep-scan chunks fit: free memory minus a safety margin (the larger
// of 2 GiB and 15% of what is free -- the worker box is also expected to load
// multi-GB models, and exhausting RAM would make the OS start paging, which
// would turn the measurement into garbage), in DEEP_CHUNK_BYTES chunks.
export function planDeepChunks(freeBytes: number): { chunks: number; reserveBytes: number } | { skipped: string } {
  const reserveBytes = Math.max(DEEP_RESERVE_MIN_BYTES, Math.floor(freeBytes * DEEP_RESERVE_FRACTION));
  const budget = freeBytes - reserveBytes;
  const chunks = Math.min(DEEP_MAX_CHUNKS, Math.floor(budget / DEEP_CHUNK_BYTES));
  if (chunks < 2) {
    return {
      skipped: `not enough free RAM for a deep scan (${(freeBytes / GIB).toFixed(1)} GiB free, ${(reserveBytes / GIB).toFixed(1)} GiB kept as a safety margin)`,
    };
  }
  return { chunks, reserveBytes };
}

export function deepFromChunks(
  chunkMiB: number,
  read: number[],
  write: number[],
  copy: number[]
): MemorySpeedDeep {
  const readStat = summarize(read);
  const slowChunks = read.filter((v) => v < readStat.medianGBs * DEEP_SLOW_FRACTION).length;
  return {
    chunkMiB,
    chunks: read.length,
    testedMiB: chunkMiB * read.length,
    read: readStat,
    write: summarize(write),
    copy: summarize(copy),
    slowChunks,
    chunkReadGBs: read.slice(0, DEEP_MAP_MAX_ENTRIES).map((v) => Math.round(v * 10) / 10),
  };
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

type Op = "read" | "write" | "copy";

interface RunOutcome {
  gbs: number;
  windowNs: number;
}

interface Pending {
  resolve: (msg: Record<string, unknown>) => void;
  reject: (err: Error) => void;
}

// Persistent worker threads sharing one control block (barrier + per-thread
// timestamps). Reused across every row, op and chunk of a job so thread
// startup, JIT warm-up and wasm compilation are paid once, not per row.
class ThreadPool {
  private readonly workers: ThreadWorker[] = [];
  private readonly pending: Array<Map<number, Pending>> = [];
  private readonly ctl: Int32Array;
  private readonly times: BigInt64Array;
  private reqId = 0;
  private gen = 0;

  constructor(readonly size: number) {
    this.ctl = new Int32Array(new SharedArrayBuffer(8));
    this.times = new BigInt64Array(new SharedArrayBuffer(size * 16));
  }

  async start(): Promise<void> {
    const ctlBuf = this.ctl.buffer as SharedArrayBuffer;
    const timesBuf = this.times.buffer as SharedArrayBuffer;
    const readies: Array<Promise<void>> = [];
    for (let i = 0; i < this.size; i++) {
      const worker = new ThreadWorker(WORKER_SCRIPT_URL, { workerData: { ctl: ctlBuf, times: timesBuf } });
      const waiting = new Map<number, Pending>();
      this.workers.push(worker);
      this.pending.push(waiting);
      readies.push(
        new Promise<void>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error("memory-speed worker thread did not become ready in time")), WORKER_READY_TIMEOUT_MS);
          worker.on("message", (msg: Record<string, unknown>) => {
            if (msg.ready) {
              clearTimeout(timer);
              resolve();
              return;
            }
            const p = waiting.get(msg.reqId as number);
            if (!p) return;
            waiting.delete(msg.reqId as number);
            if (typeof msg.error === "string") p.reject(new Error(msg.error));
            else p.resolve(msg);
          });
          worker.on("error", (err: Error) => {
            clearTimeout(timer);
            reject(err);
            for (const p of waiting.values()) p.reject(err);
            waiting.clear();
          });
        })
      );
    }
    await Promise.all(readies);
  }

  private call(i: number, msg: Record<string, unknown>): Promise<Record<string, unknown>> {
    const reqId = ++this.reqId;
    return new Promise((resolve, reject) => {
      this.pending[i].set(reqId, { resolve, reject });
      this.workers[i].postMessage({ ...msg, reqId });
    });
  }

  async attach(region: number, memory: SharedWasmMemory, threads: number): Promise<void> {
    await Promise.all(Array.from({ length: threads }, (_, i) => this.call(i, { cmd: "attach", region, memory })));
  }

  async detach(region: number, threads: number): Promise<void> {
    await Promise.all(Array.from({ length: threads }, (_, i) => this.call(i, { cmd: "detach", region })));
  }

  // Every thread writes its own slice of [0, bytes) once.
  async commit(region: number, bytes: number, threads: number): Promise<void> {
    const slices = splitRange(bytes, threads);
    await Promise.all(slices.map((s, i) => this.call(i, { cmd: "commit", region, start: s.start, end: s.end })));
  }

  // Runs one op over `region` (size `bytes`; copy goes first half -> second
  // half) on `threads` threads, `passes` times, all starting at a barrier.
  // GB/s = total bytes moved / (latest end - earliest start).
  async run(op: Op, region: number, bytes: number, threads: number, passes: number): Promise<RunOutcome> {
    const span = op === "copy" ? bytes / 2 : bytes;
    const delta = op === "copy" ? bytes / 2 : 0;
    const slices = splitRange(span, threads);
    Atomics.store(this.ctl, 0, 0);
    const gen = ++this.gen;
    await Promise.all(
      slices.map((s, i) =>
        this.call(i, { cmd: "run", op, region, start: s.start, end: s.end, delta, passes, gen, slot: i, threads })
      )
    );
    let first = this.times[0];
    let last = this.times[1];
    for (let i = 1; i < threads; i++) {
      if (this.times[2 * i] < first) first = this.times[2 * i];
      if (this.times[2 * i + 1] > last) last = this.times[2 * i + 1];
    }
    const windowNs = Number(last - first);
    if (windowNs <= 0) return { gbs: 0, windowNs: 0 };
    const perThread = slices[0].end - slices[0].start;
    const moved = perThread * threads * passes * (op === "copy" ? 2 : 1);
    return { gbs: moved / windowNs, windowNs }; // bytes per ns == GB/s
  }

  async close(): Promise<void> {
    await Promise.all(this.workers.map((w) => w.terminate().catch(() => undefined)));
  }
}

// Lets V8 reclaim big shared memories now rather than whenever it next
// happens to collect -- the worker process lives on (it serves llama.cpp for
// hours) and must not sit on gigabytes it only borrowed for a measurement.
function forceGc(): void {
  try {
    v8.setFlagsFromString("--expose-gc");
    const gc = vm.runInNewContext("gc") as (() => void) | undefined;
    gc?.();
  } catch {
    // best effort
  }
}

// ---------------------------------------------------------------------------
// RAM
// ---------------------------------------------------------------------------

// One cell: a warm-up window (JIT / caches; also tells how long one pass
// takes), then QUICK_TRIALS synchronized windows of ~TARGET_WINDOW_NS each;
// the cell is their median.
async function measureCell(pool: ThreadPool, op: Op, region: number, threads: number): Promise<number> {
  const warm = await pool.run(op, region, QUICK_REGION_BYTES, threads, 1);
  const passes = warm.windowNs > 0 ? Math.min(MAX_PASSES, Math.max(1, Math.ceil(TARGET_WINDOW_NS / warm.windowNs))) : 1;
  const samples: number[] = [];
  for (let t = 0; t < QUICK_TRIALS; t++) samples.push((await pool.run(op, region, QUICK_REGION_BYTES, threads, passes)).gbs);
  return median(samples);
}

async function measureQuickRows(
  pool: ThreadPool,
  threadCounts: number[],
  onProgress?: (detail: string) => void
): Promise<MemorySpeedThreadPoint[]> {
  const region = 0;
  const memory = allocSharedMemory(QUICK_REGION_BYTES);
  await pool.attach(region, memory, pool.size);
  await pool.commit(region, QUICK_REGION_BYTES, pool.size);
  const points: MemorySpeedThreadPoint[] = [];
  try {
    for (const threads of threadCounts) {
      const cells: Record<Op, number> = { read: 0, write: 0, copy: 0 };
      for (const op of ["read", "write", "copy"] as const) {
        onProgress?.(`measuring RAM bandwidth (${threads} thread${threads === 1 ? "" : "s"}, ${op})…`);
        cells[op] = await measureCell(pool, op, region, threads);
      }
      points.push({ threads, readGBs: cells.read, writeGBs: cells.write, copyGBs: cells.copy });
    }
  } finally {
    await pool.detach(region, pool.size).catch(() => undefined);
  }
  return points;
}

async function measureRamDeep(
  pool: ThreadPool,
  threads: number,
  onProgress?: (detail: string) => void
): Promise<{ deep: MemorySpeedDeep & { threads: number } } | { skipped: string }> {
  const plan = planDeepChunks(osFreemem());
  if ("skipped" in plan) return plan;

  // Allocate (and commit) EVERY chunk before measuring any: only then do the
  // chunks span distinct physical pages. Measuring one chunk at a time and
  // freeing it would just keep re-measuring the same few physical ranges.
  const memories: SharedWasmMemory[] = [];
  try {
    for (let i = 0; i < plan.chunks; i++) {
      onProgress?.(`deep RAM scan: allocating ${i + 1}/${plan.chunks} (${((i + 1) * DEEP_CHUNK_BYTES / GIB).toFixed(1)} GiB)…`);
      let memory: SharedWasmMemory;
      try {
        memory = allocSharedMemory(DEEP_CHUNK_BYTES);
      } catch (err) {
        log.warn(`deep RAM scan: allocation stopped at chunk ${i + 1}: ${err instanceof Error ? err.message : String(err)}`);
        break;
      }
      memories.push(memory);
      await pool.attach(i, memory, threads);
      await pool.commit(i, DEEP_CHUNK_BYTES, threads);
    }
    if (memories.length < 2) return { skipped: "could not allocate enough memory for a deep scan" };

    const read: number[] = [];
    const write: number[] = [];
    const copy: number[] = [];
    for (let i = 0; i < memories.length; i++) {
      onProgress?.(`deep RAM scan: chunk ${i + 1}/${memories.length}…`);
      const best: Record<Op, number> = { read: 0, write: 0, copy: 0 };
      for (const op of ["read", "write", "copy"] as const) {
        await pool.run(op, i, DEEP_CHUNK_BYTES, threads, 1); // untimed: kernels/caches warm
        for (let p = 0; p < DEEP_PASSES_PER_CHUNK; p++) {
          const gbs = (await pool.run(op, i, DEEP_CHUNK_BYTES, threads, 1)).gbs;
          if (gbs > best[op]) best[op] = gbs;
        }
      }
      read.push(best.read);
      write.push(best.write);
      copy.push(best.copy);
    }
    return { deep: { ...deepFromChunks(DEEP_CHUNK_BYTES / MIB, read, write, copy), threads } };
  } finally {
    for (let i = 0; i < memories.length; i++) await pool.detach(i, threads).catch(() => undefined);
    memories.length = 0;
  }
}

async function measureRamBandwidth(
  deep: boolean,
  onProgress?: (detail: string) => void
): Promise<MemorySpeedResult["ram"]> {
  const cpu = await si.cpu().catch(() => null);
  const logicalCores = cpu?.cores && cpu.cores > 0 ? cpu.cores : osCpus().length || 1;
  const physicalCores = cpu?.physicalCores && cpu.physicalCores > 0 ? cpu.physicalCores : null;
  const threadCounts = candidateThreadCounts(physicalCores, logicalCores);

  const pool = new ThreadPool(Math.max(...threadCounts));
  try {
    await pool.start();
    const points = await measureQuickRows(pool, threadCounts, onProgress);
    const ram: MemorySpeedResult["ram"] = { physicalCores, logicalCores, points };
    if (deep) {
      // The quick sweep says where reads peak -- scan at that thread count.
      const best = points.reduce((a, b) => (b.readGBs > a.readGBs ? b : a), points[0]);
      const out = await measureRamDeep(pool, best.threads, onProgress);
      if ("deep" in out) ram.deep = out.deep;
      else ram.deepSkippedReason = out.skipped;
    }
    return ram;
  } finally {
    await pool.close();
    forceGc();
  }
}

// ---------------------------------------------------------------------------
// VRAM
// ---------------------------------------------------------------------------

async function measureVramBandwidth(deep: boolean): Promise<{ vram: MemorySpeedResult["vram"]; reason?: string }> {
  if (osPlatform() !== "win32") {
    return { vram: null, reason: "VRAM bandwidth measurement is only implemented for Windows (OpenCL) right now." };
  }
  try {
    const args = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", VRAM_SCRIPT_PATH];
    if (deep) args.push("-Deep");
    const { stdout } = await execFileAsync("powershell.exe", args, {
      timeout: deep ? VRAM_DEEP_TIMEOUT_MS : VRAM_TIMEOUT_MS,
      windowsHide: true,
      maxBuffer: 4 * MIB,
    });
    // The script prints exactly one JSON line, but be defensive about stray
    // PowerShell warning noise ahead of it and take the last non-empty line.
    const lines = stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const lastLine = lines[lines.length - 1] ?? "";
    const parsed = JSON.parse(lastLine) as
      | {
          ok: true;
          deviceName: string;
          globalMemMiB: number;
          readGBs: number;
          writeGBs: number;
          copyGBs: number;
          deep?: MemorySpeedDeep;
          deepSkippedReason?: string;
        }
      | { ok: false; reason: string };
    if (!parsed.ok) return { vram: null, reason: parsed.reason };
    return {
      vram: {
        deviceName: parsed.deviceName,
        globalMemMiB: parsed.globalMemMiB,
        readGBs: parsed.readGBs,
        writeGBs: parsed.writeGBs,
        copyGBs: parsed.copyGBs,
        ...(parsed.deep ? { deep: normalizeVramDeep(parsed.deep) } : {}),
        ...(parsed.deepSkippedReason ? { deepSkippedReason: parsed.deepSkippedReason } : {}),
      },
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.warn(`VRAM bandwidth measurement failed (non-fatal, RAM result still reported): ${message}`);
    return { vram: null, reason: `VRAM measurement failed: ${message}` };
  }
}

// The PowerShell script reports per-chunk arrays; summarising here keeps the
// statistics (median / min / max / slow-chunk rule) in one place for RAM and VRAM.
function normalizeVramDeep(raw: MemorySpeedDeep & { readChunks?: number[]; writeChunks?: number[]; copyChunks?: number[] }): MemorySpeedDeep {
  const toArr = (v: unknown): number[] => (Array.isArray(v) ? v.filter((n): n is number => typeof n === "number" && Number.isFinite(n)) : typeof v === "number" ? [v] : []);
  const read = toArr(raw.readChunks);
  if (read.length === 0) return raw;
  return deepFromChunks(raw.chunkMiB, read, toArr(raw.writeChunks), toArr(raw.copyChunks));
}

export async function measureMemorySpeed(
  onProgress?: (detail: string) => void,
  opts: { deep?: boolean } = {}
): Promise<MemorySpeedResult> {
  const deep = opts.deep === true;
  const ram = await measureRamBandwidth(deep, onProgress);
  onProgress?.(deep ? "deep VRAM scan…" : "measuring VRAM bandwidth…");
  const { vram, reason } = await measureVramBandwidth(deep);
  return { measuredAt: Date.now(), mode: deep ? "deep" : "fast", ram, vram, vramUnavailableReason: reason };
}
