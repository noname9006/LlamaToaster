import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync } from "node:fs";
import type { IngestResultInput } from "../../shared/types.js";
import { deriveTestType, type SweepItem } from "../../shared/sweep.js";

export interface BenchLogger {
  debug: (...parts: unknown[]) => void;
  info: (...parts: unknown[]) => void;
  warn: (...parts: unknown[]) => void;
}

export interface BenchRunInput {
  modelPath: string;
  // One sweep combination -- see shared/sweep.ts's expandSweep. A whole
  // sweep is now one runBench call per item rather than one CSV-args call
  // for the entire sweep, so a crash on one combo can't take the rest down
  // with it (see worker/src/index.ts's per-item loop).
  item: SweepItem;
  repeats: number;
  llamaBenchPath: string;
  backend: string;
  timeoutMs?: number;
  onSpawn?: (proc: ChildProcess) => void;
  // Fired per stderr line as it streams in -- lets the caller drive live
  // phase/detail reporting without this module knowing anything about
  // phases itself.
  onStderrLine?: (line: string) => void;
  log?: BenchLogger;
}

const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000;

// Whether a given llama-bench build understands --progress (added upstream
// well after some older installed builds were released -- see
// worker/src/index.ts's marker-line parsing for what it unlocks). Probed
// once per path and cached for the worker process's lifetime: if a build
// doesn't support it, passing it anyway would fail argument parsing and
// break every run on that build, so this is checked rather than assumed.
const progressFlagSupport = new Map<string, boolean>();

export async function supportsProgressFlag(llamaBenchPath: string): Promise<boolean> {
  const cached = progressFlagSupport.get(llamaBenchPath);
  if (cached !== undefined) return cached;
  const supported = await new Promise<boolean>((resolvePromise) => {
    try {
      const proc = spawn(llamaBenchPath, ["-h"], { windowsHide: true });
      let out = "";
      proc.stdout?.on("data", (d) => (out += d.toString()));
      proc.stderr?.on("data", (d) => (out += d.toString()));
      proc.on("error", () => resolvePromise(false));
      proc.on("close", () => resolvePromise(out.includes("--progress")));
      // Guard against a build whose -h somehow hangs -- don't let a probe
      // stall the sweep.
      setTimeout(() => resolvePromise(false), 10_000).unref();
    } catch {
      resolvePromise(false);
    }
  });
  progressFlagSupport.set(llamaBenchPath, supported);
  return supported;
}

// Whether -v/--verbose is understood, probed the same way as --progress
// above and for the same reason: passing an unrecognized flag would fail
// argument parsing and break every run on that build. -v is what unlocks
// the "load_tensors: offloaded X/Y layers to GPU" line this app now reads
// for gpu_layers_loaded/total_model_layers (see worker/src/index.ts's
// parseOffloadLayers) -- confirmed live that llama-bench prints none of
// that detail at default verbosity at all.
const verboseFlagSupport = new Map<string, boolean>();

export async function supportsVerboseFlag(llamaBenchPath: string): Promise<boolean> {
  const cached = verboseFlagSupport.get(llamaBenchPath);
  if (cached !== undefined) return cached;
  const supported = await new Promise<boolean>((resolvePromise) => {
    try {
      const proc = spawn(llamaBenchPath, ["-h"], { windowsHide: true });
      let out = "";
      proc.stdout?.on("data", (d) => (out += d.toString()));
      proc.stderr?.on("data", (d) => (out += d.toString()));
      proc.on("error", () => resolvePromise(false));
      proc.on("close", () => resolvePromise(/-v,\s*--verbose/.test(out)));
      setTimeout(() => resolvePromise(false), 10_000).unref();
    } catch {
      resolvePromise(false);
    }
  });
  verboseFlagSupport.set(llamaBenchPath, supported);
  return supported;
}

export async function buildArgs(input: BenchRunInput): Promise<string[]> {
  const item = input.item;
  const args = [
    "-m",
    input.modelPath,
    "-p",
    String(item.n_prompt),
    "-n",
    String(item.n_gen),
    "-t",
    String(item.threads),
    "-ngl",
    String(item.n_gpu_layers),
    "-b",
    String(item.batch_size),
    "-ub",
    String(item.ubatch_size),
    "-ctk",
    item.cache_type_k,
    "-ctv",
    item.cache_type_v,
    "-fa",
    item.flash_attn,
    "-r",
    String(input.repeats),
    "-o",
    "json",
  ];
  if (await supportsProgressFlag(input.llamaBenchPath)) {
    args.push("--progress");
  }
  // Stderr-only (confirmed live against the real binary: doesn't affect the
  // -o json stdout this module parses) -- required for gpu_info/cpu_info's
  // sibling offload-layer detail, see supportsVerboseFlag above.
  if (await supportsVerboseFlag(input.llamaBenchPath)) {
    args.push("-v");
  }
  return args;
}

export interface BenchResult {
  stdout: string;
  stderr: string;
  code: number | null;
  signal: NodeJS.Signals | null;
  // True when this worker's own bench_timeout_ms fired and killed the
  // process -- distinct from an OS-level OOM kill, which also shows up as a
  // SIGKILL but with timedOut left false. See worker/src/index.ts's OOM
  // classification.
  timedOut: boolean;
  results: IngestResultInput[];
  // Only ever set by the llama-server/MTP path (serverBench.ts) -- a
  // human-readable summary of any readings that were computable but flagged
  // implausible (see MAX_PLAUSIBLE_*_TOKENS_PER_SECOND) or entirely missing
  // for a requested test_type, even though the item still completed. Left
  // undefined by the llama-bench-CLI path, which has no equivalent concept.
  warning?: string;
  // llama-bench's own JSON output's exact device-name strings (e.g. "AMD
  // Radeon RX 6600 XT", "AMD Ryzen 5 5600X 6-Core Processor") -- confirmed
  // live these are always present in that JSON, no flags needed. Used by
  // worker/src/index.ts's runSweepItem for the Tier-2 backend_device_name
  // upgrade (see shared/types.ts's Run.backend_device_name). Undefined on
  // the llama-server/MTP path, which has no JSON output at all, and
  // undefined here too if stdout failed to parse.
  gpu_info?: string;
  cpu_info?: string;
}

export async function runBench(input: BenchRunInput): Promise<BenchResult> {
  const args = await buildArgs(input);
  const timeoutMs = input.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const log = input.log;
  return new Promise((resolve) => {
    log?.info(
      `spawning llama-bench: ${input.llamaBenchPath} (backend=${input.backend}, timeoutMs=${timeoutMs})`
    );
    log?.debug(`llama-bench args: ${args.join(" ")}`);
    const startedAt = Date.now();
    // Spawn the executable directly (no manual `cmd /c` wrapping): Node's own
    // child_process already knows how to invoke .bat/.cmd targets safely on
    // Windows, with proper argument escaping. Re-implementing that by hand
    // here previously meant cmd.exe re-parsed the full command line itself,
    // letting shell metacharacters in any sweep value break out of argv.
    const proc: ChildProcess = spawn(input.llamaBenchPath, args, {
      windowsHide: true,
    });
    input.onSpawn?.(proc);
    let stdout = "";
    let stderr = "";
    let stderrLineBuf = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      const elapsedMs = Date.now() - startedAt;
      log?.warn(
        `llama-bench (pid ${proc.pid}) timed out after ${timeoutMs}ms (elapsed ${elapsedMs}ms), sending SIGKILL`
      );
      stderr += `\ntimed out after ${timeoutMs}ms, killing process`;
      proc.kill("SIGKILL");
    }, timeoutMs);
    proc.stdout?.on("data", (d) => (stdout += d.toString()));
    proc.stderr?.on("data", (d) => {
      const text: string = d.toString();
      stderr += text;
      if (!input.onStderrLine) return;
      stderrLineBuf += text;
      let nl = stderrLineBuf.indexOf("\n");
      while (nl !== -1) {
        const line = stderrLineBuf.slice(0, nl);
        stderrLineBuf = stderrLineBuf.slice(nl + 1);
        if (line.trim()) input.onStderrLine(line);
        nl = stderrLineBuf.indexOf("\n");
      }
    });
    proc.on("error", (err) => {
      log?.warn(`llama-bench spawn error: ${err.message}`);
      stderr += `\nspawn error: ${err.message}`;
    });
    proc.on("close", (code, signal) => {
      clearTimeout(timer);
      const elapsedMs = Date.now() - startedAt;
      log?.info(
        `llama-bench (pid ${proc.pid}) exited with code ${code}${signal ? ` signal ${signal}` : ""} ` +
          `after ${elapsedMs}ms${timedOut ? " (killed on timeout)" : ""}, stdout=${stdout.length}B stderr=${stderr.length}B`
      );
      let results: IngestResultInput[] = [];
      let gpu_info: string | undefined;
      let cpu_info: string | undefined;
      try {
        const parsed = parseLlamaBench(stdout);
        results = parsed.results;
        gpu_info = parsed.gpu_info;
        cpu_info = parsed.cpu_info;
        log?.debug(`parsed ${results.length} result row(s) from llama-bench JSON output`);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log?.warn(`failed to parse llama-bench output: ${message}`);
        stderr += `\n${message} (raw stdout: ${stdout.slice(0, 500)})`;
      }
      resolve({ stdout, stderr, code, signal, timedOut, results, gpu_info, cpu_info });
    });
  });
}

interface LlamaBenchEntry {
  build_commit?: string;
  model_filename?: string;
  model_path?: string;
  model_type?: string;
  n_batch?: number;
  n_ubatch?: number;
  n_threads?: number;
  n_gpu_layers?: number;
  n_prompt?: number;
  n_gen?: number;
  // llama-bench's JSON output names these type_k/type_v, not cache_type_k/v.
  type_k?: string;
  type_v?: string;
  flash_attn?: string | number;
  avg_ts?: number;
  stddev_ts?: number;
  // Exact device-name strings llama-bench itself reports -- confirmed live
  // always present, no flags needed (e.g. gpu_info: "AMD Radeon RX 6600 XT",
  // cpu_info: "AMD Ryzen 5 5600X 6-Core Processor"). See BenchResult's own
  // doc comment for how these get used.
  gpu_info?: string;
  cpu_info?: string;
  [key: string]: unknown;
}

export interface ParsedLlamaBench {
  results: IngestResultInput[];
  gpu_info?: string;
  cpu_info?: string;
}

export function parseLlamaBench(stdout: string): ParsedLlamaBench {
  const arr = extractJsonArray(stdout);
  if (!Array.isArray(arr)) {
    throw new Error("llama-bench did not emit a JSON array");
  }
  const results = arr.map((entry: LlamaBenchEntry) => {
    const nPrompt = entry.n_prompt ?? 0;
    const nGen = entry.n_gen ?? 0;
    return {
      test_type: deriveTestType(nPrompt, nGen),
      n_prompt: nPrompt,
      n_gen: nGen,
      n_threads: Number(entry.n_threads ?? 0),
      n_gpu_layers: entry.n_gpu_layers ?? 0,
      batch_size: entry.n_batch ?? 0,
      ubatch_size: entry.n_ubatch ?? 0,
      cache_type_k: String(entry.type_k ?? "f16"),
      cache_type_v: String(entry.type_v ?? "f16"),
      flash_attn: String(entry.flash_attn ?? "off"),
      // Always "off" here -- this whole path only ever runs for mtp:"off"
      // sweep items (an "on" item takes worker/src/serverBench.ts's
      // llama-server path instead, see worker/src/index.ts's per-item loop),
      // and llama-bench's own JSON output has no such field to read anyway.
      mtp: "off",
      avg_tps: entry.avg_ts ?? 0,
      stddev_tps: entry.stddev_ts ?? 0,
      // Memory/offload fields are filled in by the caller
      // (worker/src/index.ts's finalizeSweepItemResult) from its own
      // MemorySampler/baseline reading and parseOffloadLayers -- llama-bench's
      // JSON output doesn't report any of this itself.
      ram_peak_mib: 0,
      vram_peak_mib: 0,
      ram_avg_mib: 0,
      vram_avg_mib: null,
      ram_free_before_mib: null,
      vram_free_before_mib: null,
      system_memory_total_mb: null,
      gpu_memory_total_mb: null,
      gpu_memory_total_accuracy: "unavailable" as const,
      gpu_memory_total_source: null,
      gpu_memory_free_start_accuracy: "unavailable" as const,
      gpu_memory_free_start_source: null,
      gpu_memory_model_avg_accuracy: "unavailable" as const,
      gpu_memory_model_avg_source: null,
      gpu_memory_model_peak_accuracy: "unavailable" as const,
      gpu_memory_model_peak_source: null,
      gpu_layers_loaded: null,
      total_model_layers: null,
    };
  });
  // Identical across every entry in one invocation (one process, one model,
  // one GPU) -- any entry's value works, so the first is used unconditionally.
  return { results, gpu_info: arr[0]?.gpu_info, cpu_info: arr[0]?.cpu_info };
}

function extractJsonArray(text: string): any[] | null {
  const trimmed = text.trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed;
  } catch {
    /* fall through */
  }
  const start = trimmed.indexOf("[");
  const end = trimmed.lastIndexOf("]");
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    const parsed = JSON.parse(trimmed.slice(start, end + 1));
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export { readFileSync };
