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
  // Index into the worker's detected GPU list (see shared/types.ts's
  // RunConfig.main_gpu) -- restricts this run to one GPU on a multi-GPU
  // worker. Applied as `-sm none -mg <index>` in buildArgs below: split-mode
  // "none" is required alongside --main-gpu, since llama.cpp's default split
  // mode ("layer") still spreads the model across every visible GPU
  // regardless of --main-gpu, which only picks the KV-cache/small-tensor
  // device in that mode. Undefined means "don't pass either flag," same as
  // every run before this field existed.
  mainGpu?: number;
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
  if (input.mainGpu != null) {
    args.push("-sm", "none", "-mg", String(input.mainGpu));
  }
  // Keeps this many of the model's MoE (Mixture-of-Experts) layers' expert
  // weights on CPU RAM instead of GPU VRAM -- see shared/sweep.ts's
  // SweepItem.n_cpu_moe. 0 omits the flag entirely, same as every run before
  // this axis existed.
  if (item.n_cpu_moe > 0) {
    args.push("--n-cpu-moe", String(item.n_cpu_moe));
  }
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
  // Read from llama.cpp's own runtime output (see parseOffloadLayers below)
  // -- undefined when the line was never seen at all (item failed before
  // model load finished, or a build too old to support the required
  // verbosity flag). Populated here (rather than left to the caller) so both
  // the llama-bench path (this file) and the llama-server/MTP path
  // (serverBench.ts) share one parser and one BenchResult shape.
  offload?: OffloadResult;
  // Read from llama.cpp's own runtime output (see parseModelBufferSizes
  // below) -- undefined for the exact same reasons offload above can be:
  // never populated on a build too old for the required verbosity flag, or
  // an item that failed before tensor loading finished. Ground truth for
  // worker/src/index.ts's VRAM-discrepancy check and its always-on
  // claimed-vs-actual offload figures, preferred over the
  // external-VRAM-sample-based estimate whenever present. Split per model
  // (main vs MTP draft companion) so each model's own residency is
  // attributable independently.
  modelBufferSizes?: ModelBufferSizesByModel;
}

// llama.cpp's own one-line model-load summary, printed once *per model
// loaded* by either binary once that model's tensor loading finishes --
// confirmed live against the real binaries across -ngl 0 ("0/43"), a partial
// value ("10/43", matching the per-layer "layer N assigned to device" lines
// it also prints at this same verbosity), and full offload ("43/43"). X =
// gpu_layers_loaded, Y = total_model_layers -- Y is the GGUF's real
// transformer layer count + 1 (the output layer), not the same number as
// models.metadata.n_layer, so this is the only correct source for either
// value. Requires -v (llama-bench, see supportsVerboseFlag above) or -lv 4
// (llama-server, see serverBench.ts's buildArgs) -- absent at default
// verbosity on both binaries, confirmed live. `g` flag so parseOffloadLayers
// below can collect every occurrence, not just the first -- an MTP item
// loads TWO models (base + --model-draft companion) via llama-server, each
// printing its own line.
const OFFLOAD_LAYERS_RE = /load_tensors: offloaded (\d+)\/(\d+) layers to GPU/g;
// Non-global sibling of the above, for matching a single already-isolated
// line (e.g. worker/src/index.ts's live onStderrLine scan) without having to
// worry about a shared regex's lastIndex state across repeated .exec calls.
const OFFLOAD_LAYERS_LINE_RE = /load_tensors: offloaded (\d+)\/(\d+) layers to GPU/;

export interface OffloadInfo {
  gpu_layers_loaded: number;
  total_model_layers: number;
}

export interface OffloadResult {
  // The base/target model's offload -- always the model at `modelPath`, and
  // (for a non-MTP item) the only model loaded at all.
  main: OffloadInfo | null;
  // The MTP/--model-draft companion's own offload, only meaningful when
  // hasMtpDraft is true. Null when hasMtpDraft is false, or when only one
  // "load_tensors: offloaded" line was actually captured (e.g. a build that
  // doesn't log the draft model's own load at this verbosity) -- there's
  // nothing to disambiguate from in that case, so the single match is
  // attributed to main and draft is left unset rather than guessed.
  draft: OffloadInfo | null;
}

// A plain llama-bench item (this file's own path) always loads exactly one
// model, so a single "offloaded X/Y" line is unambiguous. An MTP item run
// through llama-server (serverBench.ts's path, hasMtpDraft true) loads two:
// the base model and its --model-draft companion, each printing its own
// line. Simply taking the first match here (a previous behavior) silently
// reported whichever model's line happened to print first as if it were the
// base model's -- confirmed live against real MTP run output that this is
// actually the *draft* model's line, not the base model's, so
// gpu_layers_loaded/total_model_layers on every MTP row previously showed
// the tiny draft head's own e.g. "5/5" instead of the real base model's e.g.
// "43/43". Disambiguated here by total layer count instead of match
// position, which is robust regardless of which model's line prints first:
// an MTP draft/companion model is, by definition and by how much smaller
// speculative-decoding heads are than the base models they're paired with,
// always the one with far fewer transformer layers -- so of at most two
// matches, the larger total_model_layers is always the base model's line and
// the smaller is always the draft's.
export function parseOffloadLayers(stderr: string, hasMtpDraft: boolean): OffloadResult {
  const matches: OffloadInfo[] = [...stderr.matchAll(OFFLOAD_LAYERS_RE)].map((m) => ({
    gpu_layers_loaded: Number(m[1]),
    total_model_layers: Number(m[2]),
  }));
  if (matches.length === 0) return { main: null, draft: null };
  if (!hasMtpDraft || matches.length < 2) {
    const main = matches.reduce((a, b) => (b.total_model_layers > a.total_model_layers ? b : a));
    return { main, draft: null };
  }
  const [main, draft] = [...matches].sort((a, b) => b.total_model_layers - a.total_model_layers);
  return { main, draft };
}

// Live sibling of parseOffloadLayers, for a caller streaming stderr one line
// at a time (worker/src/index.ts's onStderrLine) that wants to surface
// offload as soon as it's known -- right when model loading finishes, well
// before the first benchmark repeat -- rather than only after the whole
// process has exited. Only ever finds the *first* model's own line this way
// (a non-MTP llama-bench item has only one anyway); the authoritative,
// disambiguated-against-a-draft-model result still comes from
// parseOffloadLayers over the full captured stderr once the process closes.
export function matchOffloadLine(line: string): OffloadInfo | null {
  const m = OFFLOAD_LAYERS_LINE_RE.exec(line);
  if (!m) return null;
  return { gpu_layers_loaded: Number(m[1]), total_model_layers: Number(m[2]) };
}

// llama.cpp's own per-backend-buffer allocation summary, printed once per
// buffer type actually *created* while loading a model's tensors, e.g.:
//   load_tensors:   CPU_Mapped model buffer size =   137.42 MiB
//   load_tensors:        CUDA0 model buffer size =  4368.51 MiB
// Unlike OFFLOAD_LAYERS_RE above -- which reflects the ggml scheduler's
// *plan* for where each layer should go, printed before any buffer is
// actually allocated -- this line is emitted after allocation, so a weight
// buffer the scheduler assigned to the GPU but that a backend-level failure
// bounced back to the CPU shows up here as CPU_Mapped, not CUDA0. This is
// the ground-truth signal worker/src/index.ts's finalizeSweepItemResult
// prefers for its VRAM-discrepancy check over the external VRAM-sample-based
// estimate, when available. VERIFIED against real captured transcripts from
// this repo's own b10605 build (llama-bench and llama-server with a real
// base+draft MTP pair): llama-bench prints these un-prefixed while
// llama-server prefixes every line with its own logger header
// ("00.02.638.073 I load_tensors: ..."), so the regex tolerates an optional
// prefix rather than anchoring at column 0. Applied per line (no `g`/`m`
// statefulness to leak lastIndex across repeated .exec calls).
const MODEL_BUFFER_SIZE_LINE_RE = /^.*?load_tensors:\s*(\S+) model buffer size\s*=\s*([\d.]+)\s*MiB/;
const CPU_BUFFER_NAME_RE = /^CPU(_Mapped)?$/;

export interface ModelBufferSizes {
  // Sum of every non-CPU buffer (CUDA0, CUDA1, ROCm0, Vulkan0, Metal, ...).
  gpuMib: number;
  // Sum of every CPU/CPU_Mapped buffer -- includes weights the scheduler
  // itself assigned to the CPU (e.g. the untouched remainder of a partial
  // offload), not just a sysmem-fallback bounce off a GPU assignment.
  cpuMib: number;
}

// Per-model split of ModelBufferSizes, one entry per model actually loaded.
// An MTP item (llama-server + --model-draft) loads two models, each printing
// its own "offloaded X/Y" line followed by its own buffer-size lines (order
// verified live against the real binary); a plain llama-bench item loads one.
// Each buffer-size line is attributed to whichever model's "offloaded X/Y"
// line most recently preceded it, then the resulting groups are assigned
// main-vs-draft by total layer count -- the larger transformer stack is
// always the base model, the same rule parseOffloadLayers uses, so this is
// robust regardless of load order. Null members mean "no buffer lines
// captured for that model" (draft is null on every single-model run).
export interface ModelBufferSizesByModel {
  main: ModelBufferSizes | null;
  draft: ModelBufferSizes | null;
}

// Parses the whole stderr, grouping each run of buffer-size lines under the
// "offloaded X/Y" line that precedes it -- so an MTP item's base-model
// buffers and draft-model buffers are reported separately instead of the old
// aggregated-together behavior (which the draft's tiny footprint made only
// mildly wrong for the base model but impossible to report at all for the
// draft itself -- see estimateResidentGpuLayersFromBufferSizes). Null (not
// zero) when no buffer line was ever seen -- an older build without
// -v/-lv 4 support, or an item that failed before tensor loading finished --
// so callers can tell "no GPU buffer, confirmed" apart from "we have no
// idea," exactly like OffloadResult's own null already does for the
// layer-count line.
export function parseModelBufferSizes(stderr: string): ModelBufferSizesByModel | null {
  const groups: ModelBufferSizes[] = [];
  const groupTotalLayers: number[] = [];
  let current = -1;
  for (const line of stderr.split("\n")) {
    const offloadMatch = OFFLOAD_LAYERS_LINE_RE.exec(line);
    if (offloadMatch) {
      groups.push({ gpuMib: 0, cpuMib: 0 });
      groupTotalLayers.push(Number(offloadMatch[2]));
      current = groups.length - 1;
      continue;
    }
    const bufferMatch = MODEL_BUFFER_SIZE_LINE_RE.exec(line);
    if (!bufferMatch) continue;
    if (current === -1) {
      groups.push({ gpuMib: 0, cpuMib: 0 });
      groupTotalLayers.push(-1);
      current = 0;
    }
    const mib = Number(bufferMatch[2]);
    if (!Number.isFinite(mib)) continue;
    if (CPU_BUFFER_NAME_RE.test(bufferMatch[1])) groups[current].cpuMib += mib;
    else groups[current].gpuMib += mib;
  }
  const withBuffers = groups
    .map((g, i) => ({ gpuMib: g.gpuMib, cpuMib: g.cpuMib, totalLayers: groupTotalLayers[i] }))
    .filter((g) => g.gpuMib > 0 || g.cpuMib > 0);
  if (withBuffers.length === 0) return null;
  const sorted = [...withBuffers].sort((a, b) => b.totalLayers - a.totalLayers);
  return {
    main: { gpuMib: sorted[0].gpuMib, cpuMib: sorted[0].cpuMib },
    draft: sorted.length > 1 ? { gpuMib: sorted[1].gpuMib, cpuMib: sorted[1].cpuMib } : null,
  };
}

// llama-bench's -v/--verbose (see supportsVerboseFlag above) and
// llama-server's --verbosity 4 (see worker/src/serverBench.ts's buildArgs)
// both print one line per tensor while a model loads, and (for a
// CPU_REPACK-eligible quant) a second one per tensor while it repacks --
// confirmed live against a 541-tensor/402-repackable-tensor model that this
// alone is >900 near-identical lines. Harmless noise on a successful run
// (nothing here reads it), but on a *failed* one this raw stderr becomes
// the item's stored `error` verbatim (see worker/src/index.ts's
// runSweepItem), so it previously buried the one line that actually
// explained the failure underneath a tensor-by-tensor transcript. Collapses
// each contiguous run of 2+ matching lines into a single count-bearing
// summary line; a lone match, anything else, and the useful diagnostic
// lines around them (offload counts, the actual llama.cpp error) all pass
// through untouched.
const CREATE_TENSOR_LINE_RE = /^create_tensor: loading tensor /;
// A llama.cpp mmap-loading progress "." has no trailing newline, so it ends
// up glued to the front of whatever the *next* fprintf writes -- often this
// exact line -- rather than starting a line of its own; stripped as a
// leading `\.*` here rather than treated as breaking the batch.
const REPACK_TENSOR_LINE_RE = /^\.*repack: repack tensor \S+ with (\S+)\s*$/;

export function collapseTensorLoadSpam(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    if (CREATE_TENSOR_LINE_RE.test(lines[i])) {
      let j = i + 1;
      while (j < lines.length && CREATE_TENSOR_LINE_RE.test(lines[j])) j++;
      const count = j - i;
      if (count > 1) {
        out.push(`create_tensor: loading tensor -- ${count} tensors (batched)`);
        i = j;
        continue;
      }
    }
    const repackMatch = lines[i].match(REPACK_TENSOR_LINE_RE);
    if (repackMatch) {
      const quantType = repackMatch[1];
      let j = i + 1;
      while (true) {
        const m = j < lines.length ? lines[j].match(REPACK_TENSOR_LINE_RE) : null;
        if (!m || m[1] !== quantType) break;
        j++;
      }
      const count = j - i;
      if (count > 1) {
        out.push(`repack: repack tensor -- ${count} tensors with ${quantType} (batched)`);
        i = j;
        continue;
      }
    }
    out.push(lines[i]);
    i++;
  }
  return out.join("\n");
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
        const parsed = parseLlamaBench(stdout, input.item.n_cpu_moe);
        results = parsed.results;
        gpu_info = parsed.gpu_info;
        cpu_info = parsed.cpu_info;
        log?.debug(`parsed ${results.length} result row(s) from llama-bench JSON output`);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log?.warn(`failed to parse llama-bench output: ${message}`);
        stderr += `\n${message} (raw stdout: ${stdout.slice(0, 500)})`;
      }
      // false: llama-bench has no --model-draft/MTP support at all, so this
      // path only ever loads one model.
      const offload = parseOffloadLayers(stderr, false);
      // Same raw, pre-collapse stderr offload parsing needs -- see
      // parseModelBufferSizes's own doc comment.
      const modelBufferSizes = parseModelBufferSizes(stderr) ?? undefined;
      // After offload/buffer-size parsing (which needs the real line-by-line
      // stderr, unaffected by this anyway) but before returning -- this is
      // the stderr a failed item's `error` field is built from verbatim.
      stderr = collapseTensorLoadSpam(stderr);
      resolve({ stdout, stderr, code, signal, timedOut, results, gpu_info, cpu_info, offload, modelBufferSizes });
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

export function parseLlamaBench(stdout: string, nCpuMoe: number): ParsedLlamaBench {
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
      // llama-bench has no draft-model concept at all -- always 0 here, same
      // reasoning as mtp above.
      n_gpu_layers_draft: 0,
      // Unlike mtp/n_gpu_layers_draft above, this genuinely applies on the
      // llama-bench path too (see buildArgs) -- passed in from the real
      // sweep item by the caller (runBench), not hardcoded.
      n_cpu_moe: nCpuMoe,
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
