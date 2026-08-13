import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, writeFileSync, renameSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { IngestResultInput } from "../../shared/types.js";
import type { SweepItem } from "../../shared/sweep.js";
import type { BenchLogger, BenchResult } from "./bench.js";

// Drives llama-server over HTTP to benchmark MTP (multi-token-prediction)
// speculative decoding -- llama-bench itself has no --spec-type/--model-draft
// support at all (confirmed against its actual upstream source), so an
// "mtp: on" sweep item can't go through worker/src/bench.ts's normal path.
// Returns the same BenchResult shape runBench does so the caller
// (worker/src/index.ts) can reuse its existing OOM classification and
// results-reporting logic unchanged regardless of which path ran.

export interface ServerBenchRunInput {
  modelPath: string;
  // --model-draft target, only for a Gemma-4-style base model whose MTP head
  // ships as a separate companion file -- omitted for Qwen-style models
  // where the head is already baked into modelPath itself.
  mtpModelPath?: string;
  item: SweepItem;
  repeats: number;
  llamaServerPath: string;
  port: number;
  timeoutMs?: number;
  onSpawn?: (proc: ChildProcess) => void;
  // Called twice per repeat (once right before its /completion request goes
  // out, once right after its result is known) -- this module drives its own
  // repeat loop directly, so (unlike bench.ts's llama-bench path) there's no
  // subprocess stderr to regex-scrape for progress markers. liveTps is the
  // most recently measured reading for whichever phase is being reported,
  // carried forward from the prior repeat when this repeat hasn't produced
  // a fresh one yet (see runServerBench's lastPpLive/lastTgLive).
  onProgress?: (phase: "loading" | "processing" | "generating", detail: string, liveTps?: number) => void;
  log?: BenchLogger;
  // Overrides where the per-model "working" prompt offset (see
  // loadOffsetStore below) is persisted -- only ever set by tests; real
  // callers rely on the default (a file next to this worker's install).
  offsetStorePath?: string;
}

const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000;
const READY_POLL_INTERVAL_MS = 500;
const READY_TIMEOUT_MS = 120_000;
const STOP_GRACE_MS = 5000;
// Padding above n_prompt+n_gen so llama-server's own context accounting
// (special tokens, its internal bookkeeping) never rejects a request that's
// exactly at the context-size edge.
const CONTEXT_MARGIN = 256;
// Cycled through (not repeated verbatim) to build a synthetic prompt of
// exactly n_prompt tokens -- mirrors llama-bench's own "-p is a token count,
// not real text, content is irrelevant to what's being measured" semantics
// (see llm-benchmark-plan-v2.md), sent as a pre-tokenized array rather than
// a text string so the token count is exact rather than tokenizer-dependent.
// Deliberately NOT a single token repeated hundreds of times in a row --
// confirmed live against a real MTP build that an all-identical-token
// prompt reliably drove speculative decoding into producing invalid UTF-8
// output bytes (llama-server's own response parser then hard-fails with a
// "Content-only format" 500 before any timings come back at all). Cycling
// through a range keeps the same "count is all that matters" property
// without being a maximally repetitive, out-of-distribution edge case.
const FILLER_TOKEN_RANGE_START = 100;
const FILLER_TOKEN_RANGE_SIZE = 500;

// A single /completion call, retried this many times total before the repeat
// is allowed to fail the whole item -- see isRetryableServerParseFailure.
const MAX_COMPLETION_ATTEMPTS = 3;
// Matches llama-server's PEG chat-output-parser failure ("The model produced
// output that does not match the expected Content-only format" and sibling
// messages like "...peg-gemma4 format"): a confirmed, still-open upstream bug
// (ggml-org/llama.cpp#25072), root-caused by reading llama.cpp's actual
// common/peg-parser.cpp: the "any codepoint" primitive underlying rest()/
// content() (what even the trivial always-succeed Content-only grammar is
// built from -- p.content(p.rest()) + p.end()) returns a hard, unconditional
// FAIL the instant it hits one byte sequence classified INVALID UTF-8 --
// unlike the INCOMPLETE case, this path does not check the parser's lenient
// flag at all. So there is no request field or server flag that makes this
// tolerant: ONE invalid byte anywhere in a generation fails the entire
// response, even under the simplest possible chat format. MTP/speculative
// decoding occasionally emits such a byte (observed live: a short run of
// invalid-UTF-8 characters, followed by otherwise-clean, coherent generated
// text -- consistent with the accept/reject boundary landing mid-way through
// what should have been an atomic multi-byte token sequence). Bounded retry
// is therefore a black-box mitigation, not a fix -- and per
// isRetryableServerParseFailure's caller below, only useful at all because
// each retry perturbs the prompt (a byte-for-byte identical retry request
// under the temperature:0 greedy decoding below is provably pointless --
// confirmed live: 3 consecutive identical attempts against the same prompt
// produced the exact same failure every time, since nothing left in the
// pipeline is random). Deliberately narrow (matches this exact message
// shape) so a genuinely different server error still fails fast.
const RETRYABLE_SERVER_PARSE_FAILURE = /does not match the expected .*format/i;
// Added to FILLER_TOKEN_RANGE_START on each retry attempt (see runCompletion
// below) so a retry sends a genuinely different token sequence rather than
// replaying the exact request that just failed -- see the comment above.
const RETRY_PROMPT_SHIFT = 137;

function isRetryableServerParseFailure(err: unknown): boolean {
  return err instanceof Error && RETRYABLE_SERVER_PARSE_FAILURE.test(err.message);
}

// Per-(model, draft-model) "working" prompt offset, persisted across items,
// runs, and worker restarts -- observed live that offset 0 deterministically
// hits the upstream parser bug for some models under greedy sampling, so
// every repeat/item/run was re-paying the full retry ladder from scratch
// even though the *same* later offset (e.g. 274) had already been proven to
// work moments earlier. Once any offset succeeds for a given model, it's
// tried first from then on; the 0/137/274/... ladder above is only used as a
// fallback if that remembered offset itself stops working (e.g. a different
// prompt length or a different model reusing this worker).
const DEFAULT_OFFSET_STORE_PATH = join(dirname(fileURLToPath(import.meta.url)), "..", "mtp-offsets.json");

let offsetStoreCache: Record<string, number> | null = null;
let offsetStoreCachePath: string | null = null;

function loadOffsetStore(path: string): Record<string, number> {
  if (offsetStoreCache && offsetStoreCachePath === path) return offsetStoreCache;
  try {
    offsetStoreCache = JSON.parse(readFileSync(path, "utf8")) as Record<string, number>;
  } catch {
    offsetStoreCache = {};
  }
  offsetStoreCachePath = path;
  return offsetStoreCache;
}

function offsetStoreKey(modelPath: string, mtpModelPath?: string): string {
  return mtpModelPath ? `${modelPath}::${mtpModelPath}` : modelPath;
}

function getWorkingOffset(path: string, key: string): number {
  return loadOffsetStore(path)[key] ?? 0;
}

function saveWorkingOffset(path: string, key: string, offset: number, log?: BenchLogger): void {
  const store = loadOffsetStore(path);
  if (store[key] === offset) return;
  store[key] = offset;
  try {
    mkdirSync(dirname(path), { recursive: true });
    const tmp = `${path}.tmp`;
    writeFileSync(tmp, JSON.stringify(store, null, 2), "utf8");
    renameSync(tmp, path);
  } catch (err) {
    log?.warn(
      `failed to persist working MTP prompt offset for ${key}: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// Preferred first (the last offset known to work for this model, or 0 if
// none is known yet -- reproducing this file's original always-start-at-0
// behavior exactly when nothing has been learned), then the standard
// RETRY_PROMPT_SHIFT ladder as a fallback, skipping any value already listed,
// capped at maxAttempts total.
function buildOffsetCandidates(preferred: number, maxAttempts: number): number[] {
  const candidates = [preferred];
  for (let i = 0; candidates.length < maxAttempts; i++) {
    const next = i * RETRY_PROMPT_SHIFT;
    if (!candidates.includes(next)) candidates.push(next);
  }
  return candidates;
}

function buildArgs(input: ServerBenchRunInput): string[] {
  const item = input.item;
  const args = [
    "-m",
    input.modelPath,
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
    "-c",
    String(item.n_prompt + item.n_gen + CONTEXT_MARGIN),
    "--port",
    String(input.port),
    "--host",
    "127.0.0.1",
    "--spec-type",
    "draft-mtp",
    // Exposes /metrics, the only way to confirm the draft model is actually
    // contributing accepted tokens rather than --spec-type draft-mtp above
    // silently no-opping (bad/incompatible --model-draft file, draft model
    // failing to load while the base model still starts fine, etc) -- see
    // fetchSpecDecodeMetrics below.
    "--metrics",
    // Unconditional, not probed like bench.ts's supportsVerboseFlag: default
    // verbosity (3, "info") prints none of the "load_tensors: offloaded
    // X/Y layers to GPU" detail worker/src/index.ts's parseOffloadLayers
    // needs. --help documents the scale as 0=generic, 1=error, 2=warning,
    // 3=info (default), 4=trace, 5=debug. Confirmed live against the
    // installed b10405 build (both at model load and mid-generation via a
    // real /completion request) that 4 already surfaces the offload line
    // plus every other diagnostic worth keeping (slot lifecycle, context
    // checkpoints, print_timing) while producing zero of the level-5 "debug"
    // per-token/per-draft-candidate trace lines that flooded the mirrored
    // worker log at the previously-used 999 -- see logDiagnosticOutput in
    // worker/src/index.ts.
    "--verbosity",
    "4",
  ];
  if (input.mtpModelPath) {
    args.push("--model-draft", input.mtpModelPath);
    // Confirmed live against the installed build's own --help: llama-server
    // exposes -ngld/--n-gpu-layers-draft as an independent offload count for
    // the draft model (default "auto" when omitted, which is what every run
    // before this field existed effectively got). Only meaningful alongside
    // an actual --model-draft -- pushing it with no draft model loaded has
    // nothing to apply to.
    args.push("-ngld", String(item.n_gpu_layers_draft));
  }
  return args;
}

// `exited` is the same `closed` promise runServerBench already builds from
// proc's own "close" event -- raced against each poll wait so a server that
// dies almost immediately (e.g. a context-creation error at startup) is
// reported within a poll interval instead of only after the full
// READY_TIMEOUT_MS. Previously this looped on a plain timer with no way to
// notice the process was already gone, so a near-instant crash still burned
// the whole 120s before failing with a misleading "did not become ready"
// message -- confirmed live: a quantized-V-cache-without-flash-attn context
// error killed llama-server in ~3s, but the item still took 2 minutes to
// report failure.
async function waitForReady(
  port: number,
  deadlineAt: number,
  exited: Promise<{ code: number | null; signal: NodeJS.Signals | null }>
): Promise<void> {
  const STILL_WAITING = Symbol("still-waiting");
  while (Date.now() < deadlineAt) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return;
    } catch {
      /* not up yet -- keep polling, unless the process itself already died */
    }
    const raced = await Promise.race([
      exited,
      new Promise<typeof STILL_WAITING>((r) => setTimeout(() => r(STILL_WAITING), READY_POLL_INTERVAL_MS)),
    ]);
    if (raced !== STILL_WAITING) {
      throw new Error(
        `llama-server exited before becoming ready (code ${raced.code}, signal ${raced.signal ?? "none"})`
      );
    }
  }
  throw new Error(`llama-server did not become ready on port ${port} within ${READY_TIMEOUT_MS}ms`);
}

interface CompletionTimings {
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
}

// llama-server's own pre-divided *_per_second timings have been observed
// live to report wildly implausible values under MTP speculative decoding
// -- e.g. predicted_per_second: 1000000.00 for a 128-token generation on
// hardware that otherwise measures ~14 tok/s without MTP -- consistent with
// an upstream timer bug where the elapsed-ms side of the division collapses
// to a near-zero value for some repeats. Recomputing from the raw
// token-count/elapsed-ms pair (also present in the same timings object) is
// strictly no less reliable than trusting the server's own division, and an
// absolute ceiling -- generous enough that no real single-request
// prefill/decode on the hardware this tool targets could plausibly hit it,
// yet nowhere near the ~1e6 garbage value seen live -- catches whatever's
// left uncaught by that (e.g. a corrupted *_ms field itself) instead of
// silently averaging garbage into the reported result.
const MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND = 200_000;
const MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND = 5_000;

interface RateReading {
  value: number;
  // True when the computed value is outside plausible bounds (<=0 or over
  // the ceiling) -- kept and reported rather than discarded, see the two
  // call sites below and buildPhaseSummary.
  suspect: boolean;
}

// Previously named sanitizeRate and returned undefined for an implausible
// value, silently discarding it -- a repeat whose *only* readings were all
// implausible (observed live: every tg repeat hitting the timer bug
// described above) then produced literally zero data for that test_type,
// with the item still reporting "done" (see finalizeSweepItemResult in
// worker/src/index.ts, which only checks results.length > 0). Now always
// returns the computed value when one exists at all, flagged suspect
// instead of erased, so the caller can still show a real number -- labeled
// as questionable -- instead of a silent gap. Returns undefined only when
// no value is computable at all (missing/non-finite fields).
function evaluateRate(
  n: number | undefined,
  ms: number | undefined,
  perSecondFallback: number | undefined,
  ceiling: number
): RateReading | undefined {
  const value = typeof n === "number" && typeof ms === "number" && ms > 0 ? (n / ms) * 1000 : perSecondFallback;
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return { value, suspect: value <= 0 || value > ceiling };
}

function syntheticPrompt(n: number, offset = 0): number[] {
  return Array.from(
    { length: Math.max(0, n) },
    (_, i) => FILLER_TOKEN_RANGE_START + ((i + offset) % FILLER_TOKEN_RANGE_SIZE)
  );
}

async function runCompletion(
  port: number,
  nPrompt: number,
  nGen: number,
  timeoutMs: number,
  promptOffset = 0
): Promise<CompletionTimings> {
  const res = await fetch(`http://127.0.0.1:${port}/completion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      prompt: syntheticPrompt(nPrompt, promptOffset),
      n_predict: nGen,
      // Otherwise a cache hit on a repeated identical synthetic prompt would
      // skew pp timings toward zero on every repeat after the first.
      cache_prompt: false,
      stream: false,
      // Greedy decoding (llama-server treats temp<=0 as "ignore top_k/top_p,
      // always take the argmax token"), not left at the server's stochastic
      // defaults (temp 0.8 / top_p 0.95) -- makes a benchmark repeat
      // reproducible instead of sampling a different token path every time.
      // NOT confirmed to reduce the invalid-UTF-8-byte 500 described by
      // RETRYABLE_SERVER_PARSE_FAILURE above -- live evidence points the
      // other way (the exact same failure reproduced 3/3 times once
      // sampling was made deterministic), so retries compensate by varying
      // the prompt instead of relying on this. Trade-off worth knowing:
      // greedy can degenerate into repeating one token on our already-out-
      // of-distribution filler prompt, which could measure MTP's draft-
      // acceptance rate as higher than it'd be on realistic text -- an
      // extension of the same "content is irrelevant, only token count
      // matters" approximation this whole synthetic-prompt design already
      // makes, not a new one.
      temperature: 0,
    }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`llama-server /completion failed: ${res.status} ${text.slice(0, 500)}`);
  }
  const data = (await res.json()) as { timings?: CompletionTimings };
  if (!data.timings) throw new Error("llama-server /completion response had no timings");
  return data.timings;
}

interface SpecDecodeMetrics {
  drafted: number;
  accepted: number;
}

// Prometheus counters from llama-server's /metrics (see --metrics in
// buildArgs). Per llama-server's own doc comment these are absent entirely
// (not zeroed) until the first *completed* speculative request, so a missing
// match is a distinct, reportable condition from a present-but-zero one.
const SPEC_DRAFTED_METRIC = /^llamacpp:spec_decode_num_draft_tokens_total\s+(\S+)/m;
const SPEC_ACCEPTED_METRIC = /^llamacpp:spec_decode_num_accepted_tokens_total\s+(\S+)/m;

async function fetchSpecDecodeMetrics(port: number): Promise<SpecDecodeMetrics | undefined> {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/metrics`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return undefined;
    const text = await res.text();
    const drafted = Number(SPEC_DRAFTED_METRIC.exec(text)?.[1]);
    const accepted = Number(SPEC_ACCEPTED_METRIC.exec(text)?.[1]);
    if (!Number.isFinite(drafted) || !Number.isFinite(accepted)) return undefined;
    return { drafted, accepted };
  } catch {
    // /metrics is diagnostic only -- never let it fail an otherwise-good run.
    return undefined;
  }
}

interface SpecDecodeCheck {
  // Absent when /metrics itself was unreachable or reported no counters at
  // all -- distinct from a present metrics object with drafted: 0.
  metrics?: SpecDecodeMetrics;
  // Set for every non-confirmed outcome (missing counters, 0 drafted, 0
  // accepted) so the caller can fold it into the item's stored warning --
  // the same field RunDetail.tsx already renders in red for a "done" item
  // that still has something worth flagging (see the ppSummary/tgSummary
  // .warning pattern below). Left undefined only when accepted > 0.
  warning?: string;
}

// Checks whether the draft model actually contributed accepted tokens this
// item, distinct from the pp/tg numbers above which stay plausible-looking
// even if speculative decoding silently never engaged (e.g. a --model-draft
// file llama-server accepted at startup but rejected/ignored per-request).
// Still logs unconditionally (not just the bad cases) so the worker's own
// log has a permanent record, but the *return value* is what actually
// reaches the stored run item / CSV export -- see runServerBench below.
async function checkSpecDecodeMetrics(port: number, label: string, repeats: number, log?: BenchLogger): Promise<SpecDecodeCheck> {
  const metrics = await fetchSpecDecodeMetrics(port);
  if (!metrics) {
    const warning =
      `MTP: /metrics reported no speculative-decoding counters -- cannot confirm the draft model actually ran ` +
      `(spec decode may never have engaged despite --spec-type draft-mtp)`;
    log?.warn(`${label}: ${warning}`);
    return { warning };
  }
  if (metrics.drafted === 0) {
    const warning = `MTP: draft model drafted 0 tokens across ${repeats} repeat(s) -- speculative decoding did not run`;
    log?.warn(`${label}: ${warning}`);
    return { metrics, warning };
  }
  const rate = (metrics.accepted / metrics.drafted) * 100;
  if (metrics.accepted === 0) {
    const warning =
      `MTP: draft model drafted ${metrics.drafted} tokens but 0 were accepted -- draft model is not contributing ` +
      `(likely a base/draft model mismatch)`;
    log?.warn(`${label}: ${warning}`);
    return { metrics, warning };
  }
  log?.info(`${label}: MTP draft model confirmed working -- ${metrics.accepted}/${metrics.drafted} drafted tokens accepted (${rate.toFixed(1)}%)`);
  return { metrics };
}

function average(values: number[]): { avg: number; stddev: number } {
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
  return { avg, stddev: Math.sqrt(variance) };
}

interface PhaseSummary {
  avg: number;
  stddev: number;
  sampleCount: number;
  suspectCount: number;
  suspectValues: number[];
  warning?: string;
}

// Prefers averaging only the clean (non-suspect) readings, same protective
// behavior the old sanitizeRate-based filtering had -- a single garbage
// reading (e.g. the observed 1,000,000 tok/s MTP timer glitch) can't skew
// the reported avg_tps. But unlike the old behavior, this never just
// vanishes a test_type: if every reading this item got was suspect, it
// falls back to averaging those anyway so a real (if flagged) number is
// still reported instead of the row being silently omitted. Returns
// undefined only when there is truly nothing at all to average.
function buildPhaseSummary(phase: "pp" | "tg", clean: number[], suspect: number[], ceiling: number): PhaseSummary | undefined {
  const usable = clean.length > 0 ? clean : suspect;
  if (usable.length === 0) return undefined;
  const { avg, stddev } = average(usable);
  const sampleCount = clean.length + suspect.length;
  let warning: string | undefined;
  if (suspect.length > 0) {
    const rawValues = suspect.map((v) => v.toFixed(1)).join(", ");
    warning =
      clean.length > 0
        ? `${phase}: ${suspect.length} of ${sampleCount} repeat(s) reported an implausible speed (raw: ${rawValues} tok/s, ` +
          `ceiling ${ceiling}) -- excluded from the ${avg.toFixed(2)} tok/s average above, likely the known ` +
          `llama-server MTP timing bug (ggml-org/llama.cpp#25072)`
        : `${phase}: all ${sampleCount} repeat(s) reported an implausible speed (raw: ${rawValues} tok/s, ceiling ${ceiling}) ` +
          `-- no clean reading was available, so the ${avg.toFixed(2)} tok/s average above is derived from these flagged ` +
          `values and should not be trusted; likely the known llama-server MTP timing bug (ggml-org/llama.cpp#25072)`;
  }
  return { avg, stddev, sampleCount, suspectCount: suspect.length, suspectValues: suspect, warning };
}

// Sends SIGTERM and gives llama-server a grace period to shut down cleanly
// before escalating to SIGKILL -- unlike llama-bench (which just runs to
// completion and exits on its own), llama-server is a persistent process
// this module has to explicitly stop once it has what it needs.
async function stopServer(
  proc: ChildProcess,
  closed: Promise<{ code: number | null; signal: NodeJS.Signals | null }>
): Promise<{ code: number | null; signal: NodeJS.Signals | null }> {
  if (proc.exitCode !== null || proc.signalCode !== null) {
    return { code: proc.exitCode, signal: proc.signalCode };
  }
  proc.kill("SIGTERM");
  const timedOut = Symbol("stop-timeout");
  const result = await Promise.race([closed, new Promise<typeof timedOut>((r) => setTimeout(() => r(timedOut), STOP_GRACE_MS))]);
  if (result === timedOut) {
    proc.kill("SIGKILL");
    return closed;
  }
  return result;
}

export async function runServerBench(input: ServerBenchRunInput): Promise<BenchResult> {
  const item = input.item;
  const timeoutMs = input.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const args = buildArgs(input);
  const log = input.log;
  const label = `mtp item ${item.idx}`;

  log?.info(
    `spawning llama-server for MTP bench: ${input.llamaServerPath} (port=${input.port}` +
      (input.mtpModelPath ? `, draft=${input.mtpModelPath}` : "") +
      ")"
  );
  log?.debug(`llama-server args: ${args.join(" ")}`);

  let stdout = "";
  let stderr = "";
  let timedOut = false;
  const proc: ChildProcess = spawn(input.llamaServerPath, args, { windowsHide: true });
  input.onSpawn?.(proc);
  proc.stdout?.on("data", (d) => (stdout += d.toString()));
  proc.stderr?.on("data", (d) => (stderr += d.toString()));
  proc.on("error", (err) => {
    log?.warn(`llama-server spawn error: ${err.message}`);
    stderr += `\nspawn error: ${err.message}`;
  });

  const closed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolvePromise) => {
    proc.on("close", (code, signal) => resolvePromise({ code, signal }));
  });

  const timer = setTimeout(() => {
    timedOut = true;
    log?.warn(`llama-server (pid ${proc.pid}) timed out after ${timeoutMs}ms, sending SIGKILL`);
    stderr += `\ntimed out after ${timeoutMs}ms, killing process`;
    proc.kill("SIGKILL");
  }, timeoutMs);

  try {
    input.onProgress?.("loading", "starting llama-server");
    await waitForReady(input.port, Date.now() + READY_TIMEOUT_MS, closed);

    const offsetStorePath = input.offsetStorePath ?? DEFAULT_OFFSET_STORE_PATH;
    const offsetKey = offsetStoreKey(input.modelPath, input.mtpModelPath);
    // Updated in place whenever a non-preferred offset succeeds, so later
    // repeats in *this same* item also start from it, not just future items/
    // runs (see saveWorkingOffset's persistence for those).
    let preferredOffset = getWorkingOffset(offsetStorePath, offsetKey);

    const ppSamples: number[] = [];
    const tgSamples: number[] = [];
    // Readings that came back computable but outside plausible bounds --
    // kept separately (not just dropped) so buildPhaseSummary can still
    // report an actual number, and so the raw values themselves are visible
    // for diagnosing whether this is the known llama-server MTP timer bug.
    const ppSuspectSamples: number[] = [];
    const tgSuspectSamples: number[] = [];
    // Every computable reading (clean + suspect) in original repeat order --
    // unlike ppSamples/ppSuspectSamples above (partitioned by plausibility,
    // order within a partition only), this is what actually answers "what
    // did each individual run measure," so it's kept and exported alongside
    // the avg_tps summary instead of only existing transiently in this loop.
    // A repeat that produced no computable reading at all has no entry here
    // (nothing to record), so this can be shorter than input.repeats.
    const ppRepeatSamples: number[] = [];
    const tgRepeatSamples: number[] = [];
    // Carried across repeats so a phase's live reading stays populated (one
    // repeat lag) instead of going blank the moment the *other* phase's tick
    // fires -- same "last known value" idea as bench.ts's own gap-based
    // liveTps, just seeded from this module's actual per-repeat /completion
    // timings instead of a stderr marker gap.
    let lastPpLive: number | undefined;
    let lastTgLive: number | undefined;
    // Which phase gets the tick fired *before* this repeat's /completion
    // call -- the one that's actually observable, since it's shown for the
    // whole request's wall-clock duration (tens of seconds), unlike the
    // "after" tick below (fired once the result is known, immediately
    // clobbered by the *next* repeat's "before" tick with no real gap in
    // between -- effectively a single network-RTT-wide window a client
    // polling every few seconds will almost never land on). A single
    // /completion call measures pp and tg together, so there's no genuine
    // sub-request signal to report live either way; earlier this always
    // gave the long observable slot to "processing" (whenever n_prompt>0),
    // which is why live PP tok/s reliably showed in the UI but live TG
    // tok/s never did -- confirmed live (production log showed real pp/tg
    // numbers being computed and sent every repeat, but the UI only ever
    // rendered PP). Alternating it every repeat instead means each phase
    // gets a full repeat's worth of observable time in turn, which also
    // reproduces the "1 processing, then 1 generation" per-run cadence
    // this item's phase-labeling fix originally asked for.
    let waitPhase: "processing" | "generating" = item.n_prompt > 0 ? "processing" : "generating";
    const bothPhasesPresent = item.n_prompt > 0 && item.n_gen > 0;
    for (let rep = 1; rep <= input.repeats; rep++) {
      input.onProgress?.(waitPhase, `run ${rep}/${input.repeats}`, waitPhase === "processing" ? lastPpLive : lastTgLive);
      let timings: CompletionTimings | undefined;
      const candidates = buildOffsetCandidates(preferredOffset, MAX_COMPLETION_ATTEMPTS);
      for (let attempt = 1; ; attempt++) {
        const promptOffset = candidates[attempt - 1];
        try {
          timings = await runCompletion(input.port, item.n_prompt, item.n_gen, timeoutMs, promptOffset);
          if (promptOffset !== preferredOffset) {
            preferredOffset = promptOffset;
            saveWorkingOffset(offsetStorePath, offsetKey, preferredOffset, log);
          }
          break;
        } catch (err) {
          if (attempt >= candidates.length || !isRetryableServerParseFailure(err)) throw err;
          const message = err instanceof Error ? err.message : String(err);
          log?.warn(
            `run ${rep}/${input.repeats} attempt ${attempt} (offset ${promptOffset}) hit the known llama-server ` +
              `MTP output-parsing bug (upstream ggml-org/llama.cpp#25072), retrying with offset ` +
              `${candidates[attempt]}: ${message}`
          );
        }
      }
      // Logged unconditionally (not just when something looks wrong) so a
      // run's logs always have enough raw material to judge whether MTP is
      // actually accelerating decode -- not just whether the reported number
      // happens to look plausible.
      log?.debug(
        `${label}: run ${rep}/${input.repeats} raw timings: ` +
          `prompt(n=${timings.prompt_n ?? "?"} ms=${timings.prompt_ms ?? "?"} per_second=${timings.prompt_per_second ?? "?"}) ` +
          `predicted(n=${timings.predicted_n ?? "?"} ms=${timings.predicted_ms ?? "?"} per_second=${timings.predicted_per_second ?? "?"})`
      );
      if (item.n_prompt > 0) {
        const pp = evaluateRate(timings.prompt_n, timings.prompt_ms, timings.prompt_per_second, MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND);
        if (pp == null) {
          log?.warn(
            `${label}: run ${rep}/${input.repeats} produced no computable pp reading at all (raw: n=${timings.prompt_n ?? "?"} ` +
              `ms=${timings.prompt_ms ?? "?"} per_second=${timings.prompt_per_second ?? "?"})`
          );
        } else if (pp.suspect) {
          ppSuspectSamples.push(pp.value);
          ppRepeatSamples.push(pp.value);
          log?.warn(
            `${label}: run ${rep}/${input.repeats} pp reading flagged implausible (${pp.value.toFixed(2)} tok/s, ` +
              `ceiling ${MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND}) -- kept for visibility, excluded from the clean average ` +
              `(raw: n=${timings.prompt_n ?? "?"} ms=${timings.prompt_ms ?? "?"} per_second=${timings.prompt_per_second ?? "?"})`
          );
        } else {
          ppSamples.push(pp.value);
          ppRepeatSamples.push(pp.value);
          lastPpLive = pp.value;
        }
      }
      if (item.n_gen > 0) {
        const tg = evaluateRate(timings.predicted_n, timings.predicted_ms, timings.predicted_per_second, MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND);
        if (tg == null) {
          log?.warn(
            `${label}: run ${rep}/${input.repeats} produced no computable tg reading at all (raw: n=${timings.predicted_n ?? "?"} ` +
              `ms=${timings.predicted_ms ?? "?"} per_second=${timings.predicted_per_second ?? "?"})`
          );
        } else if (tg.suspect) {
          tgSuspectSamples.push(tg.value);
          tgRepeatSamples.push(tg.value);
          log?.warn(
            `${label}: run ${rep}/${input.repeats} tg reading flagged implausible (${tg.value.toFixed(2)} tok/s, ` +
              `ceiling ${MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND}) -- kept for visibility, excluded from the clean average, ` +
              `likely the known llama-server MTP timing bug (raw: n=${timings.predicted_n ?? "?"} ` +
              `ms=${timings.predicted_ms ?? "?"} per_second=${timings.predicted_per_second ?? "?"})`
          );
        } else {
          tgSamples.push(tg.value);
          tgRepeatSamples.push(tg.value);
          lastTgLive = tg.value;
        }
      }
      // Fires right after this repeat's own result is known, so the phase
      // shown/live_tps reported reflect what actually just happened (the
      // generation half, for a combined item) instead of only ever updating
      // at the top of the next repeat. Almost always clobbered within
      // milliseconds by the *next* repeat's "before" tick above -- the one
      // case it actually persists is the very last repeat, where it's the
      // last live reading shown until the item's terminal result arrives.
      const endPhase: "processing" | "generating" = item.n_gen > 0 ? "generating" : "processing";
      input.onProgress?.(endPhase, `run ${rep}/${input.repeats}`, endPhase === "generating" ? lastTgLive : lastPpLive);
      if (bothPhasesPresent) waitPhase = waitPhase === "processing" ? "generating" : "processing";
    }
    if (item.n_prompt > 0 && ppSamples.length === 0 && ppSuspectSamples.length === 0) {
      log?.warn(`${label}: every pp repeat across ${input.repeats} repeat(s) produced no computable reading at all -- pp speed unknown for this item`);
    }
    if (item.n_gen > 0 && tgSamples.length === 0 && tgSuspectSamples.length === 0) {
      log?.warn(`${label}: every tg repeat across ${input.repeats} repeat(s) produced no computable reading at all -- tg speed unknown for this item`);
    }

    // Queried before stopServer below tears the process down -- /metrics is
    // only reachable while llama-server is still up.
    const specDecode = await checkSpecDecodeMetrics(input.port, label, input.repeats, log);

    const commonFields = {
      n_prompt: item.n_prompt,
      n_gen: item.n_gen,
      n_threads: item.threads,
      n_gpu_layers: item.n_gpu_layers,
      batch_size: item.batch_size,
      ubatch_size: item.ubatch_size,
      cache_type_k: item.cache_type_k,
      cache_type_v: item.cache_type_v,
      flash_attn: item.flash_attn,
      mtp: item.mtp,
      n_gpu_layers_draft: item.n_gpu_layers_draft,
      ram_peak_mib: 0,
      vram_peak_mib: 0,
      ram_avg_mib: 0,
      vram_avg_mib: null,
      ram_free_before_mib: null,
      vram_free_before_mib: null,
      // Memory/offload fields are filled in by the caller
      // (worker/src/index.ts's finalizeSweepItemResult), same as the ram/
      // vram placeholders above -- see worker/src/bench.ts's parseLlamaBench
      // for the identical pattern on the non-MTP path.
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
    const results: IngestResultInput[] = [];
    const warnings: string[] = [];
    const ppSummary = buildPhaseSummary("pp", ppSamples, ppSuspectSamples, MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND);
    if (ppSummary) {
      results.push({
        ...commonFields,
        test_type: "pp",
        avg_tps: ppSummary.avg,
        stddev_tps: ppSummary.stddev,
        sample_count: ppSummary.sampleCount,
        suspect_count: ppSummary.suspectCount,
        suspect_samples: ppSummary.suspectValues.length > 0 ? ppSummary.suspectValues : undefined,
        repeat_samples: ppRepeatSamples.length > 0 ? ppRepeatSamples : undefined,
      });
      if (ppSummary.warning) warnings.push(ppSummary.warning);
    }
    const tgSummary = buildPhaseSummary("tg", tgSamples, tgSuspectSamples, MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND);
    if (tgSummary) {
      results.push({
        ...commonFields,
        test_type: "tg",
        avg_tps: tgSummary.avg,
        stddev_tps: tgSummary.stddev,
        sample_count: tgSummary.sampleCount,
        suspect_count: tgSummary.suspectCount,
        suspect_samples: tgSummary.suspectValues.length > 0 ? tgSummary.suspectValues : undefined,
        repeat_samples: tgRepeatSamples.length > 0 ? tgRepeatSamples : undefined,
        // Attached to the tg row specifically -- spec decode only happens
        // during generation, and undefined (vs. 0) here means /metrics never
        // confirmed anything at all rather than confirming zero activity.
        spec_drafted: specDecode.metrics?.drafted,
        spec_accepted: specDecode.metrics?.accepted,
      });
      if (tgSummary.warning) warnings.push(tgSummary.warning);
    }
    if (item.n_prompt > 0 && !ppSummary) {
      warnings.push(`pp: no computable reading in any of ${input.repeats} repeat(s) -- pp speed unknown for this item`);
    }
    if (item.n_gen > 0 && !tgSummary) {
      warnings.push(`tg: no computable reading in any of ${input.repeats} repeat(s) -- tg speed unknown for this item`);
    }
    // Surfaced regardless of whether a tg row exists at all (e.g. a
    // pp-only item still ran with --spec-type draft-mtp) so a silently
    // no-op'd draft model is visible on the item itself (RunDetail.tsx
    // renders bench.warning in red via item.error), not just in this
    // worker's own console/log file.
    if (specDecode.warning) warnings.push(specDecode.warning);
    const warning = warnings.length > 0 ? warnings.join(" | ") : undefined;

    // Cleared *before* the graceful-shutdown wait below, not after -- results
    // are already collected at this point, so the timeout guard has nothing
    // left to protect. Clearing it late left a window where, if the whole
    // operation happened to land right at the timeoutMs boundary, the timer
    // could still fire during this wait and mark an otherwise-successful run
    // timedOut: true after the fact.
    clearTimeout(timer);
    const exitResult = await stopServer(proc, closed);
    return {
      stdout,
      stderr,
      code: results.length > 0 ? 0 : (exitResult.code ?? 1),
      signal: exitResult.signal,
      timedOut,
      results,
      warning,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log?.warn(`llama-server MTP bench failed: ${message}`);
    stderr += `\n${message}`;
    // Same reasoning as above: clear before the wait, not after, so the
    // timer can't fire a second time (redundant, but would otherwise
    // overwrite an already-correct failure reason) during cleanup.
    clearTimeout(timer);
    // Checked *before* our own kill below -- distinguishes "the process
    // already died on its own" (a real signal worth trusting, e.g. genuine
    // OOM) from "it was still alive and we're just tearing it down as
    // routine cleanup after an ordinary application-level failure" (an
    // unhealthy-but-still-responding llama-server returning a 500, a
    // malformed /completion response, a readiness-poll timeout, ...).
    // Unlike bench.ts's llama-bench path -- which never manually kills its
    // process except via the shared timeout above -- this module kills the
    // server on *every* caught error as normal cleanup, so blindly
    // reporting SIGKILL here would make worker/src/index.ts's
    // classifyFailure (signal === SIGKILL => "failed_oom") misfire on every
    // single MTP failure regardless of actual cause. Confirmed live: a
    // llama-server 500 ("model produced output that does not match the
    // expected Content-only format" -- nothing to do with memory) was
    // reported as failed_oom before this fix.
    const diedOnItsOwn = proc.exitCode !== null || proc.signalCode !== null;
    proc.kill("SIGKILL");
    const exitResult = await Promise.race([
      closed,
      new Promise<{ code: null; signal: null }>((r) => setTimeout(() => r({ code: null, signal: null }), STOP_GRACE_MS)),
    ]);
    return {
      stdout,
      stderr,
      code: exitResult.code,
      signal: diedOnItsOwn ? exitResult.signal : null,
      timedOut,
      results: [],
    };
  }
}
