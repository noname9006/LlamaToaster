// Steps 0 and 6 of the optimization flow (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md):
// the baseline and the confirm runs share one harness -- llama-server started
// at the user's target context, one cache-free request per repeat with a
// 4096-token prompt followed by 512 generated tokens -- so every candidate is
// directly comparable to "what you get with no tuning". Step T's thread sweep
// lives here too: it measures the chosen placement with llama-bench.

import type { ThreadCandidateKind } from "./threadPlan.js";

export const SPEED_PROMPT_TOKENS = 4096;
export const SPEED_GEN_TOKENS = 512;
/** BOS + template tokens + tokenizer rounding headroom below the context. */
export const SPEED_TOKEN_MARGIN = 256;
export const SPEED_MIN_REPEATS = 1;
export const SPEED_MAX_REPEATS = 10;
export const SPEED_DEFAULT_REPEATS = 3;
/** Baseline "-ngl 999" -- every layer on the GPU, llama.cpp's own defaults otherwise. */
export const BASELINE_NGL = 999;

export function isSpeedRepeats(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= SPEED_MIN_REPEATS && v <= SPEED_MAX_REPEATS;
}

/** The prompt shrinks below 4096 only when the target context can't hold
 * prompt + generation + margin. */
export function speedWorkload(targetCtx: number): { promptTokens: number; nGen: number; shrunk: boolean } {
  const room = Math.floor(targetCtx) - SPEED_GEN_TOKENS - SPEED_TOKEN_MARGIN;
  const promptTokens = Math.max(1, Math.min(SPEED_PROMPT_TOKENS, room));
  return { promptTokens, nGen: SPEED_GEN_TOKENS, shrunk: promptTokens < SPEED_PROMPT_TOKENS };
}

export interface ThreadArgs {
  t: number;
  tb: number;
  mask: string | null;
  maskBatch: string | null;
  strict: boolean;
}

export type SpeedCandidateSource = "baseline" | "fit" | "fit_kv" | "fit_ctx" | "probe" | "user";

export interface SpeedCandidate {
  label: string;
  source: SpeedCandidateSource;
  ngl: number;
  /** Tensor overrides, comma-separated (fit's own form). */
  ot: string | null;
  ctk: string;
  ctv: string;
  fa: "on" | "off" | "auto";
  /** Null = llama.cpp's own thread defaults (the baseline keeps them). */
  threads: ThreadArgs | null;
  /** Fit's prediction for this placement, for the result card. */
  predicted?: { dev_used_mib: number | null; host_used_mib: number | null; layers_gpu: number | null } | null;
}

export interface SpeedRunSpec {
  target_ctx: number;
  prompt_tokens: number;
  n_gen: number;
  repeats: number;
  /** One run_item per candidate, in this order. */
  candidates: SpeedCandidate[];
}

export const MAX_SPEED_CANDIDATES = 8;

export function baselineCandidate(): SpeedCandidate {
  return { label: "Default settings", source: "baseline", ngl: BASELINE_NGL, ot: null, ctk: "f16", ctv: "f16", fa: "auto", threads: null };
}

/** The flags a candidate adds on top of -m/-c (everything that differs from
 * llama-server's defaults, plus --fit off so nothing is re-placed). */
export function candidateServerFlags(c: SpeedCandidate): string[] {
  const out = ["-ngl", String(c.ngl), "-fa", c.fa, "-ctk", c.ctk, "-ctv", c.ctv];
  if (c.ot) out.push("-ot", c.ot);
  if (c.threads) {
    out.push("-t", String(c.threads.t), "-tb", String(c.threads.tb));
    if (c.threads.mask) out.push("-C", c.threads.mask);
    if (c.threads.maskBatch) out.push("-Cb", c.threads.maskBatch);
    if (c.threads.strict && (c.threads.mask || c.threads.maskBatch)) out.push("--cpu-strict", "1");
  }
  return out;
}

/** The flags llama-server's base arg builder doesn't already emit (it always
 * passes -t/-ngl/-b/-ub/-ctk/-ctv/-fa/-c): tensor overrides and the batch
 * thread count + affinity. */
export function candidateExtraServerArgs(c: SpeedCandidate): string[] {
  const out: string[] = [];
  if (c.ot) out.push("-ot", c.ot);
  if (c.threads) {
    out.push("-tb", String(c.threads.tb));
    if (c.threads.mask) out.push("-C", c.threads.mask);
    if (c.threads.maskBatch) out.push("-Cb", c.threads.maskBatch);
    if (c.threads.strict && (c.threads.mask || c.threads.maskBatch)) out.push("--cpu-strict", "1");
  }
  return out;
}

/** llama-bench args for one thread-sweep item on top of the base builder. */
export function threadSweepBenchArgs(placementOt: string | null, item: ThreadSweepItem): string[] {
  const out: string[] = [];
  if (placementOt) out.push("-ot", placementOt.split(",").join(";"));
  if (item.mask) {
    out.push("-C", item.mask);
    if (item.strict) out.push("--cpu-strict", "1");
  }
  return out;
}

// Single quotes: both bash and PowerShell pass the text through literally, so
// fit's -ot regexes (blk\.8\.ffn_down.*) arrive unchanged. Double quotes would
// need backslash escaping that PowerShell then keeps, corrupting the pattern.
export function shellQuote(arg: string): string {
  return /^[A-Za-z0-9_./:=,+-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, `'"'"'`)}'`;
}

/** Copyable llama-server command line for a candidate. */
export function serverCommandLine(modelFile: string, ctx: number, c: SpeedCandidate): string {
  return ["llama-server", "-m", modelFile, "-c", String(ctx), ...candidateServerFlags(c), "--fit", "off"].map(shellQuote).join(" ");
}

export function candidateKey(c: Pick<SpeedCandidate, "ngl" | "ot" | "ctk" | "ctv" | "fa" | "threads">): string {
  return JSON.stringify([c.ngl, c.ot ?? "", c.ctk, c.ctv, c.fa, c.threads ?? null]);
}

/** Drops candidates identical to an earlier one. */
export function dedupeCandidates(list: readonly SpeedCandidate[]): SpeedCandidate[] {
  const seen = new Set<string>();
  const out: SpeedCandidate[] = [];
  for (const c of list) {
    const k = candidateKey(c);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(c);
  }
  return out;
}

// --- Step T: thread sweep ---------------------------------------------------

export const THREAD_TG_TOKENS = 128;
export const THREAD_PP_TOKENS = 512;

export interface ThreadSweepItem {
  /** "t" measures generation (-p 0 -n 128), "tb" prompt processing (-p 512 -n 0). */
  role: "t" | "tb";
  kind: ThreadCandidateKind;
  label: string;
  threads: number;
  mask: string | null;
  strict: boolean;
}

export interface ThreadSweepSpec {
  placement: { ngl: number; ot: string | null; ctk: string; ctv: string; fa: "on" | "off" };
  repeats: number;
  /** One run_item per entry, in this order. */
  items: ThreadSweepItem[];
}

export const MAX_THREAD_SWEEP_ITEMS = 12;

// --- comparing against the baseline ------------------------------------------

export interface SpeedReading {
  mean: number;
  stddev: number;
  samples: number;
}

/** "+38%" style delta with the noise rule: with fewer than 2 samples on
 * either side there is no spread, so no winner is called. */
export function compareToBaseline(
  cand: SpeedReading | null,
  base: SpeedReading | null
): { pct: number | null; clear: boolean; noSpread: boolean } {
  if (!cand || !base || base.mean <= 0) return { pct: null, clear: false, noSpread: false };
  const pct = ((cand.mean - base.mean) / base.mean) * 100;
  const noSpread = cand.samples < 2 || base.samples < 2;
  const clear = !noSpread && Math.abs(cand.mean - base.mean) > cand.stddev + base.stddev;
  return { pct, clear, noSpread };
}

/** Rough minutes for a speed run, from per-token rates when known. */
export function estimateSpeedRunSeconds(input: {
  candidates: number;
  repeats: number;
  promptTokens: number;
  nGen: number;
  ppTps?: number | null;
  tgTps?: number | null;
  loadSeconds?: number;
}): number {
  const pp = input.ppTps && input.ppTps > 0 ? input.ppTps : 300;
  const tg = input.tgTps && input.tgTps > 0 ? input.tgTps : 15;
  const perRepeat = input.promptTokens / pp + input.nGen / tg;
  return input.candidates * ((input.loadSeconds ?? 30) + input.repeats * perRepeat);
}

// --- trigger validation (server) ----------------------------------------------

const CACHE_TYPES = new Set(["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]);
const MASK_RE = /^0x[0-9a-f]{1,256}$/i;
// -ot entries are regex=buffer pairs; the buffer is a device name (CPU,
// CUDA0, Vulkan1, ...). Nothing that could break out of an argv entry is
// possible -- args are passed as an array, never through a shell -- but the
// value is still bounded and restricted to the characters fit emits.
const OT_PART = String.raw`[A-Za-z0-9_.*\\()|+?^$\[\]-]+=[A-Za-z0-9_]+`;
const OT_RE = new RegExp(`^${OT_PART}(?:[,;]${OT_PART})*$`);

function threadArgsError(t: unknown, field: string): string | null {
  if (t === null) return null;
  if (!t || typeof t !== "object") return `${field} must be an object or null`;
  const a = t as Record<string, unknown>;
  for (const k of ["t", "tb"] as const) {
    if (!Number.isInteger(a[k]) || (a[k] as number) < 1 || (a[k] as number) > 1024) return `${field}.${k} must be an integer in [1, 1024]`;
  }
  for (const k of ["mask", "maskBatch"] as const) {
    if (a[k] !== null && (typeof a[k] !== "string" || !MASK_RE.test(a[k] as string))) return `${field}.${k} must be a hex mask or null`;
  }
  if (typeof a.strict !== "boolean") return `${field}.strict must be a boolean`;
  return null;
}

function placementError(c: Record<string, unknown>, field: string, faAllowed: readonly string[]): string | null {
  if (!Number.isInteger(c.ngl) || (c.ngl as number) < 0 || (c.ngl as number) > 999) return `${field}.ngl must be an integer in [0, 999]`;
  if (c.ot !== null && c.ot !== undefined && (typeof c.ot !== "string" || c.ot.length > 64_000 || !OT_RE.test(c.ot))) {
    return `${field}.ot must be a tensor-override list (pattern=BUFFER, comma-separated)`;
  }
  if (typeof c.ctk !== "string" || !CACHE_TYPES.has(c.ctk) || typeof c.ctv !== "string" || !CACHE_TYPES.has(c.ctv)) {
    return `${field}.ctk/ctv must be non-f32 cache types`;
  }
  if (typeof c.fa !== "string" || !faAllowed.includes(c.fa)) return `${field}.fa must be one of ${faAllowed.join(", ")}`;
  return null;
}

export function validateSpeedRunSpec(raw: unknown, trainedCtx: number | null): string | null {
  if (!raw || typeof raw !== "object") return "speed_run must be an object";
  const s = raw as Record<string, unknown>;
  const ceiling = trainedCtx && trainedCtx > 0 ? trainedCtx : 4_194_304;
  if (!Number.isInteger(s.target_ctx) || (s.target_ctx as number) < 1024 || (s.target_ctx as number) > ceiling) {
    return `speed_run.target_ctx must be an integer in [1024, ${ceiling}]`;
  }
  const w = speedWorkload(s.target_ctx as number);
  if (s.prompt_tokens !== w.promptTokens || s.n_gen !== w.nGen) {
    return `speed_run must measure a ${w.promptTokens}-token prompt and ${w.nGen} generated tokens at this context`;
  }
  if (!isSpeedRepeats(s.repeats)) return `speed_run.repeats must be an integer between ${SPEED_MIN_REPEATS} and ${SPEED_MAX_REPEATS}`;
  if (!Array.isArray(s.candidates) || s.candidates.length === 0 || s.candidates.length > MAX_SPEED_CANDIDATES) {
    return `speed_run.candidates must list 1..${MAX_SPEED_CANDIDATES} configs`;
  }
  for (let i = 0; i < s.candidates.length; i++) {
    const c = s.candidates[i] as Record<string, unknown>;
    const field = `speed_run.candidates[${i}]`;
    if (!c || typeof c !== "object") return `${field} must be an object`;
    if (typeof c.label !== "string" || c.label.length === 0 || c.label.length > 200) return `${field}.label must be a short string`;
    if (!["baseline", "fit", "fit_kv", "fit_ctx", "probe", "user"].includes(c.source as string)) return `${field}.source is not recognised`;
    const pe = placementError(c, field, ["on", "off", "auto"]);
    if (pe) return pe;
    const te = threadArgsError(c.threads ?? null, `${field}.threads`);
    if (te) return te;
  }
  return null;
}

const THREAD_KINDS = new Set(["default", "one_core", "half", "two_ccd", "p_plus_e", "p_only"]);

export function validateThreadSweepSpec(raw: unknown): string | null {
  if (!raw || typeof raw !== "object") return "thread_sweep must be an object";
  const s = raw as Record<string, unknown>;
  if (!s.placement || typeof s.placement !== "object") return "thread_sweep.placement must be an object";
  const pe = placementError(s.placement as Record<string, unknown>, "thread_sweep.placement", ["on", "off"]);
  if (pe) return pe;
  if (!isSpeedRepeats(s.repeats)) return `thread_sweep.repeats must be an integer between ${SPEED_MIN_REPEATS} and ${SPEED_MAX_REPEATS}`;
  if (!Array.isArray(s.items) || s.items.length === 0 || s.items.length > MAX_THREAD_SWEEP_ITEMS) {
    return `thread_sweep.items must list 1..${MAX_THREAD_SWEEP_ITEMS} candidates`;
  }
  for (let i = 0; i < s.items.length; i++) {
    const it = s.items[i] as Record<string, unknown>;
    const field = `thread_sweep.items[${i}]`;
    if (!it || typeof it !== "object") return `${field} must be an object`;
    if (it.role !== "t" && it.role !== "tb") return `${field}.role must be t or tb`;
    if (!THREAD_KINDS.has(it.kind as string)) return `${field}.kind is not recognised`;
    if (typeof it.label !== "string" || it.label.length > 100) return `${field}.label must be a short string`;
    if (!Number.isInteger(it.threads) || (it.threads as number) < 1 || (it.threads as number) > 1024) return `${field}.threads must be in [1, 1024]`;
    if (it.mask !== null && (typeof it.mask !== "string" || !MASK_RE.test(it.mask))) return `${field}.mask must be a hex mask or null`;
    if (typeof it.strict !== "boolean") return `${field}.strict must be a boolean`;
  }
  return null;
}
