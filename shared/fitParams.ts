// llama-fit-params: llama.cpp's own --fit algorithm run against meta
// allocations (no weights loaded). The optimization flow's fit map
// (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md §1.1, steps 1 / 1' / 4 / 5) is
// built entirely from its answers, so everything that turns one call into one
// stored point lives here: the argument list, the line filter the worker
// applies while the -v output streams (it is MBs per call), the parser, and
// the per-point verdict.
//
// Call rules, all verified live against b11226 (see the fixtures in
// shared/__fixtures__/fit/):
//   1. never pass -ngl for a fit-map call: fit only adjusts UNSET arguments.
//      The KV support probe passes -ngl 99 on purpose, so fit skips its
//      search and only builds the initial graph.
//   2. always pass -c: with ctx unset fit shrinks the context first.
//   3. "doesn't fit" is not an exit code -- fit still answers -ngl 0. The
//      verdict comes from the final per-device "N layers ..., U MiB used,
//      F MiB free" line compared against the margin.
//   4. its decisions are logged only under -v.

export const FIT_MARGINS_MIB = [512, 1024, 1536] as const;
export type FitMarginMib = (typeof FIT_MARGINS_MIB)[number];
export const DEFAULT_FIT_MARGIN_MIB: FitMarginMib = 1024;

export function isFitMargin(v: unknown): v is FitMarginMib {
  return typeof v === "number" && (FIT_MARGINS_MIB as readonly number[]).includes(v);
}

/** Smallest context the fit map checks (1k and 2k are excluded by design). */
export const FIT_MIN_CTX = 4096;
/** Per-call budget. The slowest call seen (a model that cannot fit at all,
 * walking every layer count) took 38 s. */
export const FIT_CALL_TIMEOUT_MS = 60_000;

/** Doublings from 4096 up to the model's trained context, plus the trained
 * context itself when it isn't a power of two. */
export function fitCtxStops(nCtxTrain: number | null | undefined): number[] {
  const ceiling = Math.max(FIT_MIN_CTX, Math.floor(nCtxTrain && nCtxTrain > 0 ? nCtxTrain : FIT_MIN_CTX));
  const stops: number[] = [];
  for (let v = FIT_MIN_CTX; v < ceiling; v *= 2) stops.push(v);
  stops.push(ceiling);
  return stops;
}

export type FitFaMode = "on" | "off";

export interface FitArgsInput {
  modelPath: string;
  ctx: number;
  fa: FitFaMode | "auto";
  ctk?: string;
  ctv?: string;
  marginMib?: number;
  /** Device selection, e.g. ["-dev", "none"] on a CPU-only run or
   * ["-sm", "none", "-mg", "1"] for one GPU of several. */
  deviceArgs?: string[];
  /** KV support probe: -ngl 99 so fit does no search (rule 1). */
  kvProbe?: boolean;
}

export function buildFitArgs(input: FitArgsInput): string[] {
  const args = [
    "-m",
    input.modelPath,
    "-c",
    String(Math.floor(input.ctx)),
    "-fa",
    input.fa,
    "-ctk",
    input.ctk ?? "f16",
    "-ctv",
    input.ctv ?? "f16",
  ];
  if (input.kvProbe) {
    args.push("-ngl", "99");
  } else {
    args.push("-fitt", String(input.marginMib ?? DEFAULT_FIT_MARGIN_MIB), "-fitc", String(FIT_MIN_CTX));
  }
  args.push("-v");
  if (input.deviceArgs?.length) args.push(...input.deviceArgs);
  return args;
}

// The -v output is mostly tensor/loader noise; these are the only lines the
// parser reads. The worker keeps only matching lines while the call streams,
// so a 2 MB log becomes ~100 lines.
const KEEP_LINE_RE =
  /common_params_fit_impl|common_memory_breakdown_print|common_fit_params|llama_kv_cache: size|compute buffer size|sched_reserve: graph|Flash Attention|failed to create|print_info: n_layer |print_info: n_ctx_train|print_info: n_expert(?:_used)? /;

export function keepFitLogLine(line: string): boolean {
  return KEEP_LINE_RE.test(line);
}

export interface FitBreakdownRow {
  name: string;
  totalMib: number | null;
  freeMib: number | null;
  selfMib: number;
  modelMib: number;
  contextMib: number;
  computeMib: number;
}

export interface FitBreakdown {
  devices: FitBreakdownRow[];
  host: FitBreakdownRow | null;
}

export interface FitFinalDevice {
  name: string;
  layers: number;
  overflowing: number;
  usedMib: number;
  freeMib: number;
}

export interface ParsedFitOutput {
  /** Fitted CLI args as printed on stdout. */
  rawArgs: string;
  ctx: number | null;
  /** -1 = every layer (fit prints "-ngl -1" when nothing needed changing). */
  ngl: number | null;
  /** Fit's -ot string, comma-separated, verbatim. */
  ot: string | null;
  nLayer: number | null;
  nCtxTrain: number | null;
  nExpert: number | null;
  nExpertUsed: number | null;
  /** First "projected to use X MiB of device memory vs. Y MiB" -- what keeping
   * every layer on the GPU would need. */
  projectedDeviceMib: number | null;
  projectedFreeMib: number | null;
  /** CPU-only: "projected to use X MiB of host memory vs. Y MiB of total". */
  projectedHostMib: number | null;
  hostTotalMib: number | null;
  /** "will leave X >= T MiB ..., no changes needed" */
  noChangesNeeded: boolean;
  /** Fit gave up (CPU-only "unable to fit model into system memory"). */
  aborted: boolean;
  finalDevices: FitFinalDevice[];
  breakdowns: FitBreakdown[];
  /** The breakdown matching the chosen placement (device self == used). */
  chosen: FitBreakdown | null;
  /** Splits of the first scheduler graph (initial parameters). For the
   * two-graph form "splits = A / B" this is A (prompt processing). */
  splits: number | null;
  faDisabled: boolean;
  error: string | null;
}

const NUM = String.raw`(-?\d+(?:\.\d+)?)`;
const DEVICE_ROW_RE = new RegExp(
  String.raw`\|\s+-\s+(.+?)\s+\|\s+` + NUM + String.raw`\s+=\s+` + NUM + String.raw`\s+\+\s+\(\s*` + NUM +
    String.raw`\s+=\s+` + NUM + String.raw`\s+\+\s+` + NUM + String.raw`\s+\+\s+` + NUM + String.raw`\s*\)`
);
const HOST_ROW_RE = new RegExp(
  String.raw`\|\s+-\s+(Host)\s+\|\s+` + NUM + String.raw`\s+=\s+` + NUM + String.raw`\s+\+\s+` + NUM +
    String.raw`\s+\+\s+` + NUM + String.raw`\s*\|`
);
const FINAL_DEVICE_RE = new RegExp(
  String.raw`common_params_fit_impl:\s+-\s+(.+?):\s+(\d+) layers(?: \((\d+) overflowing\))?,\s+` + NUM +
    String.raw` MiB used,\s+` + NUM + String.raw` MiB free`
);
const PROJECTED_DEVICE_RE = new RegExp(
  String.raw`projected to use ` + NUM + String.raw` MiB of device memory vs\. ` + NUM + String.raw` MiB of free device memory`
);
const PROJECTED_HOST_RE = new RegExp(
  String.raw`projected to use ` + NUM + String.raw` MiB of host memory vs\. ` + NUM + String.raw` MiB of total host memory`
);
const SPLITS_RE = /sched_reserve: graph(?: \([^)]*\))?: nodes = [^,]+, splits = (\d+)/;
const PRINT_INFO_RE = /print_info: (n_layer|n_ctx_train|n_expert|n_expert_used)\s+=\s+(\d+)/;

function num(s: string | undefined): number {
  return Number(s);
}

/** Splits fit's stdout ("-c 32768 -ngl 41 -ot \"...\"") into its parts. */
export function parseFitArgs(stdout: string): { ctx: number | null; ngl: number | null; ot: string | null } {
  const text = stdout.trim();
  const ctx = /(?:^|\s)-c\s+(-?\d+)/.exec(text);
  const ngl = /(?:^|\s)-ngl\s+(-?\d+|all)/.exec(text);
  const ot = /(?:^|\s)-ot\s+"([^"]*)"/.exec(text) ?? /(?:^|\s)-ot\s+(\S+)/.exec(text);
  return {
    ctx: ctx ? Number(ctx[1]) : null,
    ngl: ngl ? (ngl[1] === "all" ? -1 : Number(ngl[1])) : null,
    ot: ot ? ot[1] : null,
  };
}

export function parseFitOutput(stdout: string, log: string): ParsedFitOutput {
  const args = parseFitArgs(stdout);
  const out: ParsedFitOutput = {
    rawArgs: stdout.trim(),
    ...args,
    nLayer: null,
    nCtxTrain: null,
    nExpert: null,
    nExpertUsed: null,
    projectedDeviceMib: null,
    projectedFreeMib: null,
    projectedHostMib: null,
    hostTotalMib: null,
    noChangesNeeded: false,
    aborted: false,
    finalDevices: [],
    breakdowns: [],
    chosen: null,
    splits: null,
    faDisabled: false,
    error: null,
  };

  let current: FitBreakdown | null = null;
  // Fit prints one final device list; a multi-device box prints one line per
  // device in a row. A later list (there is none in practice) replaces it.
  let finalRun: FitFinalDevice[] = [];
  let lastWasFinal = false;

  for (const line of log.split(/\r?\n/)) {
    const info = PRINT_INFO_RE.exec(line);
    if (info) {
      const v = Number(info[2]);
      if (info[1] === "n_layer" && out.nLayer == null) out.nLayer = v;
      else if (info[1] === "n_ctx_train" && out.nCtxTrain == null) out.nCtxTrain = v;
      else if (info[1] === "n_expert" && out.nExpert == null) out.nExpert = v;
      else if (info[1] === "n_expert_used" && out.nExpertUsed == null) out.nExpertUsed = v;
      continue;
    }
    if (line.includes("common_memory_breakdown_print")) {
      if (line.includes("memory breakdown [MiB]")) {
        current = { devices: [], host: null };
        out.breakdowns.push(current);
        continue;
      }
      const host = HOST_ROW_RE.exec(line);
      if (host && current) {
        current.host = {
          name: "Host",
          totalMib: null,
          freeMib: null,
          selfMib: num(host[2]),
          modelMib: num(host[3]),
          contextMib: num(host[4]),
          computeMib: num(host[5]),
        };
        continue;
      }
      const dev = DEVICE_ROW_RE.exec(line);
      if (dev && current) {
        current.devices.push({
          name: dev[1].trim(),
          totalMib: num(dev[2]),
          freeMib: num(dev[3]),
          selfMib: num(dev[4]),
          modelMib: num(dev[5]),
          contextMib: num(dev[6]),
          computeMib: num(dev[7]),
        });
      }
      continue;
    }
    const fin = FINAL_DEVICE_RE.exec(line);
    if (fin) {
      if (!lastWasFinal) finalRun = [];
      finalRun.push({
        name: fin[1].trim(),
        layers: Number(fin[2]),
        overflowing: fin[3] ? Number(fin[3]) : 0,
        usedMib: num(fin[4]),
        freeMib: num(fin[5]),
      });
      lastWasFinal = true;
      continue;
    }
    lastWasFinal = false;
    const pd = PROJECTED_DEVICE_RE.exec(line);
    if (pd) {
      if (out.projectedDeviceMib == null) {
        out.projectedDeviceMib = num(pd[1]);
        out.projectedFreeMib = num(pd[2]);
      }
      continue;
    }
    const ph = PROJECTED_HOST_RE.exec(line);
    if (ph) {
      if (out.projectedHostMib == null) {
        out.projectedHostMib = num(ph[1]);
        out.hostTotalMib = num(ph[2]);
      }
      continue;
    }
    if (/will leave -?\d+ >= -?\d+ MiB of (?:free device|system) memory, no changes needed/.test(line)) {
      out.noChangesNeeded = true;
      continue;
    }
    const sp = SPLITS_RE.exec(line);
    if (sp) {
      if (out.splits == null) out.splits = Number(sp[1]);
      continue;
    }
    if (/Flash Attention not supported, set to disabled|Flash Attention is assigned to device CPU/.test(line)) {
      out.faDisabled = true;
      continue;
    }
    if (/unable to fit model into system memory|failed to fit params to free device memory:.*abort/.test(line)) {
      out.aborted = true;
      continue;
    }
    if (/failed to create llama_context|encountered an error while trying to fit params/.test(line)) {
      if (!out.error) {
        const m = /(?:error[^:]*: )?(failed to create llama_context[^\r\n]*)/.exec(line);
        out.error = m ? m[1].trim() : line.replace(/^.*?common_fit_params:\s*/, "").trim();
      }
    }
  }
  out.finalDevices = finalRun;

  // The chosen placement's own table: every device's self equals what the
  // final line says it uses. Without a final line (nothing needed changing)
  // the initial-parameters table is the answer.
  if (out.finalDevices.length > 0) {
    out.chosen =
      [...out.breakdowns]
        .reverse()
        .find((b) =>
          out.finalDevices.every((f) => b.devices.some((d) => d.selfMib === f.usedMib))
        ) ?? null;
  } else if (out.breakdowns.length > 0) {
    out.chosen = out.breakdowns[0];
  }
  return out;
}

export type FitVerdict = "full" | "partial" | "cpu_only" | "doesnt_fit" | "error";
export type FitShortReason = "vram" | "ram" | null;

/** One stored point of the fit map -- one (ctx, cache setup) cell. Memory
 * figures are MiB, straight from fit's own tables. */
export interface FitPoint {
  ctx: number;
  fa: FitFaMode;
  ctk: string;
  ctv: string;
  margin_mib: number;
  verdict: FitVerdict;
  reason: FitShortReason;
  /** Fitted -ngl; -1 = every layer. Null on error. */
  ngl: number | null;
  /** Layers fit put on the GPU (all devices). */
  layers_gpu: number | null;
  /** n_layer + 1 (the output layer counts), what "all layers" means. */
  layers_total: number | null;
  /** MoE: layers whose routed experts (fully or partly) stay on the CPU. */
  overflow_layers: number;
  /** Fit's own -ot, comma form; llama-bench needs ';'. */
  ot: string | null;
  dev_used_mib: number | null;
  dev_free_mib: number | null;
  host_used_mib: number | null;
  /** dev_used + host_used. */
  total_mib: number | null;
  kv_mib: number | null;
  compute_mib: number | null;
  /** What every layer on the GPU would need (first projection). */
  need_all_gpu_mib: number | null;
  splits: number | null;
  fa_disabled: boolean;
  n_expert: number | null;
  n_expert_used: number | null;
  /** True when this point was not called but inferred (a smaller cache or a
   * smaller context of a setup that already put every layer on the GPU). */
  inferred: boolean;
  raw_args: string | null;
  error: string | null;
}

export interface FitVerdictContext {
  ctx: number;
  fa: FitFaMode;
  ctk: string;
  ctv: string;
  marginMib: number;
  /** System RAM the host side may use, MiB (null = don't check). */
  ramBudgetMib?: number | null;
  /** Run with -dev none (CPU only): the answer is host memory. */
  cpuOnly?: boolean;
}

function sumRows(rows: FitBreakdownRow[], key: "selfMib" | "contextMib" | "computeMib"): number {
  return rows.reduce((a, r) => a + r[key], 0);
}

export function fitPointFromOutput(parsed: ParsedFitOutput, c: FitVerdictContext): FitPoint {
  const layersTotal = parsed.nLayer != null ? parsed.nLayer + 1 : null;
  const base: FitPoint = {
    ctx: c.ctx,
    fa: c.fa,
    ctk: c.ctk,
    ctv: c.ctv,
    margin_mib: c.marginMib,
    verdict: "error",
    reason: null,
    ngl: parsed.ngl,
    layers_gpu: null,
    layers_total: layersTotal,
    overflow_layers: 0,
    ot: parsed.ot,
    dev_used_mib: null,
    dev_free_mib: null,
    host_used_mib: null,
    total_mib: null,
    kv_mib: null,
    compute_mib: null,
    need_all_gpu_mib: parsed.projectedDeviceMib,
    splits: parsed.splits,
    fa_disabled: parsed.faDisabled,
    n_expert: parsed.nExpert,
    n_expert_used: parsed.nExpertUsed,
    inferred: false,
    raw_args: parsed.rawArgs || null,
    error: parsed.error,
  };

  const chosen = parsed.chosen;
  if (chosen) {
    const rows = [...chosen.devices, ...(chosen.host ? [chosen.host] : [])];
    base.kv_mib = sumRows(rows, "contextMib");
    base.compute_mib = sumRows(rows, "computeMib");
    base.host_used_mib = chosen.host?.selfMib ?? 0;
  }

  if (parsed.error && !parsed.aborted) return base;

  if (c.cpuOnly) {
    const host = parsed.projectedHostMib ?? base.host_used_mib;
    base.host_used_mib = host;
    base.dev_used_mib = 0;
    base.total_mib = host;
    base.layers_gpu = 0;
    // -dev none: fit's "-ngl -1" only means it changed nothing; nothing runs
    // on a GPU, and a config built from this point must say so.
    base.ngl = 0;
    base.error = null;
    if (parsed.aborted || !parsed.noChangesNeeded) {
      base.verdict = "doesnt_fit";
      base.reason = "ram";
    } else {
      base.verdict = "cpu_only";
    }
    return base;
  }

  if (parsed.noChangesNeeded && parsed.finalDevices.length === 0) {
    const devs = chosen?.devices ?? [];
    base.dev_used_mib = devs.length ? sumRows(devs, "selfMib") : parsed.projectedDeviceMib;
    base.dev_free_mib = devs.length ? devs.reduce((a, d) => a + (d.freeMib ?? 0), 0) : parsed.projectedFreeMib;
    base.layers_gpu = layersTotal;
    base.verdict = "full";
  } else if (parsed.finalDevices.length > 0) {
    const fin = parsed.finalDevices;
    base.layers_gpu = fin.reduce((a, f) => a + f.layers, 0);
    base.overflow_layers = fin.reduce((a, f) => a + f.overflowing, 0);
    base.dev_used_mib = fin.reduce((a, f) => a + f.usedMib, 0);
    base.dev_free_mib = fin.reduce((a, f) => a + f.freeMib, 0);
    const shortOnDevice = fin.some((f) => f.freeMib < c.marginMib);
    if (shortOnDevice) {
      base.verdict = "doesnt_fit";
      base.reason = "vram";
    } else if (base.layers_gpu === 0) {
      base.verdict = "cpu_only";
    } else if (layersTotal != null && base.layers_gpu >= layersTotal && base.overflow_layers === 0 && !parsed.ot) {
      base.verdict = "full";
    } else {
      base.verdict = "partial";
    }
    if (!chosen) {
      // No matching table: the RAM side is what all-on-GPU would have needed
      // minus what the device now holds -- an approximation, flagged by the
      // caller as such through host_used_mib being derived.
      if (parsed.projectedDeviceMib != null) {
        base.host_used_mib = Math.max(0, parsed.projectedDeviceMib - base.dev_used_mib);
      }
    }
  } else {
    return base;
  }

  base.error = null;
  if (base.dev_used_mib != null) base.total_mib = base.dev_used_mib + (base.host_used_mib ?? 0);
  if (
    base.verdict !== "doesnt_fit" &&
    c.ramBudgetMib != null &&
    base.host_used_mib != null &&
    base.host_used_mib > c.ramBudgetMib
  ) {
    base.verdict = "doesnt_fit";
    base.reason = "ram";
  }
  return base;
}

/** llama-bench separates several -ot entries with ';', fit emits ','. */
export function otForBench(ot: string): string {
  return ot.split(",").join(";");
}

// --- KV cache bytes, for grouping pairs that size identically -------------

/** Bytes per 32 elements of each non-f32 cache type (ggml block sizes). */
export const KV_TYPE_UNITS: Record<string, number> = {
  f16: 64,
  bf16: 64,
  q8_0: 34,
  q5_1: 24,
  q5_0: 22,
  q4_1: 20,
  q4_0: 18,
  iq4_nl: 18,
};

/** Every non-32-bit cache type llama.cpp accepts for -ctk/-ctv. f32 is never
 * used in this flow (user decision 2026-10-06). */
export const KV_CACHE_TYPES = ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"] as const;

/** A key the fit map's answer depends on: the cache's size per cell. With
 * equal K/V head dims only the sum matters; MLA-style unequal dims size K
 * and V separately. */
export function kvSizeKey(ctk: string, ctv: string, equalHeadDims = true): string {
  const k = KV_TYPE_UNITS[ctk];
  const v = KV_TYPE_UNITS[ctv];
  if (k == null || v == null) return `${ctk}/${ctv}`;
  return equalHeadDims ? `s${k + v}` : `k${k}v${v}`;
}

/** Relative cache size against f16/f16 (1 = f16/f16). */
export function kvRelativeSize(ctk: string, ctv: string): number {
  const k = KV_TYPE_UNITS[ctk] ?? 64;
  const v = KV_TYPE_UNITS[ctv] ?? 64;
  return (k + v) / 128;
}

export function isQuantizedKv(t: string): boolean {
  return t !== "f16" && t !== "bf16" && t !== "f32";
}
