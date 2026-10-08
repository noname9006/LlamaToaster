// The optimization flow's fit-map job (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md
// steps 1 / 1' / 4 / 5): every answer comes from llama-fit-params, which
// sizes a placement against meta allocations without loading weights. The
// orchestration takes its process runner and reporter as arguments so it is
// testable without a llama.cpp build.

import { spawn, type ChildProcess } from "node:child_process";
import {
  buildFitArgs,
  FIT_CALL_TIMEOUT_MS,
  fitPointFromOutput,
  keepFitLogLine,
  parseFitOutput,
  type FitFaMode,
  type FitPoint,
} from "../../shared/fitParams.js";
import { classifyKvProbe, kvCandidatePairs, kvSizeGroups, type KvSupportRow } from "../../shared/kvSupport.js";
import type { FitMapSpec } from "../../shared/optimizeFlow.js";
import type { BenchLogger } from "./bench.js";

export interface FitCallResult {
  stdout: string;
  /** stderr, already reduced to the lines the parser reads. */
  log: string;
  code: number | null;
  timedOut: boolean;
  ms: number;
}

export type FitCallFn = (args: string[]) => Promise<FitCallResult>;

/** One llama-fit-params process. -v prints MBs per call; only the decision
 * lines are kept while it streams. */
export function makeFitCaller(
  binary: string,
  opts: { timeoutMs?: number; onSpawn?: (proc: ChildProcess) => void; onExit?: () => void } = {}
): FitCallFn {
  return (args) =>
    new Promise((resolve) => {
      const started = Date.now();
      const proc = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
      opts.onSpawn?.(proc);
      let stdout = "";
      let pending = "";
      const kept: string[] = [];
      let timedOut = false;
      const keep = (line: string) => {
        const l = line.replace(/\r$/, "");
        if (keepFitLogLine(l)) kept.push(l);
      };
      proc.stdout?.on("data", (d: Buffer) => {
        if (stdout.length < 64_000) stdout += d.toString();
      });
      proc.stderr?.on("data", (d: Buffer) => {
        pending += d.toString();
        let i = pending.indexOf("\n");
        while (i !== -1) {
          keep(pending.slice(0, i));
          pending = pending.slice(i + 1);
          i = pending.indexOf("\n");
        }
        // A pathological line with no newline must not grow without bound.
        if (pending.length > 1_000_000) pending = pending.slice(-10_000);
      });
      const timer = setTimeout(() => {
        timedOut = true;
        proc.kill("SIGKILL");
      }, opts.timeoutMs ?? FIT_CALL_TIMEOUT_MS);
      const done = (code: number | null) => {
        clearTimeout(timer);
        if (pending) keep(pending);
        opts.onExit?.();
        resolve({ stdout, log: kept.join("\n"), code, timedOut, ms: Date.now() - started });
      };
      proc.on("error", (err) => {
        kept.push(`common_fit_params: encountered an error while trying to fit params: ${err.message}`);
        done(null);
      });
      proc.on("close", (code) => done(code));
    });
}

export interface FitMapReporter {
  point(p: FitPoint): void | Promise<void>;
  kvRows(rows: KvSupportRow[]): void | Promise<void>;
  progress(done: number, total: number, detail: string): void;
}

export interface FitMapInput {
  spec: FitMapSpec;
  modelPath: string;
  backend: string;
  /** Device selection args (-dev none on CPU-only, -sm none -mg N for one GPU). */
  deviceArgs: string[];
  /** Known KV rows for this machine/build/model; detection is skipped when complete. */
  kvKnown?: readonly KvSupportRow[];
  ramBudgetMib: number | null;
  call: FitCallFn;
  report: FitMapReporter;
  shouldStop: () => boolean;
  /** Resolves once the job may continue (the user's Pause holds it here). */
  waitWhilePaused?: () => Promise<void>;
  log?: BenchLogger;
}

export interface FitMapOutcome {
  points: FitPoint[];
  kv: KvSupportRow[];
  calls: number;
  stopped: boolean;
}

const FA_MODES: FitFaMode[] = ["on", "off"];
const KV_PROBE_CTX = 4096;

/** Known rows are reusable when every candidate pair has an answer at both
 * FA modes (the f16/f16 baselines included). */
export function kvRowsComplete(rows: readonly KvSupportRow[] | undefined): boolean {
  if (!rows) return false;
  const have = new Set(rows.map((r) => `${r.ctk}|${r.ctv}|${r.fa}`));
  return kvCandidatePairs().every(([k, v]) => FA_MODES.every((fa) => have.has(`${k}|${v}|${fa}`)));
}

function errorPoint(ctx: number, fa: FitFaMode, ctk: string, ctv: string, margin: number, message: string): FitPoint {
  return {
    ctx, fa, ctk, ctv, margin_mib: margin, verdict: "error", reason: null, ngl: null, layers_gpu: null, layers_total: null,
    overflow_layers: 0, ot: null, dev_used_mib: null, dev_free_mib: null, host_used_mib: null, total_mib: null, kv_mib: null,
    compute_mib: null, need_all_gpu_mib: null, splits: null, fa_disabled: false, n_expert: null, n_expert_used: null,
    inferred: false, raw_args: null, error: message,
  };
}

export async function runFitMap(input: FitMapInput): Promise<FitMapOutcome> {
  const { spec, call, report, log } = input;
  const stops = [...new Set(spec.ctx_stops)].sort((a, b) => b - a);
  const cpuOnly = spec.machine === "cpu";
  const margin = spec.margin_mib;
  const points: FitPoint[] = [];
  let kv: KvSupportRow[] = [];
  let calls = 0;
  const reuseKv = spec.kv_detect && kvRowsComplete(input.kvKnown);
  // Group count is only known after detection; estimate 12 until then.
  let total = stops.length * FA_MODES.length + (spec.kv_detect && !reuseKv ? 128 : 0) + (spec.kv_fit ? 12 * stops.length : 0);
  let done = 0;
  const tick = (detail: string) => report.progress(done, Math.max(total, done), detail);

  const fitOne = async (ctx: number, fa: FitFaMode, ctk: string, ctv: string): Promise<FitPoint> => {
    await input.waitWhilePaused?.();
    const args = buildFitArgs({ modelPath: input.modelPath, ctx, fa, ctk, ctv, marginMib: margin, deviceArgs: input.deviceArgs });
    calls++;
    const res = await call(args);
    let p: FitPoint;
    if (res.timedOut) {
      p = errorPoint(ctx, fa, ctk, ctv, margin, `llama-fit-params gave no answer within ${Math.round(FIT_CALL_TIMEOUT_MS / 1000)} s`);
    } else {
      const parsed = parseFitOutput(res.stdout, res.log);
      p = fitPointFromOutput(parsed, { ctx, fa, ctk, ctv, marginMib: margin, ramBudgetMib: input.ramBudgetMib, cpuOnly });
      if (p.verdict === "error" && !p.error) p.error = `llama-fit-params exited ${res.code ?? "abnormally"} without an answer`;
    }
    log?.info(
      `fit ${ctk}/${ctv} FA ${fa} @${ctx}: ${p.verdict}${p.reason ? ` (${p.reason})` : ""}` +
        (p.layers_gpu != null ? ` · ${p.layers_gpu}${p.layers_total ? `/${p.layers_total}` : ""} layers` : "") +
        (p.overflow_layers ? ` · experts of ${p.overflow_layers} on CPU` : "") +
        (p.dev_used_mib != null ? ` · ${Math.round(p.dev_used_mib)} MiB device` : "") +
        (p.host_used_mib != null ? ` + ${Math.round(p.host_used_mib)} MiB host` : "") +
        ` · ${res.ms} ms` +
        (p.error ? ` · ${p.error}` : "")
    );
    points.push(p);
    await report.point(p);
    done++;
    return p;
  };

  // Steps 1 / 1': the default cache, FA on and off, largest context first.
  for (const fa of FA_MODES) {
    for (const ctx of stops) {
      if (input.shouldStop()) return { points, kv, calls, stopped: true };
      tick(`fit map · f16 · FA ${fa} · ${ctx.toLocaleString("en-US")} tokens`);
      await fitOne(ctx, fa, "f16", "f16");
    }
  }

  // Step 4: which K/V pairs run on the GPU.
  if (spec.kv_detect) {
    if (reuseKv) {
      kv = [...input.kvKnown!];
      log?.info(`KV support: reusing ${kv.length} known rows for this build and model`);
    } else {
      for (const fa of FA_MODES) {
        const probe = async (ctk: string, ctv: string) => {
          await input.waitWhilePaused?.();
          calls++;
          const res = await call(
            buildFitArgs({ modelPath: input.modelPath, ctx: KV_PROBE_CTX, fa, ctk, ctv, kvProbe: true, deviceArgs: input.deviceArgs })
          );
          done++;
          return parseFitOutput(res.stdout, res.log);
        };
        if (input.shouldStop()) return { points, kv, calls, stopped: true };
        tick(`KV support · FA ${fa} · f16 / f16 baseline`);
        const baseline = await probe("f16", "f16");
        for (const [ctk, ctv] of kvCandidatePairs()) {
          if (input.shouldStop()) return { points, kv, calls, stopped: true };
          const parsed = ctk === "f16" && ctv === "f16" ? baseline : (tick(`KV support · FA ${fa} · ${ctk} / ${ctv}`), await probe(ctk, ctv));
          const row = classifyKvProbe({ ctk, ctv, fa, probe: parsed, baseline, backend: input.backend });
          kv.push(row);
        }
      }
      const okCount = kv.filter((r) => r.status === "ok" || r.status === "cuda_slow").length;
      log?.info(`KV support: ${okCount} of ${kv.length} pair/FA combinations run without falling back`);
    }
    await report.kvRows(kv);
  }

  // Step 5: one path per supported K/V size (the f16 size is step 1's path).
  if (spec.kv_fit && kv.length > 0) {
    const groups = kvSizeGroups(kv).filter((g) => !(g.rep[0] === "f16" && g.rep[1] === "f16"));
    total = done + groups.length * stops.length;
    for (const g of groups) {
      for (const ctx of stops) {
        if (input.shouldStop()) return { points, kv, calls, stopped: true };
        tick(`fit map · ${g.rep[0]} / ${g.rep[1]} · FA ${g.fa} · ${ctx.toLocaleString("en-US")} tokens`);
        await fitOne(ctx, g.fa, g.rep[0], g.rep[1]);
      }
    }
  }
  report.progress(done, done, "fit map complete");
  return { points, kv, calls, stopped: false };
}
