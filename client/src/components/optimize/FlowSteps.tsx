import { useMemo, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Model, Worker } from "../../types";
import { LtIcon } from "../ltIcons";
import { Btn, CopyCommand, Kicker, Notice, Panel, Seg, StepTitle, fmtGb } from "./ui";
import { MemorySplitBar, KvSizeBars } from "./charts";
import { BaselineSection, ConfirmSection, carryForward } from "./SpeedSections";
import { FitMapSection } from "./FitMapSection";
import { RealLoadSection } from "./RealLoadSection";
import { ThreadsSection } from "./ThreadsSection";
import type { FlowContext } from "./flowContext";
import { speedRunView, spilledMib } from "./flowData";
import { machineKind, stateLabel, type FitAnswer, type Verdict } from "./machine";
import { isActive, type FitData, type FlowRun } from "./useOptimizeFlow";
import { STEP_LABEL, nextStep, prevStep, type StepId, type StepState } from "./steps";
import { detectMoe } from "../../../../shared/moePlacement.js";
import { KV_QUALITY_WARNING } from "../../../../shared/kvSupport.js";
import { buildFitSeries, ctxLabel, kvOptionsAt, proposeKvCandidates, type FitSeries } from "../../../../shared/optimizeFlow.js";
import {
  SPEED_MAX_REPEATS,
  SPEED_MIN_REPEATS,
  batchCandidates,
  serverCommandLine,
  type SpeedCandidate,
  type ThreadArgs,
} from "../../../../shared/speedRun.js";

export function modelLabelOf(m: Model): string {
  return (m.hf_file ?? m.filename ?? m.id).replace(/\.gguf$/i, "").split(/[\\/]/).pop() ?? m.id;
}

function StepHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-6 max-w-3xl">
      <h2 className="m-0 font-display text-[32px] font-semibold leading-tight">{title}</h2>
      {children && <p className="mt-1.5 text-fg-2">{children}</p>}
    </header>
  );
}

/** Back / Next under every step. Next names the step it opens and, when that
 * step is still locked, says why instead of being a dead button. */
export function StepNav({ current, states, onPick }: { current: StepId; states: StepState[]; onPick: (id: StepId) => void }) {
  const next = nextStep(current);
  const prev = prevStep(current);
  const nextState = next ? states.find((s) => s.id === next) : null;
  return (
    <div className="mt-8 flex flex-wrap items-center gap-4 border-t border-border pt-5">
      {next && (
        <Btn kind="primary" onClick={() => onPick(next)} disabled={!!nextState?.locked} aria-describedby={nextState?.locked ? "step-next-why" : undefined}>
          Next: {STEP_LABEL[next].toLowerCase()}
        </Btn>
      )}
      {prev && (
        <button type="button" onClick={() => onPick(prev)} className="min-h-10 bg-transparent text-sm text-accent hover:text-accent-hover">
          Back
        </button>
      )}
      {nextState?.locked && (
        <span id="step-next-why" className="text-sm text-muted">
          {nextState.locked}
        </span>
      )}
    </div>
  );
}

// --- 1 Machine ----------------------------------------------------------------

interface Check {
  label: string;
  value: string;
  ok: boolean;
  fix?: ReactNode;
}

export function machineChecks(w: Worker): Check[] {
  const caps = new Set(w.capabilities ?? []);
  const kind = machineKind(w);
  const build = w.installedBuilds.find((b) => b.active) ?? null;
  const hw = w.hardware;
  const gpu = kind.gpus[0];
  const checks: Check[] = [
    {
      label: "Worker",
      value: w.status === "offline" ? "Not checking in" : stateLabel(w),
      ok: w.status !== "offline",
      fix: w.status === "offline" ? <Link to="/workers" className="text-accent">Reconnect steps</Link> : undefined,
    },
    {
      label: "Worker version",
      value: caps.has("fit-map-v1") && caps.has("speed-run-v1") ? "Supports the optimization flow" : "Too old for the optimization flow",
      ok: caps.has("fit-map-v1") && caps.has("speed-run-v1"),
      fix: caps.has("fit-map-v1") && caps.has("speed-run-v1") ? undefined : <Link to="/workers" className="text-accent">Update it</Link>,
    },
    {
      label: "llama.cpp",
      value: build ? `${build.tag}${build.fit_params_path ? "" : " — no llama-fit-params in this build"}` : "No build installed",
      ok: !!build?.fit_params_path,
      fix: build?.fit_params_path ? undefined : <Link to="/workers" className="text-accent">Install a build</Link>,
    },
    {
      label: "GPU",
      value: kind.hasGpu && gpu ? `${gpu.model}${gpu.vram_mb ? `, ${fmtGb(gpu.vram_mb)}` : ""}${kind.unified ? ", unified memory" : ""}` : "None used — CPU only",
      ok: true,
    },
  ];
  if (hw?.mem_total_bytes) checks.push({ label: "System memory", value: fmtGb(hw.mem_total_bytes / 1048576), ok: true });
  if (hw?.cpu) {
    checks.push({
      label: "CPU",
      value: `${hw.cpu.brand}, ${hw.cpu.physical_cores ?? hw.cpu.cores} cores${hw.cpu.physical_cores && hw.cpu.physical_cores !== hw.cpu.cores ? ` / ${hw.cpu.cores} threads` : ""}`,
      ok: true,
    });
  }
  return checks;
}

export function machineReady(w: Worker | undefined): boolean {
  return !!w && machineChecks(w).every((c) => c.ok);
}

export function MachineStep({
  workers,
  workerId,
  onPick,
  mainGpu,
  onMainGpu,
}: {
  workers: Worker[];
  workerId: string;
  onPick: (id: string) => void;
  mainGpu: number | undefined;
  onMainGpu: (g: number | undefined) => void;
}) {
  const worker = workers.find((w) => w.id === workerId);
  const kind = worker ? machineKind(worker) : null;
  const checks = worker ? machineChecks(worker) : [];
  const ready = checks.length > 0 && checks.every((c) => c.ok);
  return (
    <div className="flex flex-col gap-8">
      <StepHeader title="Set up the machine">
        Pick the computer that runs the tests. It needs the LlamaToaster worker running and a llama.cpp build with llama-fit-params.
      </StepHeader>
      <fieldset className="m-0 border-0 p-0">
        <legend className="mb-2.5 p-0 font-display text-lg font-semibold">Your machines</legend>
        <div role="radiogroup" aria-label="Machine" className="flex max-w-3xl flex-col">
          {workers.map((w) => {
            const on = w.id === workerId;
            const k = machineKind(w);
            return (
              <button
                key={w.id}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => onPick(w.id)}
                className={`flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-l-2 border-b-border px-3.5 py-3 text-left first:border-t first:border-t-border ${
                  on ? "border-l-accent bg-surface-raised" : "border-l-transparent hover:bg-surface"
                }`}
              >
                <span className="min-w-0 flex-[1_1_240px]">
                  <span className="block font-medium">{w.displayName}</span>
                  <span className="block text-[13px] text-muted">
                    {k.hasGpu ? `${k.gpus[0]?.model ?? "GPU"}${k.gpus[0]?.vram_mb ? ` ${Math.round(k.gpus[0].vram_mb / 1024)} GB` : ""}` : "CPU only"}
                    {w.backend ? ` · ${w.backend}` : ""}
                    {w.installedBuilds.find((b) => b.active)?.tag ? ` ${w.installedBuilds.find((b) => b.active)!.tag}` : ""}
                  </span>
                </span>
                <span className={`text-sm ${w.status === "offline" ? "text-muted" : "text-online"}`}>{stateLabel(w)}</span>
              </button>
            );
          })}
        </div>
        <Link to="/workers" className="mt-3 inline-block text-sm text-accent hover:text-accent-hover">
          Add a machine
        </Link>
        {kind && kind.gpus.length > 1 && (
          <label className="mt-3 flex items-center gap-2 text-sm text-fg-2">
            GPU
            <select
              value={mainGpu ?? ""}
              onChange={(e) => onMainGpu(e.target.value === "" ? undefined : Number(e.target.value))}
              className="h-10 border border-border-strong bg-well px-2 text-sm text-fg"
            >
              <option value="">llama.cpp default (all GPUs)</option>
              {kind.gpus.map((g, i) => (
                <option key={i} value={i}>
                  {g.model}
                </option>
              ))}
            </select>
          </label>
        )}
      </fieldset>

      {worker && (
        <section aria-labelledby="h-ready" className="max-w-3xl">
          <h3 id="h-ready" className="mb-2 font-display text-lg font-semibold">
            {ready ? `${worker.displayName} is ready` : `${worker.displayName} isn't ready yet`}
          </h3>
          <dl className="m-0">
            {checks.map((c) => (
              <div key={c.label} className="grid grid-cols-[20px_150px_minmax(0,1fr)] items-baseline gap-x-3 border-t border-border py-2 text-sm">
                <dt className="contents">
                  <span aria-hidden="true" className={c.ok ? "text-online" : "text-danger"}>
                    {c.ok ? "✓" : "✕"}
                  </span>
                  <span className="text-muted">{c.label}</span>
                </dt>
                <dd className="m-0">
                  <span className="sr-only">{c.ok ? "OK: " : "Problem: "}</span>
                  {c.value}
                  {c.fix && <> · {c.fix}</>}
                </dd>
              </div>
            ))}
          </dl>
          {!ready && <p className="mt-2 text-sm text-muted">You can still pick a model and look at earlier results; measuring needs every line above ticked.</p>}
        </section>
      )}
    </div>
  );
}

// --- 2 Model ------------------------------------------------------------------

export function ModelStep({
  models,
  modelId,
  onPick,
  loading,
  verdictOf,
  workerName,
}: {
  models: Model[];
  modelId: string;
  onPick: (id: string) => void;
  loading: boolean;
  verdictOf: (m: Model) => Verdict | null;
  workerName: string;
}) {
  return (
    <div className="flex flex-col gap-6">
      <StepHeader title="Pick a model">Models already on {workerName}. The next step checks whether it fits before anything is loaded.</StepHeader>
      {models.length === 0 ? (
        <Notice>
          {loading ? "Loading the models on this machine…" : "No models on this machine yet."}{" "}
          <Link to="/models" className="text-accent">Find and download models</Link>
        </Notice>
      ) : (
        <div role="radiogroup" aria-label="Model" className="flex max-w-4xl flex-col">
          {models.map((m) => {
            const on = m.id === modelId;
            const mo = detectMoe(m.metadata.tensor_layer_bytes ?? null);
            const v = verdictOf(m);
            return (
              <button
                key={m.id}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => onPick(m.id)}
                className={`flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-l-2 border-b-border px-3.5 py-3 text-left first:border-t first:border-t-border ${
                  on ? "border-l-accent bg-surface-raised" : "border-l-transparent hover:bg-surface"
                }`}
              >
                <span className="min-w-0 flex-[1_1_260px]">
                  <span className="block truncate font-medium">{modelLabelOf(m)}</span>
                  <span className="block text-[13px] text-muted">
                    {(m.size_bytes / 1073741824).toFixed(1)} GB
                    {typeof m.metadata.n_layer === "number" ? ` · ${m.metadata.n_layer} layers` : ""}
                    {mo.isMoe ? " · mixture of experts" : " · dense"}
                    {typeof m.metadata.trained_ctx === "number" ? ` · up to ${ctxLabel(m.metadata.trained_ctx)} context` : ""}
                  </span>
                </span>
                {v && (
                  <span className={`inline-flex items-center gap-1.5 text-[13px] ${v.tone}`}>
                    <LtIcon name={v.icon} size={15} />
                    {v.text}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
      <p className="m-0 text-sm text-muted">
        Want another model? <Link to="/models" className="text-accent">Search Hugging Face on the Models page</Link>
      </p>
    </div>
  );
}

// --- 3 Fit --------------------------------------------------------------------

export function FitStep({
  model,
  stops,
  targetCtx,
  onCtx,
  trainedCtx,
  answer,
  verdict,
  pending,
  freeVramMib,
  ramTotalMib,
  repeats,
  onRepeats,
  machine,
}: {
  model: Model;
  stops: number[];
  targetCtx: number;
  onCtx: (c: number) => void;
  trainedCtx: number | null;
  answer: FitAnswer | null;
  verdict: Verdict | null;
  pending: boolean;
  freeVramMib: number | null;
  ramTotalMib: number | null;
  repeats: number;
  onRepeats: (n: number) => void;
  machine: FlowContext["machine"];
}) {
  const gpuShare = answer && freeVramMib ? Math.min(100, (answer.dev_used_mib / freeVramMib) * 100) : null;
  const ramShare = answer && ramTotalMib ? Math.min(100, (answer.host_used_mib / ramTotalMib) * 100) : null;
  return (
    <div className="flex flex-col gap-8">
      <StepHeader title="Will it fit?">
        How much context do you need? llama.cpp&apos;s own sizing answers in seconds, without loading {modelLabelOf(model)}.
      </StepHeader>
      <div>
        <Kicker className="mb-2">Target context</Kicker>
        <Seg label="Target context" value={targetCtx} onChange={onCtx} options={stops.map((s) => ({ value: s, label: ctxLabel(s) }))} />
        <p className="mt-1.5 text-[13px] text-muted">
          {trainedCtx ? `The model goes up to ${ctxLabel(trainedCtx)}. ` : ""}Every speed test after this runs at this context.
        </p>
      </div>

      <section aria-live="polite" className="max-w-3xl">
        {verdict ? (
          <p className={`m-0 flex items-start gap-2 font-display text-2xl font-semibold leading-snug ${verdict.tone}`}>
            <LtIcon name={verdict.icon} size={22} className="mt-1 flex-none" />
            <span>
              At {ctxLabel(targetCtx)}: {verdict.text}
            </span>
          </p>
        ) : (
          <p className="m-0 text-fg-2">{pending ? "Asking llama-fit-params on the machine…" : "No answer for this machine yet."}</p>
        )}
        {answer && machine === "dgpu" && (
          <div className="mt-5 flex flex-col gap-4">
            {gpuShare != null && (
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between gap-3 text-sm">
                  <span>GPU memory</span>
                  <span className="font-mono text-[13px]">
                    {fmtGb(answer.dev_used_mib)} of {fmtGb(freeVramMib)} usable
                  </span>
                </div>
                <div role="img" aria-label={`GPU memory ${fmtGb(answer.dev_used_mib)} of ${fmtGb(freeVramMib)} usable`} className="h-2.5 bg-surface-raised">
                  <div className="h-2.5 bg-series-a" style={{ width: `${gpuShare}%` }} />
                </div>
              </div>
            )}
            {ramShare != null && (
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between gap-3 text-sm">
                  <span>System RAM</span>
                  <span className="font-mono text-[13px]">
                    {fmtGb(answer.host_used_mib)} of {fmtGb(ramTotalMib)}
                  </span>
                </div>
                <div role="img" aria-label={`System RAM ${fmtGb(answer.host_used_mib)} of ${fmtGb(ramTotalMib)}`} className="h-2.5 bg-surface-raised">
                  <div className="h-2.5 bg-series-b" style={{ width: `${ramShare}%` }} />
                </div>
              </div>
            )}
            <p className="m-0 text-[13px] text-muted">
              {answer.source === "fit" ? "llama-fit-params on this machine" : "Estimated from the model's header until llama-fit-params answers"} · flash attention
              on, f16 cache. Step 5 tries other placements; step 4 first measures what llama.cpp does with no tuning.
            </p>
          </div>
        )}
      </section>

      <div className="max-w-md">
        <Kicker className="mb-2">Repeats per speed test</Kicker>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={SPEED_MIN_REPEATS}
            max={SPEED_MAX_REPEATS}
            value={repeats}
            onChange={(e) => onRepeats(Number(e.target.value))}
            aria-label="Repeats per speed test"
            className="w-40"
          />
          <span className="font-mono text-sm">{repeats}</span>
        </div>
        <p className="mt-1 text-[13px] text-muted">Fewer is faster, more is more accurate.{repeats === 1 ? " One run measures no spread." : ""}</p>
      </div>
    </div>
  );
}

// --- 4 Baseline ---------------------------------------------------------------

export function BaselineStep({ ctx, run }: { ctx: FlowContext; run: FlowRun | null }) {
  const outcome = speedRunView(run)?.rows[0]?.outcome ?? null;
  const done = outcome?.status === "done";
  return (
    <div className="flex flex-col gap-6">
      <StepHeader title="Baseline: what you get with no tuning">
        One real load with llama.cpp&apos;s defaults. Everything after this is compared against it.
      </StepHeader>
      <BaselineSection ctx={ctx} run={run} n="A" title={`Default settings at ${ctxLabel(ctx.targetCtx)}`} />
      {done && ctx.machine === "dgpu" && outcome.vramPeakMib != null && (
        <Panel>
          <StepTitle>Where the model&apos;s memory went</StepTitle>
          <MemorySplitBar dedicatedMib={outcome.vramPeakMib} sharedMib={spilledMib(outcome)} />
          {spilledMib(outcome) != null && (
            <p className="mt-3 text-sm text-fg-2">
              With every layer forced onto the GPU, part of the model didn&apos;t fit and the driver put it in shared memory, which the GPU reads over
              PCIe. Step 5 moves that part to the CPU on purpose instead.
            </p>
          )}
        </Panel>
      )}
    </div>
  );
}

// --- 5 Context and offload ----------------------------------------------------

export function OffloadStep({
  ctx,
  fit,
  fitRun,
  realRun,
  confirmRun,
  baselineRun,
  threadRun,
  threads,
  onThreads,
  candidates,
  candidateKeys,
  onAddCandidate,
  onRemoveCandidate,
  series,
}: {
  ctx: FlowContext;
  fit: FitData | null;
  fitRun: FlowRun | null;
  realRun: FlowRun | null;
  confirmRun: FlowRun | null;
  baselineRun: FlowRun | null;
  threadRun: FlowRun | null;
  threads: ThreadArgs | null;
  onThreads: (t: ThreadArgs) => void;
  candidates: SpeedCandidate[];
  candidateKeys: Set<string>;
  onAddCandidate: (c: SpeedCandidate) => void;
  onRemoveCandidate: (key: string) => void;
  series: FitSeries[];
}) {
  const placement = candidates[0] ?? null;
  return (
    <div className="flex flex-col gap-6">
      <StepHeader title="Context and offload">
        First map how memory grows with context, flash attention on and off, without loading the model. Then load the best placements for real at{" "}
        {ctxLabel(ctx.targetCtx)} and keep the fastest.
      </StepHeader>
      <FitMapSection
        ctx={ctx}
        fitRun={fitRun}
        fit={fit}
        kv={false}
        n="A"
        title="Map context and offload"
        onTrigger={async (payload) => {
          await ctx.trigger("fit", payload);
        }}
        onAddCandidate={onAddCandidate}
        candidateKeys={candidateKeys}
      />
      {series.length > 0 && fitRun && !isActive(fitRun.test) && ctx.machine === "dgpu" && (
        <Panel>
          <RealLoadSection ctx={ctx} series={series} run={realRun} onAddCandidate={onAddCandidate} />
        </Panel>
      )}
      <ConfirmSection
        ctx={ctx}
        n="B"
        title="Measure the best placements"
        candidates={candidates}
        onRemove={onRemoveCandidate}
        run={confirmRun}
        baselineRun={baselineRun}
        resultTitle="Keeping this placement"
      />
      {threads && (
        <details className="border-t border-border pt-4">
          <summary className="cursor-pointer text-sm text-fg-2">
            Advanced: CPU threads · -t {threads.t} -tb {threads.tb}
          </summary>
          <div className="mt-4">
            <ThreadsSection ctx={ctx} value={threads} onChange={onThreads} placement={placement ?? { ...defaultPlacement(), threads }} run={threadRun} />
          </div>
        </details>
      )}
    </div>
  );
}

function defaultPlacement(): SpeedCandidate {
  return { label: "Default settings", source: "baseline", ngl: 999, ot: null, ctk: "f16", ctv: "f16", fa: "on", threads: null };
}

// --- 6 KV cache ---------------------------------------------------------------

export function KvStep({
  ctx,
  kvFit,
  kvFitRun,
  kvRun,
  base,
  candidateKeys,
}: {
  ctx: FlowContext;
  /** Step 6's own KV map, or an older session's step-5 map that included KV. */
  kvFit: FitData | null;
  kvFitRun: FlowRun | null;
  kvRun: FlowRun | null;
  base: SpeedCandidate;
  candidateKeys: Set<string>;
}) {
  const series = useMemo(() => buildFitSeries(kvFit?.points ?? [], kvFit?.kv ?? []), [kvFit]);
  const options = useMemo(() => kvOptionsAt(series, ctx.targetCtx, ctx.moe), [series, ctx.targetCtx, ctx.moe]);
  const proposed = useMemo(() => proposeKvCandidates(options, base.threads ?? ctx.threads), [options, base.threads, ctx.threads]);
  const f16 = series.find((s) => s.kind === "default");
  const at = options[0]?.point.ctx ?? ctx.targetCtx;
  const f16Point = f16?.points.find((p) => p.ctx === at) ?? null;
  const sizeRows = [
    ...(f16Point ? [{ key: "f16", label: "f16 / f16", mib: f16Point.kv_mib, note: "default" }] : []),
    ...options.map((o) => ({
      key: o.seriesId,
      label: o.label,
      mib: o.kvMib,
      note: o.scoreGain > 0 ? "fits more on the GPU" : o.scoreGain < 0 ? "fits less on the GPU" : undefined,
    })),
  ];
  return (
    <div className="flex flex-col gap-6">
      <StepHeader title="KV cache types">
        A quantized cache takes less memory, which can leave room for more layers or a longer context, and can change speed. Only pairs this build runs on
        the GPU are tried, with K at least as precise as V.
      </StepHeader>
      <FitMapSection
        ctx={ctx}
        fitRun={kvFitRun}
        fit={kvFit}
        kv
        n="A"
        title="Map KV cache types"
        onTrigger={async (payload) => {
          await ctx.trigger("kv_fit", payload);
        }}
        candidateKeys={candidateKeys}
      />
      {sizeRows.length > 1 && (
        <Panel>
          <KvSizeBars rows={sizeRows} ctxLabel={ctxLabel(at)} />
          <p className="mt-3 text-[13px] text-muted">{KV_QUALITY_WARNING}</p>
        </Panel>
      )}
      <ConfirmSection
        ctx={ctx}
        step="kv_confirm"
        n="B"
        title="Measure the best caches"
        intro={
          <>
            The {proposed.length || "best"} cache size{proposed.length === 1 ? "" : "s"} that fit best at {ctxLabel(ctx.targetCtx)}, each loaded for real next
            to your step 5 config ({base.ctk} / {base.ctv}).
          </>
        }
        candidates={proposed}
        run={kvRun}
        baselineRun={null}
        reference={base}
        refName="the step 5 config"
        emptyText="Map the KV cache types first; the sizes that fit best are proposed here."
        resultTitle="Keeping this cache"
      />
    </div>
  );
}

// --- 7 Batch sizes ------------------------------------------------------------

export function BatchStep({ ctx, run, base }: { ctx: FlowContext; run: FlowRun | null; base: SpeedCandidate }) {
  const caps = new Set(ctx.worker.capabilities ?? []);
  const candidates = useMemo(() => batchCandidates(base).filter((c) => !(c.b === 2048 && c.ub === 512)), [base]);
  const reference = useMemo(() => {
    const { b: _b, ub: _ub, ...rest } = base;
    return { ...rest, label: `${base.label} · -b 2048 -ub 512` };
  }, [base]);
  const view = speedRunView(run);
  const final = carryForward(view, "the default batch sizes")?.candidate ?? base;
  return (
    <div className="flex flex-col gap-6">
      <StepHeader title="Batch sizes">
        -b and -ub set how many prompt tokens are processed at once. They change prompt speed and VRAM; generation speed stays about the same.
      </StepHeader>
      <ConfirmSection
        ctx={ctx}
        step="batch"
        n="A"
        title="Measure micro-batch sizes"
        intro={<>Your config at -ub 256, 1024 and 2048, next to the default -b 2048 -ub 512, each loaded for real at {ctxLabel(ctx.targetCtx)}.</>}
        candidates={candidates}
        run={run}
        baselineRun={null}
        reference={reference}
        refName="the default batch sizes"
        disabledReason={caps.has("speed-batch-v1") ? null : `${ctx.worker.displayName} runs an older worker that can't measure batch sizes — update it on the Machines page.`}
        resultTitle="Best batch sizes"
      />
      {view && !isActive(run!.test) && (
        <Panel accent>
          <Kicker className="text-accent">Your config for {ctxLabel(ctx.targetCtx)}</Kicker>
          <p className="mt-2 text-sm text-fg-2">
            {ctx.modelLabel} on {ctx.worker.displayName}: {final.label}.
          </p>
          <div className="mt-3">
            <CopyCommand command={serverCommandLine(ctx.modelFile, ctx.targetCtx, final)} />
          </div>
          <div className="mt-3 flex flex-wrap gap-4 text-sm">
            <Link to="/serve" className="text-accent hover:text-accent-hover">
              Open Serve
            </Link>
            <Link to="/tests" className="text-accent hover:text-accent-hover">
              See every run of this session
            </Link>
          </div>
        </Panel>
      )}
    </div>
  );
}
