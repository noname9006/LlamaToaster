import type { DaemonMemorySide, DaemonMemorySnapshot, DaemonStatus } from "../../../../shared/daemonTypes";
import { LtIcon } from "../ltIcons";
import { Kicker, Panel, StepTitle } from "../optimize/ui";
import { formatMib, withRemainder, type BreakdownPart } from "./serveData";

// Validated with the dataviz palette script against --color-surface #1b1611: all
// six checks pass. Free stays neutral: it is the absence of a thing, not a third series.
const APP = "#d95926";
const OTHER = "#3987e5";

function Segment({ mib, total, color, title }: { mib: number; total: number; color: string; title: string }) {
  if (mib <= 0) return null;
  return <div title={title} style={{ width: `${(mib / total) * 100}%`, background: color, minWidth: 2 }} className="h-full" />;
}

function Legend({ color, label, value, outline }: { color?: string; label: string; value: string; outline?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span aria-hidden className={`inline-block h-2.5 w-2.5 ${outline ? "border border-border-strong" : ""}`} style={color ? { background: color } : undefined} />
      <span className="text-fg-2">{label}</span>
      <span className="font-mono text-fg">{value}</span>
    </div>
  );
}

function Bar({ title, side, parts, sourceNote }: { title: string; side: DaemonMemorySide; parts: BreakdownPart[]; sourceNote?: string }) {
  const total = side.total ?? 0;
  // With no per-process reading, everything in use is one unattributed block.
  const unattributed = side.app == null && side.free != null ? total - side.free : null;
  const label =
    `${title}: ${formatMib(side.total)} total. ` +
    (unattributed != null ? `${formatMib(unattributed)} in use, not split by app. ` : `This app ${formatMib(side.app)}, other apps ${formatMib(side.other)}. `) +
    `${formatMib(side.free)} free.`;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <Kicker>{title}</Kicker>
        <span className="font-mono text-xs text-muted">{formatMib(side.total)}</span>
      </div>
      {total > 0 ? (
        <div role="img" aria-label={label} className="flex h-4 gap-0.5 border border-border bg-well">
          {unattributed != null ? (
            <Segment mib={unattributed} total={total} color="#7d7264" title={`In use: ${formatMib(unattributed)}`} />
          ) : (
            <>
              <Segment mib={side.app ?? 0} total={total} color={APP} title={`This app: ${formatMib(side.app)}`} />
              <Segment mib={side.other ?? 0} total={total} color={OTHER} title={`Other apps: ${formatMib(side.other)}`} />
            </>
          )}
        </div>
      ) : (
        <p className="m-0 text-sm text-muted">No reading from this machine.</p>
      )}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
        {unattributed != null ? (
          <Legend color="#7d7264" label="In use" value={formatMib(unattributed)} />
        ) : (
          <>
            <Legend color={APP} label="This app" value={formatMib(side.app)} />
            <Legend color={OTHER} label="Other apps" value={formatMib(side.other)} />
          </>
        )}
        <Legend outline label="Free" value={formatMib(side.free)} />
      </div>
      {parts.length > 0 && (
        <p className="m-0 text-xs text-muted">
          This app: {parts.map((p) => `${p.label} ${formatMib(p.mib)}`).join(" · ")}
        </p>
      )}
      {sourceNote && <p className="m-0 text-xs text-muted">{sourceNote}</p>}
    </div>
  );
}

function vramParts(m: DaemonMemorySnapshot): BreakdownPart[] {
  return withRemainder(
    [
      { label: "weights", mib: m.breakdown.weights_gpu },
      { label: "KV cache", mib: m.breakdown.kv_gpu },
      { label: "compute", mib: m.breakdown.compute_gpu },
    ],
    m.vram?.app ?? null,
    "runtime"
  );
}

function ramParts(m: DaemonMemorySnapshot): BreakdownPart[] {
  const unified = m.unified;
  return withRemainder(
    [
      { label: "weights", mib: (m.breakdown.weights_host ?? 0) + (unified ? m.breakdown.weights_gpu ?? 0 : 0) || null },
      ...(unified ? [{ label: "KV cache", mib: m.breakdown.kv_gpu }] : []),
      { label: "compute", mib: (m.breakdown.compute_host ?? 0) + (unified ? m.breakdown.compute_gpu ?? 0 : 0) || null },
      { label: "prompt cache", mib: m.breakdown.prompt_cache },
      { label: "worker", mib: m.breakdown.worker },
    ],
    m.ram.app,
    // Live, this was 865 MiB next to 864 MiB of weights on the GPU: mostly the
    // memory-mapped model file, which the working set counts.
    "mapped file and runtime"
  );
}

export function MemoryPanel({ status, machineBusy }: { status: DaemonStatus | null; machineBusy: boolean }) {
  const m = status?.memory ?? null;
  return (
    <Panel>
      <StepTitle>Memory</StepTitle>
      {!m ? (
        <p className="m-0 text-sm text-muted">
          {machineBusy ? "Paused while a test runs on this machine, because reading memory would disturb its timing." : "Waiting for the first reading from this machine."}
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          {m.spill.spilled && (
            <p className="m-0 flex items-start gap-2 border border-warning bg-warning-bg px-3 py-2 text-sm text-warning">
              <LtIcon name="warn" className="mt-0.5 shrink-0" />
              <span>
                About {formatMib(m.spill.in_system_ram_mib)} of the GPU's buffers are in system RAM instead of VRAM. Lower GPU layers or context length to bring them back.
              </span>
            </p>
          )}
          {m.unified ? (
            <Bar title={`Unified memory${m.gpu_name ? ` · ${m.gpu_name}` : ""}`} side={m.ram} parts={ramParts(m)} sourceNote="CPU and GPU share this memory." />
          ) : (
            <>
              {m.vram ? (
                <Bar
                  title={`VRAM${m.gpu_name ? ` · ${m.gpu_name}` : ""}`}
                  side={m.vram}
                  parts={vramParts(m)}
                  sourceNote={
                    [m.vram.app == null ? "This backend gives no per-app reading." : null, m.gpu_count > 1 ? `This machine has ${m.gpu_count} GPUs, and these figures may combine them.` : null]
                      .filter(Boolean)
                      .join(" ") || undefined
                  }
                />
              ) : (
                <div>
                  <Kicker>VRAM</Kicker>
                  <p className="m-0 mt-1 text-sm text-muted">No GPU memory reading on this backend.</p>
                </div>
              )}
              <Bar title="RAM" side={m.ram} parts={ramParts(m)} sourceNote="This app's RAM is its working set, which can include weights the OS is able to drop." />
            </>
          )}
        </div>
      )}
    </Panel>
  );
}
