import type { MemorySpeedResult, MemorySpeedThreadPoint } from "../types";
import { formatDate } from "../utils";
import { Tooltip } from "./Tooltip";

// Workers page "Measure memory speed" button + result table -- shows the
// hardware ceiling this machine's RAM and (Windows/OpenCL only) VRAM can
// move bytes at, distinct from any model's actual tok/s. See
// shared/types.ts's MemorySpeedResult and worker/src/memSpeed.ts.

function gbs(n: number): string {
  return `${n.toFixed(1)} GB/s`;
}

// Index of the row holding the highest value of one column -- used to
// highlight the best read/write/copy cell, which need not be the same row
// (read and write typically peak at different thread counts).
function bestIdx(points: MemorySpeedThreadPoint[], key: keyof Omit<MemorySpeedThreadPoint, "threads">): number {
  let best = 0;
  for (let i = 1; i < points.length; i++) if (points[i][key] > points[best][key]) best = i;
  return best;
}

const BEST_CELL_CLASS = "font-semibold text-accent";

function RamTable({ ram }: { ram: MemorySpeedResult["ram"] }) {
  const { points } = ram;
  if (points.length === 0) return null;
  const bestRead = bestIdx(points, "readGBs");
  const bestWrite = bestIdx(points, "writeGBs");
  const bestCopy = bestIdx(points, "copyGBs");

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border text-xs uppercase tracking-wide text-muted">
            <th className="px-3 py-2 font-medium whitespace-nowrap">Threads</th>
            <th className="px-3 py-2 font-medium whitespace-nowrap">Read</th>
            <th className="px-3 py-2 font-medium whitespace-nowrap">Write</th>
            <Tooltip text="Counts bytes read plus bytes written -- not directly comparable to the Read/Write columns, which each count only their own direction.">
              <th className="cursor-help px-3 py-2 font-medium whitespace-nowrap underline decoration-dotted underline-offset-2">
                Copy (r+w)
              </th>
            </Tooltip>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {points.map((p, i) => (
            <tr key={p.threads}>
              <td className="px-3 py-1.5 text-fg">
                {p.threads}
                {ram.physicalCores !== null && p.threads === ram.physicalCores && (
                  <span className="ml-1 text-xs text-muted">(physical cores)</span>
                )}
                {p.threads === ram.logicalCores && p.threads !== ram.physicalCores && (
                  <span className="ml-1 text-xs text-muted">(logical)</span>
                )}
              </td>
              <td className={`px-3 py-1.5 ${i === bestRead ? BEST_CELL_CLASS : "text-fg"}`}>{gbs(p.readGBs)}</td>
              <td className={`px-3 py-1.5 ${i === bestWrite ? BEST_CELL_CLASS : "text-fg"}`}>{gbs(p.writeGBs)}</td>
              <td className={`px-3 py-1.5 ${i === bestCopy ? BEST_CELL_CLASS : "text-fg"}`}>{gbs(p.copyGBs)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VramTable({ vram }: { vram: NonNullable<MemorySpeedResult["vram"]> }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border text-xs uppercase tracking-wide text-muted">
            <th className="px-3 py-2 font-medium whitespace-nowrap">Device</th>
            <th className="px-3 py-2 font-medium whitespace-nowrap">Read</th>
            <th className="px-3 py-2 font-medium whitespace-nowrap">Write</th>
            <Tooltip text="Counts bytes read plus bytes written -- not directly comparable to the Read/Write columns, which each count only their own direction.">
              <th className="cursor-help px-3 py-2 font-medium whitespace-nowrap underline decoration-dotted underline-offset-2">
                Copy (r+w)
              </th>
            </Tooltip>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-3 py-1.5 text-fg">
              {vram.deviceName} <span className="text-muted">({(vram.globalMemMiB / 1024).toFixed(1)} GiB)</span>
            </td>
            <td className="px-3 py-1.5 font-semibold text-accent">{gbs(vram.readGBs)}</td>
            <td className="px-3 py-1.5 text-fg">{gbs(vram.writeGBs)}</td>
            <td className="px-3 py-1.5 text-fg">{gbs(vram.copyGBs)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export function MemorySpeedPanel({
  result,
  onMeasure,
  queuing,
  disabled,
}: {
  result: MemorySpeedResult | null | undefined;
  onMeasure: () => void;
  // True only for the brief window between clicking the button and the
  // server accepting the job (queuing.ts's withBusy) -- the sweep itself
  // takes several seconds and shows up as this worker's normal busy/
  // BusyPhaseCard state elsewhere on the card (phase "benchmarking", detail
  // like "measuring RAM bandwidth (4 threads)…"), not as a second spinner
  // here.
  queuing: boolean;
  disabled: boolean;
}) {
  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">Memory speed</h4>
        <div className="flex items-center gap-2">
          {result && <span className="text-xs text-muted">measured {formatDate(result.measuredAt)}</span>}
          <button
            type="button"
            onClick={onMeasure}
            disabled={disabled || queuing}
            className="flex-none rounded-lg border border-border px-2.5 py-1 text-xs font-semibold text-fg hover:border-accent/50 hover:text-accent disabled:opacity-50"
            title="Sweep RAM read/write/copy bandwidth across several thread counts, plus VRAM bandwidth where available -- a few seconds, on demand."
          >
            {queuing ? "Queuing…" : result ? "Re-measure" : "Measure memory speed"}
          </button>
        </div>
      </div>

      {!result && (
        <p className="mt-1.5 text-sm text-muted">
          Not measured yet -- click "Measure memory speed" to sweep RAM read/write/copy bandwidth (and VRAM, where
          available).
        </p>
      )}

      {result && (
        <div className="mt-2 space-y-3">
          <div>
            <p className="mb-1 text-xs text-muted">
              RAM · {result.ram.logicalCores} logical
              {result.ram.physicalCores !== null && `, ${result.ram.physicalCores} physical`} core
              {result.ram.logicalCores === 1 ? "" : "s"}
            </p>
            <RamTable ram={result.ram} />
          </div>
          <div>
            <p className="mb-1 text-xs text-muted">VRAM</p>
            {result.vram ? (
              <VramTable vram={result.vram} />
            ) : (
              <p className="text-sm text-muted">{result.vramUnavailableReason ?? "Not available on this machine."}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
