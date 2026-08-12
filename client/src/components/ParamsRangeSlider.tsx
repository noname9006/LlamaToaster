import type { ChangeEvent } from "react";

// Two-handle range filter over a fixed set of parameter-count stops (billions
// of params), replacing the old single-select "Parameters" dropdown on the
// Models page. Standard two-overlapping-native-<input type=range> technique:
// each input's own track is hidden (pointer-events: none, transparent track)
// so only its thumb is interactive, and a separate div underneath draws the
// visible track + highlighted selected span -- see the `.params-range-slider`
// rules in styles/index.css.
export const PARAMS_STOPS = [0, 1, 3, 6, 9, 12, 24, 32, Infinity];
export const PARAMS_LABELS = ["<1B", "1B", "3B", "6B", "9B", "12B", "24B", "32B", ">32B"];

export function paramsRangeValues(loIndex: number, hiIndex: number): [number, number] {
  return [PARAMS_STOPS[loIndex], PARAMS_STOPS[hiIndex]];
}

// Whole-range selection (index 0..last) means "no filter" -- matches the old
// dropdown's "All" option, including models with an unknown paramsB. Any
// narrower selection excludes unknown-paramsB models, since there's nothing
// to compare against a chosen range.
export function paramsInRange(paramsB: number | null, loIndex: number, hiIndex: number): boolean {
  if (loIndex === 0 && hiIndex === PARAMS_STOPS.length - 1) return true;
  if (paramsB === null) return false;
  const [lo, hi] = paramsRangeValues(loIndex, hiIndex);
  return paramsB >= lo && paramsB <= hi;
}

export function ParamsRangeSlider({
  loIndex,
  hiIndex,
  onChange,
}: {
  loIndex: number;
  hiIndex: number;
  onChange: (loIndex: number, hiIndex: number) => void;
}) {
  const max = PARAMS_STOPS.length - 1;

  function handleLo(e: ChangeEvent<HTMLInputElement>) {
    onChange(Math.min(Number(e.target.value), hiIndex), hiIndex);
  }
  function handleHi(e: ChangeEvent<HTMLInputElement>) {
    onChange(loIndex, Math.max(Number(e.target.value), loIndex));
  }

  const loPct = (loIndex / max) * 100;
  const hiPct = (hiIndex / max) * 100;

  return (
    <div className="w-64">
      <div className="params-range-slider relative h-4">
        <div className="absolute left-0 right-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-border" />
        <div
          className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent"
          style={{ left: `${loPct}%`, right: `${100 - hiPct}%` }}
        />
        <input type="range" min={0} max={max} step={1} value={loIndex} onChange={handleLo} aria-label="Minimum parameters" />
        <input type="range" min={0} max={max} step={1} value={hiIndex} onChange={handleHi} aria-label="Maximum parameters" />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted">
        {PARAMS_LABELS.map((label, i) => (
          <span key={label} className={i === loIndex || i === hiIndex ? "font-semibold text-fg" : undefined}>
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
