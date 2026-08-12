import { useEffect, useState, type ReactNode } from "react";
import { IconX } from "./icons";

interface SliderChipInputProps {
  label: string;
  hint?: string;
  value: number[];
  onChange: (value: number[]) => void;
  min: number;
  max: number;
  // Recommended value for this exact model/worker pairing -- the slider
  // jumps here whenever it changes, but only Add commits it to the sweep.
  suggested: number;
  suggestedLabel: string;
  disabled?: boolean;
  disabledNote?: string;
  // Optional extra content under the suggestion note (e.g. a "look this up"
  // action) -- kept as a slot rather than baked in, since what's worth
  // offering there is specific to each field's caller, not this component.
  footer?: ReactNode;
}

// Same chip-list-of-swept-values idea as ChipInput, but for fields whose
// legal range is dictated by runtime context (a worker's thread count, a
// model's real layer count) rather than free-form: a slider enforces the
// upper bound directly instead of accepting any typed number.
export function SliderChipInput({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  suggested,
  suggestedLabel,
  disabled,
  disabledNote,
  footer,
}: SliderChipInputProps) {
  const [draft, setDraft] = useState(suggested);

  // Jump to the freshly computed suggestion whenever the context that
  // produced it changes (a different model or worker picked) -- but only
  // then, so it never fights the user while they're dragging.
  useEffect(() => {
    setDraft(suggested);
  }, [suggested]);

  function clamp(n: number): number {
    if (!Number.isFinite(n)) return min;
    return Math.min(max, Math.max(min, Math.round(n)));
  }

  function add() {
    const n = clamp(draft);
    if (!value.includes(n)) onChange([...value, n]);
  }

  function remove(n: number) {
    onChange(value.filter((x) => x !== n));
  }

  return (
    <div className="min-w-0 max-w-[260px]">
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <label className="text-sm font-medium text-fg">{label}</label>
        {hint && <span className="text-xs text-muted">{hint}</span>}
      </div>
      <div className="mt-1.5 flex min-h-10 flex-wrap items-center gap-1.5 rounded-lg border border-border bg-surface px-2 py-1.5">
        {value.length === 0 && <span className="px-1 text-sm text-muted">No values added yet</span>}
        {value
          .slice()
          .sort((a, b) => a - b)
          .map((n) => (
            <span
              key={n}
              className="flex items-center gap-1 rounded-md bg-accent/15 px-2 py-0.5 text-sm text-accent"
            >
              {n}
              <button
                type="button"
                onClick={() => remove(n)}
                className="text-accent/70 hover:text-accent"
                aria-label={`Remove ${n}`}
              >
                <IconX width={12} height={12} />
              </button>
            </span>
          ))}
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          value={clamp(draft)}
          disabled={disabled}
          onChange={(e) => setDraft(Number(e.target.value))}
          className="min-w-[80px] flex-1 disabled:opacity-40"
        />
        <input
          type="number"
          min={min}
          max={max}
          value={draft}
          disabled={disabled}
          onChange={(e) => setDraft(Number(e.target.value))}
          onBlur={() => setDraft((d) => clamp(d))}
          className="w-16 rounded-lg border border-border bg-surface px-2 py-1 text-center text-sm text-fg outline-none focus:border-accent disabled:opacity-40"
        />
        <button
          type="button"
          disabled={disabled}
          onClick={add}
          className="rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-40"
        >
          Add
        </button>
      </div>
      <p className="mt-1 text-xs text-muted">{disabled && disabledNote ? disabledNote : suggestedLabel}</p>
      {footer}
    </div>
  );
}
