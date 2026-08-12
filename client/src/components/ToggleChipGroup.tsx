interface ToggleChipGroupProps {
  label: string;
  hint?: string;
  options: string[];
  value: string[];
  onChange: (value: string[]) => void;
}

// Graphical replacement for a raw JSON string-enum array field (cache
// types, flash_attn): every valid value is always visible as a toggle, so
// there's nothing to type or mistype.
export function ToggleChipGroup({ label, hint, options, value, onChange }: ToggleChipGroupProps) {
  function toggle(opt: string) {
    onChange(value.includes(opt) ? value.filter((v) => v !== opt) : [...value, opt]);
  }

  return (
    <div className="min-w-0 max-w-[260px]">
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <label className="text-sm font-medium text-fg">{label}</label>
        {hint && <span className="text-xs text-muted">{hint}</span>}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {options.map((opt) => {
          const active = value.includes(opt);
          return (
            <button
              type="button"
              key={opt}
              onClick={() => toggle(opt)}
              aria-pressed={active}
              className={`rounded-md border px-2.5 py-1 text-sm font-medium transition-colors ${
                active
                  ? "border-accent/40 bg-accent/15 text-accent"
                  : "border-border bg-surface text-muted hover:border-accent/40 hover:text-fg"
              }`}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}
