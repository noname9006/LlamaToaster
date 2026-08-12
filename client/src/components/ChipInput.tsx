import { useState, type KeyboardEvent } from "react";
import { IconX } from "./icons";

interface ChipInputProps {
  label: string;
  hint?: string;
  value: number[];
  onChange: (value: number[]) => void;
  presets?: number[];
  placeholder?: string;
}

// Graphical replacement for a raw JSON array field (n_prompt, threads, ...):
// type a number, Enter/comma/blur commits it as a removable chip. Duplicate
// and non-finite values are silently ignored rather than rejected with an
// error -- there's no invalid state to explain, just nothing to add.
export function ChipInput({ label, hint, value, onChange, presets, placeholder }: ChipInputProps) {
  const [draft, setDraft] = useState("");

  function commit() {
    const parts = draft
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!parts.length) return;
    const additions = parts
      .map((p) => Number(p))
      .filter((n) => Number.isFinite(n) && !value.includes(n));
    if (additions.length) onChange([...value, ...additions]);
    setDraft("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      commit();
    } else if (e.key === "Backspace" && !draft && value.length) {
      onChange(value.slice(0, -1));
    }
  }

  function remove(n: number) {
    onChange(value.filter((x) => x !== n));
  }

  function togglePreset(n: number) {
    onChange(value.includes(n) ? value.filter((x) => x !== n) : [...value, n]);
  }

  return (
    <div className="min-w-0 max-w-[260px]">
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <label className="text-sm font-medium text-fg">{label}</label>
        {hint && <span className="text-xs text-muted">{hint}</span>}
      </div>
      <div className="mt-1.5 flex min-h-10 flex-wrap items-center gap-1.5 rounded-lg border border-border bg-surface px-2 py-1.5 focus-within:border-accent">
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
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commit}
          placeholder={value.length ? "" : placeholder ?? "type a number, Enter to add"}
          className="min-w-16 flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-muted"
          inputMode="decimal"
        />
      </div>
      {presets && presets.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {presets.map((n) => (
            <button
              type="button"
              key={n}
              onClick={() => togglePreset(n)}
              className={`rounded-md border px-2 py-0.5 text-xs transition-colors ${
                value.includes(n)
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-border text-muted hover:border-accent/40 hover:text-fg"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
