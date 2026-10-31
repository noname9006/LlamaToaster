import type { FitMarkState } from "./useModelFit";
import { FIT_MARKS, fitHeadline } from "./fitCopy";

function markOf(state: FitMarkState): { glyph: string; label: string } {
  switch (state.kind) {
    case "unknown":
      return { glyph: "❓", label: "Will it fit? Click to check" };
    case "pending":
      return { glyph: "⏳", label: "Checking whether it fits" };
    case "lost":
      return { glyph: "⚠️", label: "The machine gave no answer. Click to try again" };
    case "answered": {
      const { result } = state;
      return result.level ? { glyph: FIT_MARKS[result.level], label: fitHeadline(result) } : { glyph: "⚠️", label: result.error ?? fitHeadline(result) };
    }
  }
}

// A <button>, never nested in a quant badge's link: it sits beside it.
export function FitMark({ state, onClick, subject }: { state: FitMarkState; onClick: () => void; subject: string }) {
  const { glyph, label } = markOf(state);
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={state.kind === "pending"}
      title={label}
      aria-label={`${subject}: ${label}`}
      className="inline-flex shrink-0 items-center rounded px-0.5 text-xs leading-none hover:bg-surface-raised disabled:cursor-default"
    >
      <span className={state.kind === "pending" ? "animate-pulse" : undefined}>{glyph}</span>
    </button>
  );
}
