import type { FitMarkState } from "./useModelFit";
import { FIT_MARKS, fitBreadGrade, fitBreadHeadline, fitHeadline } from "./fitCopy";
import { useBreadSettings, type BreadSettings } from "./breadSettings";
import { gradeSvg, uncheckedSvg } from "../../../../shared/breadMarks.js";
import type { ModelFitResult } from "../../types";

const MARK_PX = 14;

type Mark = { glyph: string; label: string } | { svg: string; label: string };

function markOf(state: FitMarkState, bread: BreadSettings): Mark {
  switch (state.kind) {
    case "unknown":
      return bread.enabled
        ? { svg: uncheckedSvg(MARK_PX), label: "Will it fit? Click to check" }
        : { glyph: "❓", label: "Will it fit? Click to check" };
    case "pending":
      return bread.enabled
        ? { svg: uncheckedSvg(MARK_PX), label: "Checking whether it fits" }
        : { glyph: "⏳", label: "Checking whether it fits" };
    case "lost":
      return { glyph: "⚠️", label: "The machine gave no answer. Click to try again" };
    case "answered": {
      const { result } = state;
      if (!result.level) return { glyph: "⚠️", label: result.error ?? fitHeadline(result) };
      if (bread.enabled) {
        const svg = breadSvg(result, bread, MARK_PX);
        if (svg) return { svg, label: fitBreadHeadline(result) };
      }
      return { glyph: FIT_MARKS[result.level], label: fitHeadline(result) };
    }
  }
}

/** The bread stack for an answer; the app is dark only. */
export function breadSvg(result: ModelFitResult, bread: BreadSettings, px: number): string | null {
  const g = fitBreadGrade(result);
  return g ? gradeSvg(g.scale, g.grade, { px, style: bread.style, dark: true }) : null;
}

// A <button>, never nested in a quant badge's link: it sits beside it.
export function FitMark({ state, onClick, subject }: { state: FitMarkState; onClick: () => void; subject: string }) {
  const bread = useBreadSettings();
  const mark = markOf(state, bread);
  const pulse = state.kind === "pending" ? "animate-pulse" : undefined;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={state.kind === "pending"}
      title={mark.label}
      aria-label={`${subject}: ${mark.label}`}
      className="inline-flex shrink-0 items-center rounded px-0.5 text-xs leading-none hover:bg-surface-raised disabled:cursor-default"
    >
      {"svg" in mark ? (
        // Built from constants in shared/breadMarks.ts, never from user input.
        <span className={`inline-flex items-center ${pulse ?? ""}`} dangerouslySetInnerHTML={{ __html: mark.svg }} />
      ) : (
        <span className={pulse}>{mark.glyph}</span>
      )}
    </button>
  );
}
