import { useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

const TOOLTIP_MARGIN = 8;
const TOOLTIP_MAX_WIDTH = 260;

// Portal-based (renders into document.body, `position: fixed`) rather than an
// in-flow `absolute` node -- table headers live inside `overflow-x-auto`
// wrappers (see Runs.tsx/RunDetail.tsx), and setting overflow-x alone makes
// the browser treat overflow-y as `auto` too (CSS Overflow spec: an axis left
// `visible` while the other isn't gets promoted to `auto`), which would clip
// or scroll-hide an in-flow tooltip that pops above/below its cell.
export function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const ref = useRef<HTMLSpanElement>(null);

  function show() {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const left = Math.min(
      Math.max(TOOLTIP_MARGIN, rect.left),
      window.innerWidth - TOOLTIP_MAX_WIDTH - TOOLTIP_MARGIN
    );
    setPos({ top: rect.bottom + 6, left });
  }

  function hide() {
    setPos(null);
  }

  return (
    <span
      ref={ref}
      onMouseEnter={show}
      onMouseLeave={hide}
      className="cursor-help decoration-dotted underline-offset-2 hover:underline"
    >
      {children}
      {pos &&
        createPortal(
          <div
            role="tooltip"
            className="pointer-events-none fixed z-50 rounded-md border border-border bg-surface-raised px-2.5 py-1.5 text-xs font-normal normal-case leading-snug text-fg shadow-lg"
            style={{ top: pos.top, left: pos.left, maxWidth: TOOLTIP_MAX_WIDTH }}
          >
            {text}
          </div>,
          document.body
        )}
    </span>
  );
}
