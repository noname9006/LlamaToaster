// The v2 design's stroke glyphs (docs/plans/app-v2.dc.html), one path each so
// they render identically at every size.
import type { SVGProps } from "react";

export const LT_ICON_PATHS = {
  grid: "M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z",
  plus: "M3 3h18v18H3zM12 8v8M8 12h8",
  list: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  bars: "M18 20V10M12 20V4M6 20v-6",
  box: "M21 8l-9-5-9 5v8l9 5 9-5zM3 8l9 5 9-5M12 13v8",
  server: "M3 4h18v6H3zM3 14h18v6H3zM7 7h.01M7 17h.01",
  sliders: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6",
  ok: "M20 6L9 17l-5-5",
  oom: "M12 2l9 5v10l-9 5-9-5V7zM9 9l6 6M15 9l-6 6",
  spill: "M12 3v12M7 10l5 5 5-5M4 21h16",
  cpu: "M5 5h14v14H5zM9 9h6v6H9zM9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4",
  unstable: "M22 12h-4l-3 9L9 3l-3 9H2",
  warn: "M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0zM12 9v4M12 17h.01",
  run: "M12 6v6l4 2M12 2a10 10 0 100 20 10 10 0 000-20z",
  queue: "M8 6h13M8 12h13M8 18h13M3 6l1.5 1.5L3 9",
  chip: "M5 5h14v14H5zM9 9h6v6H9z",
  copy: "M8 8h12v12H8zM4 16V4h12",
  menu: "M4 6h16M4 12h16M4 18h16",
  offline: "M2 2l20 20M8.5 16.5a5 5 0 017 0M5 12.9a10 10 0 015.2-2.7M19 12.9a10 10 0 00-2.4-1.7M12 20h.01",
  chevron: "M9 6l6 6-6 6",
  stop: "M6 6h12v12H6z",
} as const;

export type LtIconName = keyof typeof LT_ICON_PATHS;

export function LtIcon({ name, size = 18, ...props }: { name: LtIconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d={LT_ICON_PATHS[name]} />
    </svg>
  );
}
