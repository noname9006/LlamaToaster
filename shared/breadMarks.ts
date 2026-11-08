// The bread fit marks: slices of toast from pale to golden (and burnt),
// stacked to show a grade (shared/fitGrade.ts). Plain SVG strings, so the
// React client and the extension's content script draw the same thing.
// Designed on bread-marks-preview.html (LlamaToaster/extension repo); the operator tunes the
// style from the admin console (AppSettings.breadMarkStyle).

import type { FitGrade, FitScale } from "./fitGrade.js";

export type GoldenStyle = "shine" | "butter" | "plain";
/** Light shapes look bigger on a dark page; how the pale slices make up for it. */
export type DarkFix = "shrink" | "crust" | "both" | "none";
export type StackDirection = "ltr" | "rtl";

export interface BreadMarkStyle {
  golden: GoldenStyle;
  darkFix: DarkFix;
  /** Average gap between slices, in slice units (a slice is 12 wide). */
  step: number;
  /** Each gap is this fraction of the one in front of it; 1 = even gaps. */
  falloff: number;
  /** Which way doneness grows; the most cooked slice is always on top. */
  direction: StackDirection;
}

export const DEFAULT_BREAD_MARK_STYLE: BreadMarkStyle = {
  golden: "shine",
  darkFix: "shrink",
  step: 3.5,
  falloff: 0.8,
  direction: "ltr",
};

export const BREAD_STEP_RANGE = { min: 2.5, max: 7 } as const;
export const BREAD_FALLOFF_RANGE = { min: 0.5, max: 1 } as const;

/** Single check for every boundary that takes a style from outside (admin
 * POST, the meta-table read). */
export function isBreadMarkStyle(v: unknown): v is BreadMarkStyle {
  if (!v || typeof v !== "object") return false;
  const s = v as Record<string, unknown>;
  return (
    (s.golden === "shine" || s.golden === "butter" || s.golden === "plain") &&
    (s.darkFix === "shrink" || s.darkFix === "crust" || s.darkFix === "both" || s.darkFix === "none") &&
    typeof s.step === "number" &&
    Number.isFinite(s.step) &&
    s.step >= BREAD_STEP_RANGE.min &&
    s.step <= BREAD_STEP_RANGE.max &&
    typeof s.falloff === "number" &&
    Number.isFinite(s.falloff) &&
    s.falloff >= BREAD_FALLOFF_RANGE.min &&
    s.falloff <= BREAD_FALLOFF_RANGE.max &&
    (s.direction === "ltr" || s.direction === "rtl")
  );
}

/** 1..4 pale to deep, 5 golden (set apart by saturation), 0 burnt. */
export type Shade = 0 | 1 | 2 | 3 | 4 | 5;

const SLICE =
  "M2,13 L2,6.4 C0.4,5.8 0.3,1.2 6,1.2 C11.7,1.2 11.6,5.8 10,6.4 L10,13 Q10,13.7 9.3,13.7 L2.7,13.7 Q2,13.7 2,13 Z";

const SHADES: Record<Shade, { fill: string; crust: string }> = {
  1: { fill: "#F7EACB", crust: "#C9A469" },
  2: { fill: "#F1D9A2", crust: "#BC8F4C" },
  3: { fill: "#E8C281", crust: "#AA7A3A" },
  4: { fill: "#DDAA62", crust: "#9C6A2C" },
  5: { fill: "#F0A21E", crust: "#9A5A0C" },
  0: { fill: "#2A1C14", crust: "#B8865E" },
};
const DARK_CRUST: Partial<Record<Shade, string>> = { 1: "#6E5130", 2: "#74532B", 3: "#755027" };
const DARK_SCALE: Partial<Record<Shade, number>> = { 1: 0.9, 2: 0.93, 3: 0.96, 4: 0.98 };

function slice(x: number, s: Shade, style: BreadMarkStyle, dark: boolean): string {
  const c = SHADES[s];
  const crustFix = dark && (style.darkFix === "crust" || style.darkFix === "both") && DARK_CRUST[s] != null;
  const shrink = dark && (style.darkFix === "shrink" || style.darkFix === "both") ? DARK_SCALE[s] ?? 1 : 1;
  const crust = crustFix ? DARK_CRUST[s]! : c.crust;
  const width = crustFix ? 1.6 : 1.25;
  const sc = shrink !== 1 ? ` translate(6,7.8) scale(${shrink}) translate(-6,-7.8)` : "";
  let extra = "";
  if (s === 5 && style.golden === "shine") {
    extra = `<path d="M3.4,4.2 Q4.4,2.7 6.6,2.6" fill="none" stroke="#FFF6D8" stroke-width="1" stroke-linecap="round"/>`;
  } else if (s === 5 && style.golden === "butter") {
    extra = `<rect x="4.1" y="6.3" width="3.8" height="3.2" rx=".6" fill="#FFF4B8" stroke="#D9B648" stroke-width=".5"/>`;
  } else if (s === 0) {
    extra = `<path d="M3.6,4.6 L5.6,6.6 L4.6,8.8 M8.4,7.6 L7.2,10.4 L8,12" fill="none" stroke="#E3A672" stroke-width=".85" stroke-linecap="round" stroke-linejoin="round"/>`;
  }
  return `<g transform="translate(${round(x)},0)${sc}"><path d="${SLICE}" fill="${c.fill}" stroke="${crust}" stroke-width="${width}" stroke-linejoin="round"/>${extra}</g>`;
}

const round = (n: number) => Math.round(n * 100) / 100;

/** Offsets of each slice from the back (index 0) to the top. Gaps grow
 * toward the top slice by `falloff`; the total stays step * (n - 1). */
export function sliceOffsets(n: number, step: number, falloff: number): number[] {
  if (n < 2) return [0];
  const k = n - 1;
  const weights = Array.from({ length: k }, (_, i) => Math.pow(falloff, k - 1 - i));
  const scale = (step * k) / weights.reduce((a, b) => a + b, 0);
  const out = [0];
  for (const w of weights) out.push(out[out.length - 1] + w * scale);
  return out;
}

export interface StackOptions {
  /** Rendered height in CSS px. */
  px: number;
  style?: BreadMarkStyle;
  /** Drawn on a dark page (applies style.darkFix). */
  dark?: boolean;
}

/** The slices of a stack, back to top. */
export function stackShades(scale: FitScale, grade: FitGrade): Shade[] {
  if (grade === 0) return [0];
  if (scale === "ram") return grade >= 2 ? [4, 5] : [4];
  return ([1, 2, 3, 4, 5] as Shade[]).slice(0, grade);
}

export function stackSvg(shades: Shade[], opts: StackOptions): string {
  const style = opts.style ?? DEFAULT_BREAD_MARK_STYLE;
  const n = shades.length;
  const off = sliceOffsets(n, style.step, style.falloff);
  const last = off[n - 1];
  const body = shades
    .map((s, i) => slice(style.direction === "ltr" ? off[i] : last - off[i], s, style, !!opts.dark))
    .join("");
  const vbW = round(12 + last + 0.6);
  const vbH = 14.6;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.3 0 ${vbW} ${vbH}" height="${opts.px}" width="${round((opts.px * vbW) / vbH)}" aria-hidden="true" focusable="false" style="display:block">${body}</svg>`;
}

export function gradeSvg(scale: FitScale, grade: FitGrade, opts: StackOptions): string {
  return stackSvg(stackShades(scale, grade), opts);
}

/** Not checked yet: a dashed slice with a question mark. */
export function uncheckedSvg(px: number, color = "#ab9f8d"): string {
  const w = round((px * 12.6) / 14.6);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.3 0 12.6 14.6" height="${px}" width="${w}" aria-hidden="true" focusable="false" style="display:block"><path d="${SLICE}" fill="none" stroke="${color}" stroke-width="1.1" stroke-dasharray="1.6 1.2" stroke-linejoin="round"/><text x="6" y="11.4" font-size="8" font-weight="700" text-anchor="middle" fill="${color}" font-family="system-ui, sans-serif">?</text></svg>`;
}
