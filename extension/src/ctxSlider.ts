// The context slider's scale and tick labels. Pure arithmetic with no chrome.*
// access, so the repo-level tests (shared/ctxSlider.test.ts) can import it
// without the extension's Chrome types.

/** Labelled anchor points (tokens); the slider itself is continuous between them. */
export const CTX_STOPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 10485760];
export const CTX_MIN = CTX_STOPS[0];
export const CTX_MAX = CTX_STOPS[CTX_STOPS.length - 1];
/** 1k..1M takes 100 positions per doubling. The 1M..10M stretch is rare (Llama 4
 * Scout, Mistral Nemo's header) and gets one short segment at the end. */
const PER_DOUBLING = 100;
const MAIN_END = (CTX_STOPS.length - 2) * PER_DOUBLING; // 1M
const TAIL = 70;
export const CTX_SLIDER_MAX = MAIN_END + TAIL;
const MAIN_MAX = CTX_STOPS[CTX_STOPS.length - 2];
/** Within this many positions of an anchor the thumb snaps to it (the anchors are "magnetic"). */
const SNAP_RADIUS = 12;
const TAIL_SNAP = 6;

/** Where each labelled anchor sits on the slider. */
export function stopPositions(): number[] {
  return CTX_STOPS.map((_, i) => (i < CTX_STOPS.length - 1 ? i * PER_DOUBLING : CTX_SLIDER_MAX));
}

export function ctxFromSlider(pos: number): number {
  if (pos > MAIN_END) {
    if (CTX_SLIDER_MAX - pos <= TAIL_SNAP) return CTX_MAX;
    if (pos - MAIN_END <= TAIL_SNAP) return MAIN_MAX;
    const raw = MAIN_MAX * (CTX_MAX / MAIN_MAX) ** ((pos - MAIN_END) / TAIL);
    return Math.min(CTX_MAX, Math.round(raw / 131072) * 131072);
  }
  const nearest = Math.round(pos / PER_DOUBLING);
  if (Math.abs(pos - nearest * PER_DOUBLING) <= SNAP_RADIUS) return CTX_STOPS[nearest];
  const raw = CTX_MIN * 2 ** (pos / PER_DOUBLING);
  const step = raw < 8192 ? 256 : raw < 65536 ? 1024 : 8192;
  return Math.min(MAIN_MAX, Math.max(CTX_MIN, Math.round(raw / step) * step));
}

export function sliderFromCtx(ctx: number): number {
  const c = Math.min(CTX_MAX, Math.max(CTX_MIN, ctx));
  if (c > MAIN_MAX) return Math.round(MAIN_END + (Math.log(c / MAIN_MAX) / Math.log(CTX_MAX / MAIN_MAX)) * TAIL);
  return Math.round(Math.log2(c / CTX_MIN) * PER_DOUBLING);
}

export function ctxLabel(ctx: number): string {
  if (ctx >= 1048576) return `${+(ctx / 1048576).toFixed(2)}M`;
  return `${+(ctx / 1024).toFixed(2)}k`;
}

/** Tick labels placed where the slider thumb actually sits (22px thumb inside a
 * 1px track border: its centre travels from 12px to W-12). Up to maxPos only
 * (a slider capped at a model's maximum shows the ticks it can reach). */
export function ticksHtml(maxPos = CTX_SLIDER_MAX): string {
  const pos = stopPositions();
  return CTX_STOPS.map((c, i) => [c, pos[i]] as const)
    .filter(([, p]) => p <= maxPos)
    // The last two anchors sit close together: the 10M label hangs right of its tick.
    .map(([c, p]) => `<span class="${c === CTX_MAX ? "tail" : ""}" style="left: calc(12px + (100% - 24px) * ${p / maxPos})">${ctxLabel(c)}</span>`)
    .join("");
}
