import { describe, expect, it } from "vitest";
import { CTX_MAX, CTX_SLIDER_MAX, ctxFromSlider, sliderFromCtx, stopPositions } from "../extension/src/ctxSlider";

// The extension's context slider (extension/src/ctxSlider.ts): 1k..1M at 100
// positions per doubling, then one short 1M..10M segment at the end.
describe("extension context slider", () => {
  it("ends at 10M, with the 1M..10M stretch far shorter than a doubling", () => {
    expect(CTX_MAX).toBe(10485760);
    expect(ctxFromSlider(CTX_SLIDER_MAX)).toBe(CTX_MAX);
    const pos = stopPositions();
    expect(pos[pos.length - 1] - pos[pos.length - 2]).toBeLessThan(100);
  });

  it("round-trips every labelled stop", () => {
    for (const c of [1024, 4096, 131072, 1048576, 10485760]) expect(ctxFromSlider(sliderFromCtx(c))).toBe(c);
  });

  it("is monotonic across the whole range", () => {
    let prev = 0;
    for (let p = 0; p <= CTX_SLIDER_MAX; p++) {
      const c = ctxFromSlider(p);
      expect(c).toBeGreaterThanOrEqual(prev);
      prev = c;
    }
  });
});
