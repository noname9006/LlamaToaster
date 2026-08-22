import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { acquireRateLimitSlot, recordRateLimit, resetRateLimitState } from "./hf-rate-limit.js";

describe("acquireRateLimitSlot", () => {
  beforeEach(() => {
    resetRateLimitState();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not deadlock once a learned remaining=0 window's reset time has passed", async () => {
    // Simulate what a real 429 response leaves behind: remaining=0 and a
    // 1s reset. Before the fix, nothing ever cleared `remaining` once set,
    // so acquireRateLimitSlot looped forever even after the window reset --
    // no request could ever get through to refresh it.
    const headers = new Headers({
      RateLimit: '"api";r=0;t=1',
      "RateLimit-Policy": '"fixed window";"api";q=500;w=300',
    });
    recordRateLimit("api", headers);

    let resolved = false;
    const slot = acquireRateLimitSlot("api").then(() => {
      resolved = true;
    });

    await vi.advanceTimersByTimeAsync(5000);
    expect(resolved).toBe(true);
    await slot;
  });

  it("does not deadlock when remaining=0 is reported without a t= reset hint", async () => {
    const headers = new Headers({ RateLimit: '"api";r=0' });
    recordRateLimit("api", headers);

    let resolved = false;
    const slot = acquireRateLimitSlot("api").then(() => {
      resolved = true;
    });

    await vi.advanceTimersByTimeAsync(15000);
    expect(resolved).toBe(true);
    await slot;
  });
});
