import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { acquireRateLimitSlot, recordRateLimit, resetRateLimitState } from "./hf-rate-limit.js";
import { log } from "./log.js";

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
    const slot = acquireRateLimitSlot("api", "test").then(() => {
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
    const slot = acquireRateLimitSlot("api", "test").then(() => {
      resolved = true;
    });

    await vi.advanceTimersByTimeAsync(15000);
    expect(resolved).toBe(true);
    await slot;
  });
});

describe("per-reason attribution in low-remaining logging", () => {
  beforeEach(() => {
    resetRateLimitState();
  });

  it("tallies admitted requests by their reason tag and reports it in the low-remaining warning", async () => {
    // Mirrors the real diagnosis this was built for: a burst of production
    // log spam with no indication of *what* was calling HF. Without the
    // per-reason breakdown, this test (and the real log line) can only ever
    // say a number is low, never what's consuming it.
    const warnSpy = vi.spyOn(log, "warn").mockImplementation(() => {});
    try {
      for (let i = 0; i < 40; i++) await acquireRateLimitSlot("api", "index-scan");
      for (let i = 0; i < 3; i++) await acquireRateLimitSlot("api", "backlog-walk");
      await acquireRateLimitSlot("api", "hf-updates");

      // Simulate HF reporting we're now low on the real window -- this is
      // what actually triggers the warning line in production.
      const headers = new Headers({
        RateLimit: '"api";r=45;t=17',
        "RateLimit-Policy": '"fixed window";"api";q=1000;w=300',
      });
      recordRateLimit("api", headers);

      const warnLine = warnSpy.mock.calls.map((c) => c.join(" ")).find((line) => line.includes("low remaining"));
      expect(warnLine).toBeDefined();
      expect(warnLine).toContain("r=45");
      expect(warnLine).toContain("index-scan=40");
      expect(warnLine).toContain("backlog-walk=3");
      expect(warnLine).toContain("hf-updates=1");
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("does not carry a reason's count past its entries aging out of the 5-minute window", async () => {
    vi.useFakeTimers();
    try {
      resetRateLimitState();
      await acquireRateLimitSlot("api", "backlog-walk");
      // Advance well past the 300s fixed window so that entry prunes away.
      await vi.advanceTimersByTimeAsync(301_000);
      await acquireRateLimitSlot("api", "search-ui");

      const warnSpy = vi.spyOn(log, "warn").mockImplementation(() => {});
      try {
        const headers = new Headers({
          RateLimit: '"api";r=45;t=17',
          "RateLimit-Policy": '"fixed window";"api";q=1000;w=300',
        });
        recordRateLimit("api", headers);
        const warnLine = warnSpy.mock.calls.map((c) => c.join(" ")).find((line) => line.includes("low remaining"));
        expect(warnLine).toBeDefined();
        expect(warnLine).toContain("search-ui=1");
        expect(warnLine).not.toContain("backlog-walk");
      } finally {
        warnSpy.mockRestore();
      }
    } finally {
      vi.useRealTimers();
    }
  });
});
