import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { BUILTIN_ROCM_SUPPORT } from "../../../shared/rocmSupport";
import type { RocmSupportSnapshot } from "../types";

const getRocmSupport = vi.fn<() => Promise<RocmSupportSnapshot>>();
vi.mock("./client", () => ({ api: { getRocmSupport: () => getRocmSupport() } }));

const live: RocmSupportSnapshot = {
  list: { ...BUILTIN_ROCM_SUPPORT, linux: { supported: ["rx 7900 gre"], unsupported: [] } },
  source: "live",
  fetchedAt: "2026-09-28T00:00:00.000Z",
};

// The hook keeps a module-level in-flight promise, so each test needs a fresh module.
async function freshHook() {
  vi.resetModules();
  return (await import("./useRocmSupport")).useRocmSupport;
}

beforeEach(() => getRocmSupport.mockReset());

describe("useRocmSupport", () => {
  it("is undefined until the list arrives, then returns it", async () => {
    getRocmSupport.mockResolvedValue(live);
    const useRocmSupport = await freshHook();
    const { result } = renderHook(() => useRocmSupport());
    expect(result.current).toBeUndefined();
    await waitFor(() => expect(result.current).toEqual(live.list));
  });

  it("fetches once for several consumers", async () => {
    getRocmSupport.mockResolvedValue(live);
    const useRocmSupport = await freshHook();
    const a = renderHook(() => useRocmSupport());
    const b = renderHook(() => useRocmSupport());
    await waitFor(() => expect(a.result.current).toBeDefined());
    await waitFor(() => expect(b.result.current).toBeDefined());
    expect(getRocmSupport).toHaveBeenCalledTimes(1);
  });

  it("stays undefined (built-in fallback) when the fetch fails, and retries on the next mount", async () => {
    getRocmSupport.mockRejectedValueOnce(new Error("down"));
    const useRocmSupport = await freshHook();
    const first = renderHook(() => useRocmSupport());
    await waitFor(() => expect(getRocmSupport).toHaveBeenCalledTimes(1));
    expect(first.result.current).toBeUndefined();
    first.unmount();

    getRocmSupport.mockResolvedValueOnce(live);
    const second = renderHook(() => useRocmSupport());
    await waitFor(() => expect(second.result.current).toEqual(live.list));
    expect(getRocmSupport).toHaveBeenCalledTimes(2);
  });
});
