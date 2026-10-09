import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useWorkerStatuses } from "./useWorkerStatus";
import { api, ApiError } from "./client";
import type { Worker } from "../types";

vi.mock("./client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./client")>()),
  api: { listWorkers: vi.fn() },
}));

const listWorkers = vi.mocked(api.listWorkers);

const POLL_MS = 5000;

function worker(id: string): Worker {
  return { id, displayName: id } as unknown as Worker;
}

beforeEach(() => {
  vi.useFakeTimers();
  listWorkers.mockReset();
  listWorkers.mockResolvedValue([]);
});

afterEach(() => {
  vi.useRealTimers();
});

// Lets the immediate on-mount poll settle: it is an un-awaited async call, so
// its promise resolves on the microtask queue rather than on any timer.
async function flush() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

describe("useWorkerStatuses", () => {
  it("starts unloaded and fetches once on mount", async () => {
    listWorkers.mockResolvedValue([worker("w1")]);
    const { result } = renderHook(() => useWorkerStatuses());

    expect(result.current.loaded).toBe(false);
    expect(result.current.workers).toEqual([]);

    await flush();

    expect(listWorkers).toHaveBeenCalledTimes(1);
    expect(result.current.loaded).toBe(true);
    expect(result.current.workers).toEqual([worker("w1")]);
  });

  it("derives an id order and an id-keyed lookup from the same list", async () => {
    listWorkers.mockResolvedValue([worker("w2"), worker("w1")]);
    const { result } = renderHook(() => useWorkerStatuses());
    await flush();

    // Order is the server's, not sorted -- the page relies on that.
    expect(result.current.order).toEqual(["w2", "w1"]);
    expect(result.current.status["w1"]).toEqual(worker("w1"));
    expect(result.current.status["w2"]).toEqual(worker("w2"));
  });

  it("polls again on the interval", async () => {
    const { result } = renderHook(() => useWorkerStatuses());
    await flush();
    expect(listWorkers).toHaveBeenCalledTimes(1);

    listWorkers.mockResolvedValue([worker("late")]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });

    expect(listWorkers).toHaveBeenCalledTimes(2);
    expect(result.current.workers).toEqual([worker("late")]);
  });

  it("does not fire a second request before the interval elapses", async () => {
    renderHook(() => useWorkerStatuses());
    await flush();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS - 1);
    });

    expect(listWorkers).toHaveBeenCalledTimes(1);
  });

  it("stops polling once unmounted", async () => {
    const { unmount } = renderHook(() => useWorkerStatuses());
    await flush();
    expect(listWorkers).toHaveBeenCalledTimes(1);

    unmount();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS * 4);
    });

    // The pending timer was cleared; nothing kept the loop alive.
    expect(listWorkers).toHaveBeenCalledTimes(1);
  });

  it("does not schedule another poll when unmounted while a request is in flight", async () => {
    let resolveIt: (v: Worker[]) => void = () => {};
    listWorkers.mockReturnValue(new Promise<Worker[]>((r) => (resolveIt = r)));

    const { unmount } = renderHook(() => useWorkerStatuses());
    expect(listWorkers).toHaveBeenCalledTimes(1);

    // Unmount first, then let the in-flight request land. The `cancelled`
    // guard must stop both the state write and the reschedule -- this is the
    // shape that leaks a timer if the guard is dropped.
    unmount();
    await act(async () => {
      resolveIt([worker("w1")]);
      await vi.advanceTimersByTimeAsync(POLL_MS * 3);
    });

    expect(listWorkers).toHaveBeenCalledTimes(1);
  });

  it("keeps polling after a failed request rather than stopping the loop", async () => {
    listWorkers.mockRejectedValueOnce(new Error("network down"));
    renderHook(() => useWorkerStatuses());
    await flush();
    expect(listWorkers).toHaveBeenCalledTimes(1);

    listWorkers.mockResolvedValue([worker("recovered")]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });

    expect(listWorkers).toHaveBeenCalledTimes(2);
  });

  it("recovers its data after a transient failure", async () => {
    listWorkers.mockRejectedValueOnce(new Error("network down"));
    const { result } = renderHook(() => useWorkerStatuses());
    await flush();

    // A failed first poll leaves the hook unloaded rather than showing an
    // empty machine list as though the server had answered with none.
    expect(result.current.loaded).toBe(false);
    expect(result.current.workers).toEqual([]);

    listWorkers.mockResolvedValue([worker("w1")]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });

    expect(result.current.loaded).toBe(true);
    expect(result.current.workers).toEqual([worker("w1")]);
  });

  it("exposes a manual refresh that updates immediately", async () => {
    const { result } = renderHook(() => useWorkerStatuses());
    await flush();

    listWorkers.mockResolvedValue([worker("fresh")]);
    await act(async () => {
      await result.current.refresh();
    });

    expect(result.current.workers).toEqual([worker("fresh")]);
  });

  it("keeps a stable refresh identity across re-renders", async () => {
    const { result, rerender } = renderHook(() => useWorkerStatuses());
    await flush();
    const first = result.current.refresh;
    rerender();
    expect(result.current.refresh).toBe(first);
  });

  it("backs off after a 401 instead of re-polling every interval", async () => {
    listWorkers.mockRejectedValue(new ApiError("no session", 401));
    renderHook(() => useWorkerStatuses());
    await flush();
    expect(listWorkers).toHaveBeenCalledTimes(1);

    // First 401 doubles the wait: 10s, not the normal 5s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });
    expect(listWorkers).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });
    expect(listWorkers).toHaveBeenCalledTimes(2);

    // Second consecutive 401 doubles again: 20s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS * 4 - 1);
    });
    expect(listWorkers).toHaveBeenCalledTimes(2);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(listWorkers).toHaveBeenCalledTimes(3);
  });

  it("caps the 401 backoff at one minute and keeps retrying", async () => {
    listWorkers.mockRejectedValue(new ApiError("no session", 401));
    renderHook(() => useWorkerStatuses());
    await flush();

    // Waits 10s, 20s, 40s -- call 4 lands at t=70s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000 + 20_000 + 40_000);
    });
    expect(listWorkers).toHaveBeenCalledTimes(4);

    // The next wait would be 80s uncapped; it is held to 60s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000 - 1);
    });
    expect(listWorkers).toHaveBeenCalledTimes(4);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(listWorkers).toHaveBeenCalledTimes(5);
  });

  it("returns to the normal interval once a request succeeds after a 401", async () => {
    listWorkers.mockRejectedValueOnce(new ApiError("no session", 401));
    const { result } = renderHook(() => useWorkerStatuses());
    await flush();
    expect(listWorkers).toHaveBeenCalledTimes(1);

    // Signed in elsewhere: the backed-off retry succeeds and the data arrives.
    listWorkers.mockResolvedValue([worker("back")]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(listWorkers).toHaveBeenCalledTimes(2);
    expect(result.current.workers).toEqual([worker("back")]);

    // The streak reset, so the next poll is back on the 5s interval.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_MS);
    });
    expect(listWorkers).toHaveBeenCalledTimes(3);
  });
});
