import { describe, it, expect, vi, afterEach } from "vitest";
import { api, ApiError } from "./client";
import { WORKER_INACCESSIBLE_MESSAGE } from "../types";

// `request` is module-private, so it is exercised through the thin api.*
// wrappers -- which is also the surface every page actually calls.
// Params are declared even though the stub ignores them: without them the
// mock's recorded `calls` type is an empty tuple and the URL/init assertions
// below fail to compile under the client's tsc --noEmit build step.
function mockFetch(res: { status?: number; body?: string; ok?: boolean }) {
  const status = res.status ?? 200;
  const spy = vi.fn(async (_path: string, _init?: RequestInit) =>
    ({
      ok: res.ok ?? (status >= 200 && status < 300),
      status,
      text: async () => res.body ?? "",
    }) as unknown as Response
  );
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("successful responses", () => {
  it("parses JSON and unwraps the envelope key", async () => {
    mockFetch({ body: JSON.stringify({ models: [{ id: "m1" }] }) });
    await expect(api.listModels()).resolves.toEqual([{ id: "m1" }]);
  });

  it("unwraps a different envelope for a different endpoint", async () => {
    mockFetch({ body: JSON.stringify({ workers: [{ id: "w1" }] }) });
    await expect(api.listWorkers()).resolves.toEqual([{ id: "w1" }]);
  });

  it("sends JSON bodies with a content-type header", async () => {
    const spy = mockFetch({ body: JSON.stringify({ model: { id: "m1" } }) });
    await api.registerModel({ source: "local", filename: "x.gguf" });

    const [, init] = spy.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(init?.headers).toMatchObject({ "content-type": "application/json" });
    expect(JSON.parse(String(init?.body))).toEqual({ source: "local", filename: "x.gguf" });
  });

  it("percent-encodes path parameters so an id with a slash cannot escape the route", async () => {
    const spy = mockFetch({ body: JSON.stringify({ ok: true }) });
    await api.deleteModel("a/b?c=d");
    expect(spy.mock.calls[0][0]).toBe("/api/models/a%2Fb%3Fc%3Dd");
  });
});

describe("error mapping", () => {
  it("surfaces the server's own error message", async () => {
    mockFetch({ status: 404, body: JSON.stringify({ error: "run not found" }) });
    await expect(api.listModels()).rejects.toThrowError(
      expect.objectContaining({ name: "ApiError", message: "run not found", status: 404 })
    );
  });

  it("falls back to a status-based message when the body has no error field", async () => {
    mockFetch({ status: 500, body: JSON.stringify({ something: "else" }) });
    await expect(api.listModels()).rejects.toThrowError("request failed: 500");
  });

  it("falls back when the error field is not a string", async () => {
    mockFetch({ status: 400, body: JSON.stringify({ error: { nested: true } }) });
    await expect(api.listModels()).rejects.toThrowError("request failed: 400");
  });

  it("falls back for a non-JSON error body, such as an HTML proxy page", async () => {
    mockFetch({ status: 502, body: "<html>Bad Gateway</html>" });
    await expect(api.listModels()).rejects.toThrowError("request failed: 502");
  });

  it("falls back for an empty error body", async () => {
    mockFetch({ status: 503, body: "" });
    await expect(api.listModels()).rejects.toThrowError("request failed: 503");
  });

  it("reports a network-level failure as status 0, distinct from any HTTP status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      })
    );
    const err = await api.listModels().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
    expect((err as ApiError).message).toBe("Failed to fetch");
    expect((err as ApiError).inaccessible).toBe(false);
  });

  it("flags the worker-inaccessible message so callers can render it differently", async () => {
    mockFetch({ status: 502, body: JSON.stringify({ error: WORKER_INACCESSIBLE_MESSAGE }) });
    const err = (await api.listModels().catch((e: unknown) => e)) as ApiError;
    expect(err.inaccessible).toBe(true);
    expect(err.status).toBe(502);
  });

  it("does not flag an ordinary error as inaccessible", async () => {
    mockFetch({ status: 500, body: JSON.stringify({ error: "boom" }) });
    const err = (await api.listModels().catch((e: unknown) => e)) as ApiError;
    expect(err.inaccessible).toBe(false);
  });
});

describe("ApiError", () => {
  it("carries status and defaults inaccessible to false", () => {
    const e = new ApiError("nope", 418);
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe("ApiError");
    expect(e.status).toBe(418);
    expect(e.inaccessible).toBe(false);
  });
});
