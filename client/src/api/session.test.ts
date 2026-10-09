import { describe, it, expect, vi, beforeEach } from "vitest";
import { createUnauthorizedRecheck, isSafeReturnPath, saveReturnPath, takeReturnPath } from "./session";
import type { AuthStatus } from "../types";

function status(user: AuthStatus["user"], authEnabled = true): AuthStatus {
  return { user, authEnabled } as AuthStatus;
}

beforeEach(() => {
  sessionStorage.clear();
});

describe("isSafeReturnPath", () => {
  it.each(["/workers", "/tests/abc?tab=log", "/device?code=ABCD-1234", "/settings#sessions"])("accepts app path %s", (p) => {
    expect(isSafeReturnPath(p)).toBe(true);
  });

  it.each([
    "//evil.example",
    "/\\evil.example",
    "https://evil.example/",
    "javascript:alert(1)",
    "workers",
    "",
    "/login",
    "/login?error=oauth_state",
    "/auth/github",
    null,
    42,
  ])("rejects %s", (p) => {
    expect(isSafeReturnPath(p)).toBe(false);
  });
});

describe("saveReturnPath / takeReturnPath", () => {
  it("round-trips a path once, then is empty", () => {
    saveReturnPath("/device?code=ABCD-1234");
    expect(takeReturnPath()).toBe("/device?code=ABCD-1234");
    expect(takeReturnPath()).toBeNull();
  });

  it("does not store an unsafe path", () => {
    saveReturnPath("//evil.example");
    expect(takeReturnPath()).toBeNull();
  });

  it("does not store the default page", () => {
    saveReturnPath("/");
    expect(takeReturnPath()).toBeNull();
  });

  it("ignores a tampered stored value", () => {
    sessionStorage.setItem("lt:returnTo", "https://evil.example/");
    expect(takeReturnPath()).toBeNull();
  });
});

describe("createUnauthorizedRecheck", () => {
  it("asks the server and passes the answer on", async () => {
    const check = vi.fn(async () => status(null));
    const onStatus = vi.fn();
    await createUnauthorizedRecheck(check, onStatus)();
    expect(check).toHaveBeenCalledTimes(1);
    expect(onStatus).toHaveBeenCalledWith(status(null));
  });

  it("shares one check between concurrent 401s", async () => {
    let resolve: (s: AuthStatus) => void = () => {};
    const check = vi.fn(() => new Promise<AuthStatus>((r) => (resolve = r)));
    const onStatus = vi.fn();
    const recheck = createUnauthorizedRecheck(check, onStatus);

    const a = recheck();
    const b = recheck();
    const c = recheck();
    resolve(status(null));
    await Promise.all([a, b, c]);

    expect(check).toHaveBeenCalledTimes(1);
    expect(onStatus).toHaveBeenCalledTimes(1);
  });

  it("checks again for a 401 that arrives after the previous check finished", async () => {
    const check = vi.fn(async () => status(null));
    const recheck = createUnauthorizedRecheck(check, () => {});
    await recheck();
    await recheck();
    expect(check).toHaveBeenCalledTimes(2);
  });

  it("swallows a failed check without calling onStatus", async () => {
    const check = vi.fn(async () => {
      throw new Error("network down");
    });
    const onStatus = vi.fn();
    await expect(createUnauthorizedRecheck(check, onStatus)()).resolves.toBeUndefined();
    expect(onStatus).not.toHaveBeenCalled();
  });
});
