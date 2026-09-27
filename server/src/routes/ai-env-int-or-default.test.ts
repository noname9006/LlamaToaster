import { describe, expect, it } from "vitest";
import { envIntOrDefault } from "./ai.js";

describe("envIntOrDefault", () => {
  it("falls back on undefined (env var absent)", () => {
    expect(envIntOrDefault(undefined, 2000)).toBe(2000);
  });

  it("falls back on empty string -- a blank `KEY=` line in a .env file, not truly unset", () => {
    expect(envIntOrDefault("", 2000)).toBe(2000);
  });

  it("falls back on whitespace-only string", () => {
    expect(envIntOrDefault("   ", 2000)).toBe(2000);
  });

  it("honors an explicit zero -- operators must be able to hard-disable via 0", () => {
    expect(envIntOrDefault("0", 2000)).toBe(0);
  });

  it("parses a normal positive value", () => {
    expect(envIntOrDefault("5", 2000)).toBe(5);
  });
});
