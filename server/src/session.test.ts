import { describe, expect, it } from "vitest";
import { generateToken, hashToken, safeEqual, generateUserCode } from "./session.js";

describe("generateToken", () => {
  it("produces distinct, high-entropy base64url values", () => {
    const a = generateToken();
    const b = generateToken();
    expect(a).not.toBe(b);
    expect(a).toMatch(/^[A-Za-z0-9_-]{43}$/);
  });
});

describe("hashToken", () => {
  it("is deterministic and never returns the input verbatim", () => {
    const token = generateToken();
    expect(hashToken(token)).toBe(hashToken(token));
    expect(hashToken(token)).not.toBe(token);
    expect(hashToken(token)).toMatch(/^[0-9a-f]{64}$/); // sha256 hex
  });
});

describe("safeEqual", () => {
  it("returns true only for identical strings", () => {
    expect(safeEqual("abc123", "abc123")).toBe(true);
    expect(safeEqual("abc123", "abc124")).toBe(false);
  });

  it("returns false (not throws) on length mismatch", () => {
    expect(safeEqual("short", "a-much-longer-string")).toBe(false);
  });
});

describe("generateUserCode", () => {
  it("produces an 8-character code split ABCD-EFGH, from the reduced alphabet only", () => {
    const code = generateUserCode();
    expect(code).toMatch(/^[ABCDEFGHJKMNPQRSTUVWXYZ23456789]{4}-[ABCDEFGHJKMNPQRSTUVWXYZ23456789]{4}$/);
  });

  it("excludes ambiguous characters (0, O, 1, I, L)", () => {
    for (let i = 0; i < 50; i++) {
      const code = generateUserCode();
      expect(code).not.toMatch(/[0OIL1]/);
    }
  });
});
