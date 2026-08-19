import { randomBytes, createHash, timingSafeEqual } from "node:crypto";

// 32 bytes = 256 bits. base64url so the value is safe in a cookie, an
// Authorization header, a shell argument, and a URL without escaping.
// NEVER Math.random() -- it is seeded, predictable, and not a CSPRNG.
export function generateToken(): string {
  return randomBytes(32).toString("base64url"); // 43 chars
}

export const generateSessionId = generateToken; // browser + worker access token
export const generateRefreshToken = generateToken; // worker refresh token
export const generateEnrolmentCode = generateToken; // machine enrolment secret
export const generateState = generateToken; // OAuth CSRF state

// Plain SHA-256 is correct here and bcrypt/argon2 would be wrong: these are
// 256-bit uniformly-random values, not user-chosen passwords. There is no
// dictionary to slow down, and a per-request KDF would just add latency to
// every authenticated call.
export function hashToken(token: string): string {
  return createHash("sha256").update(token).digest("hex");
}

// Constant-time compare for anything an attacker can submit repeatedly
// (the OAuth state cookie, the enrolment user_code).
export function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  if (ab.length !== bb.length) return false; // length leak is unavoidable and harmless here
  return timingSafeEqual(ab, bb);
}

// Human-typeable short code, ONLY ever a display hint scoped to one user's
// pending enrolments (see Stage 3.3) -- never a standalone credential.
// Alphabet excludes 0/O/1/I/L to survive being read off a screen.
const CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"; // 31 chars
export function generateUserCode(): string {
  // Rejection sampling from randomBytes -- a plain `% 31` would bias the
  // first 8 letters. Still crypto, still not Math.random().
  const out: string[] = [];
  while (out.length < 8) {
    for (const b of randomBytes(16)) {
      if (b >= 248) continue; // 248 = 31 * 8, the largest multiple of 31 < 256
      out.push(CODE_ALPHABET[b % 31]);
      if (out.length === 8) break;
    }
  }
  return `${out.slice(0, 4).join("")}-${out.slice(4).join("")}`; // "ABCD-EFGH"
}
