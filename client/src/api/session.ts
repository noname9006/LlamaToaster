import type { AuthStatus } from "../types";

// Where a signed-out visitor was headed, kept across the GitHub OAuth round
// trip so sign-in lands them back on it. sessionStorage rather than a ?next=
// param: the server's OAuth callback always redirects to "/", and this way
// the return path never has to travel through the provider at all.
const RETURN_KEY = "lt:returnTo";

// Only same-origin app paths. Rejects "//evil.com" and "/\evil.com" (both
// read as protocol-relative URLs by browsers) so a stored value can never
// turn into an open redirect, and never sends sign-in back to /login or the
// server's /auth/* routes.
export function isSafeReturnPath(path: unknown): path is string {
  if (typeof path !== "string" || !path.startsWith("/")) return false;
  if (path.startsWith("//") || path.startsWith("/\\")) return false;
  if (path === "/login" || path.startsWith("/login?") || path.startsWith("/login#")) return false;
  if (path.startsWith("/auth/")) return false;
  return true;
}

export function saveReturnPath(path: string): void {
  if (!isSafeReturnPath(path) || path === "/") return;
  try {
    sessionStorage.setItem(RETURN_KEY, path);
  } catch {
    /* storage unavailable -- sign-in just lands on the default page */
  }
}

// Reads and clears in one step, so a path is used at most once.
export function takeReturnPath(): string | null {
  try {
    const path = sessionStorage.getItem(RETURN_KEY);
    sessionStorage.removeItem(RETURN_KEY);
    return isSafeReturnPath(path) ? path : null;
  } catch {
    return null;
  }
}

// A 401 from any API call means the session MAY be gone -- it can also be a
// route rejecting for its own reasons. So rather than signing out on the
// spot, ask /api/auth/status once and let the caller act on the answer.
// Concurrent 401s (several pages polling at once) share one in-flight check.
// A failed check (server unreachable) is ignored: the next 401 asks again.
export function createUnauthorizedRecheck(
  check: () => Promise<AuthStatus>,
  onStatus: (status: AuthStatus) => void
): () => Promise<void> {
  let inflight: Promise<void> | null = null;
  return () => {
    if (inflight) return inflight;
    inflight = check()
      .then(onStatus, () => {})
      .finally(() => {
        inflight = null;
      });
    return inflight;
  };
}
