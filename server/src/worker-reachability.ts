// Tracks which worker URLs have failed a live call recently, so a
// persistently-unreachable worker (e.g. one that's configured but not
// actually running yet) only makes callers pay its full request timeout
// once per RECHECK_MS, not on every single page load. Deliberately doesn't
// cache success at all -- only "known down as of X" is memoized, so a
// worker that's actually up stays exactly as live as it's always been (see
// routes/models.ts's /api/models/locations doc comment on why that matters
// there).
const RECHECK_MS = 15_000;
const lastFailureAt = new Map<string, number>();

export function isRecentlyUnreachable(url: string): boolean {
  const last = lastFailureAt.get(url);
  return last !== undefined && Date.now() - last < RECHECK_MS;
}

export function markUnreachable(url: string): void {
  lastFailureAt.set(url, Date.now());
}

export function markReachable(url: string): void {
  lastFailureAt.delete(url);
}
