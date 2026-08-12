import { WORKER_INACCESSIBLE_MESSAGE } from "../../shared/types.js";

export { WORKER_INACCESSIBLE_MESSAGE };

// Node's fetch() throws a generic `TypeError: fetch failed` for every
// connection-level failure (connection refused, DNS failure, host
// unreachable, ...) and AbortSignal.timeout() rejects with an
// Abort/TimeoutError -- neither carries anything a user should see verbatim.
// Every fetchWorker() call site in this app talks to a worker's own HTTP
// listener, which always responds with JSON (even on its own errors), so a
// raw fetch-level throw here only ever means "couldn't reach the worker at
// all", never a worker-side bug -- safe to collapse into one clean message
// rather than surfacing the exception's own wording.
export function isWorkerUnreachable(err: unknown): boolean {
  const name = (err as { name?: string } | null | undefined)?.name;
  if (name === "AbortError" || name === "TimeoutError") return true;
  return err instanceof TypeError && /fetch failed/i.test(err.message);
}

export function describeWorkerError(err: unknown): string {
  if (isWorkerUnreachable(err)) return WORKER_INACCESSIBLE_MESSAGE;
  return err instanceof Error ? err.message : String(err);
}
