// Liveness is DERIVED, never stored -- see MULTIUSER_PLAN.md §1.6. Storing a
// `workers.status` column and only ever writing it on a poll (the old
// Tailscale-discovery model) means a powered-off machine reads as whatever
// it last reported forever. Deriving it fresh from last_heartbeat_at on
// every read means a dead worker can never lie about being alive.

export const HEARTBEAT_INTERVAL_MS = 10_000;
export const OFFLINE_AFTER_MS = 35_000; // ~3 missed heartbeats
export const LEASE_MS = 60_000; // ~6 missed heartbeats

export type WorkerStatus = "offline" | "busy" | "idle";

export function deriveWorkerStatus(w: { last_heartbeat_at: number | null; active_job_id: string | null }): WorkerStatus {
  if (!w.last_heartbeat_at || Date.now() - w.last_heartbeat_at > OFFLINE_AFTER_MS) return "offline";
  return w.active_job_id ? "busy" : "idle";
}
