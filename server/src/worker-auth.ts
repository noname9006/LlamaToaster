import type { FastifyRequest } from "fastify";
import { safeEqual, hashToken } from "./session.js";
import { UnauthorizedError, BadRequestError } from "./errors.js";
import { repo } from "./db/repo.js";
import type { Worker } from "../../shared/types.js";

// Stage 3 (MULTIUSER_PLAN.md §3.4/§1.15): dual-mode, so already-deployed
// Stage 1 workers (WORKER_SHARED_TOKEN + self-announced machine_id) keep
// working unchanged during the transition, while newly-enrolled workers
// (routes/device.ts) authenticate with their own per-worker session token
// instead. The wire protocol and every route handler stay identical either
// way -- only this function's internals branch on which kind of token was
// presented.
//
// Tried in this order: a bearer token is first looked up as a worker
// session (Stage 3's real identity -- no machine_id needed, the session IS
// the identity), and only falls back to the shared-secret+self-announce path
// (Stage 1) if that lookup doesn't resolve to a live worker session. This
// means a Stage-3-only deployment (no WORKER_SHARED_TOKEN configured at all)
// works fine -- the env var is only consulted inside the fallback branch,
// never up front.
export async function authenticateWorker(req: FastifyRequest): Promise<Worker> {
  const auth = req.headers.authorization;
  const token = auth?.startsWith("Bearer ") ? auth.slice("Bearer ".length) : undefined;
  if (!token) throw new UnauthorizedError("missing worker token");

  const session = repo.sessionRepo.getByTokenHash(hashToken(token));
  if (session && session.expiresAt >= Date.now() && session.isWorker && session.workerId) {
    const worker = repo.workerRepo.getWorker(session.workerId);
    if (worker) {
      repo.sessionRepo.touch(session);
      return worker;
    }
    // Session points at a worker row that no longer exists (no FK cascade
    // on sessions.worker_id) -- fall through to the Stage 1 path rather than
    // failing outright; on a shared-secret deployment the same bearer value
    // will just never match WORKER_SHARED_TOKEN either, so this doesn't
    // widen what's accepted.
  }

  const expected = process.env.WORKER_SHARED_TOKEN;
  if (!expected || !safeEqual(token, expected)) throw new UnauthorizedError("invalid worker token");

  const body = req.body as { machine_id?: unknown; hostname?: unknown } | undefined;
  const machineId = typeof body?.machine_id === "string" ? body.machine_id.trim().slice(0, 256) : "";
  if (!machineId) throw new BadRequestError("machine_id is required");
  const hostname =
    typeof body?.hostname === "string" && body.hostname.trim() ? body.hostname.trim().slice(0, 256) : machineId;

  return repo.workerRepo.getOrCreateByMachineId(machineId, hostname);
}
