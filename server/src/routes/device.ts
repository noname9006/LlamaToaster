import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { generateEnrolmentCode, generateUserCode, hashToken } from "../session.js";
import { parseDeviceStart, parseDeviceToken } from "../validate-worker-state.js";
import { NotFoundError, ConflictError } from "../errors.js";
import type { AuthenticatedRequest } from "../auth-middleware.js";
import { userOrIpKeyGenerator } from "../auth-middleware.js";
import type { DeviceStartResponse, DeviceTokenSuccess, DeviceTokenError, DeviceStatusResponse } from "../../../shared/types.js";

const ENROLMENT_TTL_MS = 15 * 60 * 1000;

// Public (device-initiated, before any human/session is involved) --
// registered unconditionally, same reasoning as routes/auth.ts's own
// always-on routes: harmless without AUTH_ENABLED (a pending enrolment
// nobody can ever approve just expires in 15 minutes), and a Stage-1-only
// deployment's workers use the old WORKER_SHARED_TOKEN path instead anyway,
// never calling these at all.
export async function deviceRoutes(app: FastifyInstance): Promise<void> {
  // MULTIUSER_PLAN.md §3.4: doubles as re-enrolment. getByMachineId is what
  // unifies first enrolment of a brand-new machine, re-enrolment of an
  // already-owned one, AND the transition of an existing Stage 1 machine
  // (created via the shared secret, user_id still NULL) into this model --
  // it's the same "reuse the row" path either way.
  app.post(
    "/api/device/start",
    { config: { bodyLimit: 4_096, rateLimit: { max: 30, timeWindow: "1 minute" } } },
    async (req): Promise<DeviceStartResponse> => {
      const { machine_id, hostname, platform, arch } = parseDeviceStart(req.body);
      const deviceCode = generateEnrolmentCode();
      const userCode = generateUserCode();
      const expiresAt = Date.now() + ENROLMENT_TTL_MS;

      const existing = repo.workerRepo.getByMachineId(machine_id);
      if (existing) {
        // Revoking prior sessions here closes the exact hole v3.3 had:
        // re-registering never leaves an old worker session live alongside
        // the new one (§3.4's own review-finding callout, D6).
        repo.sessionRepo.revokeWorkerSessions(existing.id);
        repo.workerRepo.reissueEnrolment(existing.id, { hostname, platform, arch, deviceCode, userCode, expiresAt });
      } else {
        repo.workerRepo.createPending({ machineId: machine_id, hostname, platform, arch, deviceCode, userCode, expiresAt });
      }
      return { device_code: deviceCode, user_code: userCode, verification_uri: "/device", interval: 5, expires_in: 900 };
    }
  );

  // Not specified in the plan's own §3.4 code block (only mentioned in
  // §3.1's prose: "Worker polls POST /api/device/token with device_code
  // every 5s -> 400 {error: authorization_pending} until approved") --
  // designed here to close that gap, following RFC 8628's own device-flow
  // semantics: pending/expired/success are the only three outcomes, and a
  // successful poll is the ONE-SHOT redemption point (see
  // clearEnrolmentCode's own doc comment in repo.ts for why that -- not
  // approve() -- is where the code actually gets consumed).
  app.post(
    "/api/device/token",
    { config: { bodyLimit: 1_024, rateLimit: { max: 60, timeWindow: "1 minute" } } },
    async (req, reply): Promise<DeviceTokenSuccess | DeviceTokenError> => {
      const { device_code } = parseDeviceToken(req.body);
      const worker = repo.workerRepo.getByEnrolmentCodeHash(hashToken(device_code));
      if (!worker || worker.enrolmentExpiresAt == null || worker.enrolmentExpiresAt < Date.now()) {
        reply.code(400);
        return { error: "expired_token" };
      }
      if (!worker.approvedAt) {
        reply.code(400);
        return { error: "authorization_pending" };
      }
      // worker.userId is guaranteed set once approvedAt is (both written
      // together by workerRepo.approve).
      const { token, refresh } = repo.sessionRepo.create(worker.userId!, {
        isWorker: true,
        workerId: worker.id,
        label: worker.displayName,
      });
      repo.workerRepo.clearEnrolmentCode(worker.id);
      return { session_token: token, refresh_token: refresh! };
    }
  );
}

// Authenticated (req.user) -- registered only when AUTH_ENABLED, same gate
// and reasoning as routes/sessions.ts's sessionRoutes: these read req.user
// unconditionally, which only the auth middleware ever populates.
export async function deviceApprovalRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Querystring: { user_code?: string } }>("/api/device/status", async (req): Promise<DeviceStatusResponse> => {
    const worker = repo.workerRepo.getByUserCode(req.query.user_code ?? "");
    if (!worker) return { state: "not_found" as const };
    // Checked BEFORE expiry deliberately -- an approved code that's since
    // aged past its original 15-minute TTL (the worker hasn't redeemed it
    // yet, e.g. it's briefly offline) is still meaningfully "approved" to
    // the browser waiting on this status, not "not found". Expiry only
    // matters for a code that's STILL pending.
    if (worker.approvedAt) return { state: "approved" as const };
    if (worker.enrolmentExpiresAt == null || worker.enrolmentExpiresAt < Date.now()) {
      return { state: "not_found" as const };
    }
    return {
      state: "pending" as const,
      machine: { hostname: worker.hostname, platform: worker.platform, arch: worker.arch, gpu: worker.gpuModel },
    };
  });

  app.post<{ Body: { user_code?: string } }>(
    "/api/device/approve",
    { config: { rateLimit: { max: 20, timeWindow: "1 minute", keyGenerator: userOrIpKeyGenerator } } },
    async (req) => {
      const worker = repo.workerRepo.getByUserCode(req.body.user_code ?? "");
      if (!worker) throw new NotFoundError("invalid or expired code");
      // Same ordering fix as GET /api/device/status above, for the same
      // reason -- "already approved" is the more accurate and useful error
      // than "expired" for a code whose TTL has technically elapsed but
      // that was in fact successfully approved already.
      if (worker.approvedAt) throw new ConflictError("already approved");
      if (worker.enrolmentExpiresAt == null || worker.enrolmentExpiresAt < Date.now()) {
        throw new NotFoundError("invalid or expired code");
      }
      repo.workerRepo.approve(worker.id, (req as AuthenticatedRequest).user.id);
      return { ok: true, machine: { hostname: worker.hostname } };
    }
  );
}
