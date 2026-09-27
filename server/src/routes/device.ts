import type { FastifyInstance } from "fastify";
import { createPublicKey, randomBytes, verify } from "node:crypto";
import { repo } from "../db/repo.js";
import type { WorkerEnrolment } from "../db/repo.js";
import { generateEnrolmentCode, generateUserCode, hashToken } from "../session.js";
import { parseDeviceStart, parseDeviceToken, parseDeviceChallenge, parseRegisterKey } from "../validate-worker-state.js";
import { resolveWorkerSession } from "../worker-auth.js";
import type { DeviceStartInput } from "../validate-worker-state.js";
import { NotFoundError, ConflictError, BadRequestError, ForbiddenError, UnauthorizedError } from "../errors.js";
import { reenrolMessage } from "../../../shared/machineKey.js";
import type { AuthenticatedRequest } from "../auth-middleware.js";
import { userOrIpKeyGenerator, assertOwnsWorker } from "../auth-middleware.js";
import type {
  DeviceStartResponse,
  DeviceTokenSuccess,
  DeviceTokenError,
  DeviceStatusResponse,
  DeviceApproveResponse,
  DeviceChallengeResponse,
} from "../../../shared/types.js";

const ENROLMENT_TTL_MS = 15 * 60 * 1000;
const CHALLENGE_TTL_MS = 60 * 1000;

// user_codes are looked up across first enrolments and re-enrolments alike,
// so a fresh one must be free in both tables. Collisions are ~1 in 10^12;
// the loop is only there so one can never silently alias another code.
function freshUserCode(): string {
  for (let i = 0; i < 5; i++) {
    const code = generateUserCode();
    if (!repo.workerRepo.isUserCodeTaken(code)) return code;
  }
  throw new ConflictError("could not allocate an enrolment code -- try again");
}

// True only when the request carries a signature, over a nonce this server
// issued for this machine_id, that verifies against the key already on file
// for the machine. The nonce is consumed whether or not the signature
// verifies, so each nonce buys exactly one try.
function provesMachineKey(enrolment: WorkerEnrolment, input: DeviceStartInput): boolean {
  if (!enrolment.publicKey || !input.nonce || !input.signature) return false;
  if (!repo.workerRepo.consumeChallenge(input.machine_id, input.nonce)) return false;
  try {
    return verify(
      null,
      reenrolMessage(input.nonce, input.machine_id),
      createPublicKey(enrolment.publicKey),
      Buffer.from(input.signature, "base64")
    );
  } catch {
    return false;
  }
}

function gpuFromHardwareJson(json: string | null): string | null {
  if (!json) return null;
  try {
    return (JSON.parse(json) as { gpu?: { model?: string }[] }).gpu?.[0]?.model ?? null;
  } catch {
    return null;
  }
}

// Public (device-initiated, before any human/session is involved) --
// registered unconditionally, same reasoning as routes/auth.ts's own
// always-on routes: harmless without AUTH_ENABLED (a pending enrolment
// nobody can ever approve just expires in 15 minutes), and a Stage-1-only
// deployment's workers use the old WORKER_SHARED_TOKEN path instead anyway,
// never calling these at all.
export async function deviceRoutes(app: FastifyInstance): Promise<void> {
  // MULTIUSER_PLAN.md §3.4: doubles as re-enrolment. Three cases, by what
  // getByMachineId finds: nothing (a brand-new machine -> a pending row), an
  // unowned row (a retried first enrolment or a Stage 1 machine -> reissue
  // on that row), or an OWNED machine (-> a separate re-enrolment attempt,
  // see the security finding C1 comment below).
  app.post(
    "/api/device/start",
    // 16KB, not the old 4KB -- this now also carries a HardwareInfo blob
    // (cpu flags, gpu list, ...), same shape/validator as the heartbeat path
    // but nowhere near that route's 1MB WORKER_BODY_LIMIT (queue.ts).
    { config: { bodyLimit: 16_384, rateLimit: { max: 30, timeWindow: "1 minute" } } },
    async (req): Promise<DeviceStartResponse> => {
      const input = parseDeviceStart(req.body);
      const { machine_id, hostname, platform, arch, hardware, public_key: publicKey } = input;
      const deviceCode = generateEnrolmentCode();
      const userCode = freshUserCode();
      const expiresAt = Date.now() + ENROLMENT_TTL_MS;
      let approved = false;

      const existing = repo.workerRepo.getByMachineId(machine_id);
      const enrolment = existing ? repo.workerRepo.getEnrolmentById(existing.id) : undefined;
      if (!enrolment) {
        repo.workerRepo.createPending({ machineId: machine_id, hostname, platform, arch, deviceCode, userCode, expiresAt, hardware, publicKey });
      } else if (!enrolment.userId) {
        // Nobody owns it yet (a retried first enrolment, or a Stage 1
        // machine moving over) -- still needs a human's approval.
        repo.workerRepo.reissueEnrolment(enrolment.id, { hostname, platform, arch, deviceCode, userCode, expiresAt, hardware, publicKey });
      } else {
        // An OWNED machine reconnecting. Security finding C1: this used to
        // reuse the approved row, so knowing a machine_id was enough to mint
        // a session for its owner. Now the attempt goes into its own row and
        // the live machine is untouched -- its sessions keep working until
        // the attempt is actually redeemed. It is approved right away only
        // if the caller proves it holds the machine's key; otherwise the
        // owner has to approve it like a new machine.
        approved = provesMachineKey(enrolment, input);
        repo.workerRepo.createReenrolment({
          workerId: enrolment.id,
          ownerId: enrolment.userId,
          publicKey,
          hostname,
          platform,
          arch,
          hardware,
          deviceCode,
          userCode,
          expiresAt,
          signed: approved,
        });
      }
      return {
        device_code: deviceCode,
        user_code: userCode,
        verification_uri: "/device",
        interval: 5,
        expires_in: 900,
        ...(approved ? { approved: true } : {}),
      };
    }
  );

  // Security finding C1 -- a one-time nonce for the worker to sign with its
  // machine key before POST /api/device/start. Answers the same for a
  // machine_id the server has never seen, so it can't be used to probe
  // which machines exist. Several nonces can be live per machine, so one
  // caller asking for nonces never invalidates another's.
  app.post(
    "/api/device/challenge",
    { config: { bodyLimit: 1_024, rateLimit: { max: 30, timeWindow: "1 minute" } } },
    async (req): Promise<DeviceChallengeResponse> => {
      const { machine_id } = parseDeviceChallenge(req.body);
      const nonce = randomBytes(32).toString("base64url");
      repo.workerRepo.createChallenge(machine_id, nonce, Date.now() + CHALLENGE_TTL_MS);
      return { nonce, expires_in: CHALLENGE_TTL_MS / 1000 };
    }
  );

  // Trust-on-first-use for a machine enrolled before machine keys existed:
  // its worker registers its public key once, over its own worker session.
  // Never replaces a key already on file (409) -- from then on only the
  // owner's approval can. /api/worker/* is exempt from the user-session
  // middleware; this handler accepts an enrolled worker session only, never
  // the shared deployment secret (a shared-token machine has no owner to
  // reconnect to).
  app.post("/api/worker/register-key", { config: { bodyLimit: 2_048 } }, async (req, reply) => {
    const worker = resolveWorkerSession(req);
    if (!worker) throw new UnauthorizedError("this route requires an enrolled worker session");
    const { public_key } = parseRegisterKey(req.body);
    if (repo.workerRepo.registerKey(worker.id, public_key)) return { ok: true, registered: true };
    const current = repo.workerRepo.getEnrolmentById(worker.id);
    if (current?.publicKey === public_key) return { ok: true, registered: false };
    reply.code(409);
    return { ok: false, error: "a different key is already on file for this machine" };
  });

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
      const codeHash = hashToken(device_code);
      const worker = repo.workerRepo.getByEnrolmentCodeHash(codeHash);
      if (!worker) {
        // Not a first enrolment -- maybe a reconnect of an owned machine.
        const attempt = repo.workerRepo.getReenrolmentByCodeHash(codeHash);
        if (!attempt || attempt.expiresAt < Date.now()) {
          reply.code(400);
          return { error: "expired_token" };
        }
        if (!attempt.approvedAt) {
          reply.code(400);
          return { error: "authorization_pending" };
        }
        const target = repo.workerRepo.getEnrolmentById(attempt.workerId);
        // redeemReenrolment is the single-use point: it ends the machine's
        // old sessions and consumes the attempt in one transaction.
        if (!target?.userId || !repo.workerRepo.redeemReenrolment(attempt.id)) {
          reply.code(400);
          return { error: "expired_token" };
        }
        const { token, refresh } = repo.sessionRepo.create(target.userId, {
          isWorker: true,
          workerId: target.id,
          label: target.displayName,
        });
        return { session_token: token, refresh_token: refresh! };
      }
      if (worker.enrolmentExpiresAt == null || worker.enrolmentExpiresAt < Date.now()) {
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
    const userCode = req.query.user_code ?? "";
    const worker = repo.workerRepo.getByUserCode(userCode);
    if (!worker) {
      // A reconnect of an owned machine. Only its owner sees it at all.
      const attempt = repo.workerRepo.getReenrolmentByUserCode(userCode);
      const target = attempt ? repo.workerRepo.getEnrolmentById(attempt.workerId) : undefined;
      if (!attempt || !target || target.userId !== (req as AuthenticatedRequest).user.id) {
        return { state: "not_found" as const };
      }
      if (attempt.approvedAt) return { state: "approved" as const };
      if (attempt.expiresAt < Date.now()) return { state: "not_found" as const };
      return {
        state: "pending" as const,
        machine: {
          hostname: attempt.hostname,
          platform: attempt.platform,
          arch: attempt.arch,
          gpu: gpuFromHardwareJson(attempt.hardwareJson),
        },
        possibleDuplicate: null,
        reconnectOf: { id: target.id, displayName: target.displayName },
      };
    }
    // Checked BEFORE expiry deliberately -- an approved code that's since
    // aged past its original 15-minute TTL (the worker hasn't redeemed it
    // yet, e.g. it's briefly offline) is still meaningfully "approved" to
    // the browser waiting on this status, not "not found". Expiry only
    // matters for a code that's STILL pending.
    if (worker.approvedAt) return { state: "approved" as const };
    if (worker.enrolmentExpiresAt == null || worker.enrolmentExpiresAt < Date.now()) {
      return { state: "not_found" as const };
    }
    // Surfaced here (not just at approve time) so the approval card can show
    // the "looks like a machine you already have" warning up front, before
    // the human even reaches for the Approve button -- see
    // workerRepo.findPossibleDuplicate's own doc comment for why this exists.
    const userId = (req as AuthenticatedRequest).user.id;
    const possibleDuplicate = repo.workerRepo.findPossibleDuplicate(userId, worker.id) ?? null;
    return {
      state: "pending" as const,
      machine: { hostname: worker.hostname, platform: worker.platform, arch: worker.arch, gpu: worker.gpuModel },
      possibleDuplicate,
    };
  });

  app.post<{ Body: { user_code?: string; confirm_duplicate?: boolean; merge_into?: string } }>(
    "/api/device/approve",
    { config: { rateLimit: { max: 20, timeWindow: "1 minute", keyGenerator: userOrIpKeyGenerator } } },
    async (req): Promise<DeviceApproveResponse> => {
      const userCode = req.body.user_code ?? "";
      const worker = repo.workerRepo.getByUserCode(userCode);
      if (!worker) {
        const attempt = repo.workerRepo.getReenrolmentByUserCode(userCode);
        const target = attempt ? repo.workerRepo.getEnrolmentById(attempt.workerId) : undefined;
        if (!attempt || !target) throw new NotFoundError("invalid or expired code");
        // Enforced here, not left to the UI: a reconnect can only ever be
        // approved by the machine's existing owner (security finding C1).
        if (target.userId !== (req as AuthenticatedRequest).user.id) {
          throw new ForbiddenError("only this machine's owner can approve reconnecting it");
        }
        if (attempt.approvedAt) throw new ConflictError("already approved");
        if (attempt.expiresAt < Date.now()) throw new NotFoundError("invalid or expired code");
        if (!repo.workerRepo.approveReenrolment(attempt.id, target.userId)) throw new ConflictError("already approved");
        return { ok: true, machine: { hostname: attempt.hostname } };
      }
      // Same ordering fix as GET /api/device/status above, for the same
      // reason -- "already approved" is the more accurate and useful error
      // than "expired" for a code whose TTL has technically elapsed but
      // that was in fact successfully approved already.
      if (worker.approvedAt) throw new ConflictError("already approved");
      if (worker.enrolmentExpiresAt == null || worker.enrolmentExpiresAt < Date.now()) {
        throw new NotFoundError("invalid or expired code");
      }
      const userId = (req as AuthenticatedRequest).user.id;
      // Merge takes priority over confirm_duplicate/plain-approve when both
      // are somehow present -- the client only ever sends one. Re-checked
      // here (not just trusted from a prior GET /status or the human's
      // earlier click) since that's a separate, unauthoritative poll: a
      // stale/mismatched merge target is refused rather than trusted blindly.
      if (req.body.merge_into) {
        const duplicateOf = repo.workerRepo.findPossibleDuplicate(userId, worker.id);
        if (!duplicateOf || duplicateOf.id !== req.body.merge_into) {
          throw new BadRequestError("that machine is no longer a suggested match -- refresh and try again");
        }
        assertOwnsWorker(userId, req.body.merge_into);
        // Same reasoning /api/device/start already applies when a known
        // machine_id re-enrols (above): a merge re-issues this worker's
        // identity, so a session from whichever install is being merged away
        // from shouldn't silently stay valid alongside the new one.
        repo.workerRepo.revokeTrust(req.body.merge_into);
        const merged = repo.workerRepo.mergeEnrolment(worker.id, req.body.merge_into);
        return { ok: true, machine: { hostname: merged.hostname }, merged: true };
      }
      // A caller that already got the human's explicit "add it anyway"
      // (confirm_duplicate) skips straight to approving as a new, separate
      // worker.
      if (!req.body.confirm_duplicate) {
        const duplicateOf = repo.workerRepo.findPossibleDuplicate(userId, worker.id);
        if (duplicateOf) {
          return { ok: false, needsConfirmation: true, duplicateOf };
        }
      }
      repo.workerRepo.approve(worker.id, userId);
      return { ok: true, machine: { hostname: worker.hostname } };
    }
  );
}
