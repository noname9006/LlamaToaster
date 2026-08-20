import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./repo.js";
import type { HardwareInfo } from "../../../shared/types.js";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-device-enrolment-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;
let hashToken: (t: string) => string;

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
  ({ hashToken } = await import("../session.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

function pendingOpts(overrides: Partial<Parameters<typeof repo.workerRepo.createPending>[0]> = {}) {
  return {
    machineId: "machine-1",
    hostname: "gpu-tower",
    platform: "linux",
    arch: "x64",
    deviceCode: "raw-device-code-1",
    userCode: "ABCD-EFGH",
    expiresAt: Date.now() + 15 * 60 * 1000,
    ...overrides,
  };
}

// A stable, "same physical box" hardware fixture for findPossibleDuplicate/
// mergeEnrolment tests -- override individual fields to simulate a
// genuinely different machine.
function hw(overrides: Partial<HardwareInfo> = {}): HardwareInfo {
  return {
    platform: "linux",
    arch: "x64",
    cpu: { manufacturer: "AMD", brand: "AMD Ryzen 9 5900X", flags: [], cores: 24 },
    gpu: [{ vendor: "NVIDIA", model: "RTX 4090", vram_mb: 24576, vram_dynamic: false }],
    mem_total_bytes: 64 * 1024 * 1024 * 1024,
    ...overrides,
  };
}

describe("workerRepo.createPending", () => {
  it("creates a brand-new machine with display_name defaulting to hostname, unowned", () => {
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-create-1", userCode: "CRT1-CODE" }));
    expect(w.displayName).toBe("gpu-tower");
    expect(w.hostname).toBe("gpu-tower");
    expect(w.userId).toBeNull();
    expect(w.approvedAt).toBeNull();
    expect(w.enrolmentExpiresAt).toBeGreaterThan(Date.now());
  });

  it("the device_code is stored hashed, never as plaintext -- lookup only works via the hash", () => {
    repo.workerRepo.createPending(pendingOpts({ machineId: "m-create-2", deviceCode: "secret-raw-code", userCode: "CRT2-CODE" }));
    expect(repo.workerRepo.getByEnrolmentCodeHash("secret-raw-code")).toBeUndefined();
    expect(repo.workerRepo.getByEnrolmentCodeHash(hashToken("secret-raw-code"))).toBeDefined();
  });

  it("two different machines can never share a user_code while both are pending (partial unique index)", () => {
    repo.workerRepo.createPending(pendingOpts({ machineId: "m-collide-1", userCode: "SAME-CODE" }));
    expect(() => repo.workerRepo.createPending(pendingOpts({ machineId: "m-collide-2", userCode: "SAME-CODE" }))).toThrow();
  });
});

describe("workerRepo.getByUserCode / getByEnrolmentCodeHash", () => {
  it("resolves a pending enrolment by its human-facing user_code", () => {
    repo.workerRepo.createPending(pendingOpts({ machineId: "m-lookup-1", userCode: "LOOK-UP01" }));
    const found = repo.workerRepo.getByUserCode("LOOK-UP01");
    expect(found?.hostname).toBe("gpu-tower");
    expect(found?.approvedAt).toBeNull();
  });

  it("returns undefined for an unknown code", () => {
    expect(repo.workerRepo.getByUserCode("NOPE-NOPE")).toBeUndefined();
    expect(repo.workerRepo.getByEnrolmentCodeHash("not-a-real-hash")).toBeUndefined();
  });
});

describe("workerRepo.approve", () => {
  it("sets user_id + approvedAt, but leaves the enrolment code in place for the worker's own poll to redeem", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "approver-1", login: "approver", avatarUrl: null });
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-approve-1", deviceCode: "approve-device-code", userCode: "APRV-CODE" }));
    expect(w.approvedAt).toBeNull();

    repo.workerRepo.approve(w.id, user.id);

    const after = repo.workerRepo.getEnrolmentById(w.id)!;
    expect(after.userId).toBe(user.id);
    expect(after.approvedAt).not.toBeNull();
    // Still findable by hash -- approve() must NOT consume the code itself,
    // or the worker's own /api/device/token poll could never redeem it.
    expect(repo.workerRepo.getByEnrolmentCodeHash(hashToken("approve-device-code"))?.id).toBe(w.id);
  });
});

describe("workerRepo.clearEnrolmentCode", () => {
  it("is the true one-shot consumption point -- clears both codes so neither can be reused", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "clear-1", login: "clear-user", avatarUrl: null });
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-clear-1", deviceCode: "clear-device-code", userCode: "CLR1-CODE" }));
    repo.workerRepo.approve(w.id, user.id);

    repo.workerRepo.clearEnrolmentCode(w.id);

    expect(repo.workerRepo.getByEnrolmentCodeHash(hashToken("clear-device-code"))).toBeUndefined();
    expect(repo.workerRepo.getByUserCode("CLR1-CODE")).toBeUndefined();
    // Ownership itself survives -- clearing the codes must never un-approve the machine.
    const after = repo.workerRepo.getEnrolmentById(w.id)!;
    expect(after.userId).toBe(user.id);
    expect(after.approvedAt).not.toBeNull();
  });

  it("clearing one machine's code frees up its user_code for a NEW pending enrolment to reuse", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "reuse-1", login: "reuse-user", avatarUrl: null });
    const w1 = repo.workerRepo.createPending(pendingOpts({ machineId: "m-reuse-1", userCode: "REUS-CODE" }));
    repo.workerRepo.approve(w1.id, user.id);
    repo.workerRepo.clearEnrolmentCode(w1.id);

    // Would have thrown (partial unique index) if the code were still "pending" on w1.
    expect(() => repo.workerRepo.createPending(pendingOpts({ machineId: "m-reuse-2", userCode: "REUS-CODE" }))).not.toThrow();
  });
});

describe("workerRepo.reissueEnrolment", () => {
  it("re-enrolling a still-pending machine refreshes the code but leaves it unowned", () => {
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-reissue-pending", userCode: "RIP1-CODE", deviceCode: "old-code" }));
    const reissued = repo.workerRepo.reissueEnrolment(w.id, {
      hostname: "gpu-tower",
      platform: "linux",
      arch: "x64",
      deviceCode: "new-code",
      userCode: "RIP2-CODE",
      expiresAt: Date.now() + 15 * 60 * 1000,
    });
    expect(reissued.userId).toBeNull();
    expect(repo.workerRepo.getByEnrolmentCodeHash(hashToken("old-code"))).toBeUndefined();
    expect(repo.workerRepo.getByEnrolmentCodeHash(hashToken("new-code"))?.id).toBe(w.id);
  });

  it("re-enrolling an ALREADY-OWNED machine preserves ownership -- reconnecting never requires re-approval", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "reissue-owner", login: "owner", avatarUrl: null });
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-reissue-owned", userCode: "RIO1-CODE", deviceCode: "owned-code-1" }));
    repo.workerRepo.approve(w.id, user.id);
    repo.workerRepo.clearEnrolmentCode(w.id); // simulates a completed first enrolment

    // Machine reconnects (expired session, rebuilt disk, ...).
    const reissued = repo.workerRepo.reissueEnrolment(w.id, {
      hostname: "gpu-tower",
      platform: "linux",
      arch: "x64",
      deviceCode: "owned-code-2",
      userCode: "RIO2-CODE",
      expiresAt: Date.now() + 15 * 60 * 1000,
    });
    expect(reissued.userId).toBe(user.id); // still owned, no re-approval needed
    expect(reissued.approvedAt).not.toBeNull();
  });

  it("does not clobber a user-renamed display_name", async () => {
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-reissue-name", userCode: "RIN1-CODE" }));
    // No rename API exists yet (display_name's own "user-editable inline"
    // story per §3.3 is a client feature outside Stage 3's actual code) --
    // simulate a prior rename directly to test reissueEnrolment's own
    // contract: it must never touch display_name, regardless of how it got
    // its current value.
    const { getDb } = await import("./migrate.js");
    getDb().prepare(`UPDATE workers SET display_name = ? WHERE id = ?`).run("My Renamed Box", w.id);
    const before = repo.workerRepo.getWorker(w.id)!;
    expect(before.displayName).toBe("My Renamed Box");
    repo.workerRepo.reissueEnrolment(w.id, {
      hostname: "gpu-tower",
      platform: "linux",
      arch: "x64",
      deviceCode: "rename-code",
      userCode: "RIN2-CODE",
      expiresAt: Date.now() + 15 * 60 * 1000,
    });
    const after = repo.workerRepo.getWorker(w.id)!;
    expect(after.displayName).toBe(before.displayName);
  });
});

describe("workerRepo.findPossibleDuplicate", () => {
  it("returns undefined for a candidate with neither hostname nor hardware to compare", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-null-host", login: "dup1", avatarUrl: null });
    const approved = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-null-approved", hostname: "gpu-tower", userCode: "DUPN-CODE" }));
    repo.workerRepo.approve(approved.id, user.id);
    // getOrCreateByMachineId's INSERT never sets `hostname` (only
    // display_name) -- the one real code path that leaves it null, same as
    // an un-heartbeated Stage 1 shared-secret worker.
    const candidate = repo.workerRepo.getOrCreateByMachineId("m-dup-null-candidate", "unused-display-name");
    expect(repo.workerRepo.findPossibleDuplicate(user.id, candidate.id)).toBeUndefined();
  });

  it("finds an already-approved machine the same user owns with a matching hostname", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-match-1", login: "dup2", avatarUrl: null });
    const old = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-old", hostname: "dup-tower", userCode: "DUP1-CODE" }));
    repo.workerRepo.approve(old.id, user.id);
    const candidate = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-new", hostname: "dup-tower", userCode: "DUP1-CAND" }));

    const found = repo.workerRepo.findPossibleDuplicate(user.id, candidate.id);
    expect(found).toMatchObject({ id: old.id, displayName: "dup-tower", hostnameMatch: true, hardwareMatch: false });
  });

  it("finds a match via hardware fingerprint even when the hostname differs", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-hw-1", login: "dup-hw", avatarUrl: null });
    const old = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-hw-old", hostname: "old-name", userCode: "DUPH-CODE", hardware: hw() }));
    repo.workerRepo.approve(old.id, user.id);
    const candidate = repo.workerRepo.createPending(
      pendingOpts({ machineId: "m-dup-hw-new", hostname: "totally-different-name", userCode: "DUPH-CAND", hardware: hw() })
    );

    const found = repo.workerRepo.findPossibleDuplicate(user.id, candidate.id);
    expect(found).toMatchObject({ id: old.id, hostnameMatch: false, hardwareMatch: true });
  });

  it("does not match when neither hostname nor hardware line up", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-nomatch-1", login: "dup-nomatch", avatarUrl: null });
    const old = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-nomatch-old", hostname: "box-a", userCode: "DUPX-CODE", hardware: hw() }));
    repo.workerRepo.approve(old.id, user.id);
    const candidate = repo.workerRepo.createPending(
      pendingOpts({
        machineId: "m-dup-nomatch-new",
        hostname: "box-b",
        userCode: "DUPX-CAND",
        hardware: hw({ cpu: { manufacturer: "Intel", brand: "Intel Core i9-13900K", flags: [], cores: 32 } }),
      })
    );

    expect(repo.workerRepo.findPossibleDuplicate(user.id, candidate.id)).toBeUndefined();
  });

  it("tolerates a small difference in reported total RAM but not a large one", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-ram-1", login: "dup-ram", avatarUrl: null });
    const baseRam = 64 * 1024 * 1024 * 1024;
    const old = repo.workerRepo.createPending(
      pendingOpts({ machineId: "m-dup-ram-old", hostname: "ram-box-old", userCode: "DUPR-CODE", hardware: hw({ mem_total_bytes: baseRam }) })
    );
    repo.workerRepo.approve(old.id, user.id);

    const withinTolerance = repo.workerRepo.createPending(
      pendingOpts({
        machineId: "m-dup-ram-close",
        hostname: "ram-box-close",
        userCode: "DUPR-CAND1",
        hardware: hw({ mem_total_bytes: Math.round(baseRam * 1.02) }),
      })
    );
    expect(repo.workerRepo.findPossibleDuplicate(user.id, withinTolerance.id)).toMatchObject({ id: old.id, hardwareMatch: true });

    const outsideTolerance = repo.workerRepo.createPending(
      pendingOpts({
        machineId: "m-dup-ram-far",
        hostname: "ram-box-far",
        userCode: "DUPR-CAND2",
        hardware: hw({ mem_total_bytes: Math.round(baseRam * 1.5) }),
      })
    );
    expect(repo.workerRepo.findPossibleDuplicate(user.id, outsideTolerance.id)).toBeUndefined();
  });

  it("excludes the row itself (so a machine never flags as a duplicate of its own row)", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-self", login: "dup3", avatarUrl: null });
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-self", hostname: "self-tower", userCode: "DUP2-CODE" }));
    repo.workerRepo.approve(w.id, user.id);

    expect(repo.workerRepo.findPossibleDuplicate(user.id, w.id)).toBeUndefined();
  });

  it("never matches another user's machine, even with the identical hostname", () => {
    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-owner", login: "dup-owner", avatarUrl: null });
    const other = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-other", login: "dup-other", avatarUrl: null });
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-cross-user", hostname: "shared-name", userCode: "DUP3-CODE" }));
    repo.workerRepo.approve(w.id, owner.id);
    const candidate = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-cross-user-2", hostname: "shared-name", userCode: "DUP3-CAND" }));

    expect(repo.workerRepo.findPossibleDuplicate(other.id, candidate.id)).toBeUndefined();
  });

  it("ignores a row that has a user_id but hasn't actually been approved", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "dup-pending", login: "dup4", avatarUrl: null });
    const w = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-pending-1", hostname: "pending-tower", userCode: "DUP4-CODE" }));
    // approve() always sets user_id and approved_at together -- there's no
    // real code path that sets one without the other. Simulate it anyway to
    // confirm findPossibleDuplicate's approved_at check is load-bearing,
    // not just incidentally redundant with the user_id filter.
    const { getDb } = await import("./migrate.js");
    getDb().prepare(`UPDATE workers SET user_id = ? WHERE id = ?`).run(user.id, w.id);
    const candidate = repo.workerRepo.createPending(pendingOpts({ machineId: "m-dup-pending-2", hostname: "pending-tower", userCode: "DUP4-CAND" }));

    expect(repo.workerRepo.findPossibleDuplicate(user.id, candidate.id)).toBeUndefined();
  });
});

describe("workerRepo.mergeEnrolment", () => {
  it("re-points an existing approved worker onto a freshly-enrolling machine's identity", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "merge-1", login: "merge-user", avatarUrl: null });
    const target = repo.workerRepo.createPending(pendingOpts({ machineId: "m-merge-target", hostname: "old-box", userCode: "MRG1-CODE" }));
    repo.workerRepo.approve(target.id, user.id);
    const { getDb } = await import("./migrate.js");
    getDb().prepare(`UPDATE workers SET display_name = ? WHERE id = ?`).run("My Renamed Box", target.id);

    const pending = repo.workerRepo.createPending(
      pendingOpts({
        machineId: "m-merge-pending",
        hostname: "new-box",
        userCode: "MRG2-CODE",
        deviceCode: "merge-device-code",
        hardware: hw(),
      })
    );

    const merged = repo.workerRepo.mergeEnrolment(pending.id, target.id);

    // Continuity: same worker id, and everything a human/history would care
    // about that belonged to the TARGET survives untouched.
    expect(merged.id).toBe(target.id);
    expect(merged.displayName).toBe("My Renamed Box");
    // Transplanted from the pending row: the target now IS this machine.
    expect(merged.machineId).toBe("m-merge-pending");
    expect(merged.hostname).toBe("new-box");
    expect(merged.hardware).toMatchObject({ cpu: { brand: hw().cpu.brand } });

    // The pending row is gone -- nothing left to separately approve/delete.
    expect(repo.workerRepo.getEnrolmentById(pending.id)).toBeUndefined();

    // The connecting worker's own device_code now redeems against the
    // TARGET row (already approved), not a row that no longer exists.
    expect(repo.workerRepo.getByEnrolmentCodeHash(hashToken("merge-device-code"))?.id).toBe(target.id);

    const enrolment = repo.workerRepo.getEnrolmentById(target.id)!;
    expect(enrolment.userId).toBe(user.id);
    expect(enrolment.approvedAt).not.toBeNull();
  });
});

describe("sessionRepo.revokeWorkerSessions", () => {
  it("revokes only the given worker's own sessions, leaving user browser sessions and other workers alone", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "revoke-worker-1", login: "revoke-user", avatarUrl: null });
    const browserSession = repo.sessionRepo.create(user.id, { label: "browser" });
    const workerASession = repo.sessionRepo.create(user.id, { isWorker: true, workerId: "worker-a" });
    const workerBSession = repo.sessionRepo.create(user.id, { isWorker: true, workerId: "worker-b" });

    repo.sessionRepo.revokeWorkerSessions("worker-a");

    expect(repo.sessionRepo.getByTokenHash(hashToken(workerASession.token))).toBeUndefined();
    expect(repo.sessionRepo.getByTokenHash(hashToken(workerBSession.token))).toBeDefined();
    expect(repo.sessionRepo.getByTokenHash(hashToken(browserSession.token))).toBeDefined();
  });
});
