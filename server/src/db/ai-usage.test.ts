import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./repo.js";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-ai-usage-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

function makeUser(providerUserId: string) {
  return repo.userRepo.upsertByIdentity("github", { providerUserId, login: providerUserId, avatarUrl: null });
}

describe("aiUsageRepo", () => {
  it("recordUsage increments the (user, day, hour) bucket, starting from zero", () => {
    const user = makeUser("ai-1");
    expect(repo.aiUsageRepo.getHourCount(user.id, "2026-01-01", 5)).toBe(0);
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-01", 5);
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-01", 5);
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-01", 5);
    expect(repo.aiUsageRepo.getHourCount(user.id, "2026-01-01", 5)).toBe(3);
  });

  it("getUserDayCount sums across every hour of that day, not just one", () => {
    const user = makeUser("ai-2");
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-02", 0);
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-02", 0);
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-02", 13);
    repo.aiUsageRepo.recordUsage(user.id, "2026-01-02", 23);
    expect(repo.aiUsageRepo.getUserDayCount(user.id, "2026-01-02")).toBe(4);
    // A different day is unaffected.
    expect(repo.aiUsageRepo.getUserDayCount(user.id, "2026-01-03")).toBe(0);
  });

  it("usage is scoped per user -- one user's calls never count against another's", () => {
    const userA = makeUser("ai-3a");
    const userB = makeUser("ai-3b");
    repo.aiUsageRepo.recordUsage(userA.id, "2026-01-04", 10);
    repo.aiUsageRepo.recordUsage(userA.id, "2026-01-04", 10);
    expect(repo.aiUsageRepo.getUserDayCount(userA.id, "2026-01-04")).toBe(2);
    expect(repo.aiUsageRepo.getUserDayCount(userB.id, "2026-01-04")).toBe(0);
  });

  it("getGlobalDayCount sums across every user for that day (the circuit breaker)", () => {
    const userA = makeUser("ai-4a");
    const userB = makeUser("ai-4b");
    repo.aiUsageRepo.recordUsage(userA.id, "2026-01-05", 1);
    repo.aiUsageRepo.recordUsage(userA.id, "2026-01-05", 2);
    repo.aiUsageRepo.recordUsage(userB.id, "2026-01-05", 1);
    expect(repo.aiUsageRepo.getGlobalDayCount("2026-01-05")).toBe(3);
    // A day with no usage at all reads 0, not null/undefined/NaN (SQL SUM
    // over zero matching rows returns NULL, which must be coalesced).
    expect(repo.aiUsageRepo.getGlobalDayCount("2099-12-31")).toBe(0);
  });
});
