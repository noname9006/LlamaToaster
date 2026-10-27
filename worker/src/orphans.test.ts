import { afterEach, describe, expect, it } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { platform, tmpdir } from "node:os";
import { join } from "node:path";
import { isUnderDir, killProcessesUnder, listProcessesUnder } from "./orphans.js";

describe("isUnderDir", () => {
  // Native paths: resolve() inside isUnderDir is platform-specific.
  const builds = join(tmpdir(), "builds");

  it("matches files inside the dir only, not a sibling sharing its prefix", () => {
    expect(isUnderDir(builds, join(builds, "b1", "llama-server"))).toBe(true);
    expect(isUnderDir(builds, join(tmpdir(), "builds2", "b1", "llama-server"))).toBe(false);
    expect(isUnderDir(builds, builds)).toBe(false);
  });

  it("compares case-insensitively only on Windows", () => {
    const shouted = join(builds.toUpperCase(), "b1", "llama-server");
    expect(isUnderDir(builds, shouted, "win32")).toBe(true);
    expect(isUnderDir(builds, shouted, "linux")).toBe(false);
  });
});

// A real process tree: one long-running copy of a harmless system binary
// inside a fake builds dir, and the same binary from its normal location.
describe("killProcessesUnder", () => {
  const children: ChildProcess[] = [];
  let dir: string | undefined;
  afterEach(() => {
    for (const c of children) c.kill("SIGKILL");
    children.length = 0;
    if (dir) rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  });

  it("ends a process running from the builds dir and leaves the same binary elsewhere alone", async () => {
    const win = platform() === "win32";
    const source = win ? "C:\\Windows\\System32\\PING.EXE" : "/bin/sleep";
    const args = win ? ["-n", "120", "127.0.0.1"] : ["120"];
    dir = mkdtempSync(join(tmpdir(), "llamatoaster-orphans-test-"));
    const copy = join(dir, win ? "llama-server.exe" : "llama-server");
    copyFileSync(source, copy);

    const inside = spawn(copy, args, { stdio: "ignore", windowsHide: true });
    const outside = spawn(source, args, { stdio: "ignore", windowsHide: true });
    children.push(inside, outside);
    const insideExit = new Promise((r) => inside.once("exit", r));
    await new Promise((r) => setTimeout(r, 500));

    expect((await listProcessesUnder(dir)).map((p) => p.pid)).toEqual([inside.pid]);
    const ended = await killProcessesUnder(dir);

    expect(ended.map((p) => p.pid)).toEqual([inside.pid]);
    await insideExit;
    expect(outside.exitCode).toBeNull();
    expect(outside.killed).toBe(false);
  }, 60_000);
});
