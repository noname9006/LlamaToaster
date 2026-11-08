import { describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { dropSetAside, finishPart, keepExistingTarget, restoreSetAside, setAsidePath } from "./downloadVerify.js";

const dir = mkdtempSync(join(tmpdir(), "llamatoaster-download-test-"));
let seq = 0;
const path = () => join(dir, `f${++seq}.gguf`);
const sha = (text: string) => createHash("sha256").update(text).digest("hex");
const hashFile = async (p: string) => {
  const bytes = readFileSync(p);
  return { sha256: createHash("sha256").update(bytes).digest("hex"), byteLength: bytes.length };
};

describe("keepExistingTarget", () => {
  it("nothing there means download", async () => {
    expect(await keepExistingTarget(path(), sha("x"), hashFile)).toBeNull();
  });

  it("keeps the right file, reusing its hash", async () => {
    const target = path();
    writeFileSync(target, "good");
    expect(await keepExistingTarget(target, sha("good"), hashFile)).toEqual({ sha256: sha("good"), byteLength: 4 });
    expect(existsSync(target)).toBe(true);
  });

  it("matches the expected hash whatever its case", async () => {
    const target = path();
    writeFileSync(target, "good");
    expect(await keepExistingTarget(target, sha("good").toUpperCase(), hashFile)).not.toBeNull();
  });

  it("keeps any file when nothing says what it should be", async () => {
    const target = path();
    writeFileSync(target, "whatever");
    expect((await keepExistingTarget(target, null, hashFile))?.byteLength).toBe(8);
  });

  it("moves a wrong file aside, unharmed, and asks for a download: a retry can recover", async () => {
    const target = path();
    writeFileSync(target, "bad");
    expect(await keepExistingTarget(target, sha("good"), hashFile)).toBeNull();
    expect(existsSync(target)).toBe(false);
    expect(readFileSync(setAsidePath(target), "utf8")).toBe("bad");
    expect(setAsidePath(target).endsWith(".part")).toBe(true);
  });
});

// Audit 2026-11-01 E8: a set-aside file used to stay forever, hidden from the model
// scanners by its .part name, and a failed download lost the original from view.
describe("the set-aside file", () => {
  it("goes back when the download fails", async () => {
    const target = path();
    writeFileSync(target, "theirs");
    await keepExistingTarget(target, sha("good"), hashFile);
    expect(restoreSetAside(target)).toBe(true);
    expect(readFileSync(target, "utf8")).toBe("theirs");
    expect(existsSync(setAsidePath(target))).toBe(false);
  });

  it("never overwrites a file that has taken its place", async () => {
    const target = path();
    writeFileSync(target, "theirs");
    await keepExistingTarget(target, sha("good"), hashFile);
    writeFileSync(target, "good");
    expect(restoreSetAside(target)).toBe(false);
    expect(readFileSync(target, "utf8")).toBe("good");
  });

  it("is deleted once the replacement is verified, with nothing left behind", async () => {
    const target = path();
    writeFileSync(target, "theirs");
    await keepExistingTarget(target, sha("good"), hashFile);
    writeFileSync(`${target}.part`, "good");
    await finishPart(`${target}.part`, target, sha("good"), hashFile);
    expect(dropSetAside(target)).toBe(true);
    expect(existsSync(setAsidePath(target))).toBe(false);
    expect(readFileSync(target, "utf8")).toBe("good");
  });

  it("does nothing when there is none", () => {
    const target = path();
    expect(restoreSetAside(target)).toBe(false);
    expect(dropSetAside(target)).toBe(false);
  });

  it("stays through a failed replacement, ready to go back", async () => {
    const target = path();
    writeFileSync(target, "theirs");
    await keepExistingTarget(target, sha("good"), hashFile);
    writeFileSync(`${target}.part`, "bad");
    await expect(finishPart(`${target}.part`, target, sha("good"), hashFile)).rejects.toThrow(/SHA256 mismatch/);
    expect(restoreSetAside(target)).toBe(true);
    expect(readFileSync(target, "utf8")).toBe("theirs");
  });

  it("is cleared by the next attempt if one died just after verifying", async () => {
    const target = path();
    writeFileSync(setAsidePath(target), "theirs");
    writeFileSync(target, "good");
    expect(await keepExistingTarget(target, sha("good"), hashFile)).not.toBeNull();
    expect(existsSync(setAsidePath(target))).toBe(false);
  });

  it("survives a next attempt that can't say which file is right", async () => {
    const target = path();
    writeFileSync(setAsidePath(target), "theirs");
    writeFileSync(target, "whatever");
    expect(await keepExistingTarget(target, null, hashFile)).not.toBeNull();
    expect(existsSync(setAsidePath(target))).toBe(true);
  });
});

describe("finishPart", () => {
  it("puts a good file in place", async () => {
    const target = path();
    const part = `${target}.part`;
    writeFileSync(part, "good");
    expect(await finishPart(part, target, sha("good"), hashFile)).toEqual({ sha256: sha("good"), byteLength: 4 });
    expect(readFileSync(target, "utf8")).toBe("good");
    expect(existsSync(part)).toBe(false);
  });

  it("never lets a bad one reach the target, and leaves nothing to resume", async () => {
    const target = path();
    const part = `${target}.part`;
    writeFileSync(part, "bad");
    await expect(finishPart(part, target, sha("good"), hashFile)).rejects.toThrow(/SHA256 mismatch/);
    expect(existsSync(target)).toBe(false);
    expect(existsSync(part)).toBe(false);
  });

  it("with no expected hash, anything goes in", async () => {
    const target = path();
    const part = `${target}.part`;
    writeFileSync(part, "x");
    await finishPart(part, target, null, hashFile);
    expect(existsSync(target)).toBe(true);
  });
});
