import { existsSync, renameSync, rmSync } from "node:fs";

// Keeps bytes that don't match a download's expected hash away from the model's
// path. They used to be renamed into place first and checked afterwards, so a
// corrupted file sat at the target, and every retry then saw it as "already
// present", skipped the download and failed on the same hash forever (audit
// 2026-10-11 N14). Separate from the job itself so it can be tested with real
// files.

export interface HashedFile {
  sha256: string;
  byteLength: number;
}

type HashFile = (path: string) => Promise<HashedFile>;

function matches(sha256: string, expected: string | null): boolean {
  return expected === null || sha256.toLowerCase() === expected.toLowerCase();
}

// Named so the model scanners' ".part" rule skips it: it is not a model, and not
// a partial of this download to resume either.
export function setAsidePath(target: string): string {
  return `${target}.replaced.part`;
}

/**
 * The file already at the download's target, if it is the one wanted. One with a
 * different hash is moved aside for as long as the download runs, and null sends the
 * caller on to a fresh download. It goes back if the download fails
 * (restoreSetAside), and is deleted once the replacement is verified (dropSetAside),
 * so no multi-gigabyte file stays on disk that the scanners skip and nothing shows
 * (audit 2026-11-01 E8).
 */
export async function keepExistingTarget(target: string, expected: string | null, hashFile: HashFile): Promise<HashedFile | null> {
  if (!existsSync(target)) return null;
  const hashed = await hashFile(target);
  if (matches(hashed.sha256, expected)) {
    // An attempt can die between verifying its file and removing what that replaced.
    if (expected !== null) dropSetAside(target);
    return hashed;
  }
  renameSync(target, setAsidePath(target));
  return null;
}

/** Puts a set-aside file back, unless a file has since taken its place. True if it did. */
export function restoreSetAside(target: string): boolean {
  const aside = setAsidePath(target);
  if (!existsSync(aside) || existsSync(target)) return false;
  renameSync(aside, target);
  return true;
}

/** The replacement is verified: what it replaced can go. True if there was one. */
export function dropSetAside(target: string): boolean {
  const aside = setAsidePath(target);
  if (!existsSync(aside)) return false;
  rmSync(aside, { force: true });
  return true;
}

/**
 * Moves a finished .part into place, only if it is the file that was asked for.
 * A bad one is deleted: a resume of it could never come out right, because the
 * bad bytes are already in it.
 */
export async function finishPart(partPath: string, target: string, expected: string | null, hashFile: HashFile): Promise<HashedFile> {
  const hashed = await hashFile(partPath);
  if (!matches(hashed.sha256, expected)) {
    rmSync(partPath, { force: true });
    throw new Error(`SHA256 mismatch: expected ${expected} but got ${hashed.sha256} (download discarded, retrying starts over)`);
  }
  renameSync(partPath, target);
  return hashed;
}
