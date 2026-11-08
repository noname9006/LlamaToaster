import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Every Node test file makes its own SQLite directory under the OS temp dir, and
// on Windows a still-open handle stops most of them being removed: about 130 MB
// per run, which has filled the temp drive before and read as a wall of false
// "disk is full" failures (audit 2026-09-27 L3). So the whole run shares one temp
// root, removed here once the test processes that held those handles are gone.
export default function setup(): () => void {
  const previous = { TEMP: process.env.TEMP, TMP: process.env.TMP, TMPDIR: process.env.TMPDIR };
  const root = mkdtempSync(join(tmpdir(), "llamatoaster-vitest-"));
  process.env.TEMP = root;
  process.env.TMP = root;
  process.env.TMPDIR = root;
  return () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    try {
      rmSync(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      // A handle that outlived its process: the next clean-up gets it.
    }
  };
}
