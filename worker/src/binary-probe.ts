// BENCHMARKING_PLAN_V8.md §0.7 (flag-probe machinery) and N6's ISA
// provenance. Both memoize on binary *identity* -- (path, mtimeMs, size) --
// so an in-place llama.cpp update re-probes instead of serving a stale
// answer about a different binary that happens to live at the same path.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { statSync } from "node:fs";

const execFileAsync = promisify(execFile);

const PROBE_TIMEOUT_MS = 20_000;

function binaryIdentity(path: string): string {
  try {
    const stat = statSync(path);
    return `${path}:${stat.mtimeMs}:${stat.size}`;
  } catch {
    // A path we cannot stat gets its own identity so a later successful stat
    // (the binary appearing) is treated as a different binary, not a cache hit.
    return `${path}:missing`;
  }
}

const helpTextCache = new Map<string, Promise<string>>();

async function helpText(path: string): Promise<string> {
  const key = binaryIdentity(path);
  const cached = helpTextCache.get(key);
  if (cached) return cached;
  const pending = (async () => {
    // --help exits non-zero on some llama.cpp binaries; execFile rejects on a
    // non-zero code, but the rejection still carries the captured stdout.
    try {
      const { stdout, stderr } = await execFileAsync(path, ["--help"], { timeout: PROBE_TIMEOUT_MS });
      return `${stdout}\n${stderr}`;
    } catch (err) {
      const e = err as { stdout?: string; stderr?: string };
      return `${e.stdout ?? ""}\n${e.stderr ?? ""}`;
    }
  })();
  helpTextCache.set(key, pending);
  return pending;
}

// True when this binary's own help output advertises the flag. An
// unsupported flag disables its axis -- items become `skipped` with a
// reason -- rather than failing every item against a build that never had it.
export async function supportsFlag(path: string, flag: string): Promise<boolean> {
  const text = await helpText(path);
  if (!text.trim()) return false;
  // Match the flag as a whole token: "--spec" must not match
  // "--special-thing", and "-d" must not match "-dev".
  const escaped = flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[\\s,])${escaped}([\\s,=]|$)`, "m").test(text);
}

// Test seam + a way to force a re-probe after an install/uninstall.
export function clearBinaryProbeCache(): void {
  helpTextCache.clear();
  isaCache.clear();
}

// --- N6: ISA provenance -----------------------------------------------------

// llama.cpp's own startup banner reports the RUNNING build's compiled-in +
// runtime-detected dispatch, e.g.
//   system_info: n_threads = 8 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | NEON = 0 |
// This is the binary's actual dispatch, not the host CPU's marketing sheet:
// two CPUs running the same build tag are no longer silently comparable.
export function parseCpuIsaBanner(output: string): string | null {
  const line = output
    .split("\n")
    .find((l) => /system_info\s*:/.test(l) || (/\|/.test(l) && /\bAVX\b|\bNEON\b|\bAVX512\b/.test(l)));
  if (!line) return null;
  const features: string[] = [];
  for (const segment of line.split("|")) {
    const match = /^\s*([A-Za-z0-9_]+)\s*=\s*(\d+)\s*$/.exec(segment);
    if (!match) continue;
    const [, name, value] = match;
    // n_threads and friends are not ISA features.
    if (/^n_/.test(name)) continue;
    if (value !== "0") features.push(name);
  }
  return features.length > 0 ? features.join(" ") : null;
}

const isaCache = new Map<string, Promise<string | null>>();

// Parsed once per binary identity (§0.7's memo pattern). llama-bench prints
// the banner on a trivial invocation; anything unparseable yields null rather
// than a fabricated ISA string.
export async function readCpuIsa(path: string): Promise<string | null> {
  const key = binaryIdentity(path);
  const cached = isaCache.get(key);
  if (cached) return cached;
  const pending = (async () => {
    try {
      const { stdout, stderr } = await execFileAsync(path, ["--version"], { timeout: PROBE_TIMEOUT_MS });
      const fromVersion = parseCpuIsaBanner(`${stdout}\n${stderr}`);
      if (fromVersion) return fromVersion;
    } catch (err) {
      const e = err as { stdout?: string; stderr?: string };
      const parsed = parseCpuIsaBanner(`${e.stdout ?? ""}\n${e.stderr ?? ""}`);
      if (parsed) return parsed;
    }
    return null;
  })();
  isaCache.set(key, pending);
  return pending;
}
