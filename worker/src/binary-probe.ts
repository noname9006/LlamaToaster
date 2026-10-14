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
      // windowsHide matters here even though stdio is fully piped: a console
      // child spawned without it still *inherits* this process's console (piping
      // only swaps the std handles, not the console attachment), so llama.cpp's
      // own Windows console reconfiguration inside llama-common.dll
      // (SetConsoleOutputCP(CP_UTF8) / SetConsoleMode(VT) -- confirmed present
      // in every installed build's imports) would execute against the worker's
      // visible PowerShell window. conhost reacts to that with automatic font
      // substitution -- the user's console font silently changes and never
      // reverts. windowsHide (CREATE_NO_WINDOW) gives the probe its own hidden
      // console, keeping those calls off the real window.
      const { stdout, stderr } = await execFileAsync(path, ["--help"], { timeout: PROBE_TIMEOUT_MS, windowsHide: true });
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
  return helpHasFlag(await helpText(path), flag);
}

export function helpHasFlag(text: string, flag: string): boolean {
  if (!text.trim()) return false;
  // Match the flag as a whole token: "--spec" must not match
  // "--special-thing", and "-d" must not match "-dev".
  const escaped = flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[\\s,])${escaped}([\\s,=]|$)`, "m").test(text);
}

// How a run wants the model's weights held in host memory. Both lock them in
// RAM; "no_mmap" also reads them into an owned buffer instead of mapping the
// file (speed runs), "mmap" keeps llama.cpp's default mapping (context tests,
// perplexity).
export type ModelLoadPolicy = "mlock_no_mmap" | "mlock_mmap";

// llama.cpp replaced --mlock / --mmap 0 / --no-mmap with a single
// `-lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio>` (present in b11529,
// where the old flags are gone and rejected). Read from that build's source:
// llama-model-loader.cpp sets use_mmap only for MMAP / MMAP_MLOCK / AUTO, and
// llama-model.cpp sets use_mlock for MLOCK / MMAP_MLOCK -- so "mlock" is
// exactly the old --mlock + no-mmap pairing, "mmap+mlock" the old --mlock alone.
// Older builds get the legacy flags, each probed (§0.7): llama-bench spells
// no-mmap `--mmap 0`, llama-server/llama-perplexity `--no-mmap`.
export function modelLoadArgsFromHelp(help: string, policy: ModelLoadPolicy): string[] {
  if (helpHasFlag(help, "--load-mode")) {
    return ["--load-mode", policy === "mlock_no_mmap" ? "mlock" : "mmap+mlock"];
  }
  const args: string[] = [];
  if (helpHasFlag(help, "--mlock")) args.push("--mlock");
  if (policy === "mlock_no_mmap") {
    if (helpHasFlag(help, "--no-mmap")) args.push("--no-mmap");
    else if (helpHasFlag(help, "--mmap")) args.push("--mmap", "0");
  }
  return args;
}

export async function modelLoadArgs(path: string, policy: ModelLoadPolicy): Promise<string[]> {
  return modelLoadArgsFromHelp(await helpText(path).catch(() => ""), policy);
}

// Test seam: answer this binary's --help probes with `text` instead of
// spawning it.
export function setHelpTextForTest(path: string, text: string): void {
  helpTextCache.set(binaryIdentity(path), Promise.resolve(text));
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
      const { stdout, stderr } = await execFileAsync(path, ["--version"], { timeout: PROBE_TIMEOUT_MS, windowsHide: true });
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

export interface ListedDevice {
  /** The backend's own device name, e.g. "Vulkan0" or "CUDA1". */
  name: string;
  description: string;
  totalMib: number;
  freeMib: number;
}

// One line per device, as llama.cpp's --list-devices prints it:
//   Vulkan0: AMD Radeon RX 6600 XT (8176 MiB, 7378 MiB free)
const LIST_DEVICES_LINE_RE = /^\s*(\S+):\s+(.*?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB free\)\s*$/;

export function parseListDevices(output: string): ListedDevice[] {
  const devices: ListedDevice[] = [];
  for (const line of output.split(/\r?\n/)) {
    const m = LIST_DEVICES_LINE_RE.exec(line);
    if (!m) continue;
    devices.push({ name: m[1], description: m[2], totalMib: Number(m[3]), freeMib: Number(m[4]) });
  }
  return devices;
}

/**
 * Every device this llama.cpp build lists, with its free memory as the build
 * itself sees it -- the probe's second target compares each load's claim
 * against this, device by device. Empty when the build has no --list-devices.
 *
 * Not memoized: free memory is a property of the moment, not of the binary.
 */
export async function readListDevices(path: string): Promise<ListedDevice[]> {
  let output: string;
  try {
    const { stdout, stderr } = await execFileAsync(path, ["--list-devices"], { timeout: PROBE_TIMEOUT_MS, windowsHide: true });
    output = `${stdout}\n${stderr}`;
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string };
    output = `${e.stdout ?? ""}\n${e.stderr ?? ""}`;
  }
  return parseListDevices(output);
}

/** The devices a load can place buffers on: the `-mg` one alone when set
 * (the load runs `-sm none`), every listed device otherwise. */
export function usedListedDevices(devices: readonly ListedDevice[], mainGpu?: number): ListedDevice[] {
  if (mainGpu != null) return devices[mainGpu] ? [devices[mainGpu]] : [];
  return [...devices];
}

/** Total free VRAM on the devices a load will use (usedListedDevices); null
 * when the build lists nothing or names no device at that index. */
export async function readListDevicesFreeMib(path: string, mainGpu?: number): Promise<number | null> {
  return listDevicesFreeMib(await readListDevices(path), mainGpu);
}

export function listDevicesFreeMib(devices: readonly ListedDevice[], mainGpu?: number): number | null {
  const used = usedListedDevices(devices, mainGpu);
  return used.length === 0 ? null : used.reduce((sum, d) => sum + d.freeMib, 0);
}
