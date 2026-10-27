// One long-lived powershell.exe that runs short scripts on request, so the
// sampler's VRAM probe (every ~6s for the whole test) does not start a fresh
// powershell.exe each time. Measured on this machine: a fresh spawn costs
// ~1.4s wall, ~430ms CPU and a ~63MB process per probe, all of it landing on
// the box while the benchmark is being timed.
//
// Protocol: one line per request on stdin -- the script, base64 UTF-16LE --
// answered by its output lines and then an end marker. Requests run one at a
// time. The session exits by itself when stdin closes (the worker is gone),
// and this side kills it after IDLE_MS without a request, so an idle worker
// does not keep a PowerShell resident between tests.
//
// Any failure (no spawn, a crash, a timeout) rejects; callers fall back to a
// one-shot powershell.exe, which is exactly the old behaviour.

import { spawn, type ChildProcess } from "node:child_process";

const END = "<<<PS-SESSION-END>>>";
const IDLE_MS = 30_000;

const LOOP = [
  `$ErrorActionPreference = 'SilentlyContinue'`,
  `while ($true) {`,
  `  $line = [Console]::In.ReadLine()`,
  `  if ($null -eq $line) { break }`,
  `  try {`,
  `    $script = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($line))`,
  `    & ([scriptblock]::Create($script)) | ForEach-Object { [Console]::Out.WriteLine([string]$_) }`,
  `  } catch { }`,
  `  [Console]::Out.WriteLine('${END}')`,
  `  [Console]::Out.Flush()`,
  `}`,
].join("\n");

interface Pending {
  script: string;
  timeoutMs: number;
  resolve: (stdout: string) => void;
  reject: (err: Error) => void;
}

let child: ChildProcess | null = null;
let buffered = "";
let output: string[] = [];
let current: (Pending & { timer: NodeJS.Timeout }) | null = null;
const queue: Pending[] = [];
let idleTimer: NodeJS.Timeout | null = null;

function shutdown(reason: string): void {
  const c = child;
  child = null;
  buffered = "";
  output = [];
  if (c && c.exitCode == null) c.kill();
  if (current) {
    clearTimeout(current.timer);
    current.reject(new Error(reason));
    current = null;
  }
  for (const p of queue.splice(0)) p.reject(new Error(reason));
}

function ensureChild(): ChildProcess {
  if (child) return child;
  const encoded = Buffer.from(LOOP, "utf16le").toString("base64");
  const c = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], {
    windowsHide: true,
    stdio: ["pipe", "pipe", "ignore"],
  });
  child = c;
  // Never what keeps the worker (or a one-off script) alive.
  c.unref();
  (c.stdin as unknown as { unref?: () => void } | null)?.unref?.();
  (c.stdout as unknown as { unref?: () => void } | null)?.unref?.();
  c.stdout?.on("data", (chunk: Buffer) => {
    buffered += chunk.toString();
    const lines = buffered.split(/\r?\n/);
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      if (line !== END) {
        output.push(line);
        continue;
      }
      const done = current;
      current = null;
      const text = output.join("\n");
      output = [];
      if (done) {
        clearTimeout(done.timer);
        done.resolve(text);
      }
      next();
    }
  });
  c.on("error", () => {
    if (child === c) shutdown("powershell session failed");
  });
  c.on("exit", () => {
    if (child === c) shutdown("powershell session exited");
  });
  return c;
}

function next(): void {
  if (current) return;
  const p = queue.shift();
  if (!p) {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => shutdown("idle"), IDLE_MS);
    idleTimer.unref();
    return;
  }
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
  let c: ChildProcess;
  try {
    c = ensureChild();
  } catch (err) {
    p.reject(err instanceof Error ? err : new Error(String(err)));
    next();
    return;
  }
  const timer = setTimeout(() => shutdown("powershell session timed out"), p.timeoutMs);
  current = { ...p, timer };
  c.stdin?.write(Buffer.from(p.script, "utf16le").toString("base64") + "\n");
}

/** Runs `script` in the shared session and resolves with its output lines. */
export function runInPowerShellSession(script: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    queue.push({ script, timeoutMs, resolve, reject });
    next();
  });
}

/** Shared session first; on any failure, one fresh powershell.exe via `fallback`. */
export async function runPowerShell(
  script: string,
  timeoutMs: number,
  fallback: () => Promise<{ stdout: string }>
): Promise<string> {
  // LLAMATOASTER_PS_SESSION=0 forces the old one-shot path (A/B, escape hatch).
  if (process.env.LLAMATOASTER_PS_SESSION === "0") return (await fallback()).stdout;
  try {
    return await runInPowerShellSession(script, timeoutMs);
  } catch {
    return (await fallback()).stdout;
  }
}
