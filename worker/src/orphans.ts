import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readdir, readlink } from "node:fs/promises";
import { platform } from "node:os";
import { resolve, sep } from "node:path";

const execFileAsync = promisify(execFile);

// Every llama.cpp process this worker starts (llama-bench, llama-server,
// llama-fit-params, ...) runs from a build under its own buildsDir. A worker
// that is killed outright -- taskkill /F or a closed console window, which on
// Windows never deliver a signal -- leaves them running, holding VRAM and the
// MTP server port, and the next run then measures next to a phantom process
// (audit 2026-10-11 N10). These find and end such processes. They match on
// where the executable lives, never on its name, so a llama.cpp the user runs
// from anywhere else is left alone.

export interface RunningProcess {
  pid: number;
  path: string;
}

export function isUnderDir(dir: string, path: string, os: NodeJS.Platform = platform()): boolean {
  const prefix = resolve(dir) + sep;
  return os === "win32" ? path.toLowerCase().startsWith(prefix.toLowerCase()) : path.startsWith(prefix);
}

async function listProcesses(): Promise<RunningProcess[]> {
  const os = platform();
  if (os === "win32") {
    const { stdout } = await execFileAsync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_Process -Filter 'ExecutablePath IS NOT NULL' | ForEach-Object { \"$($_.ProcessId)`t$($_.ExecutablePath)\" }",
      ],
      { timeout: 20_000, windowsHide: true, maxBuffer: 16 * 1024 * 1024 }
    );
    return stdout
      .split(/\r?\n/)
      .map((line) => line.split("\t"))
      .filter((parts) => parts.length === 2 && /^\d+$/.test(parts[0]))
      .map(([pid, path]) => ({ pid: Number(pid), path }));
  }
  if (os === "linux") {
    const entries = await readdir("/proc");
    const found: RunningProcess[] = [];
    for (const name of entries) {
      if (!/^\d+$/.test(name)) continue;
      try {
        found.push({ pid: Number(name), path: await readlink(`/proc/${name}/exe`) });
      } catch {
        // gone already, or not ours to inspect
      }
    }
    return found;
  }
  // macOS: ps prints each process's full executable path in comm.
  const { stdout } = await execFileAsync("ps", ["-axo", "pid=,comm="], { timeout: 20_000 });
  return stdout
    .split("\n")
    .map((line) => line.trim().match(/^(\d+)\s+(.+)$/))
    .filter((m): m is RegExpMatchArray => m !== null)
    .map((m) => ({ pid: Number(m[1]), path: m[2] }));
}

export async function listProcessesUnder(dir: string): Promise<RunningProcess[]> {
  return (await listProcesses()).filter((p) => p.pid !== process.pid && isUnderDir(dir, p.path));
}

// Ends every process running from `dir`, returning what it ended. Best effort:
// a process that exits on its own in the meantime is simply skipped.
export async function killProcessesUnder(dir: string): Promise<RunningProcess[]> {
  const victims = await listProcessesUnder(dir);
  for (const p of victims) {
    try {
      if (platform() === "win32") {
        await execFileAsync("taskkill", ["/PID", String(p.pid), "/T", "/F"], { timeout: 10_000, windowsHide: true });
      } else {
        process.kill(p.pid, "SIGKILL");
      }
    } catch {
      // already gone
    }
  }
  return victims;
}
