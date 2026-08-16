// Timestamped, leveled logging for the worker. Plain console.log/error calls
// carry no timestamp and are easy to forget at the exact moment something
// interesting happens (e.g. a bench timeout-kill) -- this also mirrors every
// line to a daily file under logs/, since a worker started via
// start-hidden.vbs (Windows, no visible console) or as a background service
// otherwise leaves no trace at all to inspect after the fact.
import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

export type LogLevel = "debug" | "info" | "warn" | "error";

const LEVEL_WEIGHT: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

function isLogLevel(v: string | undefined): v is LogLevel {
  return v === "debug" || v === "info" || v === "warn" || v === "error";
}

let logDir: string | null = null;
let minLevel: LogLevel = isLogLevel(process.env.LOG_LEVEL) ? process.env.LOG_LEVEL : "info";

// Optional extra sink, active only while a run is in flight (see
// worker/src/index.ts's /run handler) -- every line that clears the level
// filter below is *also* appended here, in addition to the console and the
// shared daily file, so each run gets its own dedicated, human-readable log
// file rather than requiring someone to dig it out of the shared stream.
// Only one run executes per worker at a time (the existing `busy` flag), so
// a single module-level path is enough -- no stack/queue needed.
let runLogPath: string | null = null;

export function setRunLogFile(path: string | null): void {
  runLogPath = path;
}

export function configureLogging(dir: string, level?: string): void {
  logDir = dir;
  try {
    if (!existsSync(logDir)) mkdirSync(logDir, { recursive: true });
  } catch (err) {
    logDir = null;
    console.error(
      `${new Date().toISOString()} [ERROR] could not create log dir ${dir}: ${
        err instanceof Error ? err.message : String(err)
      } (continuing with console-only logging)`
    );
  }
  if (isLogLevel(level)) minLevel = level;
}

function logFilePath(dir: string): string {
  const day = new Date().toISOString().slice(0, 10);
  return join(dir, `worker-${day}.log`);
}

function format(level: LogLevel, parts: unknown[]): string {
  const msg = parts
    .map((p) => (typeof p === "string" ? p : p instanceof Error ? p.stack ?? p.message : JSON.stringify(p)))
    .join(" ");
  return `${new Date().toISOString()} [${level.toUpperCase()}] ${msg}`;
}

function write(level: LogLevel, parts: unknown[]): void {
  if (LEVEL_WEIGHT[level] < LEVEL_WEIGHT[minLevel]) return;
  const line = format(level, parts);
  const consoleFn = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
  consoleFn(line);
  if (logDir) {
    try {
      appendFileSync(logFilePath(logDir), line + "\n", "utf8");
    } catch {
      /* best-effort -- logging must never be why a run fails */
    }
  }
  if (runLogPath) {
    try {
      appendFileSync(runLogPath, line + "\n", "utf8");
    } catch {
      /* best-effort, same reasoning as above */
    }
  }
}

export const log = {
  debug: (...parts: unknown[]) => write("debug", parts),
  info: (...parts: unknown[]) => write("info", parts),
  warn: (...parts: unknown[]) => write("warn", parts),
  error: (...parts: unknown[]) => write("error", parts),
};
