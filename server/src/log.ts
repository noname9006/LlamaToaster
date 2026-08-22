// Timestamps for the handful of call sites that log via plain console.*
// instead of Fastify's app.log -- background services (hf-index's tick
// loop) and one-time startup code (db/migrate.ts, github-releases.ts) run
// outside any request context, so they have no `app.log` to reach for.
// Fastify's own logger already timestamps every line it produces; this
// brings the same consistency to these standalone lines. Mirrors
// worker/src/log.ts's format so `[TAG] message` lines look the same in both
// processes' output.
export type LogLevel = "debug" | "info" | "warn" | "error";

function format(level: LogLevel, parts: unknown[]): string {
  const msg = parts
    .map((p) => (typeof p === "string" ? p : p instanceof Error ? (p.stack ?? p.message) : JSON.stringify(p)))
    .join(" ");
  return `${new Date().toISOString()} [${level.toUpperCase()}] ${msg}`;
}

function write(level: LogLevel, parts: unknown[]): void {
  const line = format(level, parts);
  const consoleFn = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
  consoleFn(line);
}

export const log = {
  debug: (...parts: unknown[]) => write("debug", parts),
  info: (...parts: unknown[]) => write("info", parts),
  warn: (...parts: unknown[]) => write("warn", parts),
  error: (...parts: unknown[]) => write("error", parts),
};
