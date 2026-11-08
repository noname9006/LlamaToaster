import { repo } from "./db/repo.js";
import { HttpError } from "./errors.js";
import { log } from "./log.js";

// How many bytes one owner may add to this server's disk per hour, across
// result reports, run logs, fit answers, imports and new run rows. Registration
// is open, so without a total one account could fill the disk and take every
// tenant down, even though each single request is bounded (audit 2026-11-01 D1).
// A 20,000-item sweep writes about 40 MB of results; this is far above real use.
export const DEFAULT_INGEST_BYTES_PER_HOUR = 200 * 1024 * 1024;

// Rough size of one pre-created run_items row, including its indexes.
export const RUN_ITEM_ROW_BYTES = 400;

function limit(): number {
  const raw = process.env.INGEST_BYTES_PER_HOUR;
  if (raw == null || raw.trim() === "") return DEFAULT_INGEST_BYTES_PER_HOUR;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_INGEST_BYTES_PER_HOUR;
}

// Spends `bytes` of `scope`'s hour (a user id, or worker:<id> for an ownerless
// machine), or refuses with a 429 and spends nothing.
export function chargeIngest(scope: string, bytes: number): void {
  if (repo.takeIngestBudget(scope, bytes, limit())) return;
  log.warn(`[ingest-budget] ${scope} is over its hourly write budget -- refused ${bytes} bytes`);
  throw new HttpError("too much data written this hour -- try again later", 429);
}

// What a parsed JSON body cost, using the header when it's there (the common
// case) to avoid serializing it a second time.
export function bodyBytes(headers: Record<string, unknown>, body: unknown): number {
  const header = Number(headers["content-length"]);
  return Number.isFinite(header) && header > 0 ? header : Buffer.byteLength(JSON.stringify(body ?? null));
}
