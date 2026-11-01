// Storage for the Serve page (docs/plans/MODEL_DAEMON_PLAN.md). Everything hangs off
// a worker row and cascades with it. The usage table holds counts and speeds only:
// no prompt or completion text ever reaches this server.

import { createHash, randomBytes, randomUUID } from "node:crypto";
import { getDb } from "./migrate.js";
import {
  DEFAULT_GATEWAY,
  isLoopbackHost,
  LOOPBACK_HOST,
  type GatewaySettings,
  type SamplingSettings,
  type SettingValues,
} from "../../../shared/daemonSettings.js";
import type { DaemonActualLoad, DaemonApiKeyInfo, DaemonConfig, DaemonModelSettings } from "../../../shared/daemonTypes.js";
import type { DaemonEndpoint, DaemonUsageRow } from "../../../shared/daemonUsage.js";

export const USAGE_RETENTION_DAYS = 90;
export const MAX_USAGE_ROWS_PER_WORKER = 200_000;
export const MAX_ACTIVE_KEYS = 20;
export const MAX_LOADS_PER_WORKER = 500;

interface UsageDbRow {
  id: string;
  at: number;
  model_id: string;
  load_hash: string;
  endpoint: string;
  stream: number;
  status: number;
  cancelled: number;
  processed_tokens: number | null;
  cached_tokens: number | null;
  generated_tokens: number | null;
  prompt_ms: number | null;
  predicted_ms: number | null;
  total_ms: number;
  concurrency: number;
  watched: number;
}

export interface DaemonLoadRow {
  load_hash: string;
  model_id: string;
  build_tag: string | null;
  settings: SettingValues;
  actual: DaemonActualLoad | null;
  first_seen: number;
  requests: number;
}

function parseJson<T>(raw: string | null | undefined, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export const daemonRepo = {
  getGateway(workerId: string): { gateway: GatewaySettings; version: number } | null {
    const row = getDb().prepare(`SELECT version, gateway_json FROM daemon_config WHERE worker_id = ?`).get(workerId) as
      | { version: number; gateway_json: string }
      | undefined;
    if (!row) return null;
    return { gateway: { ...DEFAULT_GATEWAY, ...parseJson<Partial<GatewaySettings>>(row.gateway_json, {}) }, version: row.version };
  },

  /** Any change the worker must hear about goes through here. */
  bumpConfig(workerId: string, gateway?: GatewaySettings): number {
    const now = Date.now();
    const db = getDb();
    db.prepare(
      `INSERT INTO daemon_config (worker_id, version, gateway_json, updated_at)
       VALUES (?, 1, ?, ?)
       ON CONFLICT(worker_id) DO UPDATE SET
         version = version + 1,
         gateway_json = COALESCE(?, gateway_json),
         updated_at = excluded.updated_at`
    ).run(workerId, JSON.stringify(gateway ?? DEFAULT_GATEWAY), now, gateway ? JSON.stringify(gateway) : null);
    return (db.prepare(`SELECT version FROM daemon_config WHERE worker_id = ?`).get(workerId) as { version: number }).version;
  },

  /** What the worker runs on. Null until the owner has used the page: until then a
   * worker opens no port at all. */
  buildConfig(workerId: string, modelId: string | null): DaemonConfig | null {
    const stored = this.getGateway(workerId);
    if (!stored) return null;
    return {
      version: stored.version,
      gateway: stored.gateway,
      key_hashes: this.activeKeyHashes(workerId),
      sampling_model_id: modelId,
      sampling: modelId ? (this.getModelSettings(workerId, modelId)?.sampling ?? null) : null,
    };
  },

  activeKeyHashes(workerId: string): string[] {
    const rows = getDb().prepare(`SELECT sha256 FROM daemon_api_keys WHERE worker_id = ? AND revoked_at IS NULL`).all(workerId) as { sha256: string }[];
    return rows.map((r) => r.sha256);
  },

  listKeys(workerId: string): DaemonApiKeyInfo[] {
    return getDb()
      .prepare(`SELECT id, name, prefix, created_at, revoked_at FROM daemon_api_keys WHERE worker_id = ? ORDER BY created_at DESC`)
      .all(workerId) as DaemonApiKeyInfo[];
  },

  /** The only place the key exists in the clear: returned once, stored hashed. */
  createKey(workerId: string, name: string): { info: DaemonApiKeyInfo; key: string } {
    const key = `lt-${randomBytes(32).toString("base64url")}`;
    const info: DaemonApiKeyInfo = { id: randomUUID(), name, prefix: key.slice(0, 10), created_at: Date.now(), revoked_at: null };
    getDb()
      .prepare(`INSERT INTO daemon_api_keys (id, worker_id, name, prefix, sha256, created_at) VALUES (?, ?, ?, ?, ?, ?)`)
      .run(info.id, workerId, name, info.prefix, createHash("sha256").update(key).digest("hex"), info.created_at);
    this.bumpConfig(workerId);
    return { info, key };
  },

  countActiveKeys(workerId: string): number {
    return (getDb().prepare(`SELECT COUNT(*) AS n FROM daemon_api_keys WHERE worker_id = ? AND revoked_at IS NULL`).get(workerId) as { n: number }).n;
  },

  /** Always succeeds, even for the last key on an open endpoint: revoking must never
   * be blocked. The endpoint falls back to this machine only, instead of staying open. */
  revokeKey(workerId: string, keyId: string): boolean {
    const db = getDb();
    const changed = db.prepare(`UPDATE daemon_api_keys SET revoked_at = ? WHERE id = ? AND worker_id = ? AND revoked_at IS NULL`).run(Date.now(), keyId, workerId).changes > 0;
    if (!changed) return false;
    const stored = this.getGateway(workerId);
    if (stored && !isLoopbackHost(stored.gateway.host) && this.countActiveKeys(workerId) === 0) {
      this.bumpConfig(workerId, { ...stored.gateway, host: LOOPBACK_HOST });
    } else {
      this.bumpConfig(workerId);
    }
    return true;
  },

  getModelSettings(workerId: string, modelId: string): DaemonModelSettings | null {
    const row = getDb()
      .prepare(`SELECT settings_json, sampling_json, alias, mmproj_filename, draft_model_id FROM daemon_model_settings WHERE worker_id = ? AND model_id = ?`)
      .get(workerId, modelId) as { settings_json: string; sampling_json: string; alias: string | null; mmproj_filename: string | null; draft_model_id: string | null } | undefined;
    if (!row) return null;
    return {
      settings: parseJson<SettingValues>(row.settings_json, {}),
      sampling: parseJson<SamplingSettings>(row.sampling_json, { values: {} }),
      alias: row.alias,
      mmproj_filename: row.mmproj_filename,
      draft_model_id: row.draft_model_id,
    };
  },

  saveModelSettings(workerId: string, modelId: string, s: DaemonModelSettings): void {
    getDb()
      .prepare(
        `INSERT INTO daemon_model_settings (worker_id, model_id, settings_json, sampling_json, alias, mmproj_filename, draft_model_id, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(worker_id, model_id) DO UPDATE SET
           settings_json = excluded.settings_json,
           sampling_json = excluded.sampling_json,
           alias = excluded.alias,
           mmproj_filename = excluded.mmproj_filename,
           draft_model_id = excluded.draft_model_id,
           updated_at = excluded.updated_at`
      )
      .run(workerId, modelId, JSON.stringify(s.settings), JSON.stringify(s.sampling), s.alias, s.mmproj_filename, s.draft_model_id, Date.now());
  },

  saveSampling(workerId: string, modelId: string, sampling: SamplingSettings): void {
    getDb()
      .prepare(
        `INSERT INTO daemon_model_settings (worker_id, model_id, settings_json, sampling_json, updated_at)
         VALUES (?, ?, '{}', ?, ?)
         ON CONFLICT(worker_id, model_id) DO UPDATE SET sampling_json = excluded.sampling_json, updated_at = excluded.updated_at`
      )
      .run(workerId, modelId, JSON.stringify(sampling), Date.now());
  },

  hasLoad(workerId: string, loadHash: string): boolean {
    return !!getDb().prepare(`SELECT 1 FROM daemon_loads WHERE worker_id = ? AND load_hash = ?`).get(workerId, loadHash);
  },

  recordLoad(workerId: string, load: { load_hash: string; model_id: string; build_tag: string | null; settings: SettingValues; actual: DaemonActualLoad | null }): void {
    getDb()
      .prepare(
        `INSERT INTO daemon_loads (worker_id, load_hash, model_id, build_tag, settings_json, actual_json, first_seen)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(worker_id, load_hash) DO UPDATE SET actual_json = COALESCE(excluded.actual_json, actual_json)`
      )
      .run(workerId, load.load_hash, load.model_id, load.build_tag, JSON.stringify(load.settings), load.actual ? JSON.stringify(load.actual) : null, Date.now());
    // Oldest go first. A real machine never gets near this; the hash is reported by
    // the worker, so a broken or hostile one could otherwise add rows forever.
    getDb()
      .prepare(
        `DELETE FROM daemon_loads WHERE worker_id = ? AND load_hash NOT IN
           (SELECT load_hash FROM daemon_loads WHERE worker_id = ? ORDER BY first_seen DESC, rowid DESC LIMIT ?)`
      )
      .run(workerId, workerId, MAX_LOADS_PER_WORKER);
  },

  listLoads(workerId: string, modelId: string): DaemonLoadRow[] {
    const rows = getDb()
      .prepare(
        `SELECT l.load_hash, l.model_id, l.build_tag, l.settings_json, l.actual_json, l.first_seen,
                (SELECT COUNT(*) FROM daemon_usage u WHERE u.worker_id = l.worker_id AND u.load_hash = l.load_hash) AS requests
         FROM daemon_loads l WHERE l.worker_id = ? AND l.model_id = ? ORDER BY l.first_seen DESC LIMIT 50`
      )
      .all(workerId, modelId) as { load_hash: string; model_id: string; build_tag: string | null; settings_json: string; actual_json: string | null; first_seen: number; requests: number }[];
    return rows.map((r) => ({
      load_hash: r.load_hash,
      model_id: r.model_id,
      build_tag: r.build_tag,
      settings: parseJson<SettingValues>(r.settings_json, {}),
      actual: parseJson<DaemonActualLoad | null>(r.actual_json, null),
      first_seen: r.first_seen,
      requests: r.requests,
    }));
  },

  /** Idempotent by id: a worker that lost the acknowledgement sends the batch again. */
  insertUsage(workerId: string, rows: DaemonUsageRow[]): void {
    if (rows.length === 0) return;
    const db = getDb();
    const stmt = db.prepare(
      `INSERT OR IGNORE INTO daemon_usage
         (worker_id, id, at, model_id, load_hash, endpoint, stream, status, cancelled, processed_tokens, cached_tokens,
          generated_tokens, prompt_ms, predicted_ms, total_ms, concurrency, watched)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );
    db.transaction(() => {
      for (const r of rows) {
        stmt.run(
          workerId, r.id, r.at, r.model_id, r.load_hash, r.endpoint, r.stream ? 1 : 0, r.status, r.cancelled ? 1 : 0,
          r.processed_tokens, r.cached_tokens, r.generated_tokens, r.prompt_ms, r.predicted_ms, r.total_ms, r.concurrency, r.watched ? 1 : 0
        );
      }
    })();
  },

  /** Newest first. */
  listUsage(workerId: string, modelId: string, opts: { since: number; loadHash?: string | null; limit: number }): DaemonUsageRow[] {
    const hashClause = opts.loadHash ? "AND load_hash = ?" : "";
    const args: (string | number)[] = [workerId, modelId, opts.since];
    if (opts.loadHash) args.push(opts.loadHash);
    args.push(opts.limit);
    const rows = getDb()
      .prepare(`SELECT * FROM daemon_usage WHERE worker_id = ? AND model_id = ? AND at >= ? ${hashClause} ORDER BY at DESC LIMIT ?`)
      .all(...args) as UsageDbRow[];
    return rows.map((r) => ({
      id: r.id,
      at: r.at,
      model_id: r.model_id,
      load_hash: r.load_hash,
      endpoint: r.endpoint as DaemonEndpoint,
      stream: r.stream === 1,
      status: r.status,
      cancelled: r.cancelled === 1,
      processed_tokens: r.processed_tokens,
      cached_tokens: r.cached_tokens,
      generated_tokens: r.generated_tokens,
      prompt_ms: r.prompt_ms,
      predicted_ms: r.predicted_ms,
      total_ms: r.total_ms,
      concurrency: r.concurrency,
      watched: r.watched === 1,
    }));
  },

  clearUsage(workerId: string, modelId: string): number {
    const db = getDb();
    const removed = db.prepare(`DELETE FROM daemon_usage WHERE worker_id = ? AND model_id = ?`).run(workerId, modelId).changes;
    db.prepare(`DELETE FROM daemon_loads WHERE worker_id = ? AND model_id = ?`).run(workerId, modelId);
    return removed;
  },

  /** Age limit first, then the per-machine cap. */
  pruneUsage(): number {
    const db = getDb();
    const cutoff = Date.now() - USAGE_RETENTION_DAYS * 86_400_000;
    let removed = db.prepare(`DELETE FROM daemon_usage WHERE at < ?`).run(cutoff).changes;
    // A load no remaining row points at has nothing left to explain. One still
    // running is written back by its next telemetry report.
    db.prepare(
      `DELETE FROM daemon_loads WHERE first_seen < ? AND NOT EXISTS
         (SELECT 1 FROM daemon_usage u WHERE u.worker_id = daemon_loads.worker_id AND u.load_hash = daemon_loads.load_hash)`
    ).run(cutoff);
    const heavy = db
      .prepare(`SELECT worker_id FROM daemon_usage GROUP BY worker_id HAVING COUNT(*) > ?`)
      .all(MAX_USAGE_ROWS_PER_WORKER) as { worker_id: string }[];
    for (const { worker_id } of heavy) {
      removed += db
        .prepare(
          `DELETE FROM daemon_usage WHERE worker_id = ? AND at < (
             SELECT at FROM daemon_usage WHERE worker_id = ? ORDER BY at DESC LIMIT 1 OFFSET ?
           )`
        )
        .run(worker_id, worker_id, MAX_USAGE_ROWS_PER_WORKER).changes;
    }
    return removed;
  },

  /** Load and unload are last-wins: anything still pending is superseded. */
  supersedeJobs(workerId: string): { cancelledPending: number; claimedLoadIds: string[] } {
    const db = getDb();
    const cancelledPending = db
      .prepare(
        `UPDATE worker_jobs SET status = 'cancelled', completed_at = ?
         WHERE worker_id = ? AND status = 'pending' AND job_type IN ('daemon_load', 'daemon_unload')`
      )
      .run(Date.now(), workerId).changes;
    const claimed = db
      .prepare(`SELECT id FROM worker_jobs WHERE worker_id = ? AND status = 'claimed' AND job_type = 'daemon_load'`)
      .all(workerId) as { id: string }[];
    return { cancelledPending, claimedLoadIds: claimed.map((c) => c.id) };
  },

  latestJob(workerId: string): { id: string; type: "daemon_load" | "daemon_unload"; status: string; error: string | null } | null {
    const row = getDb()
      .prepare(`SELECT id, job_type, status, error FROM worker_jobs WHERE worker_id = ? AND job_type IN ('daemon_load', 'daemon_unload') ORDER BY created_at DESC, rowid DESC LIMIT 1`)
      .get(workerId) as { id: string; job_type: "daemon_load" | "daemon_unload"; status: string; error: string | null } | undefined;
    return row ? { id: row.id, type: row.job_type, status: row.status, error: row.error } : null;
  },
};
