// What must survive a worker restart or an unreachable server: the gateway config
// (agents keep working without the VPS), the last load (restored on start when asked
// for), and usage rows not yet acknowledged. Holds key hashes and sampling settings,
// no keys and no prompts beyond the owner's own configured system prompt.

import { readFileSync, renameSync, writeFileSync, chmodSync } from "node:fs";
import type { DaemonConfig, DaemonLoadSpec } from "../../../shared/daemonTypes.js";
import type { DaemonUsageRow } from "../../../shared/daemonUsage.js";
import type { SpillBaseline } from "../../../shared/gpuSpill.js";

export const MAX_SPOOLED_USAGE = 5000;
export const MAX_USAGE_BATCH = 200;
/** Baselines kept: one per file, backend, GPU and build a probe has measured. */
export const MAX_SPILL_BASELINES = 200;

interface StoreFile {
  config: DaemonConfig | null;
  last_spec: DaemonLoadSpec | null;
  usage: DaemonUsageRow[];
  usage_dropped: number;
  /** What a probe measured, keyed by spillBaselineKey; the oldest go first. */
  spill_baselines: Record<string, SpillBaseline & { at: number }>;
}

export class DaemonStore {
  private data: StoreFile = { config: null, last_spec: null, usage: [], usage_dropped: 0, spill_baselines: {} };
  private dirty = false;

  constructor(private readonly path: string, private readonly warn: (msg: string) => void) {
    try {
      const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<StoreFile>;
      this.data = {
        config: parsed.config ?? null,
        last_spec: parsed.last_spec ?? null,
        usage: Array.isArray(parsed.usage) ? parsed.usage.slice(-MAX_SPOOLED_USAGE) : [],
        usage_dropped: typeof parsed.usage_dropped === "number" ? parsed.usage_dropped : 0,
        spill_baselines: parsed.spill_baselines && typeof parsed.spill_baselines === "object" ? parsed.spill_baselines : {},
      };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "ENOENT") warn(`daemon state unreadable, starting fresh: ${(err as Error).message}`);
    }
  }

  get config(): DaemonConfig | null {
    return this.data.config;
  }

  get lastSpec(): DaemonLoadSpec | null {
    return this.data.last_spec;
  }

  get usageDropped(): number {
    return this.data.usage_dropped;
  }

  spillBaseline(key: string): SpillBaseline | null {
    const b = this.data.spill_baselines[key];
    return b && Number.isFinite(b.mib) && Number.isFinite(b.toleranceMib) ? { mib: b.mib, toleranceMib: b.toleranceMib } : null;
  }

  setSpillBaseline(key: string, baseline: SpillBaseline): void {
    const all = this.data.spill_baselines;
    // Re-inserted, so the key a probe just measured is the newest.
    delete all[key];
    all[key] = { mib: baseline.mib, toleranceMib: baseline.toleranceMib, at: Date.now() };
    const keys = Object.keys(all);
    for (const oldest of keys.slice(0, Math.max(0, keys.length - MAX_SPILL_BASELINES))) delete all[oldest];
    this.flush();
  }

  setConfig(config: DaemonConfig): void {
    this.data.config = config;
    this.flush();
  }

  setLastSpec(spec: DaemonLoadSpec | null): void {
    this.data.last_spec = spec;
    this.flush();
  }

  addUsage(row: DaemonUsageRow): void {
    this.data.usage.push(row);
    if (this.data.usage.length > MAX_SPOOLED_USAGE) {
      this.data.usage.shift();
      this.data.usage_dropped++;
    }
    this.dirty = true;
  }

  usageBatch(): DaemonUsageRow[] {
    return this.data.usage.slice(0, MAX_USAGE_BATCH);
  }

  ackUsage(ids: string[]): void {
    if (ids.length === 0) return;
    const acked = new Set(ids);
    this.data.usage = this.data.usage.filter((r) => !acked.has(r.id));
    this.dirty = true;
  }

  /** Rows are flushed on a timer, so a hard kill loses at most a few seconds of counts. */
  flushIfDirty(): void {
    if (this.dirty) this.flush();
  }

  flush(): void {
    this.dirty = false;
    try {
      const tmp = `${this.path}.tmp`;
      writeFileSync(tmp, JSON.stringify(this.data), { mode: 0o600 });
      renameSync(tmp, this.path);
      chmodSync(this.path, 0o600);
    } catch (err) {
      this.warn(`could not save daemon state: ${(err as Error).message}`);
    }
  }
}
