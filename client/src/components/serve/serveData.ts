// Pure helpers for the Serve page.

import type { DaemonPhase, DaemonStatus } from "../../../../shared/daemonTypes";
import { ALL_INTERFACES_HOST, type SettingValues } from "../../../../shared/daemonSettings";
import type { Model, ModelDirFile, Worker } from "../../types";
import type { PillTone } from "../StatusPill";

export function formatMib(mib: number | null | undefined): string {
  if (mib == null) return "n/a";
  if (Math.abs(mib) >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${Math.round(mib)} MiB`;
}

export function formatTokens(n: number | null | undefined): string {
  if (n == null) return "n/a";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1024)}k`;
  return Math.round(n).toLocaleString("en-US");
}

export function formatSpeed(tps: number | null | undefined): string {
  if (tps == null) return "n/a";
  return `${tps >= 100 ? Math.round(tps) : tps.toFixed(1)} tok/s`;
}

export function formatShare(share: number | null | undefined): string {
  return share == null ? "n/a" : `${Math.round(share * 100)}%`;
}

export function modelFilename(m: Model): string {
  return m.source === "local" ? m.filename ?? m.id : m.hf_file ?? m.filename ?? m.id;
}

export function modelLabel(m: Model): string {
  return (modelFilename(m).split(/[\\/]/).pop() ?? m.id).replace(/\.gguf$/i, "");
}

function fileForModel(m: Model, files: ModelDirFile[]): ModelDirFile | undefined {
  const name = modelFilename(m);
  return files.find((f) => f.path === name) ?? (/^[0-9a-f]{64}$/.test(m.id) ? files.find((f) => f.sha256?.toLowerCase() === m.id) : undefined);
}

const MMPROJ_FILE = /mmproj/i;

/** Models this machine can load: on its disk, and not a vision projector. */
export function modelsOnMachine(models: Model[], worker: Worker | undefined): Model[] {
  const files = worker?.modelFiles ?? [];
  return models.filter((m) => !MMPROJ_FILE.test(modelFilename(m)) && fileForModel(m, files) != null).sort((a, b) => modelLabel(a).localeCompare(modelLabel(b)));
}

export function projectorFiles(worker: Worker | undefined): string[] {
  return (worker?.modelFiles ?? []).filter((f) => MMPROJ_FILE.test(f.path)).map((f) => f.path).sort();
}

/** The URL an agent uses. A bind to every interface has no one address, so it
 * names the machine's first one. */
export function baseUrl(host: string, port: number, status: DaemonStatus | null): string {
  const shown = host === ALL_INTERFACES_HOST ? status?.interfaces[0]?.address ?? "<this machine's address>" : host;
  return `http://${shown}:${port}/v1`;
}

export const PHASE_VIEW: Record<DaemonPhase, { label: string; tone: PillTone }> = {
  idle: { label: "Nothing loaded", tone: "muted" },
  loading: { label: "Loading", tone: "accent" },
  ready: { label: "Ready", tone: "success" },
  sleeping: { label: "Sleeping", tone: "muted" },
  stopping: { label: "Unloading", tone: "accent" },
  // The message beside it says why: a test, or waiting for the queue at startup.
  suspended: { label: "Paused", tone: "warning" },
  failed: { label: "Failed", tone: "danger" },
  crashed: { label: "Crashed", tone: "danger" },
};

export function isLive(phase: DaemonPhase | undefined): boolean {
  return phase === "ready" || phase === "sleeping" || phase === "loading";
}

/** There is a model to call off: running, coming back after a test, or restarting. */
export function canUnload(phase: DaemonPhase | undefined): boolean {
  return isLive(phase) || phase === "suspended" || phase === "crashed";
}

function stable(values: SettingValues): string {
  return JSON.stringify(Object.keys(values).sort().map((k) => [k, values[k]]));
}

/** Does the form differ from what is running, so a reload would change anything? */
export function settingsDiffer(a: SettingValues | null | undefined, b: SettingValues | null | undefined): boolean {
  return stable(a ?? {}) !== stable(b ?? {});
}

export interface BreakdownPart {
  label: string;
  mib: number;
}

/** What the app's share holds, with whatever llama.cpp did not account for as "Other". */
export function withRemainder(parts: { label: string; mib: number | null }[], appMib: number | null, remainderLabel: string): BreakdownPart[] {
  const known = parts.filter((p): p is BreakdownPart => p.mib != null && p.mib > 0);
  if (appMib == null) return known;
  const rest = appMib - known.reduce((sum, p) => sum + p.mib, 0);
  // Counter resolution and rounding make a few MiB either way meaningless.
  return rest >= 8 ? [...known, { label: remainderLabel, mib: rest }] : known;
}

/** One line for a load configuration in the usage picker. */
export function loadSummary(load: { settings: SettingValues; actual: { gpu_layers: number | null; total_layers: number | null; n_ctx: number | null; n_slots: number | null } | null; build_tag: string | null; requests: number }): string {
  const parts: string[] = [];
  const a = load.actual;
  if (a?.gpu_layers != null) parts.push(`${a.gpu_layers}${a.total_layers != null ? `/${a.total_layers}` : ""} layers`);
  const ctx = a?.n_ctx ?? (typeof load.settings.ctx_size === "number" ? load.settings.ctx_size : null);
  if (ctx != null) parts.push(`ctx ${formatTokens(ctx)}`);
  if (a?.n_slots != null) parts.push(`${a.n_slots} slot${a.n_slots === 1 ? "" : "s"}`);
  if (load.build_tag) parts.push(load.build_tag);
  parts.push(`${load.requests} request${load.requests === 1 ? "" : "s"}`);
  return parts.join(" · ");
}
