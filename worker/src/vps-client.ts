import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { RunItemUpdateInput, ModelDownloadCallbackInput } from "../../shared/types.js";

export function writeRawJson(runDir: string, runId: string, data: unknown): string {
  if (!existsSync(runDir)) mkdirSync(runDir, { recursive: true });
  const path = join(runDir, `${runId}.json`);
  writeFileSync(path, JSON.stringify(data, null, 2), "utf8");
  return path;
}

// Single endpoint for both tiers of per-item reporting (see shared/types.ts's
// RunItemTickInput/RunItemTerminalInput) -- the caller decides how to treat
// failures: worker/src/index.ts awaits-and-swallows for ticks, and wraps
// this in its own retry-with-backoff for terminal outcomes.
export async function postRunItemUpdate(
  vpsUrl: string,
  runId: string,
  idx: number,
  payload: RunItemUpdateInput,
  timeoutMs = 5000
): Promise<void> {
  const res = await fetch(`${vpsUrl}/api/runs/${runId}/items/${idx}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`item update failed (${res.status}): ${text}`);
  }
}

// Reports a model download's terminal outcome (success or failure) once the
// background transfer in worker/src/index.ts's POST /models/download
// settles. Same retry-with-backoff posture as postRunItemUpdate's terminal
// use above -- this is what actually gets a completed download registered
// as a Model server-side, so losing it silently would leave a fully
// downloaded file on disk that the app never learns about.
export async function postModelDownloadResult(
  vpsUrl: string,
  payload: ModelDownloadCallbackInput,
  timeoutMs = 10_000
): Promise<void> {
  const res = await fetch(`${vpsUrl}/api/models/download-callback`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`model download callback failed (${res.status}): ${text}`);
  }
}
