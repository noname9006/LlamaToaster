import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { RunItemUpdateInput } from "../../shared/types.js";

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
