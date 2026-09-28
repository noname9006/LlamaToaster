import { open } from "node:fs/promises";
import { parseGgufInfo, type GgufInfo } from "../../shared/gguf.js";

// The GGUF header parser itself lives in shared/gguf.ts (the server also
// reads headers, from Hugging Face -- docs/PLAN_MODEL_METADATA_TRUST.md).
// This is just the local-file byte source for it.
export { quantLabelFromFileType, type GgufInfo } from "../../shared/gguf.js";

// Reads a GGUF file's metadata header to find the model's transformer layer
// count and (if present) its MTP/nextn layer count. Returns nulls on any
// parse failure or unrecognized file rather than throwing -- callers treat a
// missing value as "unknown", same fail-soft posture as the rest of
// hardware/model detection in this worker.
export async function readGgufInfo(filePath: string): Promise<GgufInfo> {
  let fh;
  try {
    fh = await open(filePath, "r");
  } catch (err) {
    return parseGgufInfo({
      read: () => Promise.reject(err),
      size: () => Promise.reject(err),
    });
  }
  const handle = fh;
  try {
    return await parseGgufInfo({
      read: async (into, length) => (await handle.read(into, 0, length, null)).bytesRead,
      size: async () => (await handle.stat()).size,
    });
  } finally {
    await handle.close().catch(() => {});
  }
}
