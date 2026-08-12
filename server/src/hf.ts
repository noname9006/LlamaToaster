// Shared Hugging Face lookups -- used both by the /api/hf/* routes (client's
// model-download picker) and by the AI assistant's tool calls (server/src/routes/ai.ts),
// so the assistant grounds model suggestions in live HF data instead of its own
// training data, which goes stale as new model generations ship.

export interface HfSearchResult {
  id: string;
  downloads: number;
  likes: number;
  // Repo creation date HF's own search API reports (ISO string) -- powers
  // the "Newest" sort option. null if the API response omitted it.
  created_at: string | null;
}

export interface HfFileEntry {
  path: string;
  size_bytes: number;
  quant: string | null;
}

export const HF_REPO_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const QUANT_PATTERN = /(?:^|[-_.])((?:IQ|Q)[0-9]+(?:_[A-Z0-9]+)*|F16|F32|BF16)(?=[-_.]|$)/i;

export function parseQuant(filename: string): string | null {
  const base = filename.split("/").pop() ?? filename;
  const m = base.match(QUANT_PATTERN);
  return m ? m[1].toUpperCase() : null;
}

export async function searchHfGgufModels(query: string, timeoutMs: number): Promise<HfSearchResult[]> {
  const q = query.trim();
  if (!q) return [];
  const url = `https://huggingface.co/api/models?search=${encodeURIComponent(q)}&filter=gguf&limit=20`;
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`HF search failed: ${res.status}`);
  const data = (await res.json()) as { id: string; downloads?: number; likes?: number; createdAt?: string }[];
  return data.map((m) => ({
    id: m.id,
    downloads: m.downloads ?? 0,
    likes: m.likes ?? 0,
    created_at: typeof m.createdAt === "string" ? m.createdAt : null,
  }));
}

export async function listHfGgufFiles(repoId: string, timeoutMs: number): Promise<HfFileEntry[]> {
  const url = `https://huggingface.co/api/models/${repoId}/tree/main?recursive=true`;
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`HF tree fetch failed: ${res.status}`);
  const data = (await res.json()) as { type: string; path: string; size?: number; lfs?: { size?: number } }[];
  return data
    .filter((e) => e.type === "file" && e.path.toLowerCase().endsWith(".gguf"))
    .map((e) => ({ path: e.path, size_bytes: e.lfs?.size ?? e.size ?? 0, quant: parseQuant(e.path) }));
}

export interface HfGgufMeta {
  // Real total parameter count, computed by HF itself from the repo's GGUF
  // tensor shapes (repo-wide, not per-quant-file -- quantization changes
  // bytes-per-parameter, not the element count). null if the repo has no
  // "gguf" section yet (non-GGUF repo, or HF hasn't parsed it) or the fetch
  // failed -- fail-soft, same posture as worker/src/gguf.ts's layer-count
  // reader: callers treat this as "unknown", never block on it.
  param_count: number | null;
  // Repo-level "last modified" timestamp (ISO string) HF's own API reports --
  // the same value HF's own UI surfaces as a repo card's "Updated X ago".
  // Not a per-file commit date (the file-tree API doesn't expose that
  // without one extra request per file); good enough for "is this repo newer
  // than what I downloaded" purposes. null on fetch failure.
  last_modified: string | null;
}

export async function getHfGgufMeta(repoId: string, timeoutMs: number): Promise<HfGgufMeta> {
  try {
    const url = `https://huggingface.co/api/models/${repoId}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { param_count: null, last_modified: null };
    const data = (await res.json()) as { gguf?: { total?: number }; lastModified?: string };
    const total = data.gguf?.total;
    return {
      param_count: typeof total === "number" ? total : null,
      last_modified: typeof data.lastModified === "string" ? data.lastModified : null,
    };
  } catch {
    return { param_count: null, last_modified: null };
  }
}
