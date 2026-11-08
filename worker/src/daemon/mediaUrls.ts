// llama-server downloads a remote image_url itself, even with --offline: measured on
// b11609, a local listener received the request. An agent passing along a URL from a
// page it read could make this machine call its own router. So media must arrive
// inline as a data: URL, which is what Ollama requires too.
//
// This walks only where media can be, not every key named "url": a tool call's
// arguments are full of those (a web_fetch tool's `input.url`), and refusing them
// would break ordinary agents.

const MEDIA_KEYS = ["image_url", "video_url", "audio_url", "file_url"];
const MAX_DEPTH = 8;

function isRemote(url: unknown): url is string {
  return typeof url === "string" && !/^\s*data:/i.test(url);
}

function record(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function remoteInPart(part: unknown, depth: number): string | null {
  const p = record(part);
  if (!p || depth > MAX_DEPTH) return null;
  for (const key of MEDIA_KEYS) {
    const value = p[key];
    if (isRemote(value)) return value;
    const url = record(value)?.url;
    if (isRemote(url)) return url;
  }
  // Anthropic: {type: "image", source: {type: "url", url}}.
  const sourceUrl = record(p.source)?.url;
  if (isRemote(sourceUrl)) return sourceUrl;
  // Message-shaped items and Anthropic tool results nest more parts under content.
  if (Array.isArray(p.content)) {
    for (const inner of p.content) {
      const found = remoteInPart(inner, depth + 1);
      if (found) return found;
    }
  }
  return null;
}

/** The first media URL llama-server would fetch by itself, or null. */
export function findRemoteMedia(body: Record<string, unknown>): string | null {
  for (const list of [body.messages, body.input]) {
    if (!Array.isArray(list)) continue;
    for (const item of list) {
      const found = remoteInPart(item, 0);
      if (found) return found;
    }
  }
  return null;
}
