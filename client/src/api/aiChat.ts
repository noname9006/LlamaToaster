// Talks to LlamaToaster's own /api/ai/chat proxy (server/src/routes/ai.ts),
// not the AI provider directly -- the server holds the API key (AI_API_KEY
// in a .env file) and forwards to whatever OpenAI-compatible provider is
// configured (AI_BASE_URL/AI_MODEL), streaming the SSE response straight
// through. The browser never sees the key.

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export class AiChatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AiChatError";
  }
}

function describeHttpError(status: number, body: string): string {
  let message = body;
  try {
    const parsed = JSON.parse(body) as { error?: string };
    if (typeof parsed.error === "string") message = parsed.error;
  } catch {
    /* body wasn't JSON -- fall back to using it as-is */
  }
  return message || `Request failed (${status}).`;
}

// Streams one assistant reply. Calls onDelta with each text chunk as it
// arrives; resolves once the stream ends. Pass `signal` from an
// AbortController to support a Stop button -- an aborted fetch rejects with
// a DOMException named "AbortError", which callers can check for to
// distinguish "user cancelled" from a real failure.
export async function streamChatCompletion(
  messages: ChatMessage[],
  onDelta: (text: string) => void,
  signal?: AbortSignal,
  onModel?: (model: string) => void
): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/ai/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages }),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new AiChatError(
      `Could not reach LlamaToaster's own server (${err instanceof Error ? err.message : String(err)}).`
    );
  }

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new AiChatError(describeHttpError(res.status, text));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      if (!data) continue;
      let parsed: { choices?: { delta?: { content?: string } }[]; error?: { message?: string }; model?: string };
      try {
        parsed = JSON.parse(data);
      } catch {
        continue; // partial/malformed chunk -- keep reading rather than aborting the stream
      }
      if (parsed.error?.message) throw new AiChatError(parsed.error.message);
      if (parsed.model) onModel?.(parsed.model);
      const delta = parsed.choices?.[0]?.delta?.content;
      if (delta) onDelta(delta);
    }
  }
}
