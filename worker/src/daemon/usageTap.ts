// Reads token counts and timings out of a response on its way to the client,
// without changing a byte of it and without keeping any of the text. Measured on
// b11609:
//   chat and text completions: `timings` on the JSON body, or on the final SSE chunk
//   /v1/responses: `timings` only when streamed; `usage` either way
//   /v1/messages: no timings at all, Anthropic-shaped `usage`
// So a row may carry counts with no speeds, and no speed is ever estimated.

export interface ExtractedUsage {
  processed_tokens: number | null;
  cached_tokens: number | null;
  generated_tokens: number | null;
  prompt_ms: number | null;
  predicted_ms: number | null;
}

export const EMPTY_USAGE: ExtractedUsage = {
  processed_tokens: null,
  cached_tokens: null,
  generated_tokens: null,
  prompt_ms: null,
  predicted_ms: null,
};

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
}

function usageFields(usage: Record<string, unknown>): Partial<ExtractedUsage> {
  const out: Partial<ExtractedUsage> = {};
  // Anthropic: input_tokens already excludes the cache read.
  if ("cache_read_input_tokens" in usage) {
    const cached = num(usage.cache_read_input_tokens);
    const input = num(usage.input_tokens);
    if (cached != null) out.cached_tokens = cached;
    if (input != null) out.processed_tokens = input;
  } else {
    // OpenAI (prompt_tokens) and Responses (input_tokens) both count the cache in.
    const prompt = num(usage.prompt_tokens) ?? num(usage.input_tokens);
    const details = (usage.prompt_tokens_details ?? usage.input_tokens_details) as Record<string, unknown> | undefined;
    const cached = num(details?.cached_tokens);
    if (prompt != null) {
      out.processed_tokens = Math.max(0, prompt - (cached ?? 0));
      if (cached != null) out.cached_tokens = cached;
    }
  }
  const generated = num(usage.completion_tokens) ?? num(usage.output_tokens);
  if (generated != null) out.generated_tokens = generated;
  return out;
}

/** Later events win field by field, since streams report parts separately. */
export function extractUsage(event: unknown, into: ExtractedUsage = { ...EMPTY_USAGE }): ExtractedUsage {
  if (!event || typeof event !== "object") return into;
  const e = event as Record<string, unknown>;
  const nested = e.message && typeof e.message === "object" ? (e.message as Record<string, unknown>) : null;
  const response = e.response && typeof e.response === "object" ? (e.response as Record<string, unknown>) : null;
  const usage = e.usage ?? nested?.usage ?? response?.usage;
  if (usage && typeof usage === "object") Object.assign(into, usageFields(usage as Record<string, unknown>));
  // Timings are the server's own measured figures, so they beat usage counts.
  const timings = e.timings ?? response?.timings;
  if (timings && typeof timings === "object") {
    const t = timings as Record<string, unknown>;
    const processed = num(t.prompt_n);
    const cached = num(t.cache_n);
    const generated = num(t.predicted_n);
    const promptMs = num(t.prompt_ms);
    const predictedMs = num(t.predicted_ms);
    if (processed != null) into.processed_tokens = processed;
    if (cached != null) into.cached_tokens = cached;
    if (generated != null) into.generated_tokens = generated;
    if (promptMs != null) into.prompt_ms = promptMs;
    if (predictedMs != null) into.predicted_ms = predictedMs;
  }
  return into;
}

const MAX_SSE_LINE = 1 << 20;
const MAX_JSON_BODY = 8 << 20;

/** Feed every response chunk; call finish() at the end. */
export class UsageTap {
  private readonly stream: boolean;
  private readonly usage: ExtractedUsage = { ...EMPTY_USAGE };
  private lineBuffer = "";
  private jsonChunks: Buffer[] = [];
  private jsonBytes = 0;

  constructor(contentType: string | undefined) {
    this.stream = (contentType ?? "").includes("text/event-stream");
  }

  feed(chunk: Buffer): void {
    if (!this.stream) {
      if (this.jsonBytes <= MAX_JSON_BODY) {
        this.jsonBytes += chunk.length;
        if (this.jsonBytes <= MAX_JSON_BODY) this.jsonChunks.push(chunk);
        else this.jsonChunks = [];
      }
      return;
    }
    this.lineBuffer += chunk.toString("utf8");
    let at: number;
    while ((at = this.lineBuffer.indexOf("\n")) !== -1) {
      this.sseLine(this.lineBuffer.slice(0, at));
      this.lineBuffer = this.lineBuffer.slice(at + 1);
    }
    if (this.lineBuffer.length > MAX_SSE_LINE) this.lineBuffer = "";
  }

  private sseLine(line: string): void {
    if (!line.startsWith("data:")) return;
    // Parsing every token's chunk would cost more than the rest of the proxy.
    if (!line.includes('"timings"') && !line.includes('"usage"')) return;
    try {
      extractUsage(JSON.parse(line.slice(5)), this.usage);
    } catch {
      // Not JSON (the [DONE] marker) or cut off: nothing to read.
    }
  }

  finish(): ExtractedUsage {
    if (!this.stream && this.jsonChunks.length > 0) {
      try {
        extractUsage(JSON.parse(Buffer.concat(this.jsonChunks).toString("utf8")), this.usage);
      } catch {
        // A non-JSON error body has no counts.
      }
      this.jsonChunks = [];
    }
    return this.usage;
  }
}
