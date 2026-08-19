import type { FastifyInstance } from "fastify";
import { HF_REPO_PATTERN, searchHfGgufModels, listHfGgufFiles } from "../hf.js";
import { repo } from "../db/repo.js";
import { resolveAuthUser } from "../auth-middleware.js";
import { loadExportRows, type ResultExportRow } from "./results.js";
import type { CommunityAggregateFilters } from "../../../shared/types.js";

// extra_content.google.thought_signature is Gemini-specific (see the
// thought-signature comment in the /api/ai/chat route's stream loop below)
// -- omitted entirely for any other provider's tool calls, which never set it.
interface AiToolCall {
  id: string;
  function: { name: string; arguments: string };
  extra_content?: { google?: { thought_signature?: string } };
}

interface AiChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: AiToolCall[];
  tool_call_id?: string;
}

interface AiConfig {
  apiKey: string;
  baseUrl: string;
  models: string[];
}

interface MinimalLogger {
  warn: (obj: Record<string, unknown>, msg: string) => void;
}

const HF_TOOL_TIMEOUT_MS = 15_000;
// A model that keeps calling tools instead of answering shouldn't hang the
// request forever -- cut it off and surface whatever it's found so far.
const MAX_TOOL_ROUNDS = 4;

// §2.6: an unmetered proxy to a paid LLM, with a client-controlled message
// array and up to MAX_TOOL_ROUNDS tool rounds per request, is the
// highest-probability launch-day incident this app has -- one signed-up user
// could drain the operator's provider budget in an afternoon with nothing
// else in place. maxToolRounds is already enforced as MAX_TOOL_ROUNDS above;
// everything else is enforced in the route handler below via aiUsageRepo
// (server/src/db/repo.ts), backed by the ai_usage table.
const AI_LIMITS = {
  perUserPerHour: 30,
  perUserPerDay: 150,
  globalPerDay: Number(process.env.AI_GLOBAL_DAILY_CAP ?? 2000), // circuit breaker
  maxMessages: 40,
  maxTotalChars: 60_000, // whole conversation, enforced server-side
};

// UTC, matching ai_usage's own (day, hour) bucketing -- deliberately not
// the server's local time, which would make the daily/hourly reset moment
// depend on deploy timezone.
function utcDayAndHour(now: number): { day: string; hour: number } {
  const d = new Date(now);
  return { day: d.toISOString().slice(0, 10), hour: d.getUTCHours() };
}

// Given to the model so it can look up real, current Hugging Face listings --
// repo ids, download/like counts, actual GGUF file sizes per quant -- instead
// of guessing from training data, which goes stale fast as new model
// generations ship (see hf.ts's header comment).
const TOOLS = [
  {
    type: "function",
    function: {
      name: "search_huggingface_gguf_models",
      description:
        "Search Hugging Face for GGUF model repos matching a query. Use this before naming a specific " +
        "model in an answer -- it returns real, currently-existing repo ids with download/like counts, " +
        "which is far more reliable than recalling a model name from training data that may be outdated " +
        "or no longer the current generation.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search terms, e.g. 'qwen instruct 4b' or 'gemma gguf'." },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_huggingface_gguf_files",
      description:
        "List the actual .gguf files in a specific Hugging Face repo, with real file sizes in bytes and " +
        "parsed quantization (e.g. Q4_K_M). Use this to pick a quant that actually fits a given RAM/VRAM " +
        "budget instead of guessing a size.",
      parameters: {
        type: "object",
        properties: {
          repo_id: { type: "string", description: "e.g. 'bartowski/Qwen_Qwen3.5-4B-GGUF'" },
        },
        required: ["repo_id"],
      },
    },
  },
  // Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4): the assistant's only window
  // into other users' data, and it can only ever see k-anonymised aggregates
  // -- every row repo.communityRepo returns already describes five or more
  // people, never fewer, and never a username or machine id. Requires
  // sign-in (see runTool below); with AUTH_ENABLED off or no session, both
  // tools return an error result instead of data.
  {
    type: "function",
    function: {
      name: "list_community_facets",
      description:
        "Discover what community benchmark data currently exists (models, backends, platforms, GPUs) " +
        "that has enough contributors to be shown -- use this BEFORE get_community_aggregates to find a " +
        "valid filter value instead of guessing one. Every value returned already has 5+ contributors; a " +
        "value with fewer never appears here at all. Requires the user to be signed in.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "get_community_aggregates",
      description:
        "Get anonymised, aggregated benchmark throughput (avg tok/s, avg RAM/VRAM) contributed by OTHER " +
        "users who opted in to sharing, optionally filtered by model_id/backend/platform/gpu_model. Every " +
        "row already represents 5 or more distinct contributors (rows describing fewer are suppressed " +
        "server-side) -- never attribute a row to an individual, never state or imply how many people are " +
        "in a group beyond the contributor_count field itself, and never combine two rows to try to narrow " +
        "a group down further. Requires the user to be signed in.",
      parameters: {
        type: "object",
        properties: {
          model_id: { type: "string", description: "Filter to one model's sha256 id (see the Registered models section above)." },
          backend: { type: "string", description: "e.g. 'cuda', 'vulkan', 'metal', 'cpu'." },
          platform: { type: "string", description: "e.g. 'linux', 'darwin', 'win32'." },
          gpu_model: { type: "string", description: "e.g. 'RTX 4090'." },
        },
      },
    },
  },
];

// callerId: the authenticated caller's user id, or undefined with
// AUTH_ENABLED off / no session (§4.6's same convention throughout this
// file). The two community tools (§5.4) need it both to exclude the
// caller's own runs from an aggregate (point 3 of §5.4) and because there is
// no "other users" concept to query at all without a real caller identity.
async function runTool(name: string, rawArguments: string, callerId: string | undefined): Promise<unknown> {
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(rawArguments || "{}");
  } catch {
    return { error: "could not parse tool arguments as JSON" };
  }

  if (name === "search_huggingface_gguf_models") {
    const query = typeof args.query === "string" ? args.query : "";
    try {
      return { results: await searchHfGgufModels(query, HF_TOOL_TIMEOUT_MS) };
    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err) };
    }
  }

  if (name === "list_huggingface_gguf_files") {
    const repoId = typeof args.repo_id === "string" ? args.repo_id : "";
    if (!HF_REPO_PATTERN.test(repoId)) return { error: "invalid repo id" };
    try {
      return { files: await listHfGgufFiles(repoId, HF_TOOL_TIMEOUT_MS) };
    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err) };
    }
  }

  if (name === "list_community_facets") {
    if (!callerId) return { error: "sign in to use community benchmark data" };
    return repo.communityRepo.listFacets(callerId);
  }

  if (name === "get_community_aggregates") {
    if (!callerId) return { error: "sign in to use community benchmark data" };
    const filters: CommunityAggregateFilters = {
      modelId: typeof args.model_id === "string" ? args.model_id : undefined,
      backend: typeof args.backend === "string" ? args.backend : undefined,
      platform: typeof args.platform === "string" ? args.platform : undefined,
      gpuModel: typeof args.gpu_model === "string" ? args.gpu_model : undefined,
    };
    return { aggregates: repo.communityRepo.listAggregates(callerId, filters) };
  }

  return { error: `unknown tool ${name}` };
}

// Read fresh on every call (not cached at module load) -- these come from
// .env (see index.ts's process.loadEnvFile()), which is optional, so the
// assistant simply reports "not configured" rather than failing to boot when
// it's absent.
//
// AI_MODEL is a comma-separated priority list ("first,second,third") -- see
// callProviderWithFallback below for how the list is walked and models that
// fail are put in cooldown.
function aiConfig(): AiConfig | null {
  const apiKey = process.env.AI_API_KEY;
  const baseUrl = process.env.AI_BASE_URL;
  const rawModel = process.env.AI_MODEL;
  if (!apiKey || !baseUrl || !rawModel) return null;
  const models = rawModel
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);
  if (models.length === 0) return null;
  return { apiKey, baseUrl, models };
}

// How long a model that failed every attempt is skipped for, before the
// priority list is given another chance to use it -- module-level (not
// per-request) so the cooldown is actually shared across requests, and reset
// on process restart. "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,openrouter/free"
// with a bad first entry means every request eats the full fallback chain
// otherwise.
const MODEL_COOLDOWN_MS = 15 * 60 * 1000;
// Per-attempt timeout, not overall -- "a couple of requests with a timeout"
// means each attempt gets its own clock, not one budget shared across
// retries and fallbacks.
const MODEL_ATTEMPT_TIMEOUT_MS = 12_000;
const MODEL_ATTEMPTS_PER_MODEL = 2;

const modelCooldownUntil = new Map<string, number>();

// Priority order, with anything still cooling down pushed out of the way --
// unless *everything* is cooling down, in which case ignore cooldowns
// entirely rather than fail the request outright (a stale cooldown shouldn't
// be able to take the assistant down completely).
function candidateModels(models: string[]): string[] {
  const now = Date.now();
  const available = models.filter((m) => (modelCooldownUntil.get(m) ?? 0) <= now);
  return available.length > 0 ? available : models;
}

// Tries each model in priority order (first, second, third, ...), skipping
// any still in cooldown from a recent failure. Each model gets up to
// MODEL_ATTEMPTS_PER_MODEL attempts, each bounded by MODEL_ATTEMPT_TIMEOUT_MS
// -- a model that exhausts its attempts is put in cooldown for
// MODEL_COOLDOWN_MS and the next model in the list is tried. A 401/403 is
// treated as an API-key problem (applies to every model on this provider,
// not this one specifically) and returned immediately without burning
// through the rest of the list.
async function callProviderWithFallback(
  config: AiConfig,
  messages: AiChatMessage[],
  toolChoice: "required" | "auto",
  outerSignal: AbortSignal,
  log: MinimalLogger
): Promise<{ upstream: Response; model: string }> {
  const models = candidateModels(config.models);
  let lastResponse: Response | null = null;
  let lastError: unknown;

  for (const model of models) {
    for (let attempt = 0; attempt < MODEL_ATTEMPTS_PER_MODEL; attempt++) {
      if (outerSignal.aborted) throw new DOMException("aborted", "AbortError");

      const timeoutController = new AbortController();
      const timeout = setTimeout(() => timeoutController.abort(), MODEL_ATTEMPT_TIMEOUT_MS);
      const onOuterAbort = () => timeoutController.abort();
      outerSignal.addEventListener("abort", onOuterAbort);

      try {
        const upstream = await callProvider(config, model, messages, toolChoice, timeoutController.signal);
        if (upstream.ok) {
          modelCooldownUntil.delete(model);
          return { upstream, model };
        }
        if (upstream.status === 401 || upstream.status === 403) return { upstream, model };

        // Read the body now, not just the status -- every attempt but the
        // very last one used to get its body silently .cancel()ed here,
        // discarded before anything ever logged it, so a failure on model 1
        // or 2 of a 3-model list only ever showed up as a bare status code
        // with no way to tell "model not found" apart from "bad request
        // shape" apart from "provider outage". Rebuild a fresh Response from
        // the captured text so the rest of this function's (and the route
        // handler's) `upstream.text()`/`.status`/`.ok` reads still work
        // exactly as before if this ends up being the one ultimately
        // returned to the caller.
        const bodyText = await upstream.text().catch(() => "");
        lastResponse?.body?.cancel().catch(() => {});
        lastResponse = new Response(bodyText, { status: upstream.status, statusText: upstream.statusText, headers: upstream.headers });
        log.warn({ model, attempt, status: upstream.status, body: bodyText.slice(0, 2000) }, "ai chat: model attempt failed");
      } catch (err) {
        if (outerSignal.aborted) throw err;
        lastError = err;
        log.warn(
          { model, attempt, err: err instanceof Error ? err.message : String(err) },
          "ai chat: model attempt errored"
        );
      } finally {
        clearTimeout(timeout);
        outerSignal.removeEventListener("abort", onOuterAbort);
      }
    }

    modelCooldownUntil.set(model, Date.now() + MODEL_COOLDOWN_MS);
    log.warn({ model, cooldownMs: MODEL_COOLDOWN_MS }, "ai chat: model exhausted retries, entering cooldown");
  }

  if (lastResponse) return { upstream: lastResponse, model: models[models.length - 1] };
  throw lastError ?? new Error("no configured AI models are available");
}

// Providers report errors as either {"error": "message"} or (OpenAI-style)
// {"error": {"message": "..."}} -- unwrap to a single string either way so
// the client only ever has to deal with the one shape the rest of this
// app's routes already use, instead of double-encoding the raw upstream body
// inside a JSON string field.
function extractProviderErrorMessage(body: string): string {
  try {
    const parsed = JSON.parse(body) as { error?: { message?: string } | string };
    if (typeof parsed.error === "string") return parsed.error;
    if (parsed.error?.message) return parsed.error.message;
  } catch {
    /* not JSON -- use the raw body */
  }
  return body;
}

function isValidMessages(value: unknown): value is AiChatMessage[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (m) =>
        m &&
        typeof m === "object" &&
        (m.role === "system" || m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    )
  );
}

async function callProvider(
  config: AiConfig,
  model: string,
  messages: AiChatMessage[],
  toolChoice: "required" | "auto",
  signal: AbortSignal
): Promise<Response> {
  return fetch(`${config.baseUrl.replace(/\/+$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({ model, messages, tools: TOOLS, tool_choice: toolChoice, stream: true }),
    signal,
  });
}

interface StreamDeltaToolCall {
  index?: number;
  id?: string;
  function?: { name?: string; arguments?: string };
  extra_content?: { google?: { thought_signature?: string } };
}

interface StreamChunk {
  choices?: { delta?: { content?: string | null; tool_calls?: StreamDeltaToolCall[] } }[];
  error?: { message?: string };
}

// Parses an OpenAI-compatible SSE stream into JSON-decoded events, same
// "data: {...}" / "data: [DONE]" framing the client (aiChat.ts) already
// parses -- this just does it server-side so a provider's own stream can be
// re-emitted to the browser as it arrives instead of being buffered whole.
async function* iterSseEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
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
        try {
          yield JSON.parse(data) as StreamChunk;
        } catch {
          continue; // partial/malformed chunk -- keep reading rather than aborting the stream
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// Moved server-side from client/src/components/ChatPanel.tsx (§2.7) -- the
// text itself is unchanged, only where it lives. The server is now the only
// thing that can ever set the system message; a client-supplied one is
// discarded outright (see the request handler below), which is what makes
// this instruction actually enforceable rather than just a convention the
// stock client happens to follow.
const SERVER_SYSTEM_PROMPT =
  "You are an assistant embedded in LlamaToaster, a local LLM inference benchmarking tool for " +
  "llama.cpp / llama-bench. Help the user choose llama.cpp parameters (threads, n_gpu_layers, " +
  "batch/ubatch size, KV cache quantization, flash attention), suggest GGUF models that fit their " +
  "hardware, and analyze the benchmark data below to say which configuration performs best on " +
  "their machine(s).\n\n" +
  "Grounding: you have search_huggingface_gguf_models and list_huggingface_gguf_files tools. Your " +
  "training data's knowledge of model releases is stale — treat any belief about whether a model " +
  "exists yet, its parameter count, or its file size as unverified until a tool confirms it this " +
  "turn. Never say a model \"isn't released yet\" or \"isn't available\" based on memory; search " +
  "first. If a tool call fails or returns nothing useful, say so plainly instead of falling back to " +
  "a guess.\n\n" +
  "Community data: you also have list_community_facets and get_community_aggregates, which return " +
  "OTHER users' benchmark throughput, anonymised and aggregated across everyone who ran a given " +
  "model/backend/platform/GPU combination. Every row you get back already represents 5 or more " +
  "distinct people — you will never be given a row for fewer, so there is nothing to protect by " +
  "refusing to use one. But you must never try to make a row MORE identifying than it already is: " +
  "never say or imply how many people are behind a number beyond echoing its own contributor_count, " +
  "never attribute a result to a specific person, never speculate about who a contributor might be, " +
  "and never chain two aggregate rows together to narrow a group down toward one individual. If asked " +
  "who submitted a result, or for data about a specific person, explain that this tool only returns " +
  "anonymised group aggregates and cannot identify individuals.\n\n" +
  "Style: keep replies short — a few sentences or a short bullet list. Don't use a markdown table " +
  "unless the user is comparing multiple options side by side or explicitly asks for one. Skip " +
  "preamble and caveats; lead with the answer.";

const MAX_CONTEXT_RESULT_ROWS = 150;

// Mirrors client/src/utils.ts's formatFlashAttn exactly -- duplicated rather
// than shared since it's a single four-line pure function with no other
// server-side use; not worth a shared/ module for.
function formatFlashAttn(value: string): string {
  if (value === "true" || value === "1") return "on";
  if (value === "false" || value === "0") return "off";
  return value;
}

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.6): the caller's own machines
// once authenticated; every machine in single-tenant mode (userId undefined
// -- AUTH_ENABLED off), unchanged from before this scoping existed.
function hardwareSummary(userId: string | undefined): string {
  const workers = userId ? repo.workerRepo.listWorkersForUser(userId) : repo.workerRepo.listWorkers();
  if (workers.length === 0) return "(no machines connected)";
  return workers
    .map((w) => {
      const hw = w.hardware;
      const cpu = hw ? hw.cpu.brand || hw.cpu.manufacturer || "unknown CPU" : "unknown CPU";
      const gpu = hw && hw.gpu.length > 0 ? hw.gpu.map((g) => g.model).join(", ") : "no discrete GPU reported";
      const activeBuild = w.installedBuilds.find((b) => b.active)?.tag;
      return `- ${w.displayName} — ${w.backend ?? "unknown"} backend, ${w.platform ?? "?"}/${w.arch ?? "?"}, CPU: ${cpu}, GPU: ${gpu}, build ${activeBuild ?? "none"} (${w.status})`;
    })
    .join("\n");
}

function modelsSummary(): { text: string; count: number } {
  const models = repo.listModels();
  if (models.length === 0) return { text: "(none registered)", count: 0 };
  const text = models
    .map((m) => {
      const family = typeof m.metadata.arch === "string" ? m.metadata.arch : "unknown arch";
      const quant = typeof m.metadata.quant === "string" ? `, ${m.metadata.quant}` : "";
      const repoLabel = m.hf_repo ? ` — ${m.hf_repo}` : "";
      return `- ${m.filename} (${family}${quant})${repoLabel}`;
    })
    .join("\n");
  return { text, count: models.length };
}

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.6): the caller's own results once
// authenticated; every result in single-tenant mode, same as hardwareSummary
// above.
function resultsSummary(userId: string | undefined): string {
  const rows: ResultExportRow[] = loadExportRows(userId);
  if (rows.length === 0) return "(no benchmark results yet)";
  const sorted = [...rows].sort((a, b) => b.created_at - a.created_at);
  const capped = sorted.slice(0, MAX_CONTEXT_RESULT_ROWS);
  const header =
    "worker | backend | model | test | n_prompt | n_gen | threads | ngl (loaded/total) | batch | ubatch | ctk | ctv | fa | avg_tps | ram_peak_mib | vram_peak_mib | vram_total_mib";
  const lines = capped.map(
    (r) =>
      `${r.worker_name} | ${r.backend_type}${r.backend_device_name ? ` (${r.backend_device_name})` : ""} | ${r.model_filename} | ${r.test_type} | ${r.n_prompt} | ${r.n_gen} | ${r.n_threads} | ` +
      `${r.n_gpu_layers} (${r.gpu_layers_loaded ?? "?"}/${r.total_model_layers ?? "?"}) | ${r.batch_size} | ${r.ubatch_size} | ${r.cache_type_k} | ${r.cache_type_v} | ${formatFlashAttn(r.flash_attn)} | ` +
      `${r.avg_tps.toFixed(2)} | ${r.ram_peak_mib} | ${r.vram_peak_mib ?? "n/a"} | ${r.gpu_memory_total_mib ?? "n/a"}`
  );
  const scope = rows.length > capped.length ? `most recent ${capped.length} of ${rows.length} total` : `${rows.length} total`;
  return `(${scope})\n${header}\n${lines.join("\n")}`;
}

// Server-side port of client/src/api/aiContext.ts's buildContextSnapshot
// (§2.7) -- same output shape, but reads the DB directly instead of an HTTP
// round-trip to this server's own API. userId now genuinely scopes hardware/
// results to the caller (§4.6) -- modelsSummary stays unscoped, since the
// model catalog is global (§4.3). Exported for ai.test.ts's own isolation
// coverage -- otherwise unused outside this module.
export function buildContextSnapshot(userId: string | undefined): string {
  const { text: models, count: modelCount } = modelsSummary();
  return [
    "## Hardware",
    hardwareSummary(userId),
    "",
    `## Registered models (n=${modelCount})`,
    models,
    "",
    "## Recent benchmark results",
    resultsSummary(userId),
  ].join("\n");
}

export async function aiRoutes(app: FastifyInstance): Promise<void> {
  // Lets the client show "set up the assistant" vs. the chat UI, and which
  // model is active, without ever exposing the key itself. Also surfaces the
  // caller's own remaining §2.6 quota (when authenticated) so the limit is
  // visible in the chat panel before it bites, per the plan's own
  // requirement -- omitted entirely with AUTH_ENABLED off/unauthenticated,
  // since there's no per-user budget to report in that mode.
  app.get("/api/ai/status", async (request) => {
    const config = aiConfig();
    const authed = resolveAuthUser(request);
    const quota = authed
      ? (() => {
          const { day, hour } = utcDayAndHour(Date.now());
          return {
            remainingHour: Math.max(0, AI_LIMITS.perUserPerHour - repo.aiUsageRepo.getHourCount(authed.user.id, day, hour)),
            remainingDay: Math.max(0, AI_LIMITS.perUserPerDay - repo.aiUsageRepo.getUserDayCount(authed.user.id, day)),
          };
        })()
      : undefined;
    return { configured: config !== null, model: config?.models[0], quota };
  });

  // Proxies to the configured OpenAI-compatible provider so the API key
  // never leaves the server -- the browser only ever talks to this route.
  // Runs a tool-calling loop where every round is itself streamed from the
  // provider (see iterSseEvents above): a round that turns out to be tool
  // calls has its fragments reassembled internally and never reaches the
  // client, while a round that's the final answer has its content deltas
  // forwarded live as they arrive, so the reply appears token-by-token
  // instead of all at once after the whole generation finishes.
  app.post<{ Body: { messages?: unknown } }>("/api/ai/chat", async (request, reply) => {
    const config = aiConfig();
    if (!config) {
      return reply.code(501).send({
        error: "AI assistant not configured on this server -- set AI_API_KEY, AI_BASE_URL, and AI_MODEL in a .env file at the repo root, then restart the server.",
      });
    }
    const rawMessages = request.body?.messages;
    if (!isValidMessages(rawMessages)) {
      return reply.code(400).send({ error: "messages must be a non-empty array of {role, content}" });
    }

    // §2.7: the server owns the system prompt -- ANY client-supplied system
    // message is discarded outright, never merged with or allowed to
    // override SERVER_SYSTEM_PROMPT below. This is what actually makes that
    // instruction enforceable (previously the browser built the whole
    // conversation including the system message, so `curl`ing this endpoint
    // directly could send anything).
    const userMessages = rawMessages.filter((m) => m.role !== "system");
    if (userMessages.length !== rawMessages.length) {
      request.log.warn({ discarded: rawMessages.length - userMessages.length }, "ai chat: client-supplied system message(s) discarded");
    }

    if (userMessages.length > AI_LIMITS.maxMessages) {
      return reply.code(400).send({ error: `too many messages in this conversation (max ${AI_LIMITS.maxMessages})` });
    }
    const totalChars = userMessages.reduce((sum, m) => sum + m.content.length, 0);
    if (totalChars > AI_LIMITS.maxTotalChars) {
      return reply.code(400).send({ error: `conversation too long (max ${AI_LIMITS.maxTotalChars} characters total)` });
    }

    // §2.6 budget: only enforced against a REAL user (ai_usage.user_id has a
    // real FK to users(id) -- there's nothing to attribute usage to with
    // AUTH_ENABLED off, so Stage 1's original unmetered behavior applies
    // unchanged in that mode). The global circuit breaker, by contrast,
    // protects the operator's own provider budget regardless of who's
    // calling, so it's checked either way -- it just never accumulates at
    // all in Stage-1-only mode, since nothing ever gets recorded.
    const authed = resolveAuthUser(request);
    const { day, hour } = utcDayAndHour(Date.now());
    if (repo.aiUsageRepo.getGlobalDayCount(day) >= AI_LIMITS.globalPerDay) {
      return reply.code(503).send({ error: "the assistant is over its daily budget, try again tomorrow" });
    }
    if (authed) {
      if (repo.aiUsageRepo.getHourCount(authed.user.id, day, hour) >= AI_LIMITS.perUserPerHour) {
        return reply
          .code(429)
          .send({ error: `hourly AI usage limit reached (${AI_LIMITS.perUserPerHour}/hour) -- try again later` });
      }
      if (repo.aiUsageRepo.getUserDayCount(authed.user.id, day) >= AI_LIMITS.perUserPerDay) {
        return reply
          .code(429)
          .send({ error: `daily AI usage limit reached (${AI_LIMITS.perUserPerDay}/day) -- try again tomorrow` });
      }
      repo.aiUsageRepo.recordUsage(authed.user.id, day, hour);
    }

    // Aborts the upstream request if the client disconnects (closes the tab,
    // hits Stop) -- avoids burning the configured provider's quota on a
    // generation nobody's listening to anymore. Deliberately reply.raw (the
    // response socket), not request.raw: the *request* stream's 'close'
    // fires as soon as Fastify finishes reading the incoming body -- long
    // before a response exists -- and would abort every call almost
    // immediately. reply.raw only closes when the client actually stops
    // waiting for the response.
    // Logging below is deliberately metadata-only: round/tool/timing/outcome
    // info, never message content -- no prompt text (system/user/assistant)
    // and no generated reply text ever gets written to the log, only counts,
    // lengths, and structured tool arguments (a search query / repo id, not
    // conversational content).
    const controller = new AbortController();
    const requestStartedAt = Date.now();
    let finished = false;
    reply.raw.on("close", () => {
      controller.abort();
      if (!finished) request.log.info("ai chat: client disconnected before a final answer");
    });

    request.log.info({ messageCount: userMessages.length, authed: authed !== null }, "ai chat: request started");

    // Server-built system message, prepended unconditionally -- see §2.7's
    // header comment above. buildContextSnapshot is synchronous (direct DB
    // reads, no network round-trip), unlike the client-side version it
    // replaces.
    const conversation: AiChatMessage[] = [
      { role: "system", content: `${SERVER_SYSTEM_PROMPT}\n\n${buildContextSnapshot(authed?.user.id)}` },
      ...userMessages,
    ];

    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      // Round 0 forces a tool call ("required", not "auto") -- left to its own
      // judgment, a model tends to answer straight from training data for
      // anything it feels "confident" about, which for a fast-moving model
      // landscape is exactly the wrong call (e.g. confidently claiming a model
      // that's been out for months "isn't released yet"). Forcing one real
      // lookup up front means every reply is grounded in at least one live
      // fact before the model gets to talk; later rounds go back to "auto" so
      // it can choose to look up more or wrap up.
      const toolChoice = round === 0 ? "required" : "auto";
      const roundStartedAt = Date.now();
      let upstream: Response;
      let usedModel: string;
      try {
        ({ upstream, model: usedModel } = await callProviderWithFallback(
          config,
          conversation,
          toolChoice,
          controller.signal,
          request.log
        ));
      } catch (err) {
        request.log.error({ round, err }, "ai chat: could not reach any configured provider/model");
        return reply.code(502).send({
          error: `Could not reach the AI provider: ${err instanceof Error ? err.message : String(err)}`,
        });
      }

      if (!upstream.ok) {
        const text = await upstream.text().catch(() => "");
        const message = extractProviderErrorMessage(text) || `AI provider returned ${upstream.status}`;
        const prefix = upstream.status === 401 || upstream.status === 403 ? "AI provider rejected the configured API key: " : "";
        request.log.error({ round, model: usedModel, status: upstream.status, text }, "ai chat: provider returned an error");
        return reply.code(upstream.status || 502).send({ error: `${prefix}${message}` });
      }
      if (!upstream.body) {
        request.log.error({ round, model: usedModel }, "ai chat: provider returned no response body");
        return reply.code(502).send({ error: "AI provider returned an empty response" });
      }

      // Accumulated across this round's stream: tool_calls arrive in
      // fragments (id/name only on their first chunk, arguments dribbled in
      // piece by piece, keyed by index) that have to be reassembled before
      // they're runnable, same as the OpenAI streaming tool-call contract
      // generally. Content, by contrast, is forwarded to the client the
      // moment it arrives (see streamingToClient below) -- that's the whole
      // point of streaming a round that turns out to be the final answer.
      let content = "";
      const toolCallFragments = new Map<
        string,
        { id: string; name: string; arguments: string; thoughtSignature?: string }
      >();
      let lastFragmentKey: string | null = null;
      let streamingToClient = false;

      try {
        for await (const chunk of iterSseEvents(upstream.body)) {
          if (chunk.error?.message) throw new Error(chunk.error.message);
          const delta = chunk.choices?.[0]?.delta;
          if (!delta) continue;

          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              let key: string = typeof tc.index === "number" ? `idx:${tc.index}` : lastFragmentKey ?? "idx:0";
              const existing = toolCallFragments.get(key);
              // A key already holding a fragment with a *different* call id
              // than this chunk means two distinct parallel tool calls
              // collided on the same (or missing) index -- observed with
              // Gemini's parallel tool calls via an OpenAI-compat shim, which
              // stamps every call in a round with index 0/undefined instead
              // of unique indices. Left alone, their arguments get appended
              // together into one invalid JSON blob (e.g.
              // `{"query":"a"}{"query":"b"}`), which the provider then
              // rejects wholesale on the next round once that malformed tool
              // call is echoed back in the conversation history. Re-key by
              // the call's own id -- still unique even when index isn't --
              // so each call gets its own fragment.
              if (existing && tc.id && existing.id && existing.id !== tc.id) {
                key = `id:${tc.id}`;
              }
              lastFragmentKey = key;

              const fragment = toolCallFragments.get(key) ?? { id: "", name: "", arguments: "" };
              if (tc.id) fragment.id = tc.id;
              if (tc.function?.name) fragment.name = tc.function.name;
              if (tc.function?.arguments) fragment.arguments += tc.function.arguments;
              // Gemini attaches a per-call thought_signature that must be
              // echoed back verbatim on this same tool call in the next
              // round's assistant message, or Gemini 3 rejects the request
              // with "Function call is missing a thought_signature" -- see
              // https://ai.google.dev/gemini-api/docs/thinking#signatures.
              // Captured the same way id/name are (last non-empty value
              // wins) since it arrives as one opaque blob, not chunked.
              if (tc.extra_content?.google?.thought_signature) {
                fragment.thoughtSignature = tc.extra_content.google.thought_signature;
              }
              toolCallFragments.set(key, fragment);
            }
          }

          // Guarded on toolCallFragments.size === 0: a correctly-implemented
          // OpenAI-compatible provider never mixes content and tool_calls
          // within one message, but if one somehow did, committing to
          // streaming raw content to the browser and then trying to run
          // tools instead would mean writing a second HTTP response onto an
          // already-hijacked socket. Once we've started forwarding content
          // for this round, that round is the final answer, full stop.
          if (typeof delta.content === "string" && delta.content.length > 0 && toolCallFragments.size === 0) {
            content += delta.content;
            if (!streamingToClient) {
              streamingToClient = true;
              finished = true;
              // reply.hijack() takes the raw socket over from Fastify
              // entirely, so headers must go straight on reply.raw --
              // reply.header() only buffers into Fastify's own header store,
              // which is flushed by reply.send(), which we never call on
              // this path.
              reply.hijack();
              reply.raw.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
              // Lets the client attribute this reply to the model that
              // actually produced it (fallback chains mean that isn't always
              // config.models[0]) -- sent once, before any content delta.
              reply.raw.write(`data: ${JSON.stringify({ model: usedModel })}\n\n`);
            }
            reply.raw.write(`data: ${JSON.stringify({ choices: [{ delta: { content: delta.content } }] })}\n\n`);
          }
        }
      } catch (err) {
        if (controller.signal.aborted) return; // client disconnected -- nothing left to tell it
        request.log.error({ round, err }, "ai chat: error while streaming from provider");
        if (streamingToClient) {
          try {
            reply.raw.write(
              `data: ${JSON.stringify({ error: { message: err instanceof Error ? err.message : String(err) } })}\n\n`
            );
            reply.raw.end();
          } catch {
            /* socket already gone */
          }
          return;
        }
        return reply.code(502).send({
          error: `AI provider stream failed: ${err instanceof Error ? err.message : String(err)}`,
        });
      }

      const roundLatencyMs = Date.now() - roundStartedAt;

      if (toolCallFragments.size > 0 && !streamingToClient) {
        const toolCalls: AiToolCall[] = Array.from(toolCallFragments.values()).map((tc) => ({
          id: tc.id,
          function: { name: tc.name, arguments: tc.arguments },
          ...(tc.thoughtSignature ? { extra_content: { google: { thought_signature: tc.thoughtSignature } } } : {}),
        }));
        request.log.info(
          { round, model: usedModel, toolChoice, roundLatencyMs, toolCallCount: toolCalls.length },
          "ai chat: round produced tool calls"
        );
        conversation.push({ role: "assistant", content, tool_calls: toolCalls });
        for (const call of toolCalls) {
          const toolStartedAt = Date.now();
          request.log.info(
            { round, tool: call.function.name, args: call.function.arguments },
            "ai chat: tool call started"
          );
          const result = await runTool(call.function.name, call.function.arguments, authed?.user.id);
          const isError = Boolean(result && typeof result === "object" && "error" in result);
          request.log.info(
            { round, tool: call.function.name, toolLatencyMs: Date.now() - toolStartedAt, ok: !isError },
            "ai chat: tool call finished"
          );
          conversation.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
        }
        continue; // let the model see the tool results and respond again
      }

      // Final answer -- either it streamed content above, or the provider
      // sent an empty round (no content, no tool_calls); either way this
      // round ends the request.
      if (!streamingToClient) {
        finished = true;
        reply.hijack();
        reply.raw.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
        reply.raw.write(`data: ${JSON.stringify({ model: usedModel })}\n\n`);
        reply.raw.write(`data: ${JSON.stringify({ choices: [{ delta: { content: "" } }] })}\n\n`);
      }
      const totalLatencyMs = Date.now() - requestStartedAt;
      request.log.info(
        { round, model: usedModel, toolChoice, roundLatencyMs, totalLatencyMs, contentLength: content.length },
        "ai chat: request completed with a final answer"
      );
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
      return;
    }

    request.log.warn(
      { rounds: MAX_TOOL_ROUNDS, totalLatencyMs: Date.now() - requestStartedAt },
      "ai chat: gave up after max tool-call rounds without a final answer"
    );
    finished = true;
    reply.hijack();
    reply.raw.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
    reply.raw.write(
      `data: ${JSON.stringify({ choices: [{ delta: { content: "(stopped after several tool calls without a final answer -- try rephrasing.)" } }] })}\n\n`
    );
    reply.raw.write("data: [DONE]\n\n");
    reply.raw.end();
  });
}
