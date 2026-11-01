// Every setting the Serve page can change, as data. This table is the only
// route from the browser to llama-server's argv: the UI renders it, the server
// validates against it, and the worker builds flags from it. There is no
// raw-arguments path, which is what keeps --tools (exec_shell_command, write_file),
// --log-prompts-dir, --slot-save-path and friends out without a deny list.
//
// A setting left unset passes no flag, so the build's own default stands. All
// flags below were read from `llama-server --help`, build 11609. Older builds lack
// some; DaemonStatus.supported_settings says which, per binary.

export type SettingValue = number | string | boolean;
export type SettingValues = Record<string, SettingValue>;

export type LoadSettingGroup = "context" | "placement" | "cpu" | "cache" | "chat" | "vision" | "speculative" | "serving";

export const LOAD_GROUP_LABELS: Record<LoadSettingGroup, string> = {
  context: "Context and batching",
  placement: "GPU placement",
  cpu: "CPU and memory",
  cache: "Prompt cache",
  chat: "Chat format",
  vision: "Vision",
  speculative: "Speculative decoding",
  serving: "Serving",
};

interface BaseDef {
  key: string;
  label: string;
  help: string;
  /** Rendered behind "Show advanced". */
  advanced?: boolean;
}

export interface LoadIntDef extends BaseDef {
  kind: "int" | "float";
  group: LoadSettingGroup;
  flag: string;
  min: number;
  max: number;
  step?: number;
  /** A slider instead of a bare number box. */
  slider?: boolean;
  /** Log-spaced slider stops, for context sizes. */
  logScale?: boolean;
}

export interface LoadEnumDef extends BaseDef {
  kind: "enum";
  group: LoadSettingGroup;
  flag: string;
  options: readonly string[];
}

export interface LoadToggleDef extends BaseDef {
  kind: "toggle";
  group: LoadSettingGroup;
  /** Probed against --help; also the "on" form. */
  flag: string;
  /** Absent for a one-way switch: off then means "pass nothing". */
  offFlag?: string;
}

export interface LoadTextDef extends BaseDef {
  kind: "text";
  group: LoadSettingGroup;
  flag: string;
  pattern: RegExp;
  maxLength: number;
  placeholder?: string;
  /** Extra check past the pattern. Returns the problem, or null. */
  check?: (value: string) => string | null;
}

export type LoadSettingDef = LoadIntDef | LoadEnumDef | LoadToggleDef | LoadTextDef;

const CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] as const;

// Passed as argv with no shell, so these patterns are not guarding quoting. They
// stop a value being read as another flag, and keep the free-form fields narrow.
const DEVICE_LIST = /^[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$/;
const NUMBER_LIST = /^\d+(\.\d+)?(,\d+(\.\d+)?)*$/;
const TENSOR_OVERRIDE = /^[A-Za-z0-9_.*+?|()[\]\\^$][A-Za-z0-9_.*+?|()[\]\\^$-]*=[A-Za-z0-9_]+(,[A-Za-z0-9_.*+?|()[\]\\^$][A-Za-z0-9_.*+?|()[\]\\^$-]*=[A-Za-z0-9_]+)*$/;
const TEMPLATE_NAME = /^[a-z0-9][a-z0-9._-]*$/;

function jsonObjectProblem(value: string): string | null {
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? null : "must be a JSON object";
  } catch {
    return "must be valid JSON";
  }
}

export const LOAD_SETTINGS: readonly LoadSettingDef[] = [
  // Context and batching
  { key: "ctx_size", kind: "int", group: "context", flag: "--ctx-size", label: "Context length", min: 0, max: 8388608, step: 1024, slider: true, logScale: true, help: "Tokens of context. 0 takes it from the model. Larger contexts cost KV cache memory." },
  { key: "parallel", kind: "int", group: "context", flag: "--parallel", label: "Parallel requests", min: 1, max: 64, slider: true, help: "Server slots. With a unified KV cache they share the context; each extra slot lowers the speed of the others." },
  { key: "batch_size", kind: "int", group: "context", flag: "--batch-size", label: "Batch size", min: 1, max: 1048576, help: "Logical maximum batch size." },
  { key: "ubatch_size", kind: "int", group: "context", flag: "--ubatch-size", label: "Physical batch size", min: 1, max: 1048576, help: "Physical maximum batch size. Sets the compute buffer and the logits reservation." },
  { key: "flash_attn", kind: "enum", group: "context", flag: "--flash-attn", label: "Flash Attention", options: ["auto", "on", "off"], help: "Whether to use Flash Attention." },
  { key: "cache_type_k", kind: "enum", group: "context", flag: "--cache-type-k", label: "KV cache type K", options: CACHE_TYPES, help: "Data type of the K half of the KV cache. Non-float types need Flash Attention off on most backends." },
  { key: "cache_type_v", kind: "enum", group: "context", flag: "--cache-type-v", label: "KV cache type V", options: CACHE_TYPES, help: "Data type of the V half of the KV cache." },
  { key: "kv_unified", kind: "toggle", group: "context", flag: "--kv-unified", offFlag: "--no-kv-unified", label: "Unified KV cache", advanced: true, help: "One KV buffer shared by all slots. On by default when the slot count is automatic." },
  { key: "kv_unified_per_slot", kind: "int", group: "context", flag: "--kv-unified-per-slot", label: "Context per slot", min: 1, max: 8388608, advanced: true, help: "Context limit per slot. Without a context length, the shared pool is sized to slots × this." },
  { key: "kv_offload", kind: "toggle", group: "context", flag: "--kv-offload", offFlag: "--no-kv-offload", label: "KV cache on GPU", help: "Keep the KV cache in GPU memory." },
  { key: "swa_full", kind: "toggle", group: "context", flag: "--swa-full", label: "Full-size SWA cache", advanced: true, help: "Use a full-size cache for sliding-window attention layers." },
  { key: "context_shift", kind: "toggle", group: "context", flag: "--context-shift", offFlag: "--no-context-shift", label: "Context shift", advanced: true, help: "Drop the oldest tokens instead of stopping when a generation fills the context." },

  // GPU placement
  { key: "n_gpu_layers", kind: "text", group: "placement", flag: "--n-gpu-layers", label: "GPU layers", pattern: /^(auto|all|\d{1,4})$/, maxLength: 8, placeholder: "auto", help: "Layers in VRAM: a number, all, or auto." },
  { key: "fit", kind: "enum", group: "placement", flag: "--fit", label: "Fit to memory", options: ["on", "off"], help: "Let llama.cpp adjust unset settings to fit in device memory." },
  { key: "fit_target", kind: "text", group: "placement", flag: "--fit-target", label: "Fit margin (MiB)", pattern: NUMBER_LIST, maxLength: 64, placeholder: "1024", advanced: true, help: "Memory to leave free per device when fitting." },
  { key: "fit_ctx", kind: "int", group: "placement", flag: "--fit-ctx", label: "Fit minimum context", min: 1, max: 8388608, advanced: true, help: "Smallest context fitting may choose." },
  { key: "device", kind: "text", group: "placement", flag: "--device", label: "Devices", pattern: DEVICE_LIST, maxLength: 128, placeholder: "Vulkan0", advanced: true, help: "Comma-separated devices to offload to, or none." },
  { key: "split_mode", kind: "enum", group: "placement", flag: "--split-mode", label: "Split mode", options: ["none", "layer", "row", "tensor"], advanced: true, help: "How to split the model across GPUs." },
  { key: "main_gpu", kind: "int", group: "placement", flag: "--main-gpu", label: "Main GPU", min: 0, max: 64, advanced: true, help: "The GPU to use when split mode is none." },
  { key: "tensor_split", kind: "text", group: "placement", flag: "--tensor-split", label: "Tensor split", pattern: NUMBER_LIST, maxLength: 64, placeholder: "3,1", advanced: true, help: "Share of the model per GPU." },
  { key: "cpu_moe", kind: "toggle", group: "placement", flag: "--cpu-moe", label: "Experts on CPU", advanced: true, help: "Keep all Mixture of Experts weights in the CPU." },
  { key: "n_cpu_moe", kind: "int", group: "placement", flag: "--n-cpu-moe", label: "Experts on CPU (layers)", min: 0, max: 4096, advanced: true, help: "Keep the expert weights of the first N layers in the CPU." },
  { key: "moe_cache_mib", kind: "int", group: "placement", flag: "--moe-cache-mib", label: "Expert cache (MiB)", min: 0, max: 1048576, advanced: true, help: "GPU cache for experts kept in the CPU." },
  { key: "n_cpu_ffn", kind: "int", group: "placement", flag: "--n-cpu-ffn", label: "FFN on CPU (layers)", min: 0, max: 4096, advanced: true, help: "Keep the dense FFN weights of the first N layers in the CPU." },
  { key: "override_tensor", kind: "text", group: "placement", flag: "--override-tensor", label: "Tensor overrides", pattern: TENSOR_OVERRIDE, maxLength: 512, placeholder: "blk\\.1\\..*=CPU", advanced: true, help: "pattern=buffer type, comma-separated." },
  { key: "op_offload", kind: "toggle", group: "placement", flag: "--op-offload", offFlag: "--no-op-offload", label: "Offload host operations", advanced: true, help: "Offload host tensor operations to the device." },
  { key: "no_host", kind: "toggle", group: "placement", flag: "--no-host", label: "Bypass host buffer", advanced: true, help: "Skip the host buffer so extra buffers can be used." },
  { key: "repack", kind: "toggle", group: "placement", flag: "--repack", offFlag: "--no-repack", label: "Weight repacking", advanced: true, help: "Repack weights for the CPU backend." },

  // CPU and memory
  { key: "threads", kind: "int", group: "cpu", flag: "--threads", label: "CPU threads", min: 1, max: 1024, slider: true, help: "Threads used for generation." },
  { key: "threads_batch", kind: "int", group: "cpu", flag: "--threads-batch", label: "Prompt threads", min: 1, max: 1024, advanced: true, help: "Threads for batch and prompt processing." },
  { key: "load_mode", kind: "enum", group: "cpu", flag: "--load-mode", label: "Load mode", options: ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"], help: "How weights are loaded. mlock keeps them from being swapped. Tests on this app use mlock." },
  { key: "lazy_mode", kind: "enum", group: "cpu", flag: "--lazy-mode", label: "Lazy tensors", options: ["auto", "on", "off"], advanced: true, help: "Read very large per-layer tensors from disk on demand." },
  { key: "numa", kind: "enum", group: "cpu", flag: "--numa", label: "NUMA", options: ["distribute", "isolate", "numactl"], advanced: true, help: "Optimizations for NUMA systems." },
  { key: "prio", kind: "enum", group: "cpu", flag: "--prio", label: "Thread priority", options: ["-1", "0", "1", "2", "3"], advanced: true, help: "low (-1), normal (0), medium (1), high (2), realtime (3). Realtime can starve the rest of the machine, including its desktop." },
  { key: "prio_batch", kind: "enum", group: "cpu", flag: "--prio-batch", label: "Prompt thread priority", options: ["0", "1", "2", "3"], advanced: true, help: "normal (0), medium (1), high (2), realtime (3). Realtime can starve the rest of the machine, including its desktop." },
  { key: "poll", kind: "int", group: "cpu", flag: "--poll", label: "Polling level", min: 0, max: 100, advanced: true, help: "How hard threads spin while waiting for work." },

  // Prompt cache
  { key: "cache_prompt", kind: "toggle", group: "cache", flag: "--cache-prompt", offFlag: "--no-cache-prompt", label: "Prompt caching", help: "Reuse the common prefix of the previous prompt." },
  { key: "cache_ram", kind: "int", group: "cache", flag: "--cache-ram", label: "Prompt cache size (MiB)", min: -1, max: 16777216, help: "Maximum RAM for saved prompts. -1 no limit, 0 off." },
  { key: "cache_reuse", kind: "int", group: "cache", flag: "--cache-reuse", label: "Cache reuse chunk", min: 0, max: 8388608, advanced: true, help: "Smallest chunk to try reusing by KV shifting." },
  { key: "ctx_checkpoints", kind: "int", group: "cache", flag: "--ctx-checkpoints", label: "Context checkpoints", min: 0, max: 1024, advanced: true, help: "Checkpoints per slot." },
  { key: "checkpoint_min_step", kind: "int", group: "cache", flag: "--checkpoint-min-step", label: "Checkpoint spacing", min: 0, max: 8388608, advanced: true, help: "Smallest gap between checkpoints, in tokens." },
  { key: "cache_idle_slots", kind: "toggle", group: "cache", flag: "--cache-idle-slots", offFlag: "--no-cache-idle-slots", label: "Save idle slots", advanced: true, help: "Save idle slots to the prompt cache when a new task starts." },
  { key: "slot_prompt_similarity", kind: "float", group: "cache", flag: "--slot-prompt-similarity", label: "Slot match similarity", min: 0, max: 1, step: 0.01, advanced: true, help: "How closely a prompt must match a slot to use it." },

  // Chat format
  { key: "jinja", kind: "toggle", group: "chat", flag: "--jinja", offFlag: "--no-jinja", label: "Jinja templates", advanced: true, help: "Use the Jinja template engine for chat." },
  { key: "chat_template", kind: "text", group: "chat", flag: "--chat-template", label: "Chat template", pattern: TEMPLATE_NAME, maxLength: 64, placeholder: "from the model", advanced: true, help: "Name of a built-in template. Empty uses the model's own." },
  { key: "chat_template_kwargs", kind: "text", group: "chat", flag: "--chat-template-kwargs", label: "Template arguments", pattern: /^\{[\s\S]*\}$/, maxLength: 4096, placeholder: '{"enable_thinking":false}', check: jsonObjectProblem, advanced: true, help: "A JSON object passed to the template." },
  { key: "reasoning", kind: "enum", group: "chat", flag: "--reasoning", label: "Reasoning", options: ["auto", "on", "off"], help: "Whether the model reasons before answering." },
  { key: "reasoning_format", kind: "enum", group: "chat", flag: "--reasoning-format", label: "Reasoning format", options: ["auto", "none", "deepseek", "deepseek-legacy"], advanced: true, help: "Where thoughts appear in the response." },
  { key: "reasoning_effort", kind: "enum", group: "chat", flag: "--reasoning-effort", label: "Reasoning effort", options: ["default", "minimal", "low", "medium", "high", "xhigh", "max"], advanced: true, help: "Effort level given to the template." },
  { key: "reasoning_budget", kind: "int", group: "chat", flag: "--reasoning-budget", label: "Reasoning budget", min: -1, max: 8388608, help: "Tokens for thinking. -1 unlimited, 0 ends at once." },
  { key: "skip_chat_parsing", kind: "toggle", group: "chat", flag: "--skip-chat-parsing", offFlag: "--no-skip-chat-parsing", label: "Raw output", advanced: true, help: "Put everything in content, including reasoning and tool calls." },
  { key: "prefill_assistant", kind: "toggle", group: "chat", flag: "--prefill-assistant", offFlag: "--no-prefill-assistant", label: "Prefill assistant", advanced: true, help: "Treat a trailing assistant message as a prefill." },

  // Vision. The projector file itself is DaemonLoadSpec.mmproj_filename.
  { key: "mmproj_offload", kind: "toggle", group: "vision", flag: "--mmproj-offload", offFlag: "--no-mmproj-offload", label: "Projector on GPU", help: "Offload the multimodal projector to the GPU." },
  { key: "image_min_tokens", kind: "int", group: "vision", flag: "--image-min-tokens", label: "Image minimum tokens", min: 0, max: 1048576, advanced: true, help: "For vision models with dynamic resolution." },
  { key: "image_max_tokens", kind: "int", group: "vision", flag: "--image-max-tokens", label: "Image maximum tokens", min: 0, max: 1048576, advanced: true, help: "For vision models with dynamic resolution." },

  // Speculative decoding. The draft model itself is DaemonLoadSpec.draft_model_id.
  { key: "spec_type", kind: "text", group: "speculative", flag: "--spec-type", label: "Method", pattern: /^(none|draft-simple|draft-eagle3|draft-mtp|draft-dflash|draft-dspark|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache)(,(none|draft-simple|draft-eagle3|draft-mtp|draft-dflash|draft-dspark|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache))*$/, maxLength: 128, placeholder: "none", help: "Types of speculative decoding, comma-separated." },
  { key: "n_gpu_layers_draft", kind: "text", group: "speculative", flag: "--n-gpu-layers-draft", label: "Draft GPU layers", pattern: /^(auto|all|\d{1,4})$/, maxLength: 8, placeholder: "auto", help: "Draft model layers in VRAM." },
  { key: "spec_draft_n_max", kind: "int", group: "speculative", flag: "--spec-draft-n-max", label: "Draft tokens (max)", min: 0, max: 1024, help: "Tokens to draft per step." },
  { key: "spec_draft_n_min", kind: "int", group: "speculative", flag: "--spec-draft-n-min", label: "Draft tokens (min)", min: 0, max: 1024, advanced: true, help: "Fewest draft tokens to use." },
  { key: "spec_draft_p_min", kind: "float", group: "speculative", flag: "--spec-draft-p-min", label: "Draft probability cutoff", min: 0, max: 1, step: 0.01, help: "Lowest greedy probability to keep drafting." },
  { key: "cache_type_k_draft", kind: "enum", group: "speculative", flag: "--cache-type-k-draft", label: "Draft cache type K", options: CACHE_TYPES, advanced: true, help: "KV cache K type for the draft model." },
  { key: "cache_type_v_draft", kind: "enum", group: "speculative", flag: "--cache-type-v-draft", label: "Draft cache type V", options: CACHE_TYPES, advanced: true, help: "KV cache V type for the draft model." },

  // Serving. The API identifier is DaemonLoadSpec.alias.
  { key: "sleep_idle_seconds", kind: "int", group: "serving", flag: "--sleep-idle-seconds", label: "Sleep after idle (s)", min: 1, max: 31536000, help: "Free the model's memory after this long with no requests. The next request reloads it." },
  { key: "timeout", kind: "int", group: "serving", flag: "--timeout", label: "Read/write timeout (s)", min: 1, max: 86400, advanced: true, help: "Between llama-server and the gateway." },
  { key: "threads_http", kind: "int", group: "serving", flag: "--threads-http", label: "HTTP threads", min: 1, max: 1024, advanced: true, help: "Threads handling HTTP requests." },
  { key: "embedding", kind: "toggle", group: "serving", flag: "--embedding", label: "Embeddings only", advanced: true, help: "Only for dedicated embedding models. Chat stops working." },
  { key: "reranking", kind: "toggle", group: "serving", flag: "--reranking", label: "Reranking", advanced: true, help: "Enable the rerank endpoint." },
  { key: "pooling", kind: "enum", group: "serving", flag: "--pooling", label: "Pooling", options: ["none", "mean", "cls", "last", "rank"], advanced: true, help: "Pooling for embeddings." },
  { key: "rope_freq_base", kind: "float", group: "serving", flag: "--rope-freq-base", label: "RoPE frequency base", min: 0, max: 10000000, advanced: true, help: "Overrides the model's value." },
  { key: "rope_freq_scale", kind: "float", group: "serving", flag: "--rope-freq-scale", label: "RoPE frequency scale", min: 0, max: 1000, advanced: true, help: "Expands context by a factor of 1/N." },
];

const LOAD_BY_KEY = new Map(LOAD_SETTINGS.map((d) => [d.key, d]));

export interface ValidationResult {
  values: SettingValues;
  errors: string[];
}

function checkNumber(label: string, raw: unknown, def: { kind: "int" | "float"; min: number; max: number }): { value?: number; error?: string } {
  const n = typeof raw === "string" && raw.trim() !== "" ? Number(raw) : raw;
  if (typeof n !== "number" || !Number.isFinite(n)) return { error: `${label} must be a number` };
  if (def.kind === "int" && !Number.isInteger(n)) return { error: `${label} must be a whole number` };
  if (n < def.min || n > def.max) return { error: `${label} must be between ${def.min} and ${def.max}` };
  return { value: n };
}

/** Unknown keys are errors, not silently dropped: the worker refuses them too. */
export function validateLoadSettings(raw: unknown): ValidationResult {
  const errors: string[] = [];
  const values: SettingValues = {};
  if (raw == null) return { values, errors };
  if (typeof raw !== "object" || Array.isArray(raw)) return { values, errors: ["settings must be an object"] };
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    const def = LOAD_BY_KEY.get(key);
    if (!def) {
      errors.push(`unknown setting: ${key.slice(0, 64)}`);
      continue;
    }
    if (value === null || value === undefined || value === "") continue;
    switch (def.kind) {
      case "int":
      case "float": {
        const { value: n, error } = checkNumber(def.label, value, def);
        if (error) errors.push(error);
        else values[key] = n as number;
        break;
      }
      case "enum":
        if (typeof value === "string" && def.options.includes(value)) values[key] = value;
        else errors.push(`${def.label} must be one of ${def.options.join(", ")}`);
        break;
      case "toggle":
        if (typeof value === "boolean") values[key] = value;
        else errors.push(`${def.label} must be on or off`);
        break;
      case "text": {
        if (typeof value !== "string" || value.length > def.maxLength || !def.pattern.test(value)) {
          errors.push(`${def.label} has an invalid value`);
          break;
        }
        const problem = def.check?.(value);
        if (problem) errors.push(`${def.label} ${problem}`);
        else values[key] = value;
        break;
      }
    }
  }
  return { values, errors };
}

/** Settings in `values` the binary's --help does not list. */
export function unsupportedSettings(values: SettingValues, supported: ReadonlySet<string>): string[] {
  return Object.keys(values).filter((key) => {
    if (supported.has(key)) return false;
    // Off with no off-form passes nothing, so it runs on any build.
    const def = LOAD_BY_KEY.get(key);
    return !(def?.kind === "toggle" && !def.offFlag && values[key] === false);
  });
}

/** Flags for a validated value set. Order follows the table, so the same settings
 * always give the same argv. */
export function buildLoadArgs(values: SettingValues): string[] {
  const args: string[] = [];
  for (const def of LOAD_SETTINGS) {
    const value = values[def.key];
    if (value === undefined) continue;
    if (def.kind === "toggle") {
      if (value === true) args.push(def.flag);
      else if (def.offFlag) args.push(def.offFlag);
    } else {
      args.push(def.flag, String(value));
    }
  }
  return args;
}

// --- Sampling: applied per request by the gateway, no restart ---

export type SamplingEndpoint = "chat" | "completions" | "responses" | "messages";

export interface SamplingDef extends BaseDef {
  kind: "int" | "float";
  min: number;
  max: number;
  step?: number;
  /** Disabled position, shown to explain what "off" means. */
  offAt?: number;
  /** Field on /props default_generation_settings.params holding the build's default. */
  defaultField: string;
  /** Request field per endpoint. An endpoint not listed never carries it. */
  fields: Partial<Record<SamplingEndpoint, string>>;
}

// Read back from /slots while in flight on b11609: /v1/messages honours temperature,
// top_k and top_p, and drops the rest. The other three honour everything.
const LLAMA_ONLY = (field: string) => ({ chat: field, completions: field, responses: field });
const ALL_NAMED = (field: string) => ({ chat: field, completions: field, responses: field, messages: field });

export const SAMPLING_SETTINGS: readonly SamplingDef[] = [
  { key: "temperature", kind: "float", min: 0, max: 2, step: 0.01, label: "Temperature", defaultField: "temperature", fields: ALL_NAMED("temperature"), help: "Higher is more random." },
  { key: "max_tokens", kind: "int", min: 1, max: 8388608, label: "Limit response length", defaultField: "n_predict", fields: { chat: "max_tokens", completions: "max_tokens", responses: "max_output_tokens" }, help: "Most tokens to generate. Empty means no limit." },
  { key: "seed", kind: "int", min: -1, max: 2147483647, label: "Seed", defaultField: "seed", fields: LLAMA_ONLY("seed"), help: "-1 picks a random seed." },
  { key: "top_k", kind: "int", min: 0, max: 1000, offAt: 0, label: "Top K", defaultField: "top_k", fields: ALL_NAMED("top_k"), help: "Sample from the K most likely tokens." },
  { key: "top_p", kind: "float", min: 0, max: 1, step: 0.01, offAt: 1, label: "Top P", defaultField: "top_p", fields: ALL_NAMED("top_p"), help: "Sample from the tokens making up this much probability." },
  { key: "min_p", kind: "float", min: 0, max: 1, step: 0.01, offAt: 0, label: "Min P", defaultField: "min_p", fields: LLAMA_ONLY("min_p"), help: "Drop tokens below this share of the top token's probability." },
  { key: "repeat_penalty", kind: "float", min: 0, max: 3, step: 0.01, offAt: 1, label: "Repeat penalty", defaultField: "repeat_penalty", fields: LLAMA_ONLY("repeat_penalty"), help: "Penalize recently seen tokens." },
  { key: "repeat_last_n", kind: "int", min: -1, max: 65536, offAt: 0, label: "Repeat window", defaultField: "repeat_last_n", fields: LLAMA_ONLY("repeat_last_n"), help: "Tokens the repeat penalty looks back over." },
  { key: "presence_penalty", kind: "float", min: -2, max: 2, step: 0.01, offAt: 0, label: "Presence penalty", defaultField: "presence_penalty", fields: LLAMA_ONLY("presence_penalty"), help: "Penalize tokens that appeared at all." },
  { key: "frequency_penalty", kind: "float", min: -2, max: 2, step: 0.01, offAt: 0, label: "Frequency penalty", defaultField: "frequency_penalty", fields: LLAMA_ONLY("frequency_penalty"), help: "Penalize tokens by how often they appeared." },
  { key: "typical_p", kind: "float", min: 0, max: 1, step: 0.01, offAt: 1, advanced: true, label: "Typical P", defaultField: "typical_p", fields: LLAMA_ONLY("typical_p"), help: "Locally typical sampling." },
  { key: "top_n_sigma", kind: "float", min: -1, max: 100, step: 0.1, offAt: -1, advanced: true, label: "Top N sigma", defaultField: "top_n_sigma", fields: LLAMA_ONLY("top_n_sigma"), help: "Keep tokens within N standard deviations." },
  { key: "dynatemp_range", kind: "float", min: 0, max: 5, step: 0.01, offAt: 0, advanced: true, label: "Dynamic temperature range", defaultField: "dynatemp_range", fields: LLAMA_ONLY("dynatemp_range"), help: "Vary temperature by this much." },
  { key: "dynatemp_exponent", kind: "float", min: 0, max: 10, step: 0.01, advanced: true, label: "Dynamic temperature exponent", defaultField: "dynatemp_exponent", fields: LLAMA_ONLY("dynatemp_exponent"), help: "Shape of the variation." },
  { key: "dry_multiplier", kind: "float", min: 0, max: 10, step: 0.01, offAt: 0, advanced: true, label: "DRY multiplier", defaultField: "dry_multiplier", fields: LLAMA_ONLY("dry_multiplier"), help: "Strength of the DRY repetition penalty." },
  { key: "dry_base", kind: "float", min: 1, max: 10, step: 0.01, advanced: true, label: "DRY base", defaultField: "dry_base", fields: LLAMA_ONLY("dry_base"), help: "Base of the DRY penalty." },
  { key: "dry_allowed_length", kind: "int", min: 0, max: 1024, advanced: true, label: "DRY allowed length", defaultField: "dry_allowed_length", fields: LLAMA_ONLY("dry_allowed_length"), help: "Repeats up to this long go unpenalized." },
  { key: "dry_penalty_last_n", kind: "int", min: -1, max: 65536, advanced: true, label: "DRY window", defaultField: "dry_penalty_last_n", fields: LLAMA_ONLY("dry_penalty_last_n"), help: "Tokens DRY looks back over." },
  { key: "xtc_probability", kind: "float", min: 0, max: 1, step: 0.01, offAt: 0, advanced: true, label: "XTC probability", defaultField: "xtc_probability", fields: LLAMA_ONLY("xtc_probability"), help: "Chance of cutting the most likely tokens." },
  { key: "xtc_threshold", kind: "float", min: 0, max: 1, step: 0.01, advanced: true, label: "XTC threshold", defaultField: "xtc_threshold", fields: LLAMA_ONLY("xtc_threshold"), help: "Tokens above this probability can be cut." },
  { key: "mirostat", kind: "int", min: 0, max: 2, offAt: 0, advanced: true, label: "Mirostat", defaultField: "mirostat", fields: LLAMA_ONLY("mirostat"), help: "0 off, 1 Mirostat, 2 Mirostat 2.0. Top K, Top P and Typical P are ignored while on." },
  { key: "mirostat_tau", kind: "float", min: 0, max: 100, step: 0.01, advanced: true, label: "Mirostat target entropy", defaultField: "mirostat_tau", fields: LLAMA_ONLY("mirostat_tau"), help: "Tau." },
  { key: "mirostat_eta", kind: "float", min: 0, max: 10, step: 0.01, advanced: true, label: "Mirostat learning rate", defaultField: "mirostat_eta", fields: LLAMA_ONLY("mirostat_eta"), help: "Eta." },
];

const SAMPLING_BY_KEY = new Map(SAMPLING_SETTINGS.map((d) => [d.key, d]));

export interface SamplingSettings {
  values: Record<string, number>;
  system_prompt?: string;
  stop?: string[];
  /** Apply over what the client sent, instead of only filling gaps. */
  override_client?: boolean;
}

export const EMPTY_SAMPLING: SamplingSettings = { values: {} };

export const MAX_SYSTEM_PROMPT_CHARS = 65536;
const MAX_STOP_STRINGS = 16;
const MAX_STOP_CHARS = 256;

export function validateSampling(raw: unknown): { sampling: SamplingSettings; errors: string[] } {
  const errors: string[] = [];
  const sampling: SamplingSettings = { values: {} };
  if (raw == null) return { sampling, errors };
  if (typeof raw !== "object" || Array.isArray(raw)) return { sampling, errors: ["sampling must be an object"] };
  const r = raw as Record<string, unknown>;
  if (r.values != null && (typeof r.values !== "object" || Array.isArray(r.values))) errors.push("sampling values must be an object");
  for (const [key, value] of Object.entries((r.values ?? {}) as Record<string, unknown>)) {
    const def = SAMPLING_BY_KEY.get(key);
    if (!def) {
      errors.push(`unknown sampling setting: ${key.slice(0, 64)}`);
      continue;
    }
    if (value === null || value === undefined || value === "") continue;
    const { value: n, error } = checkNumber(def.label, value, def);
    if (error) errors.push(error);
    else sampling.values[key] = n as number;
  }
  if (r.system_prompt != null && r.system_prompt !== "") {
    if (typeof r.system_prompt !== "string" || r.system_prompt.length > MAX_SYSTEM_PROMPT_CHARS) errors.push("system prompt is too long");
    else sampling.system_prompt = r.system_prompt;
  }
  if (r.stop != null) {
    if (!Array.isArray(r.stop) || r.stop.length > MAX_STOP_STRINGS || r.stop.some((s) => typeof s !== "string" || s === "" || s.length > MAX_STOP_CHARS)) {
      errors.push(`stop strings must be at most ${MAX_STOP_STRINGS} non-empty strings of ${MAX_STOP_CHARS} characters`);
    } else if (r.stop.length > 0) sampling.stop = r.stop as string[];
  }
  if (r.override_client != null) {
    if (typeof r.override_client !== "boolean") errors.push("override_client must be true or false");
    else if (r.override_client) sampling.override_client = true;
  }
  return { sampling, errors };
}

/** A request body with the page's sampling settings applied. Pure: the input is
 * left untouched, and with nothing to apply it is returned as it came. */
export function applySampling(body: Record<string, unknown>, endpoint: SamplingEndpoint, s: SamplingSettings | null): Record<string, unknown> {
  if (!s) return body;
  const out: Record<string, unknown> = { ...body };
  const force = s.override_client === true;
  for (const [key, value] of Object.entries(s.values)) {
    const field = SAMPLING_BY_KEY.get(key)?.fields[endpoint];
    if (field && (force || out[field] === undefined)) out[field] = value;
  }
  if (s.stop) {
    const field = endpoint === "messages" ? "stop_sequences" : endpoint === "responses" ? null : "stop";
    if (field && (force || out[field] === undefined)) out[field] = s.stop;
  }
  if (s.system_prompt) {
    if (endpoint === "chat" && Array.isArray(out.messages)) {
      const hasSystem = out.messages.some((m) => {
        const role = (m as { role?: unknown } | null)?.role;
        return role === "system" || role === "developer";
      });
      if (!hasSystem) out.messages = [{ role: "system", content: s.system_prompt }, ...out.messages];
      else if (force) {
        out.messages = [{ role: "system", content: s.system_prompt }, ...out.messages.filter((m) => {
          const role = (m as { role?: unknown } | null)?.role;
          return role !== "system" && role !== "developer";
        })];
      }
    } else if (endpoint === "messages" && (force || out.system === undefined)) out.system = s.system_prompt;
    else if (endpoint === "responses" && (force || out.instructions === undefined)) out.instructions = s.system_prompt;
  }
  return out;
}

// --- Identity and bounds of the load itself ---

export const ALIAS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/;

/** A default API identifier from a model file name. */
export function defaultAlias(filename: string): string {
  const base = filename.split(/[\\/]/).pop() ?? filename;
  const stem = base.replace(/\.gguf$/i, "").replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "");
  return (stem || "model").slice(0, 128);
}

export const DAEMON_DEFAULT_PORT = 8642;
export const LOOPBACK_HOST = "127.0.0.1";
export const ALL_INTERFACES_HOST = "0.0.0.0";

export function isLoopbackHost(host: string): boolean {
  return host === LOOPBACK_HOST || host === "::1" || host === "localhost";
}

const IPV4 = /^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$/;

export interface GatewaySettings {
  host: string;
  port: number;
  cors_origins: string[];
  /** Load the last model when the worker starts. */
  autostart: boolean;
}

export const DEFAULT_GATEWAY: GatewaySettings = { host: LOOPBACK_HOST, port: DAEMON_DEFAULT_PORT, cors_origins: [], autostart: false };

export function validateGateway(raw: unknown, keyCount: number): { gateway: GatewaySettings; errors: string[] } {
  const errors: string[] = [];
  const gateway: GatewaySettings = { ...DEFAULT_GATEWAY, cors_origins: [] };
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return { gateway, errors: ["gateway must be an object"] };
  const r = raw as Record<string, unknown>;
  if (typeof r.host === "string" && IPV4.test(r.host)) gateway.host = r.host;
  else errors.push("host must be an IPv4 address");
  if (Number.isInteger(r.port) && (r.port as number) >= 1024 && (r.port as number) <= 65535) gateway.port = r.port as number;
  else errors.push("port must be between 1024 and 65535");
  if (r.autostart != null) {
    if (typeof r.autostart === "boolean") gateway.autostart = r.autostart;
    else errors.push("autostart must be true or false");
  }
  if (r.cors_origins != null) {
    const origins = r.cors_origins;
    const valid =
      Array.isArray(origins) &&
      origins.length <= 32 &&
      origins.every((o) => typeof o === "string" && /^https?:\/\/[A-Za-z0-9.-]+(:\d{1,5})?$/.test(o));
    if (valid) gateway.cors_origins = origins as string[];
    else errors.push("allowed origins must look like https://host:port, at most 32");
  }
  if (!isLoopbackHost(gateway.host) && keyCount === 0) {
    errors.push("create an API key before opening the endpoint beyond this machine");
  }
  return { gateway, errors };
}
