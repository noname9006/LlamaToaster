import type { Model } from "./types";

// One-time carry-over for browser storage keys written under this app's old
// "llama-bench:" name (pre-rename to LlamaToaster) so existing users don't
// silently lose AI chat history, remembered sweeps, panel width, etc. Safe to
// call every startup -- once the old-prefixed keys are gone, it's a no-op.
export function migrateLegacyStorageKeys(): void {
  const OLD_PREFIX = "llama-bench:";
  const NEW_PREFIX = "llamatoaster:";
  for (const storage of [localStorage, sessionStorage]) {
    const oldKeys: string[] = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key?.startsWith(OLD_PREFIX)) oldKeys.push(key);
    }
    for (const oldKey of oldKeys) {
      const newKey = NEW_PREFIX + oldKey.slice(OLD_PREFIX.length);
      if (storage.getItem(newKey) === null) {
        storage.setItem(newKey, storage.getItem(oldKey)!);
      }
      storage.removeItem(oldKey);
    }
  }
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatDate(ms: number): string {
  return new Date(ms).toLocaleString();
}

// "Updated 3 days ago" / "Updated 2 months ago" style relative label, same
// rough granularity as Hugging Face's own repo cards. Returns null for an
// unparseable/missing timestamp so callers can just omit the label.
export function formatRelativeTime(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return null;
  const diffDays = Math.floor((Date.now() - then) / (1000 * 60 * 60 * 24));
  if (diffDays <= 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 30) return `${diffDays} days ago`;
  const diffMonths = Math.floor(diffDays / 30);
  if (diffMonths < 12) return `${diffMonths} month${diffMonths > 1 ? "s" : ""} ago`;
  const diffYears = Math.floor(diffMonths / 12);
  return `${diffYears} year${diffYears > 1 ? "s" : ""} ago`;
}

export function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function shortId(id: string, len = 12): string {
  return id.slice(0, len);
}

// flash_attn has been stored as "on"/"off" (the toggle's real options),
// "true" (a past seeding bug -- see NewRun.tsx's sanitizeSweep), and
// possibly a bare boolean echoed back by llama-bench's own JSON output --
// normalize whatever a given item/result happens to hold so every display
// (and Compare's cross-run combo matching) treats them as the same value.
export function formatFlashAttn(value: string): string {
  if (value === "true" || value === "1") return "on";
  if (value === "false" || value === "0") return "off";
  return value;
}

export function hfRepoUrl(repo: string): string {
  return `https://huggingface.co/${repo}`;
}

export function hfFileUrl(repo: string, file: string): string {
  return `https://huggingface.co/${repo}/blob/main/${file}`;
}

// "Author" here means the HF namespace a model was published under (the
// part before the "/" in e.g. "bartowski/Llama-3.1-8B-GGUF") -- locally
// registered models have no such namespace.
export function modelAuthor(m: Model): string {
  return m.hf_repo ? m.hf_repo.split("/")[0] : "local";
}

// Best-effort family detection: prefer metadata.arch if the model was
// registered with one, otherwise pattern-match the repo id / filename
// against common architecture family names. Not exhaustive -- new families
// ship constantly -- so an unmatched model just falls back to "Other"
// rather than blocking filtering/sorting on it.
const FAMILY_PATTERNS: [RegExp, string][] = [
  [/llama-?4/i, "Llama 4"],
  [/llama-?3/i, "Llama 3"],
  [/llama-?2/i, "Llama 2"],
  [/llama/i, "Llama"],
  [/gemma/i, "Gemma"],
  [/qwen/i, "Qwen"],
  [/mixtral/i, "Mixtral"],
  [/mistral/i, "Mistral"],
  [/phi-?\d/i, "Phi"],
  [/deepseek/i, "DeepSeek"],
  [/command-?r/i, "Command R"],
  [/granite/i, "Granite"],
  [/falcon/i, "Falcon"],
  [/starcoder/i, "StarCoder"],
  [/codellama/i, "CodeLlama"],
  [/vicuna/i, "Vicuna"],
  [/wizardlm/i, "WizardLM"],
  [/tinyllama/i, "TinyLlama"],
  [/stablelm/i, "StableLM"],
  [/internlm/i, "InternLM"],
  [/baichuan/i, "Baichuan"],
  [/chatglm|glm-?\d/i, "GLM"],
  [/dbrx/i, "DBRX"],
  [/olmo/i, "OLMo"],
  [/smollm/i, "SmolLM"],
  [/exaone/i, "EXAONE"],
  [/minicpm/i, "MiniCPM"],
  [/nemotron/i, "Nemotron"],
  [/mpt/i, "MPT"],
  [/bloom/i, "BLOOM"],
  [/jamba/i, "Jamba"],
  [/\byi-?\d/i, "Yi"],
  [/gpt-?oss/i, "GPT-OSS"],
  [/gpt-?2/i, "GPT-2"],
];

export function modelFamily(m: Model): string {
  if (typeof m.metadata.arch === "string" && m.metadata.arch.trim()) return m.metadata.arch;
  const haystack = `${m.hf_repo ?? ""} ${m.filename}`;
  for (const [pattern, name] of FAMILY_PATTERNS) {
    if (pattern.test(haystack)) return name;
  }
  return "Other";
}

// Best-effort parameter count in billions, parsed from the repo id /
// filename (the near-universal "...-8B-..." / "...-2b-it..." naming
// convention). An optional single letter before the digits (e.g. Gemma's
// "E2B"/"E4B" "effective params" naming) is tolerated but not captured.
// Returns null when nothing matches rather than guessing.
//
// Deliberately unreliable for families that name themselves after an
// "effective"/marketing parameter count rather than the real total (e.g.
// Gemma 3n's "E2B" is ~2B *effective* compute but ~5B real parameters) --
// this is only ever a fallback for models with no stored metadata.param_count
// (see modelParamsB below), which comes from Hugging Face's own GGUF tensor
// data and is authoritative when present.
const PARAM_COUNT_PATTERN = /(?:^|[-_ .])[A-Za-z]?(\d+(?:\.\d+)?)[bB](?:[-_ .]|$)/;

// Split out from modelParamsB so Hugging Face search results (which only
// have a repo id, not yet a filename -- files aren't fetched until a repo
// row is expanded) can reuse the same parsing without a Model object.
export function paramsBFromText(text: string): number | null {
  const match = text.match(PARAM_COUNT_PATTERN);
  return match ? Number(match[1]) : null;
}

export function modelParamsB(m: Model): number | null {
  if (typeof m.metadata.param_count === "number") {
    return Math.round((m.metadata.param_count / 1e9) * 10) / 10;
  }
  return paramsBFromText(`${m.hf_repo ?? ""} ${m.filename}`);
}

export function formatParamsB(b: number | null): string {
  return b === null ? "—" : `${b}B`;
}
