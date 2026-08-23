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

// "model (4 GB)" / "model (1 GB shared)" for a HardwareInfo.gpu entry.
// vram_dynamic (see shared/types.ts's HardwareInfo) means shared/unified
// memory, typically an iGPU -- labeled "shared" rather than presented with
// the same confidence as a discrete GPU's fixed VRAM pool, since the
// reported number there is an estimate/allocation, not a dedicated
// capacity. Falls back to just the model name when vram_mb is
// null/undefined (driver didn't report it, or this is from a worker running
// old code that predates the field).
export function formatGpuLabel(g: { model: string; vendor: string; vram_mb?: number | null; vram_dynamic?: boolean }): string {
  const name = g.model || g.vendor || "unknown";
  if (g.vram_mb == null) return name;
  const vram = formatBytes(g.vram_mb * 1024 * 1024);
  return `${name} (${vram}${g.vram_dynamic ? " shared" : ""})`;
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

// Ultra-short "3d" / "2w" / "1mo" style relative label for a dense table
// column, distinct from formatRelativeTime's prose "3 days ago" (which needs
// the room a card/list description line has, not a fixed-width cell) --
// takes an epoch-ms number since that's what Model.created_at actually is,
// not the ISO string formatRelativeTime expects from HF search results.
export function formatShortRelativeTime(ms: number): string {
  const diffSeconds = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (diffSeconds < 60) return "now";
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d`;
  const diffWeeks = Math.floor(diffDays / 7);
  if (diffDays < 30) return `${diffWeeks}w`;
  const diffMonths = Math.floor(diffDays / 30);
  if (diffMonths < 12) return `${diffMonths}mo`;
  const diffYears = Math.floor(diffMonths / 12);
  return `${diffYears}y`;
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

// navigator.clipboard is only available in a "secure context" (HTTPS, or
// localhost) -- this app is normally reached over plain HTTP on a Tailscale
// IP (see README's "Tailscale-only orchestrator" / BIND_HOST notes), where
// it's simply undefined, so a bare `navigator.clipboard?.writeText(...)`
// silently no-ops instead of copying anything. Falls back to the older
// execCommand("copy") technique (via a temporary offscreen textarea), which
// still works without a secure context.
export async function copyToClipboard(text: string): Promise<boolean> {
  if (window.isSecureContext && navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to the legacy fallback below
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.top = "-1000px";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(textarea);
  return ok;
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
