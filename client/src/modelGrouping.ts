// Shared model-grouping logic for the Models page and the New Run model
// picker, so "which files count as the same model" and "which draft file is
// this model's MTP companion" are answered identically in both places.
import { modelAuthor, modelFamily } from "./utils";
import { isMtpDraftModel } from "./types";
import type { Model } from "./types";

// Mirrors server/src/hf.ts's QUANT_PATTERN -- used here only to strip a
// quant token off a *local* model's filename (an HF-backed model's grouping
// key comes from its repo name instead, which never carries a quant suffix).
// Group 1 is unsloth's "Dynamic" quant prefix (UD-Q4_K_XL, UD-IQ2_M, ...) --
// optional, and kept separate from group 2 (the actual quant code) so
// modelBaseLabel below can still strip the whole thing as one unit via the
// full match, while extractQuant reassembles "UD-" + code for display.
const QUANT_PATTERN = /(?:^|[-_.])(UD[-_])?((?:IQ|Q)[0-9]+(?:_[A-Z0-9]+)*|F16|F32|BF16)(?=[-_.]|$)/i;

// Best-effort "same model, different file" identity: for an HF-backed model
// this is the repo name (quant variants of one model overwhelmingly live as
// separate files in one shared repo, e.g. bartowski's *-GGUF repos), with a
// trailing "-GGUF" convention-suffix stripped; for a manually-registered
// local model (no repo to key off) it's the filename with its extension and
// any quant token removed.
// Extracts the quant token a filename carries (e.g. "Q4_K_M", or "UD-Q4_K_XL"
// for an unsloth Dynamic quant), for display as its own badge -- unlike
// modelBaseLabel below, which strips it. Neither the worker's GGUF parsing
// nor the download-callback route ever populates metadata.quant, so this
// filename regex (mirrors server/src/hf.ts's parseQuant, used for the HF
// search file list) is the only source there is for a registered model.
export function extractQuant(filename: string): string | null {
  const m = filename.match(QUANT_PATTERN);
  if (!m) return null;
  return (m[1] ? "UD-" : "") + m[2].toUpperCase();
}

export function modelBaseLabel(m: Model): string {
  if (m.hf_repo) {
    const repoName = m.hf_repo.split("/")[1] ?? m.hf_repo;
    return repoName.replace(/[-_]gguf$/i, "").trim();
  }
  return m.filename
    .replace(/\.(gguf|bin)$/i, "")
    .replace(QUANT_PATTERN, "")
    .replace(/[-_.]+$/, "")
    .trim();
}

function modelGroupKey(m: Model): string {
  return `${modelAuthor(m)}::${modelFamily(m)}::${modelBaseLabel(m)}`.toLowerCase();
}

// Rough "bits per weight" derived from a quant code's leading digit, purely
// to order quant siblings within a group the same way Hugging Face's own
// quant tables do -- smallest/most-compressed first, not alphabetically or
// by an incidental size quirk. Not an exact bits-per-weight computation
// (K-quants/IQ-quants have several sub-variants at the same nominal bit
// count, e.g. Q4_0 vs Q4_K_S vs Q4_K_M) -- buildModelGroups' sort below
// breaks ties between those with the file's actual size_bytes, which is
// still a faithful (and simpler than hardcoding every sub-variant) proxy
// for "which of these same-bit-count quants is more compressed."
function quantBitRank(quant: string | null): number {
  if (!quant) return Infinity; // no quant token at all -- sort last, nothing to rank
  const code = quant.replace(/^UD-/, "");
  if (/^F32$/i.test(code)) return 32;
  if (/^(F16|BF16)$/i.test(code)) return 16;
  const m = code.match(/^I?Q(\d+)/i);
  return m ? Number(m[1]) : Infinity;
}

export interface ModelQuantEntry {
  base: Model;
  // Draft/MTP companion files sharing the base model's hf_repo -- there's no
  // explicit pairing field anywhere, so shared hf_repo is the only real
  // signal available (mirrors how these files actually ship, e.g. unsloth's
  // "<repo>/MTP/mtp-gemma-4-12B-it.gguf" alongside "<repo>/gemma-4-12B-it-*.gguf").
  drafts: Model[];
}

export interface ModelGroup {
  key: string;
  label: string;
  author: string;
  family: string;
  quants: ModelQuantEntry[];
}

export function buildModelGroups(models: Model[]): ModelGroup[] {
  const bases = models.filter((m) => !isMtpDraftModel(m));
  const drafts = models.filter((m) => isMtpDraftModel(m));

  const groups = new Map<string, ModelGroup>();
  for (const base of bases) {
    const key = modelGroupKey(base);
    let group = groups.get(key);
    if (!group) {
      group = { key, label: modelBaseLabel(base), author: modelAuthor(base), family: modelFamily(base), quants: [] };
      groups.set(key, group);
    }
    group.quants.push({
      base,
      drafts: base.hf_repo ? drafts.filter((d) => d.hf_repo === base.hf_repo) : [],
    });
  }
  for (const group of groups.values()) {
    group.quants.sort((a, b) => {
      const ra = quantBitRank(extractQuant(a.base.hf_file ?? a.base.filename));
      const rb = quantBitRank(extractQuant(b.base.hf_file ?? b.base.filename));
      return ra !== rb ? ra - rb : a.base.size_bytes - b.base.size_bytes;
    });
  }
  return Array.from(groups.values());
}
