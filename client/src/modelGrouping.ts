// Shared model-grouping logic for the Models page and the New Run model
// picker, so "which files count as the same model" and "which draft file is
// this model's MTP companion" are answered identically in both places.
import { modelAuthor, modelFamily } from "./utils";
import { isMtpDraftModel } from "./types";
import type { Model } from "./types";

// Mirrors server/src/hf.ts's QUANT_PATTERN -- used here only to strip a
// quant token off a *local* model's filename (an HF-backed model's grouping
// key comes from its repo name instead, which never carries a quant suffix).
const QUANT_PATTERN = /(?:^|[-_.])((?:IQ|Q)[0-9]+(?:_[A-Z0-9]+)*|F16|F32|BF16)(?=[-_.]|$)/i;

// Best-effort "same model, different file" identity: for an HF-backed model
// this is the repo name (quant variants of one model overwhelmingly live as
// separate files in one shared repo, e.g. bartowski's *-GGUF repos), with a
// trailing "-GGUF" convention-suffix stripped; for a manually-registered
// local model (no repo to key off) it's the filename with its extension and
// any quant token removed.
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
    group.quants.sort((a, b) => a.base.size_bytes - b.base.size_bytes);
  }
  return Array.from(groups.values());
}
