// Which file under model_dir a registered Model actually is. Pure, so it can be
// tested: the caller supplies the cache entries and what exists on disk.
//
// Identity outranks location. A file that merely sits at the model's name may be
// a different file (another repo's same-named quant, or one replaced in place),
// and results run against it would be filed under this model's hash (audit
// 2026-10-11 N11).

export interface ModelPathModel {
  id: string;
  source?: string;
  filename?: string | null;
  hf_repo?: string | null;
  hf_file?: string | null;
}

export interface ModelPathEntry {
  path: string;
  sha256?: string | null;
  hf_model_id?: string | null;
  state?: string;
}

export type ModelPathChoice =
  // Found. `proven` is false only for a file matched by name with nothing known
  // about what it holds, which a caller with another source may decline.
  | { kind: "found"; path: string; proven: boolean }
  // A file sits at the model's name, and the cache knows it is a different one.
  | { kind: "different"; path: string; actual: string }
  | { kind: "none" };

function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/**
 * `name` is the model's own relative path. Match order: the file's hash, then its
 * Hugging Face provenance, then the path itself, and last (unless `exactOnly`)
 * a same-named file elsewhere, the "flat instead of in a subfolder" hand-dropped
 * case. A basename collision resolves to the first match.
 */
export function chooseModelPath(
  model: ModelPathModel,
  name: string,
  entries: ModelPathEntry[],
  exists: (relativePath: string) => boolean,
  opts: { exactOnly?: boolean } = {}
): ModelPathChoice {
  const live = entries.filter((e) => e.state !== "missing");
  const shaId = /^[0-9a-f]{64}$/i.test(model.id) ? model.id.toLowerCase() : null;
  const wantHfId = model.hf_repo && model.hf_file ? `${model.hf_repo}/${model.hf_file}` : null;
  const hashIs = (e: ModelPathEntry, sha: string) => e.sha256?.toLowerCase() === sha;

  if (shaId) {
    const byHash = live.find((e) => hashIs(e, shaId));
    if (byHash) return { kind: "found", path: byHash.path, proven: true };
  }
  if (wantHfId) {
    // Not one whose hash contradicts this model's: a repo can change a file
    // under the same name.
    const byHf = live.find((e) => e.hf_model_id === wantHfId && !(shaId && e.sha256 && !hashIs(e, shaId)));
    if (byHf) return { kind: "found", path: byHf.path, proven: true };
  }

  if (exists(name)) {
    const here = live.find((e) => e.path === name);
    if (shaId && here?.sha256 && !hashIs(here, shaId)) {
      return { kind: "different", path: name, actual: here.sha256.toLowerCase() };
    }
    if (wantHfId && here?.hf_model_id && here.hf_model_id !== wantHfId) {
      return opts.exactOnly
        ? { kind: "different", path: name, actual: here.hf_model_id }
        : { kind: "found", path: name, proven: false };
    }
    // Nothing else could answer for a model with no Hugging Face source, so the
    // name is as much proof as it will ever have.
    return { kind: "found", path: name, proven: !wantHfId };
  }

  if (opts.exactOnly) return { kind: "none" };
  const wantBase = baseName(name);
  const byBase = live.find((e) => baseName(e.path) === wantBase && !(shaId && e.sha256 && !hashIs(e, shaId)));
  return byBase ? { kind: "found", path: byBase.path, proven: false } : { kind: "none" };
}
