import { useEffect, useRef, useState } from "react";
import { IconChevronDown, IconRefreshCw } from "./icons";
import { Tooltip } from "./Tooltip";
import { formatParamsB, formatRelativeTime, hfRepoUrl, modelAuthor, modelParamsB } from "../utils";
import { buildModelGroups } from "../modelGrouping";
import type { Model } from "../types";

// True when HF reports this repo as touched after we downloaded our copy --
// a best-effort "you might be behind" signal, not proof the actual .gguf
// bytes changed (a repo's lastModified also moves on README/tag edits).
function hasPossibleUpdate(model: Model, hfLastModified: string | null | undefined): boolean {
  if (!hfLastModified) return false;
  const hfMs = new Date(hfLastModified).getTime();
  return !Number.isNaN(hfMs) && hfMs > model.created_at;
}

function ModelMeta({ model, hfLastModified }: { model: Model; hfLastModified: string | null | undefined }) {
  const author = modelAuthor(model);
  const paramsB = modelParamsB(model);
  const updatedLabel = formatRelativeTime(hfLastModified);
  const updateAvailable = hasPossibleUpdate(model, hfLastModified);
  return (
    <div className="flex items-center gap-1.5 text-xs text-muted">
      <span>{author}</span>
      {paramsB !== null && (
        <>
          <span aria-hidden="true">·</span>
          <span>{formatParamsB(paramsB)}</span>
        </>
      )}
      {updatedLabel && (
        <>
          <span aria-hidden="true">·</span>
          <span>Updated {updatedLabel}</span>
        </>
      )}
      {updateAvailable && (
        <Tooltip text="Hugging Face reports this repo was touched after you downloaded this file -- there may be a newer version. Check the repo on HF before assuming this is stale.">
          <span className="inline-flex items-center gap-1 rounded-full bg-warning-bg px-1.5 py-0.5 font-medium text-warning">
            <IconRefreshCw width={10} height={10} />
            update?
          </span>
        </Tooltip>
      )}
    </div>
  );
}

export function ModelPicker({
  models,
  value,
  mtpValue,
  onSelect,
  hfUpdates,
  className,
}: {
  models: Model[];
  value: string;
  // Currently-picked MTP companion (if any) -- only used to highlight which
  // draft row (if any) matches the current selection.
  mtpValue?: string;
  // A row for an MTP/draft companion picks its paired base model AND that
  // companion in one action -- draftId is set then, otherwise undefined.
  onSelect: (modelId: string, draftId?: string) => void;
  hfUpdates: Record<string, string | null>;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const groups = buildModelGroups(models);
  const selected = models.find((m) => m.id === value);

  function pick(modelId: string, draftId?: string) {
    onSelect(modelId, draftId);
    setOpen(false);
  }

  return (
    <div ref={rootRef} className={`relative flex flex-col gap-1.5 text-sm ${className ?? ""}`}>
      <span className="text-muted">Model</span>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-lg border border-border bg-surface px-3 py-1.5 text-left text-fg outline-none focus:border-accent"
      >
        {selected ? (
          <span className="flex min-w-0 flex-col">
            <span className="truncate">{selected.filename}</span>
            <ModelMeta model={selected} hfLastModified={hfUpdates[selected.id]} />
          </span>
        ) : (
          <span className="text-muted">select…</span>
        )}
        <IconChevronDown width={16} height={16} className={`shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="absolute top-full z-40 mt-1 max-h-96 w-full min-w-[26rem] overflow-y-auto rounded-lg border border-border bg-surface-raised shadow-lg">
          {groups.length === 0 && <p className="px-3 py-2.5 text-sm text-muted">No models registered yet.</p>}
          {groups.map((group) => (
            <div key={group.key} className="border-b border-border last:border-b-0">
              <div className="px-3 pt-2 pb-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted">
                {group.author} · {group.family} · {group.label}
              </div>
              {group.quants.map(({ base }) => (
                <button
                  key={base.id}
                  type="button"
                  onClick={() => pick(base.id)}
                  className={`flex w-full flex-col gap-0.5 px-3 py-2 text-left hover:bg-white/5 ${
                    base.id === value ? "bg-accent/10" : ""
                  }`}
                >
                  <span className="truncate text-fg">{base.filename}</span>
                  <ModelMeta model={base} hfLastModified={hfUpdates[base.id]} />
                </button>
              ))}
              {group.quants.map(({ base, drafts }) =>
                drafts.map((draft) => (
                  <button
                    key={draft.id}
                    type="button"
                    onClick={() => pick(base.id, draft.id)}
                    title={`Select ${base.filename} with ${draft.filename} as its MTP draft model`}
                    className={`flex w-full flex-col gap-0.5 border-l-2 border-border py-2 pl-8 pr-3 text-left hover:bg-white/5 ${
                      base.id === value && draft.id === mtpValue ? "bg-accent/5" : ""
                    }`}
                  >
                    <span className="truncate text-fg">
                      {!/^mtp\//i.test(draft.filename) && <span className="text-muted">MTP/</span>}
                      {draft.filename}
                    </span>
                    <ModelMeta model={draft} hfLastModified={hfUpdates[draft.id]} />
                  </button>
                ))
              )}
            </div>
          ))}
        </div>
      )}
      {selected?.hf_repo && (
        <a
          href={hfRepoUrl(selected.hf_repo)}
          target="_blank"
          rel="noreferrer"
          className="self-start text-xs text-accent hover:underline"
        >
          view on HF ↗
        </a>
      )}
    </div>
  );
}
