import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import type { Model, Worker } from "../../types";
import { isMtpDraftModel } from "../../types";
import { LtIcon } from "../ltIcons";
import { modelVerdict } from "./machine";

// The v2 Models screen's matrix (docs/plans/app-v2.dc.html): every model on
// any of your machines against every machine, with the quick fit verdict at
// 8K context. An estimate -- New test's fit map replaces it with llama.cpp's
// own sizing for one pairing.

const FIT_TABLE_CTX = 8192;

function label(m: Model): string {
  return (m.hf_file ?? m.filename ?? m.id).replace(/\.gguf$/i, "").split(/[\\/]/).pop() ?? m.id;
}

export function ModelFitTable() {
  const [models, setModels] = useState<Model[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [locations, setLocations] = useState<Record<string, string[]>>({});
  const [open, setOpen] = useState(true);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([api.listModels(), api.listWorkers(), api.getModelLocations().catch(() => ({ locations: {} }))])
      .then(([m, w, l]) => {
        if (cancelled) return;
        setModels(m);
        setWorkers(w);
        setLocations(l.locations);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = useMemo(() => models.filter((m) => !isMtpDraftModel(m) && (locations[m.id]?.length ?? 0) > 0), [models, locations]);
  if (rows.length === 0 || workers.length === 0) return null;

  return (
    <section aria-labelledby="h-fit-table" className="mb-6 border border-border bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <h2 id="h-fit-table" className="m-0 font-display text-lg font-semibold">
          Fit per machine
        </h2>
        <span className="font-mono text-xs text-muted">quick estimate at 8K context</span>
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="text-sm text-accent hover:text-accent-hover">
          {open ? "Hide" : "Show"}
        </button>
      </div>
      {open && (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-border-strong font-mono text-xs text-muted">
                <th scope="col" className="px-4 py-2.5 font-normal">Model</th>
                {workers.map((w) => (
                  <th key={w.id} scope="col" className="px-2 py-2.5 font-normal">
                    {w.displayName}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.id} className="border-b border-border last:border-b-0">
                  <th scope="row" className="px-4 py-2.5 text-left font-medium">
                    <span className="block text-sm">{label(m)}</span>
                    <span className="block font-mono text-xs font-normal text-muted">
                      {(m.size_bytes / 1073741824).toFixed(1)} GB{typeof m.metadata.n_layer === "number" ? ` · ${m.metadata.n_layer} layers` : ""}
                    </span>
                  </th>
                  {workers.map((w) => {
                    const here = locations[m.id]?.includes(w.id);
                    const trained = typeof m.metadata.trained_ctx === "number" ? m.metadata.trained_ctx : Infinity;
                    const v = modelVerdict(m, w, Math.min(FIT_TABLE_CTX, trained));
                    return (
                      <td key={w.id} className="px-2 py-2.5 text-sm">
                        {v ? (
                          <span className={`inline-flex items-center gap-1.5 ${v.tone}`}>
                            <LtIcon name={v.icon} size={16} />
                            {v.text}
                          </span>
                        ) : (
                          <span className="text-muted">—</span>
                        )}
                        {!here && <span className="block font-mono text-xs text-muted">not on this machine</span>}
                        {here && w.status !== "offline" && (
                          <Link to={`/benchmark?worker=${encodeURIComponent(w.id)}&model=${encodeURIComponent(m.id)}`} className="block font-mono text-xs text-accent">
                            map it
                          </Link>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
