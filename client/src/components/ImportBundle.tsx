// BENCHMARKING_PLAN_V8.md N7 -- the import half of reproducible exchange.
//
// Validation is PER ROW: tampering one field rejects exactly that row and the
// rest of the bundle still imports. Imported rows are badged and never merge
// into local profile scoring unless opted in for that specific import, and a
// bundle mixing methodology versions surfaces that rather than averaging them.

import { useState } from "react";
import { api } from "../api/client";
import type { ImportResponse } from "../types";

export function ImportBundle() {
  const [result, setResult] = useState<ImportResponse | null>(null);
  const [error, setError] = useState("");
  const [optIn, setOptIn] = useState(false);
  const [busy, setBusy] = useState(false);

  async function handleFile(file: File): Promise<void> {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const bundle = JSON.parse(await file.text()) as unknown;
      setResult(await api.importBundle(bundle, optIn));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">Import a results bundle</span>
      <p className="mt-2 text-xs leading-relaxed text-muted">
        Each row's hash is recomputed from its canonical form and re-checked here, so a bundle that was altered
        after export fails on exactly the altered rows. Imported rows are badged wherever they appear.
      </p>
      <label className="mt-3 flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={optIn} onChange={(e) => setOptIn(e.target.checked)} />
        Include these rows in local profile scoring (off by default — imported measurements come from someone
        else's machine)
      </label>
      <input
        type="file"
        accept="application/json,.json"
        disabled={busy}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
        }}
        className="mt-3 block text-xs text-muted"
      />

      {error && <p className="mt-2 text-xs text-danger">{error}</p>}

      {result && (
        <div className="mt-3 rounded-lg border border-border bg-surface-raised p-3 text-xs leading-relaxed text-muted">
          <p>
            Imported <b className="text-fg">{result.imported_rows}</b> row
            {result.imported_rows === 1 ? "" : "s"}
            {result.rejected_rows.length > 0 && (
              <>
                {" "}
                · <b className="text-danger">{result.rejected_rows.length} rejected</b>
              </>
            )}
            .
          </p>
          {result.mixed_vintages && (
            <p className="mt-1">
              This bundle mixes methodology versions ({result.method_versions.join(", ")}). They are surfaced
              together and never averaged together.
            </p>
          )}
          {result.rejected_rows.map((row) => (
            <p key={row.index} className="mt-1 text-danger">
              Row {row.index}: {row.reason}
            </p>
          ))}
          <p className="mt-1">{result.notice}</p>
        </div>
      )}
    </div>
  );
}
