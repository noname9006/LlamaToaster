import { useEffect, useState } from "react";
import type { ChartConfiguration } from "chart.js";
import { api } from "../api/client";
import { Chart } from "../components/Chart";
import type { Run, ResultRow } from "../types";
import { shortId, formatFlashAttn } from "../utils";

interface ColumnDef {
  id: string;
  name: string;
}

// Carries the same suspect-measurement metadata RunDetail.tsx flags
// (see resultSuspectTitle there) alongside the plain number this page used
// to store -- suspect_count/suspect_samples are only ever populated by the
// llama-server/MTP path (worker/src/serverBench.ts) for a reading that was
// computable but outside plausible bounds (most often the known
// llama-server MTP timer bug). Kept as a small struct rather than reading
// straight off ResultRow so a row missing from one run cleanly falls back
// to "—" the same way it always did.
interface CellValue {
  avg_tps: number;
  sample_count?: number;
  suspect_count?: number;
  suspect_samples?: number[];
}
interface RowDef {
  key: string;
  label: string;
  values: Record<string, CellValue>;
}

function isSuspectCell(v: CellValue | undefined): boolean {
  return !!v?.suspect_count;
}

// Mirrors RunDetail.tsx's resultSuspectTitle wording so the same measurement
// reads identically wherever it's shown.
function cellSuspectTitle(v: CellValue): string | undefined {
  if (!v.suspect_count) return undefined;
  const raw = v.suspect_samples?.map((n) => n.toFixed(1)).join(", ") ?? "unknown";
  const total = v.sample_count ?? v.suspect_count;
  // See RunDetail.tsx's resultSuspectTitle -- an all-suspect result has no
  // clean reading to fall back on, so the shown average is actually derived
  // from these flagged values, not excluding them. Kept in sync with that
  // function's wording rather than the old single unconditional phrasing.
  return v.suspect_count >= total
    ? `All ${total} repeat(s) reported an implausible speed -- no clean reading was available, so this average is ` +
        `derived from those flagged values and should not be trusted (raw: ${raw} tok/s) -- likely the known ` +
        `llama-server MTP timing bug`
    : `${v.suspect_count} of ${total} repeat(s) reported an implausible speed and were excluded from this average ` +
        `(raw flagged values: ${raw} tok/s) -- likely the known llama-server MTP timing bug`;
}

// Reads the app's actual dark-theme tokens (client/src/styles/index.css)
// rather than hardcoding hex values a second time, so this chart's colors
// stay in sync if the theme ever changes.
function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// One base color per compared run (cycled if more than the theme's own
// named accents are selected) -- suspect bars within a dataset are
// overridden to the warning color below instead of using this, so a
// flagged bar is visually distinct from its own run's clean bars, not just
// from other runs' bars.
function datasetPalette(): string[] {
  return [
    cssVar("--color-accent", "#6ea8fe"),
    cssVar("--color-success", "#86ef9e"),
    cssVar("--color-danger", "#ff9b9b"),
    "#c9a6ff",
    "#7fe0e0",
    "#ffb27f",
  ];
}

function comboKey(r: ResultRow): string {
  return [
    r.test_type,
    r.n_prompt,
    r.n_gen,
    r.n_threads,
    r.n_gpu_layers,
    r.batch_size,
    r.ubatch_size,
    r.cache_type_k,
    r.cache_type_v,
    formatFlashAttn(r.flash_attn),
    r.mtp,
    // Only actually varies on an mtp:"on" row (see shared/sweep.ts's
    // SweepItem.n_gpu_layers_draft), but included unconditionally here same
    // as every other field -- without it, two rows differing only in draft
    // offload would collide onto the same map key in render() below and one
    // would silently disappear from the comparison instead of showing as its
    // own row.
    r.n_gpu_layers_draft,
  ].join("|");
}

function comboLabel(r: ResultRow): string {
  // Unlike the other swept params (already omitted from this deliberately
  // terse label), mtp gets called out when "on" -- it's a different
  // execution path (llama-server, not llama-bench) with a large expected
  // speed delta, not just another quantization/config knob. ngld only
  // follows mtp itself since it's meaningless without it.
  return (
    `${r.test_type} p${r.n_prompt} t${r.n_threads} ngl${r.n_gpu_layers}` +
    (r.mtp === "on" ? ` mtp ngld${r.n_gpu_layers_draft}` : "")
  );
}

export function Compare() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [columns, setColumns] = useState<ColumnDef[]>([]);
  const [rows, setRows] = useState<RowDef[]>([]);

  useEffect(() => {
    (async () => {
      const list = await api.listRuns();
      setRuns(list.filter((r) => r.status === "done"));
    })();
  }, []);

  function toggleSelected(id: string) {
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  }

  async function render() {
    // Match rows across runs by their sweep combo (test_type + every swept
    // param), not by array position -- different runs can have different
    // sweeps, in a different order, with different sizes.
    const all = await Promise.all(
      selected.map(async (id) => {
        const d = await api.getRun(id);
        return { id, name: `${d.run.worker_name} ${d.run.id.slice(0, 8)}`, results: d.results ?? [] };
      })
    );

    if (!all.length) {
      setColumns([]);
      setRows([]);
      return;
    }

    const byRun = all.map((a) => {
      const map = new Map<string, ResultRow>();
      for (const r of a.results) map.set(comboKey(r), r);
      return map;
    });

    const keyOrder: string[] = [];
    const seen = new Set<string>();
    const labelByKey = new Map<string, string>();
    for (const a of all) {
      for (const r of a.results) {
        const k = comboKey(r);
        if (!seen.has(k)) {
          seen.add(k);
          keyOrder.push(k);
          labelByKey.set(k, comboLabel(r));
        }
      }
    }

    setColumns(all.map((a) => ({ id: a.id, name: a.name })));
    setRows(
      keyOrder.map((k) => {
        const values: Record<string, CellValue> = {};
        all.forEach((a, i) => {
          const r = byRun[i].get(k);
          if (r) {
            values[a.id] = {
              avg_tps: r.avg_tps,
              sample_count: r.sample_count,
              suspect_count: r.suspect_count,
              suspect_samples: r.suspect_samples,
            };
          }
        });
        return { key: k, label: labelByKey.get(k) ?? k, values };
      })
    );
  }

  const chartConfig: ChartConfiguration<"bar"> | null = rows.length
    ? (() => {
        const palette = datasetPalette();
        const warningColor = cssVar("--color-warning", "#ffd479");
        return {
          type: "bar",
          data: {
            labels: rows.map((r) => r.label),
            datasets: columns.map((c, i) => ({
              label: c.name,
              data: rows.map((r) => r.values[c.id]?.avg_tps ?? null) as number[],
              // Per-bar override, not per-dataset -- a suspect reading only
              // turns *that* bar the warning color, so it's visually
              // distinct from this same run's own clean bars, not just from
              // other runs' bars.
              backgroundColor: rows.map((r) =>
                isSuspectCell(r.values[c.id]) ? warningColor : palette[i % palette.length]
              ),
            })),
          },
          options: {
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } },
            plugins: {
              tooltip: {
                callbacks: {
                  label(context) {
                    const row = rows[context.dataIndex];
                    const col = columns[context.datasetIndex];
                    const cell = row?.values[col?.id ?? ""];
                    const base = `${context.dataset.label}: ${context.formattedValue} tok/s`;
                    if (!cell || !isSuspectCell(cell)) return base;
                    return [base, `⚠ ${cellSuspectTitle(cell)}`];
                  },
                },
              },
            },
          },
        };
      })()
    : null;

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Compare runs</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Select 2+ runs to compare avg tokens/s side by side. Runs with different sweep configs
        are matched by test config, not position — a config missing from a run shows as "—".
        A <span className="text-warning">⚠ highlighted</span> bar or cell means at least one
        repeat behind that average was flagged as an implausible reading (most often the known
        llama-server MTP timing bug) — hover it for the raw measured values.
      </p>

      {runs.length === 0 ? (
        <p className="mt-4 text-sm text-muted">No completed runs yet.</p>
      ) : (
        <div className="mt-4 flex flex-wrap gap-2">
          {runs.map((r) => (
            <label
              key={r.id}
              className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${
                selected.includes(r.id) ? "border-accent/40 bg-accent/10" : "border-border"
              }`}
            >
              <input type="checkbox" checked={selected.includes(r.id)} onChange={() => toggleSelected(r.id)} />
              <code className="text-muted">{shortId(r.id)}</code>
              <span className="text-fg">{r.worker_name}</span>
            </label>
          ))}
        </div>
      )}

      <button
        type="button"
        onClick={render}
        disabled={selected.length < 2}
        className="mt-4 rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
      >
        Compare
      </button>

      {chartConfig && (
        <div className="relative mt-4 h-72 rounded-xl border border-border bg-surface p-4">
          <Chart config={chartConfig} />
        </div>
      )}

      {rows.length > 0 && (
        <div className="mt-4 overflow-x-auto rounded-xl border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-border text-xs uppercase tracking-wide text-muted">
                <th className="px-4 py-2.5 font-medium">config</th>
                {columns.map((c) => (
                  <th key={c.id} className="px-4 py-2.5 font-medium">
                    {c.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row) => (
                <tr key={row.key}>
                  <td className="px-4 py-2.5 text-fg">{row.label}</td>
                  {columns.map((c) => {
                    const cell = row.values[c.id];
                    const suspect = isSuspectCell(cell);
                    return (
                      <td
                        key={c.id}
                        className={`px-4 py-2.5 ${suspect ? "text-warning" : "text-muted"}`}
                        title={cell ? cellSuspectTitle(cell) : undefined}
                      >
                        {cell !== undefined ? `${cell.avg_tps.toFixed(2)}${suspect ? " ⚠" : ""}` : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
