// BENCHMARKING_PLAN_V8.md N3 -- model-vs-model comparison across weight
// quants: a table of per-model best-config rows side by side, plus a Pareto
// scatter of speed × memory.
//
// The fairness checks are BLOCKING and re-checked per member. A drifted
// member is marked FATAL in this view rather than quietly shifting the answer
// — a comparison is the one place a silent confound poisons exactly the
// conclusion being sold. Aborting the group keeps completed members
// comparable: the ones that finished cleanly still render.
//
// Identical quant LABELS with different file hashes display separately.
// Labels lie, hashes don't.

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ComparisonResponse } from "../types";

export function ComparisonView({ comparisonId }: { comparisonId: string }) {
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getComparison(comparisonId)
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError("");
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [comparisonId]);

  if (error) return <p className="text-sm text-danger">Could not load this comparison: {error}</p>;
  if (!data) return <p className="text-sm text-muted">Loading comparison…</p>;

  const maxPeak = Math.max(...data.pareto.map((p) => p.peakMib), 1);
  const maxTg = Math.max(...data.pareto.map((p) => p.tg), 1);

  return (
    <div className="flex flex-col gap-4">
      {data.drifted_members.length > 0 && (
        <p className="rounded-lg border border-danger/40 bg-danger-bg px-3.5 py-2.5 text-xs leading-relaxed text-danger">
          <b>{data.drifted_members.length} member(s) drifted.</b> The interesting variable in a comparison is the
          model file, so everything else is held fixed — same machine, build, backend, GPU, flags, repeats and
          methodology version. A drifted member fails loudly rather than shifting the conclusion; the members that
          finished cleanly below stay comparable with each other.
        </p>
      )}

      <div className="overflow-x-auto rounded-xl border border-border bg-surface">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-[10px] uppercase tracking-wide text-muted">
              <th className="px-3 py-2">model</th>
              <th className="px-3 py-2">quant</th>
              <th className="px-3 py-2 text-right">PP tok/s</th>
              <th className="px-3 py-2 text-right">TG tok/s</th>
              <th className="px-3 py-2 text-right">peak VRAM</th>
              <th className="px-3 py-2 text-right">verified ctx</th>
              <th className="px-3 py-2 text-right">PPL / KLD</th>
              <th className="px-3 py-2">status</th>
            </tr>
          </thead>
          <tbody>
            {data.members.map((member) => (
              <tr key={member.run_id} className="border-b border-border/40">
                <td className="px-3 py-2 font-mono text-fg" title={member.file_sha256 ?? undefined}>
                  {member.model_filename}
                </td>
                <td className="px-3 py-2 text-muted">
                  {member.quant_label ?? "—"}
                  {member.file_sha256 && (
                    <span className="ml-1 font-mono text-[9px] text-muted" title="file hash — identical labels on different files stay distinct rows">
                      {member.file_sha256.slice(0, 8)}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {member.pp != null ? member.pp.toFixed(0) : "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {member.tg != null ? member.tg.toFixed(1) : "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {member.vram_peak_mib != null ? `${member.vram_peak_mib} MiB` : "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {member.verified_ctx_tokens != null ? member.verified_ctx_tokens.toLocaleString() : "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {member.ppl != null ? (
                    <span title={member.dataset_hash ?? undefined}>
                      {member.ppl.toFixed(3)}
                      {member.kld_vs_baseline != null && ` / ${member.kld_vs_baseline.toFixed(4)}`}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="px-3 py-2">
                  {member.violations.length > 0 ? (
                    <span
                      title={member.violations.map((v) => v.message).join(" · ")}
                      className="rounded-full bg-danger-bg px-2 py-0.5 text-[10px] font-bold text-danger"
                    >
                      drifted — {member.violations.map((v) => v.field).join(", ")}
                    </span>
                  ) : (
                    <span className="text-muted">{member.status}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.pareto.length > 0 && (
        <div className="rounded-xl border border-border bg-surface p-4">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">
            Pareto — speed × memory across models
          </span>
          <div className="relative mt-3 h-48 rounded-lg border border-border bg-bg">
            {data.pareto.map((point) => (
              <span
                key={point.model_id}
                title={`${point.label}: ${point.tg.toFixed(1)} tok/s, ${point.peakMib} MiB${point.dominated ? " (dominated)" : " (on the frontier)"}`}
                style={{
                  left: `${(point.peakMib / maxPeak) * 88 + 4}%`,
                  bottom: `${(point.tg / maxTg) * 80 + 8}%`,
                }}
                className={
                  point.dominated
                    ? "absolute -translate-x-1/2 rounded-full border border-border bg-raised px-2 py-0.5 text-[10px] text-muted opacity-60"
                    : "absolute -translate-x-1/2 rounded-full border border-accent bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent"
                }
              >
                {point.label}
              </span>
            ))}
          </div>
          <p className="mt-2 text-[10px] text-muted">← lower memory · higher generation speed ↑</p>
          {/* Never colour-only. */}
          <p className="sr-only">
            {data.pareto
              .map(
                (p) =>
                  `${p.label}: ${p.tg.toFixed(1)} tokens per second at ${p.peakMib} mebibytes, ${p.dominated ? "dominated by another configuration" : "on the frontier"}`
              )
              .join(". ")}
          </p>
        </div>
      )}

      <p className="text-[11px] leading-relaxed text-muted">{data.quality_disclaimer}</p>
    </div>
  );
}
