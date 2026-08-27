// BENCHMARKING_PLAN_V8.md N3 -- the comparison TRIGGER: 2-5 models already
// registered on the target worker (file present in its local cache, not
// merely bookmarked), one frozen grid, priced as one total before submit.
// Members ride the ordinary trigger route with a shared comparison_id, so
// every §0.5 guard and the blocking fairness check apply unchanged.

import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { isMtpDraftModel } from "../types";
import type { Model, Worker } from "../types";
import { priceMatrix, ETA_UNAVAILABLE } from "../../../shared/pricing";
import { MAX_COMPARISON_MEMBERS, MIN_COMPARISON_MEMBERS } from "../../../shared/comparison";
import { formatBytes } from "../utils";

const KV_PAIR_CHOICES: { label: string; pair: [string, string] }[] = [
  { label: "f16 / f16", pair: ["f16", "f16"] },
  { label: "q8_0 / q8_0", pair: ["q8_0", "q8_0"] },
  { label: "q4_0 / q4_0", pair: ["q4_0", "q4_0"] },
];

export function NewComparison({ onCreated }: { onCreated: (comparisonId: string) => void }) {
  const [open, setOpen] = useState(false);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [workerId, setWorkerId] = useState("");
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [nPrompt, setNPrompt] = useState(512);
  const [nGen, setNGen] = useState(128);
  const [repeats, setRepeats] = useState(5);
  const [nGpuLayers, setNGpuLayers] = useState(99);
  const [kvPair, setKvPair] = useState<[string, string]>(["f16", "f16"]);
  const [priceLabel, setPriceLabel] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    api.listWorkers().then(setWorkers).catch(() => {});
    api.listModels().then(setModels).catch(() => {});
  }, [open]);

  const worker = workers.find((w) => w.id === workerId);

  // "Already registered on the target worker" -- file present in its local
  // cache, not merely bookmarked. models.id IS the file's sha256.
  const availableModels = useMemo(() => {
    if (!worker) return [];
    const hashes = new Set(
      (worker.modelFiles ?? []).map((f) => f.sha256).filter((h): h is string => typeof h === "string")
    );
    return models.filter((m) => hashes.has(m.id) && !isMtpDraftModel(m));
  }, [worker, models]);

  useEffect(() => {
    // Switching workers invalidates any selection scoped to the old one.
    setSelectedModelIds([]);
    setPriceLabel("");
  }, [workerId]);

  function toggleModel(id: string): void {
    setSelectedModelIds((cur) => {
      if (cur.includes(id)) return cur.filter((x) => x !== id);
      if (cur.length >= MAX_COMPARISON_MEMBERS) return cur;
      return [...cur, id];
    });
  }

  // §0.6 generalized to N models -- comparisons are the most expensive
  // object in the product, so the confirm screen shows TOTAL hours, never
  // per-model hours.
  async function priceTotal(): Promise<void> {
    if (!worker || selectedModelIds.length < MIN_COMPARISON_MEMBERS) {
      setPriceLabel("");
      return;
    }
    setPriceLabel("Pricing the full matrix…");
    try {
      let totalSeconds = 0;
      let anyUnpriced = false;
      for (const modelId of selectedModelIds) {
        const rates = await api.getModelRates(modelId, worker.id);
        const priced = priceMatrix([{ nPrompt, nGen, repeats, ppRate: rates.pp, tgRate: rates.tg }]);
        if (priced.seconds == null) anyUnpriced = true;
        else totalSeconds += priced.seconds;
      }
      const hours = totalSeconds / 3600;
      setPriceLabel(
        anyUnpriced
          ? `~${hours.toFixed(1)} h+ total across ${selectedModelIds.length} models (${ETA_UNAVAILABLE.toLowerCase()} for at least one -- no measured rate yet)`
          : `~${hours.toFixed(1)} h total across ${selectedModelIds.length} models`
      );
    } catch {
      setPriceLabel(ETA_UNAVAILABLE);
    }
  }

  async function submit(): Promise<void> {
    if (!worker || selectedModelIds.length < MIN_COMPARISON_MEMBERS) return;
    setBusy(true);
    setMsg("Creating comparison…");
    const comparisonId = crypto.randomUUID();
    // The frozen grid every member shares -- the interesting variable is the
    // model file, so everything else stays fixed by construction (one
    // single-value axis each), which is also what makes the blocking
    // fairness check trivially satisfied across members.
    const sweep = {
      n_prompt: [nPrompt],
      n_gen: [nGen],
      threads: [0],
      n_gpu_layers: [nGpuLayers],
      batch_size: [2048],
      ubatch_size: [512],
      cache_type_k: [kvPair[0]],
      cache_type_v: [kvPair[1]],
      flash_attn: ["on"],
      mtp: ["off"],
      n_gpu_layers_draft: [0],
      n_cpu_moe: [0],
      repeats,
    };
    const failures: string[] = [];
    for (const modelId of selectedModelIds) {
      try {
        await api.triggerRun({
          model_id: modelId,
          worker_id: worker.id,
          kind: "sweep",
          sweep,
          comparison_id: comparisonId,
        });
      } catch (err) {
        const name = models.find((m) => m.id === modelId)?.filename ?? modelId;
        failures.push(`${name}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    setBusy(false);
    if (failures.length === 0) {
      setMsg(`Comparison started across ${selectedModelIds.length} models.`);
      setSelectedModelIds([]);
      onCreated(comparisonId);
    } else {
      setMsg(`Some members were refused — ${failures.join("; ")}`);
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent"
      >
        + New comparison
      </button>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-fg">New model-vs-model comparison</h3>
        <button type="button" onClick={() => setOpen(false)} className="text-xs text-muted hover:text-fg">
          Cancel
        </button>
      </div>
      <p className="mt-1 text-[11px] leading-relaxed text-muted">
        Select {MIN_COMPARISON_MEMBERS}–{MAX_COMPARISON_MEMBERS} models already downloaded to one machine. Same
        worker, build, and grid for every member — the model file is the only variable, so a fairness check blocks
        anything else that drifts.
      </p>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Machine
          <select
            value={workerId}
            onChange={(e) => setWorkerId(e.target.value)}
            className="rounded-lg border border-border bg-bg px-2.5 py-1.5 text-sm text-fg"
          >
            <option value="">Choose a machine…</option>
            {workers.map((w) => (
              <option key={w.id} value={w.id}>
                {w.displayName} ({w.status})
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-muted">
          KV cache pair (frozen grid)
          <select
            value={KV_PAIR_CHOICES.findIndex(({ pair }) => pair[0] === kvPair[0] && pair[1] === kvPair[1])}
            onChange={(e) => setKvPair(KV_PAIR_CHOICES[Number(e.target.value)].pair)}
            className="rounded-lg border border-border bg-bg px-2.5 py-1.5 text-sm text-fg"
          >
            {KV_PAIR_CHOICES.map((c, i) => (
              <option key={c.label} value={i}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {workerId && (
        <div className="mt-3">
          <span className="text-xs text-muted">
            Models already on this machine ({availableModels.length} found)
          </span>
          {availableModels.length === 0 ? (
            <p className="mt-1 text-[11px] text-muted">
              This machine hasn't reported any downloaded model files with a hash yet — nothing here qualifies as
              "already registered." Older workers that predate file hashing won't appear as a source here either.
            </p>
          ) : (
            <div className="mt-1.5 flex flex-col gap-1.5">
              {availableModels.map((m) => (
                <label
                  key={m.id}
                  className={`flex cursor-pointer items-center gap-2 rounded-lg border px-2.5 py-1.5 text-sm ${
                    selectedModelIds.includes(m.id) ? "border-accent/40 bg-accent/10" : "border-border"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedModelIds.includes(m.id)}
                    disabled={!selectedModelIds.includes(m.id) && selectedModelIds.length >= MAX_COMPARISON_MEMBERS}
                    onChange={() => toggleModel(m.id)}
                  />
                  <span className="text-fg">{m.filename}</span>
                  <span className="text-[11px] text-muted">
                    {m.metadata.quant ?? "?"} · {formatBytes(m.size_bytes)}
                  </span>
                </label>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          n_prompt
          <input
            type="number"
            min={1}
            value={nPrompt}
            onChange={(e) => setNPrompt(Number(e.target.value) || 1)}
            className="rounded-lg border border-border bg-bg px-2 py-1 font-mono text-sm text-fg"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          n_gen
          <input
            type="number"
            min={1}
            value={nGen}
            onChange={(e) => setNGen(Number(e.target.value) || 1)}
            className="rounded-lg border border-border bg-bg px-2 py-1 font-mono text-sm text-fg"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          repeats
          <input
            type="number"
            min={1}
            max={25}
            value={repeats}
            onChange={(e) => setRepeats(Number(e.target.value) || 1)}
            className="rounded-lg border border-border bg-bg px-2 py-1 font-mono text-sm text-fg"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          n_gpu_layers
          <input
            type="number"
            min={0}
            value={nGpuLayers}
            onChange={(e) => setNGpuLayers(Number(e.target.value) || 0)}
            className="rounded-lg border border-border bg-bg px-2 py-1 font-mono text-sm text-fg"
          />
        </label>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={priceTotal}
          disabled={!worker || selectedModelIds.length < MIN_COMPARISON_MEMBERS}
          className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
        >
          Price total
        </button>
        {priceLabel && <span className="text-xs text-muted">{priceLabel}</span>}
      </div>

      <button
        type="button"
        onClick={submit}
        disabled={busy || !worker || selectedModelIds.length < MIN_COMPARISON_MEMBERS}
        className="mt-3 rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
      >
        {busy ? "Starting…" : `Start comparison (${selectedModelIds.length} model${selectedModelIds.length === 1 ? "" : "s"})`}
      </button>

      {msg && <p className="mt-2 text-[11px] text-muted">{msg}</p>}
    </div>
  );
}
