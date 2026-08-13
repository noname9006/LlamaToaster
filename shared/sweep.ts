import type { SweepConfig, TestType } from "./types.js";

export interface SweepItem {
  idx: number;
  n_prompt: number;
  n_gen: number;
  threads: number;
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  // GPU layers offloaded for the MTP/draft companion model (llama-server's
  // -ngld/--n-gpu-layers-draft) -- only meaningful for an item whose mtp is
  // "on" *and* whose base model needs a separate --model-draft file (see
  // worker/src/serverBench.ts's buildArgs, which only ever pushes -ngld
  // alongside an actual --model-draft). Ignored (but still present, since
  // every SweepItem is a full cross-product) on every other item.
  n_gpu_layers_draft: number;
}

// Fixed field order matches the -p -n -t -ngl -b -ub -ctk -ctv -fa argument
// order llama-bench itself would expand a CSV-args sweep in, with mtp last
// since it has no llama-bench CLI equivalent at all -- an "on" item routes to
// worker/src/serverBench.ts (llama-server) instead of llama-bench, see
// worker/src/index.ts's per-item loop. n_gpu_layers_draft nests innermost of
// all, after mtp: it's only ever relevant to an "on" item, so grouping it
// last means a sweep that leaves it at a single value (the common case when
// MTP is off or unused) produces no duplicate items -- only a sweep that
// deliberately adds multiple draft-ngl values pays the full cross-product,
// including pointless duplicates of every mtp:"off" item that differ only in
// an ignored field. The server (pre-creating run_items at trigger time) and
// the worker (spawning one process per item) both call this on the exact
// same sweep JSON, so they must independently produce the identical ordered
// list -- there's no registration round trip to reconcile the two
// otherwise. repeats is intentionally not an axis here: it stays an
// in-process repeat count per item rather than a separate item, since an OOM
// at context creation fails identically on every repeat of the same combo.
export function expandSweep(sweep: Omit<SweepConfig, "model_id">): SweepItem[] {
  const items: SweepItem[] = [];
  for (const n_prompt of sweep.n_prompt) {
    for (const n_gen of sweep.n_gen) {
      for (const threads of sweep.threads) {
        for (const n_gpu_layers of sweep.n_gpu_layers) {
          for (const batch_size of sweep.batch_size) {
            for (const ubatch_size of sweep.ubatch_size) {
              for (const cache_type_k of sweep.cache_type_k) {
                for (const cache_type_v of sweep.cache_type_v) {
                  for (const flash_attn of sweep.flash_attn) {
                    for (const mtp of sweep.mtp) {
                      for (const n_gpu_layers_draft of sweep.n_gpu_layers_draft) {
                        items.push({
                          idx: items.length,
                          n_prompt,
                          n_gen,
                          threads,
                          n_gpu_layers,
                          batch_size,
                          ubatch_size,
                          cache_type_k,
                          cache_type_v,
                          flash_attn,
                          mtp,
                          n_gpu_layers_draft,
                        });
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return items;
}

// llama-bench's JSON output has no "test_type" field at all -- pp/tg/pg is
// implied by which of n_prompt/n_gen is zero (a combined "pg" item only
// occurs if the sweep uses both nonzero for the same combo).
export function deriveTestType(nPrompt: number, nGen: number): TestType {
  if (nGen === 0) return "pp";
  if (nPrompt === 0) return "tg";
  return "pg";
}
