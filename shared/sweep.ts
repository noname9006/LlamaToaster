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
  // llama.cpp's --n-cpu-moe N -- keeps the MoE (Mixture-of-Experts) weights
  // of the first N layers on CPU RAM instead of GPU VRAM, independent of
  // both mtp above and the base model's own -ngl (see worker/src/bench.ts
  // and worker/src/serverBench.ts's buildArgs, which push this on either
  // path regardless of MTP). 0 omits the flag entirely -- today's behavior.
  // Only meaningful for a model detected as MoE (see shared/types.ts's
  // ModelMetadata.expert_count); NewRun.tsx disables the control otherwise.
  n_cpu_moe: number;
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
// an ignored field. n_cpu_moe nests innermost of everything, even after
// n_gpu_layers_draft -- unlike that field it isn't MTP-only (--n-cpu-moe
// applies on both the llama-bench and llama-server paths regardless of mtp),
// but it's still only ever meaningful for a MoE model, so grouping it last
// keeps the same "a sweep that leaves it at a single value adds no duplicate
// items" property for the common (non-MoE) case. The server (pre-creating run_items at trigger time) and
// the worker (spawning one process per item) both call this on the exact
// same sweep JSON, so they must independently produce the identical ordered
// list -- there's no registration round trip to reconcile the two
// otherwise. repeats is intentionally not an axis here: it stays an
// in-process repeat count per item rather than a separate item, since an OOM
// at context creation fails identically on every repeat of the same combo.
// llama.cpp hard-refuses to create a context for any combo where flash
// attention is off but either KV cache side is quantized -- confirmed live
// (build b10442): "llama_init_from_model: quantized V cache requires
// flash_attn to be enabled", thrown from both the llama-bench and
// llama-server paths identically. There's no partial credit for only one
// side being f16 either -- ctk=f16/ctv=q8_0/fa=off fails the exact same way
// as ctk=q8_0/ctv=q8_0/fa=off, so both sides are checked. A sweep that
// cross-products flash_attn:[on,off] against any quantized cache type (a
// natural thing to do, e.g. comparing fa on vs off at a fixed cache
// setting) previously still produced these as real items: each one spawned
// a whole process just to die in a few seconds with a context-creation
// error, and produced no usable pp/tg row -- see worker/src/bench.ts and
// worker/src/serverBench.ts, which have no way to know this in advance
// per-item. Skipped here instead, in the one place both the server (item
// pre-creation) and worker (item execution) independently call this on the
// same sweep JSON, so both stay in agreement on the smaller item count
// without any registration round trip.
function isValidCombo(cache_type_k: string, cache_type_v: string, flash_attn: string): boolean {
  return flash_attn !== "off" || (cache_type_k === "f16" && cache_type_v === "f16");
}

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
                    if (!isValidCombo(cache_type_k, cache_type_v, flash_attn)) continue;
                    for (const mtp of sweep.mtp) {
                      for (const n_gpu_layers_draft of sweep.n_gpu_layers_draft) {
                        for (const n_cpu_moe of sweep.n_cpu_moe) {
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
                            n_cpu_moe,
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
