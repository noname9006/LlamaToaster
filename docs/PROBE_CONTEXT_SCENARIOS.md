# How the context test behaves, per architecture

The probe's context phase asks one question — *how many tokens can this machine
actually hold for this model?* — and answers it by loading the model
repeatedly. What it costs to answer, how many loads it spends, and where it
converges all depend on the model's attention geometry, because that is what
decides how many bytes a token of context costs.

This document is the scenario map: what the ladder does for each architecture
family, worked against real numbers, and where the estimator's model of that
family is thin.

Companion reading: `shared/probeLadder.ts` (the search), `shared/vramEstimate.ts`
(the cost model), `worker/src/gguf.ts` (where the geometry is read from).

---

## 1. The mechanism, in brief

- **The grid.** Context is searched on a power-of-two ladder from 1024 up to
  the model's trained context, with the ceiling always included exactly even
  when it isn't a power of two (`ctxLadderStops`). A `basic` probe only ever
  tests those stops; `fine` walks the same stops first, then bisects inside
  whichever adjacent pair bracketed the boundary, down to 1/16 of that
  bracket's width.
- **The anchor.** The phase seeds at the stop nearest `maxAffordableContext`
  for the currently-pinned layer count — a pure calculation, no load spent. A
  good anchor is worth two or three loads; a bad one costs the loads it takes
  the walk to climb out of.
- **The walk.** One notch up on success, one notch down on failure, converging
  the moment it reverses. Each step **doubles** the KV cache, so the cost of a
  step is not constant — the last step is as expensive as everything before it
  combined.
- **The budget.** 24 loads by default, shared with the layer phase and not
  pre-split. A badly anchored context phase is spending the layer phase's
  loads.
- **The verdict.** The largest context that passed, at the placement that
  achieved it.

## 2. What a token of context costs

```
kvBytesPerToken = nLayer x nHeadKv x (headDimK x bytes(ctk) + headDimV x bytes(ctv))
```

Every term comes from the GGUF header, and every one of them is architecture:

| term | GGUF key | what changes it |
|---|---|---|
| `nLayer` | `<arch>.block_count` | depth |
| `nHeadKv` | `<arch>.attention.head_count_kv` | MHA vs GQA vs MQA |
| `headDimK/V` | `<arch>.attention.key_length` / `.value_length` | falls back to `n_embd / n_head`, dropping confidence a level |
| `bytes(...)` | run setting, not the file | `f16` = 2 B, `q8_0` = 1.0625 B, `q4_0` = 0.5625 B |

Two further terms modify it structurally: `sliding_window` (only some layers
keep a full-length cache) and `shared_kv_layers` (some trailing layers keep
none at all).

The cache is split across pools exactly the way weights are: llama.cpp offloads
the **last** `ngl` blocks first, so the GPU holds the KV of that same suffix and
system RAM holds the rest (`computeDualPoolFit`). A context that doesn't fit in
VRAM does not necessarily fail — it moves.

---

## 3. Scenarios

### A. GQA MoE, no SWA — *context is cheap until suddenly it isn't*

**Reference case, measured.** Qwen3.6-35B-A3B-UD-IQ4_NL: 40 layers, `n_head`
16, `n_head_kv` 2 (8:1 GQA), head dim 256, trained context 262144.

```
KV/token (whole model, f16) = 80 KiB

  ctx    1,024 ->     80 MiB        ctx  131,072 -> 10,240 MiB
  ctx    8,192 ->    640 MiB        ctx  262,144 -> 20,480 MiB
  ctx   32,768 ->  2,560 MiB
```

The shape to notice: across the first five stops the KV cache is a rounding
error next to ~17 GiB of weights, and by the trained ceiling it is **larger
than the weights**. The context ladder is therefore nearly free at the bottom
and brutally expensive at the top, and which regime a machine sits in decides
how the probe behaves on it.

On an 8 GiB card this model is weights-bound, and context is the *wrong* axis
to search — confirmed by measurement: at 4 layers offloaded, going from a
1,024-token context to a 131,072-token one cost 762 MiB of total GPU footprint
and essentially no throughput (11.60 → 11.38 tok/s, same sweep).

**Ladder:** climbs many stops, cheaply, converges high.
**Gotcha:** a mode that resolves context *first* (`max_context`, `balanced`)
pins a large context and then finds no room left to add layers —
`growLayersPhase` correctly returns null and the mode ends having optimised the
axis that didn't matter. On a weights-bound model, prefer `max_gpu`.

### B. Dense MHA, no GQA — *KV-bound from the first stop*

`nHeadKv == nHead` (Llama-1 era, Mistral-7B-v0.1, many 2023 fine-tunes). Same
formula, 8-16x the per-token cost of case A.

**Ladder:** the anchor lands low, the walk reverses after one or two steps, and
the phase converges in 2-3 loads. `MaxCtxEstimate.binding` reports `"kv"` — the
honest signal that adding GPU layers will not buy context here, only a smaller
cache type will.
**Gotcha:** none, really. This is the case the ladder handles in the fewest
loads.

### C. SWA hybrid — *context grows sub-linearly, and the pattern is a guess*

Gemma-2/3, Cohere2, Llama-4, olmo2, openai-moe. Most layers attend only within
a sliding window; a repeating minority are full-attention ("global"). Only the
global layers' caches grow with context, so doubling the context roughly
doubles `globals/nLayer` of the cache rather than all of it.

`swaGlobalLayersInSuffix` counts globals within the GPU-resident suffix,
anchored on the model's final block (always global) and stepping backward by
the pattern period.

**Ladder:** climbs much further than the layer count alone suggests, often to
the trained ceiling.
**Gotchas, in order of how much they bite:**

1. **The period is a fallback when the file doesn't declare one.**
   `SWA_PATTERN_FALLBACK = 6` matches Gemma-3. Gemma-2 and openai-moe use 2;
   Cohere2, Llama-4 and olmo2 use 4. On those, assuming 6 *undercounts* global
   layers and so *underestimates* VRAM — the unsafe direction. The anchor comes
   out too high and the walk pays real loads climbing back down.
2. **Confidence always drops a level when `slidingWindow` is set**, even with a
   real period from the header, because "which layers carry no independent KV"
   still rests on the trailing-block assumption.
3. Array-typed hparams — see scenario I.

### D. Shared-KV trailing layers — *some layers cost nothing*

`<arch>.attention.shared_kv_layers`: a trailing block of layers reusing an
earlier layer's cache. Excluded from every count on both pools, using the same
suffix anchoring as the SWA pattern.

**Ladder:** behaves like a shallower model on the context axis while still
paying full price for weights. Expect it to beat an `nLayer`-based intuition by
exactly the shared fraction.
**Gotcha:** the exclusion assumes the shared layers are the model's *trailing*
N blocks. Nothing reads them layer-by-layer.

### E. Quantized KV cache — *not architecture, but the same lever*

`ctk`/`ctv` scale the per-token cost directly: `q8_0` is 0.53x `f16`, `q4_0` is
0.28x. On case A that is 10,240 MiB versus 5,440 MiB at a 131,072-token
context.

**Ladder:** the same search, anchored roughly one to two stops higher per
quantization step.
**Gotcha:** a verified context is only valid for the `(ctk, ctv)` pair it was
measured with. A ceiling verified under `q8_0` does not transfer to an `f16`
run, and nothing in the stored limit stops someone reading it that way.

### F. Multiple slots — *the ladder searches total, users think per-request*

`-c` covers **every** slot: `contextSizeForSlots` multiplies per-slot demand by
the slot count. A probe at `--parallel 4` that verifies 32,768 tokens has
verified 8,192 tokens *per concurrent request*.

**Gotcha:** the reported number is the `-c` value, not the per-request one.

### G. MLA / latent KV — *unmodelled; the estimate is far too pessimistic*

DeepSeek-V2/V3, Kimi and other multi-head-latent-attention models compress KV
into a low-rank latent whose size has nothing to do with `nHeadKv x headDim`.
**There is no MLA handling anywhere in `shared/vramEstimate.ts`** — the formula
above is applied regardless, overestimating the cache by a large factor.

**Ladder:** anchors far too low, then walks upward one stop at a time through
successes, spending loads to discover context the model always had. A `basic`
walk from 1024 to 131072 is seven loads of the budget, and the mode's layer
phase pays for them.
**Until it is modelled:** treat a verified context on an MLA model as a floor,
not a ceiling, and prefer `keep_context` or `custom` to pin a known-good value
rather than paying the ladder to rediscover it.

### H. Recurrent and hybrid SSM — *unmodelled; there may be no KV at all*

Mamba/Mamba-2, Jamba, Falcon-H1, RWKV and hybrid attention+SSM stacks carry a
**fixed-size recurrent state** per sequence. It does not grow with context.
There is no SSM handling in the estimator either, so a cost is predicted for
tokens that cost nothing.

**Ladder:** as in G, but more so — every stop succeeds until something
unrelated fails, and the estimate never explains why.
**Note:** for a *hybrid* stack the truth is in between — the attention layers
do grow — and treating the model as fully attentional is the safe direction.

### I. Array-typed hyperparameters — *the last element wins*

Some files write `head_count_kv`, `head_count`, `key_length` or `value_length`
as a **per-layer array** rather than a scalar (confirmed live on a real Gemma-4
GGUF, where `head_count_kv` alternates 8 on local layers and 1 on global ones).
`worker/src/gguf.ts` resolves an array to its **last element**, and
`shared/vramEstimate.ts` models the whole KV system with one scalar per field.

**Consequence:** the cost model uses whatever geometry the final block happens
to have, applied to all of them. For the Gemma-4 case that is the global
layer's extreme-GQA value — an *underestimate* of every local layer.
**Gotcha:** this one is silent. The confidence label does not drop for it.

---

## 4. Interaction with the host-backed fallback check

The context axis and `detectHostBackedFallback` touch the same memory, so it is
worth being explicit that they do not fight:

- A large context genuinely can push KV cache into system RAM, and it shows up
  in the same `shared` counter that spilled weights do.
- The detector's primary signal is a **slope**, not a level: how much
  system-RAM-backed memory appears *per layer added*, in units of one layer of
  this model. Overhead — including KV overhead from a big context — doesn't
  scale with layer count; spilled weights do. So growing the context cannot
  produce a spilling verdict, because the context isn't what's varying.
- That comparison is only valid between rungs at the **same context**, which is
  why the reference is restricted to same-context rungs. The layer phase pins
  context while it searches, so its rungs are mutually comparable by
  construction; a context phase (ngl fixed, ctx varying) has no valid slope
  reference and falls through to the single-rung bootstrap.
- The bootstrap judges shared usage against the rung's own *predicted
  footprint*, which includes GPU-side KV. That is deliberate: measured on the
  reference machine, 4 layers at a 131,072-token context reported 810 MiB
  shared, which is 48% of that placement's weights but only 25% of its full
  footprint. Denominated in weights alone it would be convicted; denominated in
  the footprint it is correctly left alone.

---

## 5. Quick reference

| architecture | KV per token | ladder behaviour | loads | main risk |
|---|---|---|---|---|
| Dense MHA | very high | reverses early | 2-3 | none |
| GQA dense | moderate | converges mid-ladder | 3-5 | none |
| GQA MoE (case A) | low until high ctx | climbs far | 4-7 | wrong axis searched first |
| SWA hybrid | sub-linear in ctx | climbs very far | 4-8 | fallback period wrong for non-Gemma-3 |
| Shared-KV tail | reduced | climbs further | 4-7 | trailing-block assumption |
| MLA / latent | **overestimated** | anchors far too low | 7+ | wasted budget, pessimistic result |
| Recurrent / SSM | **fabricated** | anchors far too low | 7+ | estimate describes nothing real |

Load counts are expectations for a `basic` context phase, not measurements —
only case A has been measured end to end. `fine` adds up to four more per
phase. Everything marked **bold** is a known gap in `shared/vramEstimate.ts`,
not a property of the architecture.
