# Context test redesign — how parameters to measure are selected

How the probe decides which configurations are worth loading, and what a load
does and does not tell you once it has run.

Companion reading: `docs/PROBE_CONTEXT_SCENARIOS.md` (what the ladder does per
architecture), `shared/probeLadder.ts` (the search), `shared/vramEstimate.ts`
(the cost model and the spill verdict).

**Status.** The verdict changes in §7 are implemented. Everything else —
the two-scenario structure, the reuse widening, persisting the new verdict
fields — is specification, not code.

| part | state |
|---|---|
| Resident slope as a required second opinion | implemented |
| Bootstrap residency veto | implemented |
| Pressure gate | **proposed and withdrawn** — see §7 |
| Span-1 slope references | **proposed and withdrawn** — see §7 |
| Wizard / Targets structure | spec only |
| Reuse across probes (not just batch siblings) | spec only |
| `residentSlopeRatio` / `abstained` persisted and displayed | spec only |

---

## 1. What a load actually measures

A rung answers one question: **does this configuration allocate?** It is not a
performance measurement, and nothing downstream may treat it as one.

Every rung runs the same fixed ~512-token workload — a 256-token prompt, 256
generated — whatever `-c` was set to. So a rung at 262,144 tokens allocates a
cache of that size and then reads roughly none of it:

- its prompt-processing rate is prefill of **256 tokens**;
- its time-to-first-token is that same prefill;
- its generation rate is generation against a **near-empty cache**.

None of those three observe the context they are labelled with, so none of them
may enter a verdict, a ranking, or a recommendation. They establish that a
configuration ran and produced tokens. That is their entire job.

The consequence worth stating plainly: a flat generation rate across a context
ladder does **not** mean context is cheap on that model. It means the probe
never exercised it.

---

## 2. The candidate space

Two axes, and they are not sampled alike.

| axis | grid | points | why that grid |
|---|---|---|---|
| Offload (`ngl`) | every integer, 0…n | 42 | Cost is linear and the unit is one layer's weights, so every integer is a distinct, equally-spaced hypothesis. |
| Context | ×2 from 1,024, plus the exact trained ceiling (`ctxLadderStops`) | 9 | Each step doubles the KV cache, so the grid is already logarithmic in cost — the last step is as expensive as every step below it combined. |

On the reference model that is 42 × 9 = **378 candidate configurations**
against a budget of 24 loads (`PROBE_MAX_LOADS`, admin-overridable). Selection
is the whole design; the search itself is trivial once the candidate is chosen.

---

## 3. How the next point is chosen

Three mechanisms, cheapest first.

**Reuse — 0 loads.** A stored frontier point for the same worker, model,
llama.cpp build and KV pair answers the question outright. Today reuse is
scoped to siblings under one batch root (`GET /api/tests/:id/probe-dedup`), so
asking two questions a week apart re-measures the overlap from scratch.
Widening it to all probes for that key is the single largest saving available.

**Anchor from the estimate — 0 loads.** The per-tensor GGUF estimator answers
"how much context does this layer count afford" (`maxAffordableContext`) and
its inverse (`estimateSafeNgl`) as pure arithmetic, snapped to the nearest grid
point to become the seed. A good anchor is worth two or three loads; a bad one
costs the loads the search spends climbing out of it. Nothing here touches the
GPU.

**Move — 1 load each.** Three styles, and which one applies is decided by what
is known, not by the mode:

- **Jump** (`direct`) when the estimator has a specific answer to test. A wrong
  one costs one load and brackets immediately, where a staircase toward it
  costs many.
- **Bisect** when the target is bracketed but unknown.
- **Step** one notch on the context grid (`slider_refine`), because that grid
  is already logarithmic — bisecting a log scale bisects nothing meaningful.

Convergence is reversal: the first time a direction flips, the boundary is
between the last two points and the phase ends. At `basic` granularity that is
the answer; `fine` then bisects inside the bracketing pair to a sixteenth of
its width, which costs up to four more loads and is only worth it on the
context axis, where grid points are far apart.

---

## 4. Which point, concretely

Four rules decide every candidate. Nothing else is consulted, and no rule
depends on which scenario asked.

| situation | point chosen | rationale |
|---|---|---|
| Opening | the estimator's own answer for the pinned axis | The only informed hypothesis available, and it costs nothing to compute. |
| Last point passed | +1, then +2, +4, +8… until one fails | Doubling reaches a distant boundary in log steps while still bracketing a near one immediately — the guard against an anchor that is badly too low. |
| Last point failed | the estimator's safe answer if untried, else the bracket midpoint | One jump to a computed answer beats a staircase toward it; a wrong jump costs a single load and brackets on the spot. |
| Bracket closed | stop — no load | Width 1 on the offload axis; adjacent grid stops on the context axis, or a sixteenth of that interval at `fine`. |

Applied to the reference machine, where the estimator's safe answer at floor
context is 16 layers:

| load | point | chosen because | outcome |
|---|---|---|---|
| 1 | ngl 0 · ctx 1,024 | Baseline — what this machine reports as host-backed with nothing claimed on the GPU | 571 MiB noted |
| 2 | ngl 16 · ctx 1,024 | Opening: jump to the estimator's answer | passed |
| 3 | ngl 17 · ctx 1,024 | Last point passed: step up by 1 | failed |
| — | bracket [16, 17] | Width 1 — offload boundary resolved, no load spent | boundary = 16 |
| 4 | ngl 16 · ctx 262,144 | Offload settled, so context now moves at a placement already judged clean — jump straight to the trained ceiling | passed |
| 5–6 | ngl 18, 20 · ctx 1,024 | Optional: show what crossing the boundary costs, rather than asserting it | cliff drawn |

Six loads against the 24-load budget, and the context axis moved exactly once.
The five probes on the 6 Sep 2026 reference run spent 31 loads reaching four
different answers.

---

## 5. Which axis moves first

The offload axis is resolved first, at the floor context. Three reasons, and
only the first is about speed:

| reason | consequence |
|---|---|
| Cheapest loads | A 1,024-token cache is a rounding error against the weights, so these loads are the fastest the probe can buy. |
| Uncontaminated verdict | With the cache term negligible, host-backed memory at these rungs is attributable to weights alone — which is what the offload verdict is about. |
| Context never moves at an unjudged placement | Spill on the context axis is reported but deliberately never failed, so a placement reached by growing context inherits a pass it was never tested for. Resolving offload first means that inheritance can never happen. |

That third row is why the ordering is fixed rather than a heuristic. On the
reference run, ngl 17 at 262,144 tokens passed with 4,013 MiB host-backed
without its layer count ever being judged — and four probes then reported four
different boundaries for one machine.

---

## 6. Where the anchor comes from, and where it is wrong

Every anchor traces back to the model file. A term the estimator reads wrongly
is not a wrong number on a screen; it is loads spent walking back from a bad
seed.

| term | GGUF key | effect on the anchor |
|---|---|---|
| Depth | `block_count` | Scales the cache linearly |
| KV heads | `attention.head_count_kv` | MHA against MQA is an 8–16× swing in cost per token |
| Head dims | `attention.key_length` / `.value_length` | Falls back to `n_embd / n_head`, dropping confidence a level |
| Cache width | *run setting, not the file* | f16 2 B, q8_0 1.06 B, q4_0 0.56 B — a verified ceiling is valid only for the pair it was measured with |
| Sliding window | `attention.sliding_window` | Only global layers grow with context; the period is guessed at 6 when undeclared |
| Shared KV tail | `attention.shared_kv_layers` | Trailing layers carrying no independent cache |

### Which way a bad anchor points

An anchor that is **too low** is only expensive: the search climbs through
successes, spending loads to rediscover capacity the machine always had, and
still arrives at the right answer. An anchor that is **too high** is worse than
expensive: the search opens on failures, and on a driver that accepts
overcommitted allocations silently, a plausible wrong answer is reachable.

| family | modelled | anchor points | what it costs | flagged |
|---|---|---|---|---|
| Sliding window, period declared | yes | about right | Climbs further than layer count suggests, correctly | yes — `rough` |
| Sliding window, period assumed | partly | **too high** | Failed loads while the search walks back down | yes — `rough` |
| Shared-KV tail | yes | about right | Rests on the trailing-block assumption | no |
| MLA / latent KV | **no** | **far too low** | Loads spent climbing; a pessimistic result if the budget runs out first | no |
| Recurrent / hybrid SSM | **no** | **far too low** | As above, and the estimate describes nothing real | no |
| Array-typed hparams | **mis-read** | **file-dependent** | Failed loads when the last block is the cheap one | no |

**The two that are not modelled at all.** *MLA / latent KV* (DeepSeek-V2/V3,
Kimi) compresses key and value into a single low-rank latent, so there is no
per-head cache to size — the `layers × kv-heads × head-dim` formula describes a
structure the file does not contain, and predicts a cache several times larger
than the one allocated. *Recurrent and hybrid SSM* (Mamba, Mamba-2, Jamba,
Falcon-H1, RWKV) carries a fixed-size state per sequence that does not grow
with context at all, so a per-token cost is charged for tokens that are free.
For a pure recurrent model the context axis carries no information whatever;
for a hybrid, only its attention layers grow. Both overestimate, so both anchor
far too low, and on both the honest reading of a verified context is a floor
rather than a ceiling.

**The one that is mis-read.** Some files write `head_count_kv`, `head_count` or
the head dimensions as one value *per layer* rather than a single number.
`worker/src/gguf.ts` takes the last element and the estimator applies it to
every layer — confirmed on a real Gemma-4 file whose `head_count_kv` alternates
8 on local layers and 1 on global ones. Anchoring the whole model on the final
block's geometry underestimates every local layer, and unlike the
sliding-window case nothing lowers the confidence label to say so.

**Sliding window is a different situation from all of those.** It is genuinely
modelled: local layers are capped at the window and their cost stops growing,
only global layers scale with context, and the split is computed
(`swaGlobalLayersInSuffix`) rather than assumed away. It is also the only entry
here that flags itself — confidence drops to `rough` whenever `sliding_window`
is set, even when a real period was read from the header, because which
trailing layers carry no independent cache still rests on an assumption. The
exposure is narrow and specific: when the file does not declare the period,
`SWA_PATTERN_FALLBACK` assumes 6, which is Gemma-3's. Gemma-2 and openai-moe
use 2; Cohere2, Llama-4 and olmo2 use 4. Assuming 6 on those undercounts the
global layers, which undercounts the cache, which is the one direction that
produces failed loads instead of merely slow ones.

**Priority.** Ranked by how likely a real user is to hit them, the top two
entries are both Gemma — the most common local family on the platform — and
both point the dangerous way. MLA and SSM are rare and fail gracefully. The
proportionate response for those two is not a memory model but **detection**:
recognise the architecture from the GGUF header and return
`confidence: "unknown"` rather than a confident wrong number, so the search
starts from measurement instead of fantasy.

---

## 7. What decides a rung passed

The load either allocated or it did not, and if it allocated, the driver either
put it on the GPU or silently backed it with system RAM. Two counters, read as
a pair, and not one speed measurement among them.

**Host-backed slope — primary evidence.** How much system-RAM-backed GPU memory
appears per layer added, against a rung at least two layers below at the same
context, in units of one layer's weights from the file. Overhead does not scale
with layer count; spilled weights do. The two-layer span
(`HOST_BACKED_MIN_SLOPE_SPAN`) is not conservatism — at one layer the driver's
~224 MiB allocation granule appears and disappears between adjacent rungs, and
a measured 42-setting sweep shows the clean and spilling regimes overlapping
outright at that resolution.

**Resident slope — required second opinion.** The same interval read on the
dedicated-VRAM counter: did the added layers show up *on the device*? On the
calibration sweep it separates the two regimes wider than the host counter
does, and the two agree on all nine measured intervals:

| interval | dedicated/layer | shared/layer | truth |
|---|---|---|---|
| 2 → 4 | 1.05 | 0.00 | clean |
| 4 → 6 | 0.96 | 0.00 | clean |
| 6 → 8 | 0.74 | 0.25 | clean |
| 8 → 10 | 0.20 | 0.76 | spilling |
| 10 → 12 | −0.21 | 1.18 | spilling |
| 12 → 14 | 0.18 | 0.78 | spilling |
| 14 → 18 | −0.02 | 0.99 | spilling |
| 18 → 26 | 0.06 | 0.91 | spilling |
| 26 → 41 | 0.05 | 0.89 | spilling |

Clean 0.74–1.05 against spilling −0.21–0.20: a gap of 0.54, where the shared
counter's is 0.29. Both must agree before a rung is failed. Where they disagree
the rung is left **undecided** (`abstained`) rather than convicted, and the
caller falls through to its weaker evidence exactly as it does when no counter
existed at all — because a false conviction fails a configuration that works
and moves the reported boundary.

**Residency veto — bootstrap only.** When no valid reference exists, a single
rung is judged against its own predicted footprint — and that numerator counts
bytes that were never weights. On the reference run ngl 0 reported 571 MiB
host-backed with *nothing* claimed on the GPU, so ngl 1 was charged that fixed
cost against one layer and convicted at 0.95, while its dedicated VRAM had just
grown by 401 MiB for a 420 MiB layer and its generation rate had gone up,
9.1 → 10.3 tok/s. When the dedicated counter shows the predicted footprint
largely did land (`HOST_BACKED_RESIDENT_ACQUIT_FRAC`, 0.70), no host-counter
reading may convict.

### Two rules that were proposed and withdrawn

A **pressure gate** — refusing to convict unless the GPU was actually full — is
wrong on this hardware. The calibration sweep records ngl 26 spilling 7,647 MiB
to host memory at 3.5 tok/s with the GPU **46% full**, and `growLayersPhase`'s
own notes record the driver refusing weights with 1.9 GiB of VRAM free. The
driver does not wait to run out, so "it ran out" cannot be a precondition.

Admitting a **one-layer slope reference** guarded by a minimum delta does not
work either: the granularity noise is large and positive, not small, so a
genuinely clean pair can read +1.05 layers. Any threshold admitting the real
cases admits that one too. The span stays at two, and the fix for rungs that
fall through to the bootstrap belongs in the **search order**, which can
guarantee a valid reference exists.

### Known gaps in the current implementation

- The acquit threshold's margin is thin: spilling ngl 12 reads 0.62, clean
  ngl 10 reads 0.77, the threshold sits at 0.70. It only ever acquits, which
  bounds the damage.
- No real measurement exercises the abstention branch — the two counters agree
  on all nine measured intervals, so its test is constructed.
- `residentSlopeRatio` and `abstained` are computed but not persisted, so a
  rung that abstains renders as *slope 0.61 · passed* with nothing explaining
  why. Needs a migration and a UI line.

---

## 8. What this cannot answer

A verified context is an **allocation ceiling**: the cache fit, and the process
generated tokens with it essentially empty. How the model behaves with 200,000
tokens actually resident — attention over a full cache, a KV cache the driver
parked in system RAM and now has to read on every token — is not observed by
any rung the probe runs, and cannot be inferred from the rates beside them.

Answering it means a workload that fills the cache, which is a separate sweep
with its own budget and its own honest cost. Until that exists, the probe's
context result should be read as *this much will allocate*, and nothing further.

---

## Reference run

5 probes, 36 rungs, 6 Sep 2026. Qwen3.6-35B-A3B-UD-IQ4_NL (41 claimable layers,
262,144 trained context) on a Radeon RX 6600 XT, 8 GiB, Vulkan, llama.cpp
b10819. Per-layer weight ~422 MiB, derived from the run's own reported slopes.

The 11-point calibration sweep quoted in §7 is the earlier one recorded at the
top of `shared/vramEstimate.ts` and reproduced as a fixture in
`shared/vramEstimate.test.ts`.
