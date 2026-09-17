# Context test redesign — how parameters to measure are selected

How the probe decides which configurations are worth loading, and what a load
does and does not tell you once it has run.

Companion reading: `docs/PROBE_CONTEXT_SCENARIOS.md` (what the ladder does per
architecture), `shared/probeLadder.ts` (the search), `shared/vramEstimate.ts`
(the cost model), `shared/gpuSpill.ts` (the spill verdict).

**Status (17 Sep 2026).** The search the probe runs today is §10, and the spill
verdict it uses is §7. Sections 2–5 and 9 describe the searches it replaced — the
removed `max_gpu`, `max_context` and `balanced` modes, the `fine` setting,
and the first frontier — and are kept for the reasoning behind them. §1, §6 and
§8 still hold.

| part | state |
|---|---|
| Spill as growth over an anchor load (§7) | implemented |
| Wizard / Targets structure | implemented |
| Two targets, per-context order, controls, per-phase spill (§10) | implemented |
| `max_gpu` / `max_context` / `balanced` modes, `fine` | **removed** |
| Reuse across probes (not just batch siblings) | spec only |

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

> Superseded by §10 — kept for the reasoning. The search described here no longer runs.

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

> Superseded by §10 — kept for the reasoning. The search described here no longer runs.

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

> Superseded by §10 — kept for the reasoning. The search described here no longer runs.

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
| 4 | ngl 16 · ctx 262,144 | Offload settled, so context now moves at a placement already judged clean — jump straight to the trained ceiling | **fails** if most of the memory that context added lands in system RAM (§7), and the walk bisects down |
| 5–6 | ngl 18, 20 · ctx 1,024 | Optional: show what crossing the boundary costs, rather than asserting it | cliff drawn |

Six loads against the 24-load budget, and the context axis moved exactly once.
The five probes on the 6 Sep 2026 reference run spent 31 loads reaching four
different answers.

Row 4 used to read *passed*, and that was the bug: probe 124c2ab1 "verified"
262,144 tokens on this 8 GiB card with the entire cache in system RAM. A
context whose cache is host-backed now fails, so what this row really shows is
the **largest context whose cache stayed on the GPU at 16 layers** — which on
this machine is below the trained ceiling, and is exactly why one corner is not
an answer. See §9.

---

## 5. Which axis moves first

> Superseded by §10 — kept for the reasoning. The search described here no longer runs.

The offload axis is resolved first, at the floor context. Three reasons, and
only the first is about speed:

| reason | consequence |
|---|---|
| Cheapest loads | A 1,024-token cache is a rounding error against the weights, so these loads are the fastest the probe can buy. |
| Uncontaminated verdict | With the cache term negligible, host-backed memory at these rungs is attributable to weights alone — which is what the offload verdict is about. |
| Context never moves at an unjudged placement | Spill on the context axis judges the cache, never the weights, so a placement reached by growing context inherits a layer verdict it was never tested for. Resolving offload first means that inheritance can never happen. |

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
put it on the GPU or silently backed part of it with system RAM. Not one speed
measurement is involved.

### The anchor

Every probe starts with two loads of the same model at **1 layer and 1,024
tokens** — the anchor, A. Every other load n is judged on what changed since A,
not against zero. What A already holds cancels out: overhead the driver keeps in
the process's shared memory, and any buffer that never lands in VRAM at that
placement.

One layer, never zero: the process's shared memory jumps by about a gigabyte
with the first layer on the GPU (277 → 1,272 MiB on the reference machine below)
while dedicated VRAM takes the whole claim — overhead no later load should be
charged with.

The anchor is this probe's own. The same placement has spilled 828 MiB in one
session and 189 MiB in another, so a baseline from another probe is not one.
Anchor loads are never taken from a batch sibling.

### Two accounts of the change

For each reading of n — dedicated and shared come from one sample of both
counters, so they describe the same moment:

| account | formula | moves when |
|---|---|---|
| **s** — shared grew | shared(n) − shared(A) | memory went to system RAM, *or* the driver's own overhead grew |
| **d** — claim not taken by VRAM | (claim(n) − claim(A)) − (dedicated(n) − dedicated(A)) | memory went to system RAM, *or* a buffer that sat in system RAM at A moved into VRAM (negative) |

Claim is llama.cpp's own report of every GPU buffer it allocated. Real spill
moves both accounts by the same amount; each of the other causes moves only
one. So:

```
spill     = min(s, d)                         (d alone where there is no shared counter)
tolerance = |s − d|                           how far the two accounts disagree
          + range of s and d in the phase     how far this load's readings moved
          + movement of A's readings          how far the anchor's readings moved
          + |A2 − A1|                         how far the anchor's two loads differed
spilled   = spill > tolerance
```

No constant appears anywhere: every term is measured on this probe.

### Phases

| phase | value used | why |
|---|---|---|
| ready hold (3.5 s, no request) | the last reading | a driver still paging buffers in right after load must not read as spill |
| prompt + generation | the median of paired readings | a jump for a minority of the phase widens the tolerance instead of deciding it |

Each phase of n is compared with the same phase of A. The worse phase decides.
Peaks are never used, and neither is a mean: at 20 layers dedicated VRAM held
5,318 MiB for three quarters of generation and 5,386 MiB for the rest — the
median reads what the load held, the mean (5,335) a value it never held, the
peak one moment.

### Measured

Qwen3.8-27B-UD-Q5_K_M on the reference machine (RX 6600 XT, Vulkan, llama.cpp
b11009), 1,024 tokens, MiB. Live check of the rule itself, 17 Sep 2026:

| load | claim | ready: s / d / tolerance | work: s / d / tolerance | verdict |
|---|---|---|---|---|
| A1 — 1 layer | 1,233 | 0 / 0 / 0 | 0 / 0 / 0 | anchor |
| A2 — 1 layer | 1,233 | 0 / 0 / 0 | 0 / 0 / 2 | anchor; loads differed by 0 |
| 4 layers | 1,841 | 0 / −3 / 3 | 0 / −4 / 5 | clean |
| 9 layers | 3,330 | +139 / +131 / 9 | +145 / +129 / 20 | spilled |

From the probe run of 16 Sep 2026 (whole-load peaks for shared, which the rule
no longer uses, but the steps agree):

| step | claim grew | dedicated grew | d | s |
|---|---|---|---|---|
| 0 → 1 layers | 995 | 1,003 | −8 | +995 (overhead, not spill) |
| 9 → 20 | 3,034 | 2,338 | 696 | 716 |
| 20 → 23 | 779 | 94 | 685 | 691 |
| 262,144 tok, 5 layers vs the 1k anchor | 3,090 | 3,342 | −252 | +20 (compute buffer moved *into* VRAM) |

### What spilled

A spill larger than what this context adds in KV cache and compute buffer over
the anchor means the weights are affected: the load is a layer failure, and no
smaller context is tried at that placement. So is any spill at the smallest
context, where there is nothing smaller to try. A smaller spill anywhere else
is a cache failure, and the context walk bisects down.

### A spill that does not change is not counted

A buffer that sits in system RAM at the anchor and stays there is part of the
baseline. On b11009 that is the whole 238 MiB compute buffer at 1,024 tokens:
held against zero it failed every placement at that context, 0 layers
included, and with it the whole Wizard curve. The anchor rule deliberately
does not see it. A cache spill that *grows* with context still shows, because
every context is measured against the one 1k anchor.

### What it replaced

**Against zero (14–16 Sep 2026).** llama.cpp's buffers minus dedicated VRAM,
failing past the counter's own movement. Calibrated on b10956 with
Qwen3.6-35B-A3B, where a clean load read at or just below zero:

| load | GPU buffers (MiB) | in VRAM | difference |
|---|---|---|---|
| 1,024 tok, 0 layers | 275 | 280 | −5 (clean) |
| 1,024 tok, 4 layers | 2,002 | 2,017 | −16 (clean) |
| 1,024 tok, 8 layers | 3,635 | 2,806 | 828 |
| 1,024 tok, 15 layers | 6,462 | 4,623 | 1,839 |
| 262,144 tok, 0 layers | 1,069 | 822 | 247 |
| 262,144 tok, 4 layers | 3,320 | 3,085 | 234 |

It assumed a clean load reads zero. On b11009 with Qwen3.8-27B it read +236 at
0 layers and +228 at 1, 2 and 4, and failed the whole probe.

**Slopes (before 14 Sep).** System RAM appearing per layer or per context step
against a reference rung, a bootstrap ratio without one, then a 768 MiB cache
allowance. It was systematically late: the reference was itself already
spilling, and a slope only sees growth above it. The anchor rule answers that
failure with an anchor that has almost nothing on the GPU, and by requiring
two independent accounts to agree.

### Settled along the way

- The driver does not wait for VRAM to run out. 9 layers put ~130 MiB in
  system RAM with about 3 GB of the card still free, so "was the GPU full" cannot be
  a precondition.
- `llama-server --list-devices` free is not a fit check on Windows: it read
  7,378 MiB — 90% of the card — on every load, while other processes held about
  2 GB and dedicated VRAM never passed ~5.7 GB. It stays the claim target (§10)
  by design, not as a spill test.

### Known gaps

- **An anchor that already spills** cannot be seen directly. It would show as
  the next load's s and d both growing by nearly its whole extra claim.
- **Windows CUDA** is unconfirmed: the CUDA context may land in dedicated VRAM
  in ways a 1-layer anchor does not represent.
- **No shared counter** (nvidia-smi): d alone decides, with no disagreement term.
- **No per-process dedicated counter** (Metal), or no buffer report: the load is
  unmeasured and passes on the estimate-based inference, which only warns.
  Nothing falls back to a difference against zero.
- **Shared-counter noise** has been measured on one card: 0 MiB between the two
  anchor loads and within steady phases.

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

## 9. The frontier — what the Wizard actually runs

> Superseded by §10 — kept for the reasoning. The search described here no longer runs.

§4 resolves one corner: the most layers at the cheapest context, then the
largest context at that placement. That is a real answer to a question nobody
asked. The question users trade against is *what does context cost in layers* —
and 16 layers at 16k tokens and 12 layers at 128k are both true of one machine.

The `frontier` mode answers that, as one layer boundary per context stop
(`nextFrontierRung` / `resolveFrontier`). It costs no new budget, no new table
and no new endpoint: the rungs in `probe_attempts` are the measurement, and the
curve is derived from them by the same function the search uses.

**What makes it affordable.** At a fixed layer count a larger context needs
strictly more memory, so a pass carries DOWN the context axis and a failure
carries UP it. Two stops that resolve to the same layer count therefore settle
every stop between them with no load at all — on a model whose KV cache is
cheap the whole curve costs the two ends, which is what max_gpu already spent.

**Order, and why.**

| step | why that order |
|---|---|
| Floor stop first | Cheapest loads, and the only verdict uncontaminated by cache (§5). Its answer bounds every other stop from above. |
| Ceiling stop next | Together with the floor it brackets the curve; agreement proves it flat for free. |
| Then the widest remaining step | Each remaining load goes where the curve actually bends. A span whose ends already agree needs none. |

**No reference loads.** Every rung measures its own placement (§7), so no load
is spent buying a smaller-context twin for another rung to be compared against.
An earlier version tested every candidate at the floor first, doubling the
loads per candidate, because the slope-based rule it served needed one.

**What the estimate is allowed to do.** Pick the seed, inside the bracket the
neighbours imply — nothing more. `estimateSafeNgl` does not vary with context
at all, so an unbounded target proposed 20 layers at a context where 13 had
just failed; bounding it is what keeps the walk from wandering.

**Honesty of the output.** Every stop carries how it was decided — `measured`,
`implied`, or `unmeasured` when the budget ran out — and a stop resting on a
pass whose memory placement was never measured is flagged `unverified` rather than
presented as a boundary. Rates shown beside a stop are empty-cache figures
(§1), and are labelled as such in the UI.

---

## 10. Context tests v2 — two targets, one context at a time

The Wizard (`frontier`) and Targets (`keep_context`, `fixed_offload`) find
two answers from the same loads (`shared/probeLadder.ts`, `probeOutcome`):

| target | rule |
|---|---|
| 2. Claim fits free VRAM | the most layers whose llama.cpp GPU claim is below what `llama-server --list-devices` reported free on the probe's devices, read once before the first load |
| 1. No spill | the most layers whose GPU memory all stayed in VRAM (§7), at or below target 2 |

### Order

Every mode starts with the **anchor** (§7): two loads at 1 layer and 1,024
tokens. They are the smallest context's first load and control at 1 layer, so
no search loads that point again. A failed anchor ends the probe; one whose
claim does not fit free VRAM means nothing fits at any placement.

The Wizard then settles each context stop before the next, smallest first:

1. **Target 2 at this stop.** A load that fits carries down the context axis
   and one that does not carries up, because a claim only grows with context.
   The anchor's claim is already known, so the first load at the smallest stop
   is the prediction from it, which errs low — never every layer. From two
   claims the per-layer claim predicts the boundary, and the search loads the
   prediction and its neighbour.
2. **If nothing fits here, the probe ends** — nothing fits at any larger stop.
3. **The control of the previous stop's no-spill answer**: that placement is
   loaded a second time. It runs here, after this stop's target 2, so other
   loads sit between the two, and before this stop's target 1, so this stop's
   ceiling is never an unconfirmed answer. A control that spills overturns the
   answer (a point is clean only when every load of it was), the search resumes
   at the previous stop, and the new answer gets its own control.
4. **Target 1 at this stop**, capped by this stop's target 2 and by the
   previous stop's no-spill answer.

keep_context runs steps 1 and 4 and the control on its one context.
fixed_offload runs the same rules over the context stops at its pinned layers,
with "two stops above" in place of "two layers above". custom is one load.

### Target 1: where spill starts, then two above

Spill does not grow steadily with layers or with context. On the RX 6600 XT
(`docs/research/spill-tax-dataset.json`) 5 layers spilled 159 MiB and 6
spilled nothing, at four contexts; 8 layers spilled at 1k–8k and not at 16k.
So no spill verdict is carried between contexts, and the search:

1. bisects between the highest clean and the lowest spilling layer count until
   they are adjacent — where spill starts;
2. loads the two layer counts above the first spilling one;
3. repeats from any of those that is clean.

A clean island three or more layers above where spill starts is missed on
purpose. The cap from the previous stop is the other deliberate cost: 16k never
tries 8 layers after 8k settled on 6.

### Judging a claim device by device

A load fits only if every device llama.cpp put buffers on claimed less than
that device's own free memory, as `--list-devices` reported it. The devices are
the `-mg` one alone when the test pins a GPU, and every listed device otherwise.
Comparing totals would let one device's spare room hide another's overflow — most
often an integrated GPU, whose large "free" is mostly shared system RAM, masking
a discrete card that is already full. The verdict is stored per load
(`claim_fits_free`); rows from older workers fall back to comparing totals.

### Loads that say nothing about memory

A load that never becomes ready in time, crashes without an out-of-memory
signature, or has its request rejected is stored with `load_kind = error`. It
is never offered again. Target 2 draws no fit or miss from it, so it cannot drag
a boundary down. Target 1 still counts it as not clean, since nothing clean was
seen there. An out-of-memory failure is a memory verdict: it did not fit.

### Loads stopped at ready

When the server reports ready, its claim is known. A claim not below the free
figure can be neither answer, so the load stops there — no prompt, no
generation, no spill — and is stored with `load_kind = claim_stop`.

A Wizard load searching target 2 can also be one no spill verdict could use: its
layer count is above the no-spill answer at the context below (or that stop has
none), and target 1 here never tries more layers than that — a control can only
lower it. `probeOutcome` marks such a load `nextClaimOnly`; once its claim fits
it stops at ready the same way and is stored with `load_kind = claim_only`.
Batch dedup never reuses a `claim_only` row for a load that has to judge spill.

### Spill by phase

The worker reads the process's dedicated VRAM and shared GPU memory about once a
second for the whole load, both in one sample (`worker/src/memoryTrace.ts`: one
PowerShell session on Windows running one `Get-Counter` over both counters,
because a fresh session costs ~5.5 s and `Get-Counter` itself takes ~1 s per
sample; on Linux, one reading at a time, so a slow `nvidia-smi` never overlaps
itself). The ready hold uses only readings taken before the first request. A
full load holds 3.5 s after ready before its request, then
`measureGrowthSpill` judges each phase against the same phase of the anchor
(§7). Every paired reading goes to the run's log, as `dedicated/shared` per
second, followed by s, d, spill and tolerance for each phase.

### Budget

The default is 40 loads (admin-configurable). The two anchor loads count
toward it. The real run on the RX 6600 XT
with Qwen3.6-35B-A3B needed 46 loads for the whole curve — 8 of them controls,
5 stopped at ready — so at 40 it stopped with 128k's control and all of 256k
still open. A stored budget of exactly the old default, 24, is raised to 40
once by the server's migration.

When the budget runs out, the stored ceiling (below) prefers the largest context
whose answer a control confirmed over a larger one that was only loaded once,
and the Wizard card names the context where the budget ran out.

A Targets probe that finds no no-spill answer ends as failed, but its card still
shows target 2 when there is one, and it can be applied.

### Deploying it

Deploy the server before the workers. A new worker refuses the removed modes
and a probe with no mode; a new server refuses to create them, and accepts the
new `probe_attempts` fields. An old worker on a new server keeps working with
its old search. After the server update, check the probe load budget in the
admin settings: the migration only raises a stored value of exactly 24.

A probe recorded before the anchor carries no `spill_method`; the client
re-resolves those rows without the anchor, exactly as they were searched.

A worker that authenticates with the shared deployment token cannot report
context-test results — those routes require an enrolled worker session — so an
end-to-end check needs an enrolled worker.

### Stored and shown

`model_machine_limits` records target 1 at the largest context whose answer a
control confirmed (an unconfirmed one only when none was).
Every `probe_attempts` row carries the claim, the `--list-devices` figure,
`load_kind`, the per-phase spill and tolerance (`spill_ready_*`,
`spill_work_*`), which rule judged it (`spill_method`: `anchor` or
`growth`), the worse phase's two accounts (`spill_shared_growth_mib`,
`spill_unlanded_growth_mib`) and the ladder's own bounds (`ladder_ngl_max`,
`ladder_max_ctx`), so the client re-resolves both answers with
`probeOutcome` itself. A load reused from a batch sibling keeps the verdict it
was given against that sibling's anchor. Without a `--list-devices` figure, target 2 is
skipped and target 1 is searched with every layer as its cap, opening at the
estimate.

---

## Reference run

5 probes, 36 rungs, 6 Sep 2026. Qwen3.6-35B-A3B-UD-IQ4_NL (41 claimable layers,
262,144 trained context) on a Radeon RX 6600 XT, 8 GiB, Vulkan, llama.cpp
b10819. Per-layer weight ~422 MiB, derived from the run's own reported slopes.

The zero-based measurements quoted in §7 were taken on the same card and model
with llama.cpp b10956 on 14 Sep 2026; all twelve are fixtures in
`shared/gpuSpill.test.ts`. The anchor-rule measurements are Qwen3.8-27B on the
same card with b11009, 16–17 Sep 2026, and are fixtures in the same file.
