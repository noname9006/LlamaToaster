# Benchmarking Plan v8 — User-Needs Capability Layer (Standalone Edition)

Status: **standalone**. This edition is buildable and reviewable on its own: §0 restates,
normatively, every definition this document consumes. It no longer requires
`BENCHMARKING_PLAN_V7.md` to be actionable. Where v7 exists in the tree it remains the
implementation plan for the core benchmarking phases; if that document and this one ever
diverge on anything §0 covers, **this document governs** for all Parts A and B scope.

Two parts:

* **Part A — Amendments M1–M7.** Adjustments to the core phases. Each names its host
  phase and is applied *as part of implementing that phase*, not after it.
* **Part B — Capabilities N1–N7.** New subsystems beyond the core's declared boundary.

Source material: `docs/pain1.md` (the seven questions users actually ask),
`docs/pain2.md` (the manual workflow this product replaces), `docs/pain3.md`
(methodology pitfalls).

Companion artifact: `docs/benchmark-page-mockup-v8.html` renders this document's UI
intent (Part A surfaces plus the benchmark-visible Part B items). It supersedes the
benchmark portions of `docs/site-mockup.html`; `docs/benchmark-page-mockup.html`
remains the v7 console reference.

---

## 0. Normative base

Everything in Parts A and B is defined in terms of the primitives below. They are
quoted here as specification, not by reference. Each entry notes its origin in the core
plan for traceability only.

### 0.1 Methodology version

```
results ADD method_version INTEGER   -- NULL = pre-v8-core methodology
shared/types.ts: export const METHOD_VERSION = 1;
```

`METHOD_VERSION` increments whenever measurement semantics change (warmup added,
TTFT derived→streamed, suspect filter applied to both sides of a pair, stddev floor).
It is stamped by the worker on every ingested row, surfaced wherever rows of different
vintages appear together, and rows of different `method_version` are never averaged
together in scoring or curves.

### 0.2 Engine/spec legality, and the depth rule

Legal engine pairs are enumerated, never derived:

```ts
type EngineKind = "bench" | "server";
type SpecMode   = "off" | "mtp" | "draft";
// ("bench","mtp") is not representable: llama-bench has no speculation.
export interface EnginePair { engine: EngineKind; spec: SpecMode }
export const LEGAL_ENGINE_PAIRS: readonly EnginePair[] = [
  { engine: "bench",  spec: "off"   },   // llama-bench, default path
  { engine: "server", spec: "off"   },   // llama-server without speculation
  { engine: "server", spec: "mtp"   },   // --spec-type draft-mtp
  { engine: "server", spec: "draft" },   // --spec-type draft-simple
];
```

**Depth rule:** `-d` prefills KV before each test and is llama-bench-only;
llama-server has no KV-prefill flag. Sweep validation therefore rejects
`n_depth > 0` for any pair with `engine === "server"` — a server row always carries
`n_depth = 0`, and its effective context is `n_prompt`. Part B's N1 is the one place
server-path context scaling is measured, and it works within this rule (see N1).

Cache-type strings across sweeps, estimates and validation come from the closed
nine-value set `{f32, bf16, f16, q8_0, q5_0, q5_1, q4_1, q4_0, iq4_nl}`. Rejection
copy for the depth rule names the fix:
*"n_depth is only supported on the llama-bench engine — llama-server has no KV-prefill
flag. Set n_depth to 0 for llama-server configurations."*

### 0.3 Scoring primitives

Runs of `kind='sweep'` only; single `method_version`; results carry one row per
`test_type ∈ {pp, tg}` per configuration tuple, and scoring requires both. Glossary
for everything the gates consume: a **suspect** reading is one the implausibility
filter rejects — rates physically impossible for the backend (the ~1e6 tok/s
timer-bug class) — or one timed only by wall-clock fallback; rows record
`suspect_count` out of `sample_count`.

* **Normalization.** Per metric `X̂ = X / P90(per-config medians of X)`, memory inverted.
  P90 of medians, not raw max — one repeat hit by a timer bug (~1e6 tok/s) must not pin
  the denominator.
* **Pressure.** `PRESSURE = max(VRAM_peak / VRAM_total, RAM_peak / RAM_total)` over
  non-null inputs. Degradation: VRAM+RAM available → as above; RAM only →
  `RAM_peak / RAM_total`; neither → PRESSURE undefined and only a speed card is emitted,
  with copy saying so.
* **TG reference depth.** TG compared at a pinned depth (default 50 %), nearest
  available otherwise; the depth used is stored per profile (`reference_depth`).
  Scoring tuples therefore group rows while ignoring `n_depth` — depth selects *which*
  tg row feeds the comparison, and every other §0.4 axis stays part of the tuple's
  identity.
* **Eligibility gates.** No `failed`/`failed_oom` item in the tuple's sub-run;
  `suspect_count` absent or 0 on every row; `stddev ≤ max(0.10 · mean, 0.5 tok/s)`
  (absolute floor stops near-zero means exploding the ratio); `n ≥ 3`. Rows flagged
  `swa` (§0.10) are excluded from the TG reference-depth comparison — depth past the
  window is structurally misleading — and tuples removed only by a flag tally as
  `caveat_flagged`. `skipped` items do not disqualify — they were never measured.
* **Rejection accounting.** Scoring emits tallies of which gate removed each tuple
  (`stability`, `suspect_samples`, `missing_pp_or_tg`, `caveat_flagged`). Zero-profile
  outcomes render the tallies, never silence. If every tuple failed only stability, an
  additional `unverified` Max Speed card may be emitted with that gate waived, visibly
  distinct and excluded from downstream defaults.

### 0.4 Canonical config hash

One function in shared code; server and worker agree with no reconciliation path.

```ts
export function configHash(item: ConfigHashInput): string;
```

Keys are sorted; every axis has an explicit default so adding an axis never changes an
existing configuration's hash. **Included:** `n_prompt, n_gen, n_depth, threads,
n_gpu_layers, batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn,
n_gpu_layers_draft, n_cpu_moe, engine, concurrency`. **Excluded:** `spec`,
`spec_type`, `spec_n_max`, `spec_n_min` — the exclusion is what makes twin joins work.

### 0.5 Run identity, kinds, grouping

* `runs.root_run_id`: denormalized at creation, immutable, points at the run itself for
  standalone runs. Chain-scoped reads are indexed equality predicates.
* `runs.kind` values: `'tuning' | 'refine' | 'sweep' | 'runtime'`, NULL = standalone.
  N2 adds `'probe'`.
* Duplicate-trigger guard: at trigger, a caller's non-terminal run matching
  `(user_id, model_id, worker_id)` with the same non-NULL kind yields 409; roots only
  (`root_run_id IS NULL OR root_run_id = id`); NULL-kind matches only NULL-kind. The
  409 body names the active root and kind:
  *"you already have a \<kind\> run for this model on that machine (\<root id\>).
  Stop it first or wait for it to finish."*
* Budgets: `MAX_SWEEP_ITEMS = 20_000` (hard 400), `WARN_SWEEP_ITEMS = 4_096`
  (201 with warning), `MAX_AXIS_VALUES = 64`, `repeats ∈ 1..25`; chain quotas:
  depth ≤ 3, ≤ 3 active roots per user, 48 h wall clock enforced continuously against
  running roots. Quota composition with Part B: probes are exempt from the
  active-roots quota (minutes-long by construction); a comparison group counts once,
  not once per member.

### 0.6 ETA pricing rule

Pre-flight and live ETAs price from, in order: (1) the most recent
`{engine:"server", spec:"off"}` pp rate for `(model_id, worker_id[, llama_cpp_build])`;
(2) the parent sweep's llama-bench pp for the same model, rendered with the mandatory
label *"derived from llama-bench"*; (3) neither → `ETA unavailable`. Generation time
prices symmetrically: server-measured tg first, then labeled bench-derived tg, then
that component renders unavailable. No cell renders a number without naming its source.

### 0.7 Flag-probe machinery

`supportsFlag(path, flag): Promise<boolean>` probes a binary's help output once per
binary *identity* — memo keyed `(path, mtimeMs, size)` so in-place llama.cpp updates
re-probe. An unsupported flag disables its axis (items become `skipped` with reason)
rather than failing every item. Used for server flags across Parts A/B and by N1/N4.

### 0.8 Twin join and speedup statuses

A speculative row joins its `{engine:"server", spec:"off"}` baseline on
`(root_run_id, config_hash, test_type, n_depth, method_version)`, where `config_hash`
excludes spec fields (§0.4). The columns this join consumes:

```
results ADD prompt_offset INTEGER                 -- pinned prompt-content offset
results ADD spec_type TEXT, spec_n_max INTEGER, spec_n_min INTEGER
results ADD config_hash TEXT                      -- §0.4 hash; excludes spec fields
results ADD speedup REAL, speedup_status TEXT     -- ok | unverified | unavailable
CREATE INDEX IF NOT EXISTS idx_results_config_hash ON results(config_hash);
```

Both sides must match on `prompt_offset` — the pinned
prompt-content offset each server row records, since acceptance rate and TG depend on
the exact token sequence — and on method
version. `speedup_status ∈ ok | unverified | unavailable`; any suspect-only side,
wall-clock fallback, zero drafted/accepted counters, offset mismatch, or version
mismatch ⇒ `unverified`. An offset mismatch additionally writes the
`spec_pair_prompt_mismatch` flag (§0.10). Acceptance rate = accepted/drafted guarded
at 0.

### 0.9 Privacy floor

Community aggregates require `HAVING COUNT(DISTINCT user_id) >= 5`. Contributions from
fewer distinct users stay private-by-default. N7 extends this floor to curves and
comparisons.

### 0.10 Caveat flags

```
results ADD caveat_flags TEXT      -- JSON string[]
```

Closed registry, each rendered with its reason: `swa` (depth exceeds the sliding
window), `context_unverified`, `kv_estimate_rough`, `spec_pair_prompt_mismatch`,
`thermally_throttled` (M6), `cache_evicted` and `context_shift` (N1). Flagged rows are
kept, never deleted; consumers decide exclusions — e.g. §0.3's TG reference-depth rule
for `swa`.

### 0.11 Schema-evolution conventions

Every `ADD COLUMN` in this document lands in the ordered column-migration list
applied at boot; every new table and its indexes land in a `createXxx(database)`
helper invoked after the column migrations; and any new FK column defaults to NULL
(SQLite's requirement with foreign keys on). Parts A and B follow this split
throughout.

### 0.12 Observability conventions

Every Part A/B subsystem emits structured events on the existing request-scoped child
logger — counters with fields, never free-text lines; correlation rides the existing
request-id plumbing; no PII beyond what rows already store. Minimum set:
`probe_finished {outcome, loads_used}`, `quality_job_finished {outcome}`,
`curve_served {points, cache_hit}`, `thermally_flagged_ratio` (once per completed
run, driving N6's >⅓ threshold visibility), `comparison_member_failed
{member_run_id, reason}` (fairness drift is a user-facing FATAL and exactly what gets
debugged after the fact), and `import_rejected {reason, row_id}`.

---

## Pain crosswalk

| # | Pain (pain1 unless noted) | Answered in | How |
|---|---|---|---|
| 1 | How fast is it *really*, on my machine? | Core Test B | llama-server runtime runs, streamed TTFT/E2E, priced ETAs (§0.6) |
| 2 | Memory: peaks, KV at 8k→128k | M1 + N1/N2 | Inverse affordability estimate; curves and verified limits |
| 3 | Optimal settings — optimal *for what?* | Core Test A + M2/M3 | chained tune→refine→sweep ranked by stated goal |
| 4 | Speed-vs-quality across weight quants | N3+N4 | model-vs-model comparison + perplexity subsystem |
| 5 | What happens as context grows? | N1 | context-scaling curves, both engines |
| 6 | Largest *usable* configuration | M1 + N2 | estimate, then boundary verification |
| 7 | Reproducibility | §0 + N7 | methodology plumbing; exchange format |
| — | Concurrency knee (pain3 #4) | N5 | load driver + knee finder |
| — | Thermal/sustained state (pain3 #7) | M6 + N6 | sensor telemetry; policy |
| — | CPU ISA provenance (pain3 #6) | N6 | banner-derived ISA + row provenance |

---

# Part A — Amendments to the core phases

Each amendment is applied while implementing its host phase. None grows a core phase
into a new subsystem.

## M1 — Affordability inversion: `maxAffordableContext`

**Host:** the KV memory estimator phase — new subsection immediately after it.

The forward estimator runs config → KV bytes. Pain #6 asks the inverse: *given this
GPU, these weights, this KV type — what is the largest context that still fits and
stays practical?*

`shared/vramEstimate.ts` gains:

```ts
export interface MaxCtxInput {
  totalMib: number;                  // VRAM total (RAM total for CPU workers)
  weightsMib: number | null;         // VRAM-resident weight share for the chosen
                                     // placement (ngl / n-cpu-moe); null when only a
                                     // full-offload number exists
  scratchMib?: number;               // graphs + driver scratch; default 256
  activationsHeadroomFrac?: number;  // default 0.10
  parallelSlots?: number;            // default 1; KV splits across slots
  // Per-token KV inputs, identical fields to the forward estimator:
  nLayer: number; nHeadKv: number; headDimK?: number; headDimV?: number;
  nEmbd?: number; nHead?: number;    // consumed only by the dK/dV fallback below
  cacheTypeK: string; cacheTypeV: string;
  slidingWindow?: number;
}
export interface MaxCtxEstimate {
  tokens: number;
  confidence: "good" | "rough" | "unknown";  // same words, same table as forward
  binding: "kv" | "weights-placement" | null;
}
```

Formula (all integers floored at the end):

```
usableMib     = totalMib · (1 − activationsHeadroomFrac) − weightsMib − scratchMib
kvBytesPerTok = nLayer · nHeadKv · (dK·bytesPerVal(K) + dV·bytesPerVal(V))
tokens        = usableMib · 2^20 / (kvBytesPerTok · max(1, parallelSlots))
```

where `dK = headDimK ?? nEmbd/nHead` and `dV = headDimV ?? headDimK ?? dK` — the
fallback consumes the optional inputs (`nEmbd` and `nHead` are required together for
it; absent ⇒ confidence drops a level), exactly as the forward estimator derives them.

* **Binding semantics.** `weights-placement` when
  `weightsMib + scratchMib > totalMib · (1 − activationsHeadroomFrac)` — the model
  barely seats before any KV exists; `kv` otherwise; `null` when confidence is
  `unknown`. Max Context's peak-sum tie-break likewise degrades gracefully: with no
  memory totals it is omitted and order stays stable by estimated tokens descending.
* **SWA handling — direction matters because this is the inverse.** The forward
  estimator's naive all-layers-full-context error *overestimates cost*; here the same
  naivety would *overestimate affordable tokens*, which is the unsafe direction. When
  `slidingWindow` is present, assume the Gemma-style ~1 global : 5 local interleave.
  With `g = ceil(nLayer/6)` global layers, local layers contribute their window-bound
  bytes and the global budget is solved:

```
tokens = max(0, usableMib·2^20 / kvBytesPerTok − (nLayer − g) · sw) / g
```

  Confidence stays `rough` and the note states the assumed ratio.
* MTP companion under-count inherited from the forward table (`rough`).
* `weightsMib === null` → interpolate from any prior run's
  `vram_free_before_mib/gpu_memory_model_peak` pair for this model+machine when one
  exists; else return confidence `unknown` with the full-`trained_ctx` load as a
  conservative floor candidate. Never invent a weights number.
* **Advisory, never gating** — same rule as the forward estimator. It ranks and
  annotates; it does not silently exclude configs from sweeps or profiles.

Consumers: profile-card fit lines (**M3**), questionnaire feasibility readout (**M2**),
Test B pre-commit cell flagging, N2's probe candidate selection.

**Exit (worked example, all inputs named).** Machine: RX 6600 XT,
`totalMib = 8192`, slots 1. Model: Qwen3-30B-A3B Q4_K_M — 48 layers, `n_head_kv = 4`,
`head_dim_k = head_dim_v = 128`, no sliding window, MoE placement resident share
`weightsMib = 2969`, `scratchMib = 256`, headroom 10 %. Then
`usable = 8192·0.9 − 2969 − 256 ≈ 4148 MiB`;
f16/f16: `kvBytesPerTok = 48·4·(128+128)·2 = 98 304 B (96 KiB)` → **≈ 44 200 tokens**;
q8_0/q8_0: `48·4·(136+136) = 52 224 B (51 KiB)` → **≈ 83 300 tokens**; confidence
`good`; binding `kv`. Unknown-metadata models get confidence `unknown`, never a
fabricated number.

## M2 — Goal questionnaire

**Host:** Benchmark page UI phase — new section before its information-architecture
section. Three visible questions + one collapsed optional:

**Q1 · What matters most?** — `Balanced` (default) | `Max tok/s` | `Max context`.
Selecting `Max context` reveals a **usable-speed floor** (chips 40/50/60 %, default
≥ 50 % of best measured TG) — without it, argmax-context degenerates into "quantize
everything and crawl".

**Q2 · Target context?** — token input; default `min(32k, trained_ctx)`;
"don't know" allowed (`target_unverified` marker shown instead of a fit line).
Clamped by `trained_ctx` inline (typing 65 536 against a 32 768-cap shows the clamp
immediately). Live feasibility via **M1**: *"your card affords ~44 k at f16/f16,
~83 k at q8_0/q8_0"*. Anchors depth-axis copy and Test B size pre-selection.

**Q3 · Workload shape?** — `Chat (generation-heavy)` | `Feed documents (prompt-heavy)`
| `Even` (default). Sets `(wPP, wTG)`: chat `0.25/0.75`, docs `0.7/0.3`,
even `0.5/0.5`.

Optional, collapsed: **KV quality tolerance** — `q4_0 ok` (default) | `q8_0 ok` |
`f16 only`. Prunes forbidden pairs from the grid before expansion (**M4**) — fewer
tests, cheaper chain, and Low Memory can no longer recommend a cache quality the user
opted out of.

Rules:

* **Skippable.** "Just optimize" accepts every default = exactly the previous
  behavior, bit-for-bit (see the pinned reduction in M3).
* **Answers are intent, stored as configuration**: root run config under a `goals` key
  — stored verbatim in the run's own config, so results stay reproducible without
  reference to whatever the defaults were that day.
* **Re-scoring is pure post-processing.** Changing goals re-runs scoring over stored
  results — never re-measurement. Completion CTA shows
  *"scored for: Max context · 32 k · chat"* with a change-goal affordance.
* Answers appear in stage editors as *Held* fields with reasons ("depth % of your
  clamped 32 k target", "KV axis pruned: f16 only").
* Presets carry the block too (**M5**).

**Exit.** Trigger accepts a `goals` block and echoes it on the created root; a
skipped questionnaire produces a trigger payload byte-identical to today's.

## M3 — Goal-parameterized profile scoring

**Host:** replaces the fixed-weight profile table and its closing "weights stay fixed"
sentence in the scoring appendix. Self-contained; uses §0.3 primitives.

Let `(wPP, wTG)` be Q3's weights and `S = wPP·P̂P + wTG·T̂G` the combined speed score.
Let `FIT = min(1, maxAffordableContext(M1).tokens / target_ctx)` when both the estimate
is computable (`confidence ≠ unknown`) and a target was stated; else `FIT = null`.

| Profile | Shown when | Rule |
|---|---|---|
| Max Speed | goal `Max tok/s` or `Balanced` | `argmax S` |
| Balanced | goal `Balanced` or default skip | `argmax( (0.4 + f)·S + 0.2·(1 − PRESSURE) )`, where `f = 0.4·FIT` when FIT computable, else `f = 0.4` (redistributed to the speed coefficient) |
| Max Context | goal `Max context` only | among configs with `T̂G ≥ speed-floor` (M2): `argmax estimated affordable tokens (M1)`, tie-break lower `VRAM_peak + RAM_peak` |
| Low Memory | whenever PRESSURE is computable for ≥ 1 config; on machines reporting no memory totals the §0.3 degradation applies (speed card + stated reason, no silent collapse) | unchanged from §0.3 eligibility: among `T̂G ≥ 0.9 and P̂P ≥ 0.8`: `argmin PRESSURE`, tie-break lower peak-sum |

**Pinned reduction.** With defaults (`Even`, FIT null) Balanced computes
`argmax(0.8·S + 0.2·(1 − PRESSURE))` = `argmax(0.4·P̂P + 0.4·T̂G + 0.2·(1−PRESSURE))` —
byte-for-byte the previous behavior, which is what "skippable" means.

Every card carries a **fit line** from M1: *"affords your 32 k target: yes, ~1.9×
headroom"* / *"tops out near 44 k — below your target"*, with the confidence word. The
fit line annotates; it reorders nothing except the Max Context card, whose goal asked
for estimates as the criterion — and whose copy says so explicitly ("ranked on
estimated fit, not measured at your target — verify with a probe"). Card membership
follows the goal exactly as the Shown-when column states; hidden cards are named, not
silently absent, and switching goals re-scores instantly without re-measurement.

Weights are user intent, not physics — the defaults hold until usage data says
otherwise, and any combination a user picks is honored verbatim.

**Exit.** Scoring fixture: `Even` + no target reproduces the fixed-weight ranking
exactly (the pinned reduction); stating a target reorders cards only through FIT; Max
Context honors the floor and its copy names the estimate basis.

## M4 — Quality-tolerance pruning of the KV axis

**Host:** appended after the sweep-stage KV-default-set appendix.

M2's optional tolerance removes forbidden pairs at grid-build time — before expansion,
so live count, cost estimate, and invalid-combination breakdown all reflect it. One
inherited rule is inviolable: whatever the tolerance, one unquantized pair survives so
the flash-attention-off axis keeps something to vary against.

* Recommended grids collapse accordingly (`q8_0 ok` drops `q8_0/q4_0`; `f16 only`
  leaves `f16/f16` alone), and the stage editor says the axis shrank
  ("KV fixed: f16/f16 — sweep is now FA × ngl × depth shaped") rather than letting the
  count imply it.
* Expert mode prunes quantized sides but retains other *unquantized* pairs (f32, bf16)
  — the tolerance speaks to quantization, not to width. Only the recommended set
  collapses to the single f16/f16 value.

**Exit.** Grid-builder tests: each tolerance prunes exactly its forbidden pairs; ≥ 1
unquantized pair always survives; live count equals post-prune expansion.

## M5 — Goals travel with presets

**Host:** preset section of the Benchmark-page UI phase — exception paragraph directly
after "grids only".

M2's answers (goal, target context, workload shape, KV tolerance) are saved and loaded
with presets, inside config under the `goals` key. They are intent about the
*workload*, not properties of a machine — the same reasoning that keeps grids portable.
Loading applies them to the questionnaire controls verbatim, marked like every other
loaded value. **Legacy presets** saved before this amendment lack the key: loading one
treats `goals` as unset (questionnaire shows defaults, marked *unset*, never fabricated
answers); saving afterwards writes the block.

**Exit.** Preset round-trip restores the goals block verbatim; a pre-goals preset
loads with the questionnaire controls marked *unset*.

## M6 — Thermal/power telemetry

**Host:** runtime-results columns block and caveat-flag list of the streaming phase.

### Schema

```
results ADD gpu_temp_c_max INTEGER      -- adapter peak °C; NULL where no sensor
results ADD gpu_clock_mhz_min INTEGER   -- lowest sampled sclk (MHz)
results ADD gpu_clock_samples TEXT      -- JSON number[] MHz, sampled on the VRAM cadence (~6 s)
```

Aggregates alone cannot detect a trend — min/max cannot see "sagged between halves" —
so the sample series is part of the spec, not a nice-to-have. At ~6 s sampling a
30-minute item yields ~300 samples (~2 KB JSON).

### Detection rule (computable, deterministic)

Eligible when the item has **≥ 4 repeats** and ≥ 4 clock samples (each half needs ≥ 2
samples). An odd sample count puts the extras in the first half (floor split) —
stated so the rule stays exactly reproducible. Split the sample array chronologically
in half — with uniform cadence this
approximates repeat halves. The window is the timed work, not the spawn: server items
bracket sampling from the first timed request through process exit (model-load idle
clocks would otherwise mask or fabricate sag); llama-bench items approximate with the
whole item, since repeats are binary-internal. Flag
`thermally_throttled` when `mean(secondHalf) ≤ 0.95 · mean(firstHalf)` **and**
`gpu_temp_c_max IS NOT NULL`. The flagged row is **kept, not failed** — burst-vs-
sustained is visible data; policy lands in N6. Items with < 4 repeats are simply
ineligible for the flag (stated, not silent).

**Writer ownership.** The worker computes the flag, `gpu_clock_mhz_min` and
`gpu_temp_c_max` at item end from its own sample buffer and reports them with the
result; the server stores them verbatim and never re-derives. One writer per column —
mirroring how every other worker-derived field in this plan behaves.

### Source of truth — vendor-split, mirroring the VRAM reader's dispatch

There is no cross-platform GPU sensor source, and `systeminformation`'s own fields were
already proven unusable on the reference box for memory. Sources per platform/vendor:

| Platform / vendor | Source | Notes |
|---|---|---|
| NVIDIA, any OS | `nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits` | Same binary and exec pattern as the existing CUDA VRAM path; ~200–500 ms/exec — acceptable at VRAM cadence |
| **ROCm, Linux** | `rocm-smi --showtemp --showclocks --json` (parse JSON; fall back to text parsing) | The canonical ROCm CLI shipped with the stack; adapter-level sclk + temp. Second fallback: the same amdgpu hwmon attrs below, which remain valid under ROCm kernels |
| AMD Linux, generic/vulkan (stock kernel driver) | sysfs hwmon: `/sys/class/drm/card*/device/hwmon/*/temp1_input` (millidegrees °C) and `freq1_input` (Hz) | Free, no install; documented in the kernel's amdgpu thermal page — the same interface class the VRAM reader already cites |
| AMD Windows (vulkan / win-rocm) | **No vendor-agnostic source.** WMI thermal zones are motherboard-level; PDH has no sensor counters. Options: LibreHardwareMonitorLib (new dependency) or ADLX SDK | v1 ships NULL with availability declared; library choice deferred to implementation (open question) |
| Apple metal | NULL in v1 (`powermetrics` needs sudo) | Declared unavailable |
| Intel / other | NULL in v1 unless LHM adopted | Declared unavailable |

Provenance: extend the measurement-source enum with `sensor_nvidia_smi`,
`sensor_rocm_smi`, `sensor_amdgpu_hwmon`, `sensor_lhm`; track worst-source across an
item's samples exactly like memory streams do. Heartbeat reports sensor availability
per backend so machine cards can declare "clock · temp available" up front — a later
flag never surprises a machine that could never produce one.

Sampling piggybacks the existing VRAM tick (every third 2 s sampler tick); sysfs reads
are free, nvidia-smi/rocm-smi execs fit the existing timeout budget.

Retention: the sample series is this plan's first bulky column (~2 KB per item). It
may be pruned 30 days after ingest — detection already ran at ingest time, and
`gpu_clock_mhz_min`, `gpu_temp_c_max` and the caveat flag persist indefinitely, which
is everything later readers (scoring, N6 policy) consume.

**Exit.** Sampler fixture: a sagging second half sets the flag; < 4 repeats or
< 4 samples never flags; a sensorless backend writes NULL columns and declares its
availability on the heartbeat.

## M7 — Threads held with a stated reason

**Host:** doc-level change to the Tune-coarse panel's Held list.

`threads (machine default)` becomes held-with-reason: Test A targets fully offloaded
configs where `-t` barely bites; on CPU-bound rows it is the dominant variable and
remains an Expert-mode axis, with ISA provenance handled by N6. Copy-only — verified
in the stage-editor review, no separate exit.

---

# Part B — Capabilities beyond the core

Everything here consumes §0 plus Part A. Boundary: measure speed and memory always,
quality only as a labeled synthetic proxy.

## N1 — Context curves as a first-class deliverable

**Goal.** Answer pain #5 with an artifact: for a given (model, machine, build,
engine), a curve of `context → pp, tg, ttft, e2e, vram_peak`, renderable on the model
page and exportable.

**Why it is not just a view.** The llama-bench half falls out of the depth axis almost
for free (§0.2: bench rows carry `n_depth`). The server half does not exist, and it
must work inside §0.2's rule: server rows keep `n_depth = 0`; effective context is
`n_prompt`, and curve x-values are `n_depth + n_prompt`.

**Measurement choreography per point (server path).** Three request classes, never
averaged together:

1. **Warm/discard** — tiny `n_predict: 8` request on a short **nonce prompt distinct
   from the measured one**, so it absorbs CUDA-graph capture / pipeline compile
   without seeding the measured prefix into the cache; excluded from statistics by
   construction.
2. **Cold timed prefill — one per point.** Full prompt, `stream: true`,
   `n_predict: 1`, `ignore_eos`. First-chunk arrival is the TTFT data point for this
   context size. This is the measurement the old "untimed prefill" sketch would have
   destroyed: prefill growth is the curve's spine, so the prefill request is *timed
   data*, not setup. TTFT is therefore single-sample per point (`ttft_n = 1`) and the
   UI labels it as such (no p50/p95 pretense on cold columns). The same response's
   `timings.prompt_ms ÷ prompt_n` is the point's **pp** value — this is how the curve
   keeps a pp column while §0.2 keeps `n_depth = 0` on server rows.
3. **Warm repeats × (repeats − 1)** — identical prompt with `cache_prompt: true`;
   generation/E2E stats come from these. Because the prefix is cached, a warm repeat
   should report `timings.prompt_n == 0`.

**Eviction detector (concrete).** Any **warm repeat** (class 3) whose response reports
`timings.prompt_n > 0` re-prefilled — the cache did not hold. (The cold request of
class 2 legitimately reports the full prompt; it is never evaluated by this detector.)
Flag the row
`cache_evicted` in `caveat_flags`, exclude it from tg statistics, drop it from the
curve with an explanatory tooltip. Silent fast-repeats are exactly the failure mode
this detector exists to prevent.

**Context shift.** Probe `--no-context-shift` via §0.7; if unsupported and logs show a
shift event, flag the row — shifted contexts silently corrupt TTFT comparability.

**Slot accounting.** `-c = slots × (n_depth + n_prompt + n_gen + margin)` — per-slot
demand is the full prompt + generation budget; at `parallel 1` this equals the
existing sizing.

**Sizing ladder.** Points at `{2k, 4k, 8k, 16k, 32k, 64k, 128k}`, clamped to both
`trained_ctx` and M1 affordability; points beyond either render as
"unavailable — reason", never silent gaps.

**Artifact & storage.** No new table: a curve is a deterministic grouping over
`results` (rows sharing `(model_id, worker_id, llama_cpp_build, engine,
method_version)`, ordered by effective context). Choreographed points stamp
`METHOD_VERSION = 2` — cold-timed prefill plus warm-repeat statistics is a real
semantics change under §0.1 — which is what keeps ordinary runtime rows' warm-biased
TTFT out of curves without a dedicated marker column. Serve at
`GET /api/models/:id/curve?worker&build&engine`; compute on read, cache client-side.
The endpoint serves the caller's own rows; other tenants' contributions surface only
behind the §0.9 floor. Read path: an
`idx_results_curve` index on exactly the five grouping keys (§0.11 helper) keeps
compute-on-read an index range scan rather than a table scan. When two choreographed
points share an effective context (a re-measure of the same cell), the greatest
`(created_at, id)` wins — a total order, since second-resolution timestamps can
collide — and the superseded point renders greyed, never averaged.
Materialize later only if profiling demands it.

**UI.** Model page Curves tab: PP/TG/TTFT vs ctx with caveat shading, the user's M2
target drawn as a vertical marker. Curve CSV export.

**Trigger.** The bench half falls out of depth-axis sweeps; the server half needs a
home. The Curves tab exposes **"Measure missing points"**: it prices uncovered ladder
cells under §0.6, then enqueues one runtime-kind run through the ordinary trigger
route (so every §0.5 guard applies unchanged) whose items execute exactly the
choreography above (and therefore stamp `METHOD_VERSION = 2`). Cells already covered
render as measured; nothing auto-runs.

**Exit.** Curves render for a model with ≥ 3 measured points on both engines; nothing
beyond `trained_ctx` appears as measured; a cache-evicted row carries its flag and
drops out with a tooltip; cold-TTFT points are visibly labeled single-shot.

## N2 — Largest-usable-config advisor (estimate → verify)

**Goal.** Turn M1's estimate into a verified number, closing pain #6's "loads
successfully ≠ actually usable" gap with evidence.

**Mechanism.** A new run kind `'probe'` (§0.5 enum extended):

```
runs.kind ADD VALUE 'probe'
-- enum: 'tuning'|'refine'|'sweep'|'runtime'|'probe'; NULL = standalone
```

Engine pinned: **llama-server** (the tok/s floor below is meaningless on llama-bench,
which has no request lifecycle). A probe run loads the model at a candidate context
and performs a short timed interaction. Worker execution path, explicit (same pattern
as N4's job type):

```ts
{ type: "run_probe", payload: {
    model_id: string,
    candidateCtx: number,
    placement: { ngl: number; nCpuMoe?: number; slots: number },
    kvPair: [string, string],
} }
```

Capability gating: workers advertise `probe-v1`; trigger and dispatch refuse to
enqueue the job without it — the version-skew-gate pattern, so an older worker fails
loudly at scheduling instead of choking on an unknown job type mid-fleet. Refusal
copy: *"that machine runs an older worker build — update it to probe on this
model."* Results
ingest through a worker-authed `POST /api/runs/:id/probe-result` that writes
`model_machine_limits` in one idempotent transaction (re-report replaces via the
UNIQUE key).

**Security.** Ingestion requires an enrolled worker session — the per-worker bearer
credential established when the machine was approved, never the shared deployment
secret (which the chain-advance rule refuses for the same reason) — because these
rows feed *verified* claims other tenants see. Authorization binds by path, not
payload: `worker_id` is derived from the run named in the URL, so a spoofed identity
cannot poison another machine's verified ceiling. Payload bounds mirror the sweep-axis
table — `ngl`/`n-cpu-moe` within `[0, 1024]`, `slots ∈ [1, 64]`, `candidateCtx` within
`[256, 4 194 304]`, cache types drawn from the known set — and out-of-bounds yields
400, never a silent clamp. Upstream trigger quotas bound how many runs can exist to
replay against; idempotency makes replays harmless.

Steps:

1. Candidate = M1 estimate for the **placement under test** — the same `ngl` /
   `n-cpu-moe` the estimate assumed; verification is placement-specific too. Requires
   `confidence ∈ {good, rough}`. When confidence is `unknown`, offer instead the
   conservative floor candidate (full-`trained_ctx` load) explicitly labeled
   conservative.
2. Spawn with `-c = candidate`, generate `n_gen = 256` streamed; success = no OOM, no
   spill (`vram_peak_mib` within total), gen tok/s above floor (default 1 tok/s —
   excludes swap-thrash "success").
3. On failure retry once at `candidate × 0.75`; on success with > 25 % headroom
   optionally probe once more at `min(trained_ctx, candidate × 1.33)` — three loads
   max, ever.
4. Persist — a re-probe replaces the stale row through the same UNIQUE key:

```sql
CREATE TABLE IF NOT EXISTS model_machine_limits (
  id TEXT PRIMARY KEY,
  worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
  model_id  TEXT NOT NULL REFERENCES models(id)  ON DELETE CASCADE,
  llama_cpp_build TEXT NOT NULL,
  kv_type   TEXT NOT NULL,            -- the K/V pair probed
  placement_hash TEXT NOT NULL,       -- hash of the placement the estimate assumed
                                      -- (ngl / n-cpu-moe / slots)
  verified_ctx_tokens INTEGER NOT NULL,
  margin_observed_frac REAL,          -- headroom left at the verified point
  method_version INTEGER,
  created_at INTEGER NOT NULL,
  UNIQUE(worker_id, model_id, llama_cpp_build, kv_type, placement_hash)
);
```

**Interactions (stated, all benign).** The duplicate-trigger guard serializes probes
on the same `(user, model, worker)` triple (§0.5): a second probe on that triple —
including a different KV type — is refused with the same 409 until the running one
finishes; acceptable for runs that last minutes. Probes
are short, so the continuous wall-clock scan never touches them in practice. Probes
are tenant-visible runs, badged (open question below leans visible-but-badged).

**Consumers.** Profile cards upgrade fit lines from *"estimate"* to *"verified up to X
tokens on this machine"* when a matching row exists; M2's feasibility readout prefers
verified over estimated; N3 comparison tables show the verified ceiling per quant.

**Honesty rules.** Verification is per (machine, build, KV pair, placement) — changing
any invalidates. Rows carry `method_version`; stale rows surface with a re-probe
affordance, never silently.

**Exit.** On the worked example's machine: a probe verifies a ceiling at or below the
≈ 44 k estimate and records its observed margin (illustratively, 41 k at 6 %); the
card shows the verified number; changing the KV type yields a fresh row, not a reused
one.

## N3 — Model-vs-model comparison (weight quants)

**Goal.** Pain #4 verbatim: *"Q4_K_M vs Q5_K_M vs Q6_K without downloading and testing
everything yourself."* Different GGUF files → multi-model orchestration on top of what
the core already measures — not a new measurement.

**Design.**

* **Trigger.** Selects 2–5 models (hard cap) **already registered on the target
  worker** — file present in its local cache, not merely bookmarked — plus one frozen
  grid (typically a trimmed sweep: KV collapsed to the tolerance pair per M2/M4,
  depth at the questionnaire target), because the interesting variable is the model
  file. Members are created through the ordinary trigger route with `kind='sweep'`,
  so every §0.5 guard composes unchanged.
* **Fairness rules (blocking, at trigger and re-checked per member):** same worker,
  same `llama_cpp_build`, same backend/GPU, same flags, same `method_version`, same
  repeats. Drift → FATAL for the affected member, marked in the view — comparisons are
  the one place a silent confound poisons exactly the conclusion being sold.
* **Orchestration.** Members share a root group (§0.5); worker FIFO serializes them;
  no drainer — nothing derives from anything. Members carry `kind = 'sweep'` (trimmed
  sweeps), so §0.5's duplicate guard composes exactly with ordinary runs. New column
  `runs.comparison_id TEXT` groups members; NULL otherwise.
* **Cost gate.** Price the full matrix from prior rates (§0.6 generalized to N
  models). Comparisons are the most expensive object in the product; the confirm screen
  shows total hours, not per-model hours.
* **Quant provenance.** Label parsed from GGUF metadata — verified present:
  `readGgufInfo` extracts `general.file_type` → filename-style quant codes
  (`worker/src/gguf.ts`), stored on the model record. Identical labels with different
  hashes display separately — labels lie, hashes don't.

**Output.** Comparison table (per-model best-config rows side by side) plus a Pareto
scatter across models: speed × memory, gaining N4's quality column once available.

**Storage.** Zero schema beyond `runs.comparison_id` — every measurement is an
ordinary `results` row on its own model_id. Results with a non-null `comparison_id`
are **excluded from profile scoring** (§0.3): comparison grids are trimmed by design,
and they must not compete with Test A's full-grid profiles. `comparison_id` gets its
own index — the comparison view is the hottest group-by after root grouping.

**Exit.** Three quants of one base compared on one machine → table + scatter render;
mid-group build swap fails the drifted member loudly; aborting the group keeps
completed members comparable.

## N4 — Quality subsystem (perplexity/KL)

**Goal.** Give pain #4's quality half a real measurement while keeping the boundary
honest: this is the promised "new subsystem with its own stage".

**Design.**

* New worker job type:

```ts
{ type: "measure_quality", payload: {
    model_id: string,
    ctxTokens: number,
    kvPair: [string, string],
    datasetHash: string,     // pins which corpus the number means
} }
```

  implemented as a driven `llama-perplexity` invocation (flags probed via §0.7; `-c`
  sized per slot math; existing budgets apply). Workers advertise a `quality-v1`
  capability string and dispatch refuses enqueue without it, mirroring the probe gate.
  Ingestion runs through a worker-authed `POST /api/runs/:id/quality-result` that
  writes the row and computes the KLD baseline pairing server-side in one idempotent
  transaction.

  **Security.** Same session rule as probes — enrolled worker session, no shared-
  secret fallback; `worker_id` derived from the URL's run, never the payload.
  Validation before any write: `datasetHash` matches `sha256:[0-9a-f]{64}`,
  `ctx_tokens` within the same bounds as probe contexts, `ppl > 0` and `kld_vs_baseline
  ≥ 0` and both finite — malformed readings are a 400, never stored as data.
* **Dataset discipline.** Bundled default corpus with a pinned hash — only
  redistributably licensed text qualifies (record the license beside the hash). User
  corpora allowed, hashed and recorded. A perplexity number without its corpus hash is
  meaningless and the UI never shows one.
* **Storage** — FK-complete per convention (every new FK gets an explicit ON DELETE):

```sql
CREATE TABLE IF NOT EXISTS quality_results (
  id TEXT PRIMARY KEY,
  root_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
  llama_cpp_build TEXT NOT NULL,
  ctx_tokens INTEGER NOT NULL,
  cache_type_k TEXT NOT NULL, cache_type_v TEXT NOT NULL,
  ppl REAL, kld_vs_baseline REAL,
  dataset_hash TEXT NOT NULL,
  method_version INTEGER,
  created_at INTEGER NOT NULL,
  UNIQUE(root_run_id, model_id, ctx_tokens, cache_type_k, cache_type_v, dataset_hash)
);
```

  Worker retries replace through that key — without it, a retried job accumulates
  duplicate rows and the f16 baseline pairing faces an ambiguous "which one?"

* **Baseline pairing.** KLD against the f16-KV, same-ctx, same-dataset row of the same
  model+build — mirroring §0.8's twin discipline. No baseline row → ppl only,
  `kld_vs_baseline` null. Deleting the baseline marks dependent KLD rows orphaned
  rather than silently re-baselining.
* **Where it appears.** N3's comparison gains a quality column; profile cards gain an
  optional footnote ("KV q8_0/q8_0: KLD 0.004 vs f16 at 32 k") — never a score, never
  mixed into any argmax. Ranking on quality stays out until the calibration gate
  passes; the measurement informs, the user decides.
* **Boundary amended, not repealed.** *"Speed and memory are always measured. Quality
  is measured only as a synthetic proxy (perplexity/KLD against a pinned corpus) and is
  clearly labeled as such — it is still not your workload."*

**Cost warning.** Perplexity at long ctx is quadratic prefill work; price it like any
item; default to one ctx point (the questionnaire target), not a ladder.

**Hard gate.** N4 never feeds ranking until calibrated against a sanity cohort
(known-degenerate pairs, e.g. q4_0/q4_0 vs f16 at matched ctx) reviewed by a human who
confirms the metric moves the expected direction.

**Exit.** Same model, f16 vs q4_0 KV at 32 k → KLD pair recorded with dataset hash;
comparison view renders it; deleting the baseline orphans dependent rows loudly.

## N5 — Concurrency knee

**Goal.** Pain3 #4: find where continuous batching turns throughput gains into latency
collapse — per model+machine, as data instead of folklore.

**Design.**

1. Extract `worker/src/loadDriver.ts` — N simultaneous streaming requests, per-slot
   and aggregate tok/s, TTFT p50/p95 from raw samples, `-c = slots × (n_depth +
   n_prompt + n_gen + margin)` sizing rule carried over verbatim. Each stream's prompt
   carries a distinct nonce suffix so the prefix cache cannot dedupe concurrent
   requests — otherwise later streams read artificially fast against a warm prefix.
   This extraction is a prerequisite regardless of N5: the streaming client must not
   grow into it ad hoc.
2. Knee finder: slots ∈ `{1, 2, 4, 8}` (Expert-editable), prompt/gen cells fixed from
   the M2 target; report per-slot TTFT p95 and aggregate tps;
   **knee = smallest slot count whose TTFT p95 exceeds 2× its slots=1 value.**
   Stored as ordinary `results` rows with `concurrency` set (column exists in the
   runtime-columns phase) — the knee is a derived read, not a stored verdict.

**Non-goals inside the goal:** no closed-loop think-time simulation, no multi-client
network realism — this characterizes the server, not the LAN.

**Exit.** TTFT-p95-vs-slots chart renders with the knee marked; past the knee,
aggregate tps visibly trades against per-user latency in the same chart.

## N6 — Sustained-state policy & CPU fidelity

**Goal.** Finish what M6 started — it added telemetry; this adds policy — and close
pain3 #6.

* **Steady-state options.** Per-run toggle "discard first M repeats" (default off;
  M = 1), available **only when `repeats ≥ M + 3`** so post-discard n still satisfies
  the n ≥ 3 gate (§0.3) — the control is disabled with that reason otherwise.
  Implemented as a scoring-side exclusion recorded in config, never post-hoc trimming
  outside the stored configuration.
* **Throttle-aware suggestions.** When > ⅓ of a run's items flag
  `thermally_throttled` (denominator: all non-cancelled items, skipped included), the
  completion CTA offers a priced re-run of flagged items after cooldown. Never
  automatic.
* **ISA provenance.** Source: llama.cpp's own startup banner — binaries print
  `system_info: ... | AVX = 1 | AVX512 = 0 | NEON = ...` reflecting the *running
  build's* compiled-in + runtime-detected dispatch. The worker parses it once per
  binary identity (§0.7 memo pattern), heartbeats report it as `cpu_isa`, and
  CPU-bound result rows store it. Two CPUs running the same build tag are no longer
  silently comparable — and the ISA recorded is the binary's actual dispatch, not the
  host CPU's marketing sheet.
* **Threads parity.** Expert mode documents `-t` as dominant on CPU-bound rows (M7);
  a CPU-focused preset template ships — threads varied, ngl pinned at 0 — giving
  pain2's workflow a first-class home.

**Exit.** A throttled run's CTA offer re-runs only flagged items; two workers
differing only in AVX512 support produce rows whose provenance explains a > 2× PP gap
on the same quant.

## N7 — Reproducible exchange

**Goal.** Pain #7 beyond one install: let results travel without losing their meaning.

* **Export bundle** (per run/root): measurements, full config incl. the M2 `goals`
  block, `method_version`, build tag, anonymized hardware class (GPU name + VRAM
  class; no serials), dataset hashes for quality rows, and `config_hash` **recomputed
  at export time from each item's canonical form via §0.4** — not read back from
  `results.config_hash`, which exists only where the twin join wrote it.
* **Import** validates: hash mismatch rejects exactly the offending row;
  `method_version` mixing follows §0.1 display rules; imported rows are badged and
  never merge into local profile scoring unless opted in per import.
* **Community aggregates** extend the §0.9 floor to curves and comparisons;
  contributions from < 5 distinct users stay private-by-default.
* **Methods section as artifact**: exports embed a rendered summary of the pipeline
  (warmup, streaming clock, suspect filter, stddev floor) keyed to `method_version` —
  every shared number carries its own methods section.

**Exit.** Round-trip export→import on a second instance preserves byte-equal hashes
and renders identical curves; tampering one field breaks exactly that row's validation.

## Accessibility of the new surfaces

The core UI phase sets the accessibility bar; every Part A/B addition meets it,
specifically: questionnaire groups are labelled fieldsets (goal/floor/workload chips
carry `role="radio"` + `aria-checked`; the floor reveal and KV-tolerance section are
disclosures with `aria-expanded`); the target-context clamp announces via
`aria-live="polite"`; tolerance-pruned KV chips stay in the accessibility tree — struck
through visually, `aria-disabled` with the reason in an accessible description,
exactly like the profile-source rule; fit lines and confidence words are text, never
colour-only; the concurrency chart pairs its bars with an `sr-only` summary sentence
("knee at slots = 4 — TTFT p95 doubles past this point"); and "Measure missing points"
renders `aria-disabled` with its reason when no ladder cells are uncovered.

---

## Order & dependencies

Part A lands **inside** the core phase order — each M names its host:

```
M1 → with KV-estimator phase     M2 → with Benchmark-page start
M3, M4, M5 → with Benchmark page M6 → with runtime-results phase
M7 → doc-only with Benchmark page
```

Capability prerequisites, phrased as capabilities (core plan sections in parens for
navigation only):

```
N1  context curves      ← streaming client + warmup; depth axis; flag probes (P4/P3/P1.5)
N2  usable-config probe ← M1; server path w/o speculation (P1)
N3  model comparison    ← safety net, chains/grouping, presets (P0–P2, P6.4)
N4  quality             ← N3 for presentation, else standalone; needs flag-probes (§0.7) + budgets
N5  concurrency knee    ← streaming client; load-driver extraction folded in here
N6  sustained/CPU       ← M6 thermal columns
N7  exchange            ← all prior (format stability argues for last)
```

Hard gates:

* Nothing in the core phases blocks on Part B; Part B gates on the named capabilities
  exiting green.
* **N4 never feeds ranking** until the N4 calibration gate passes.
* **N3 fairness checks are blocking**, at trigger and per member.
* **N1's server-path choreography** ships as separate code paths; its warm-cache
  repeats may not silently change ordinary runtime runs' reuse semantics.

## Non-goals

Training/fine-tuning evaluation; subjective human preference capture; multi-GPU
tensor/pipeline split benchmarking (upstream support too fluid); network-realistic
load testing; anything that puts a quality number inside a ranking argmax.

## Open questions

* Does `llama-perplexity` expose KL directly, or do we diff logits ourselves?
  (Determines whether N4's KLD ships day one or lands ppl-only.)
* Should probe runs (N2) be tenant-visible runs in the Runs list, or system-scoped?
  (Leans visible-but-badged — trust.)
* Bundle format for N7: JSON lines vs SQLite attach? (Decide at N7 start.)
* AMD Windows sensors (M6): adopt LibreHardwareMonitorLib or ADLX, or stay NULL?
  (Decide when the first AMD-Windows user needs the flag; NULL + declared availability
  is the honest default until then.)

## Audit note

* **Pass 1** folded: the M1 formula/input mismatches and its unsubstantiated exit
  example (now a fully worked one), M6's aggregate-only schema (series added, detection
  made computable, sources enumerated incl. ROCm), N1's warm-prefill contradiction
  (cold timed prefill restored as data; eviction given a concrete detector), N4's
  missing FKs, N6's gate interaction, N2's unspecified engine and unknown-confidence
  fallback, M3's redistribution ambiguity (pinned reduction), M5's legacy-preset
  behavior, and N7's hash provenance.
* **Pass 2** folded: M1's SWA inversion direction (naive math overestimates affordable
  tokens — the unsafe direction — now solved on an assumed global-layer share with
  `rough` confidence), M4's Expert-mode collapse nuance (tolerance prunes
  quantization, not width), and M6/M7 flag eligibility requiring ≥ 4 repeats.
* **Pass 3** folded: `caveat_flags` consumed everywhere but defined nowhere in this
  document (now §0.10 with its registry); the Low Memory "always shown" row contradicting
  §0.3's no-memory degradation (now conditional, with the degraded state stated); the
  hwmon frequency unit mislabel (`freq1_input` is Hz, not mHz); N1's pp column left
  source-less (now derived from the cold request's own timings); slot sizing referenced
  but never defined standalone (formula inlined in N1/N5); the `caveat_flagged` tally
  having no corresponding gate (swa-exclusion named as the gate); M1's binding bullet
  using an undefined `headroom` shorthand plus a both-or-neither note on the fallback
  inputs; N2 claiming "identical" probe dedupe where the guard actually serializes all
  same-triple probes (reworded) while leaving placement unpinned (now pinned, with
  `placement_hash` added to the verification key and UNIQUE constraint); N3 member runs
  carrying no declared kind (`kind='sweep'` now stated, making the duplicate guard's
  composition exact); and two dangling shorthand references ("modified-state rule",
  "(§N…)" anchors) reworded for standalone readers.
* **Pass 4** folded: N1's curve grouping unable to distinguish choreographed points
  from ordinary runtime rows sharing the same keys (points now stamp
  `METHOD_VERSION = 2`, so §0.1's never-average rule segregates them by construction);
  N3's `kind='sweep'` members silently flowing into profile scoring (results with a
  non-null `comparison_id` are now excluded — trimmed grids must not compete with Test
  A's full-grid profiles); N5's concurrent identical prompts letting the prefix cache
  dedupe streams (distinct nonce suffix per stream); §0.6 pricing only the pp side of
  a Test B cell (generation now prices symmetrically); §0.8's offset mismatch setting
  `unverified` without writing its registry flag; and §0.2's engine-pair list written
  in shorthand that was not valid TypeScript.
* **Pass 5** folded: M6's detection window left unspecified — model-load idle clocks
  inside the sampled span could mask or fabricate sag (server items now bracket the
  timed-repeat window; llama-bench items state their whole-item approximation); N1's
  eviction detector literally reading "any repeat" would have condemned the legitimate
  cold prefill (now scoped to warm repeats); N2 describing the duplicate guard's 409
  rejection as "queueing"; and N2's exit example citing a "~44 k ± rough" estimate
  when the worked example it references carries confidence `good`.
* **Passes 11–14 (targeted security / performance / observability / accessibility
  lenses)** folded: result ingestion routes had no named auth model and no payload
  validation (enrolled-worker-session required, shared secret refused, `worker_id`
  bound by path not payload, bounds on contexts/placement/cache-types/hash format,
  malformed readings 400-not-stored); the curve read path and comparison view lacked
  indexes (`idx_results_curve`, `idx_runs_comparison`); M6's sample series was the
  first bulky column with no retention story (prunable at 30 days; aggregates and flag
  persist); Part B emitted no signals at all (§0.12 minimum structured-event set); and
  none of the new UI surfaces had been checked against the core accessibility bar
  (consolidated conformance block added). The stale full-site mockup was annotated as
  superseded on its benchmark surfaces.
* **Pass 15** folded, via an API-surface / DDL-parity / term-definition lens: N2 and
  N3 never bound their actions to routes (both now state they ride the ordinary
  `POST /api/runs/trigger`, inheriting every §0.5 guard); §0.8 made twin-join
  semantics normative without carrying the columns the join consumes (the five-column
  DDL block plus its index now lives in §0.8); and two security-critical terms were
  used before definition — "enrolled worker session" (now defined at first use:
  per-worker bearer credential from machine approval, never the shared secret) and
  the "known set" of cache types (now the closed nine-value list in §0.2).
* **Pass 16** folded, via a failure-copy inventory / migration-replay drill /
  full mockup diff: three decision-hinging rejections had no user-facing copy (the
  depth-on-server 400, the duplicate-trigger 409, and the capability refusal — each
  now states its message, all naming the fix or the blocker); the upgrade/replay
  drill came back clean (ALTER defaults fill legacy rows, NULL semantics cover every
  historical column, M1's interpolation inputs pre-date this plan); and the
  element-by-element mockup diff found exactly one drift — the Max Context card was
  missing the mandatory quality-not-measured disclaimer every profile card must
  carry (now added alongside its estimate-basis copy).
* **Pass 17** folded: byte-level encoding integrity verified on all three artifacts
  (zero replacement characters; balanced fences; balanced tag pairs) after many edit
  round-trips, and a cross-reference gap closed — the document never pointed to its
  own companion mockup while three HTML candidates existed, one of them partly
  superseded (header now names the authoritative file and its scope).
* **Pass 18** folded, via a retry/idempotency and duplicate-key lens: quality
  ingestion had no idempotency rule, so a retried job would accumulate duplicate rows
  and make the f16 baseline pairing ambiguous (`root_run_id` made NOT NULL — quality
  work is run-scoped like everything else — plus a six-column UNIQUE key that retries
  replace through); curve points re-measuring the same cell had no selection rule
  (later `created_at` wins, superseded point greyed, never averaged); M6's odd-sample
  split was unspecified despite the rule claiming exact reproducibility (floor split,
  extras to the first half); and N2's Exit contained an "e.g." where the executable-
  criteria convention demands an assertion (rewritten: ceiling at or below estimate,
  margin recorded).
* **Pass 20** folded: a hedging-language sweep over every Exit paragraph came back
  clean (all fourteen assertive; the three remaining "e.g."s sit in illustrative
  prose, not criteria), and §0.12's event set — written before some later features —
  was missing an event for comparison fairness failures (`comparison_member_failed`
  added; a mid-group FATAL is precisely what gets debugged after the fact).
* **Pass 19** folded, via concurrency-walkthrough and clock-authority lenses: the
  interleaving walkthrough came out clean by construction (worker FIFO serializes all
  job kinds, each item owns its process, cross-machine measurement lands under
  different grouping keys); N1's duplicate-point tie-break was not a total order —
  second-resolution `created_at` collisions would make the "deterministic" rule
  nondeterministic (now `(created_at, id)`); and M6 never assigned a writer for its
  flag and aggregates (worker derives at item end, server stores verbatim — one
  writer per column).
* **Pass 6** folded: N2 specified a job payload for quality measurement but none for
  probes themselves (a `run_probe` worker job now mirrors N4's pattern); N1 had no
  trigger story — bench-half points fall out of sweeps, but nothing said how
  server-half points ever come to exist (the Curves tab now exposes "Measure missing
  points", enqueueing runtime-kind items that execute the choreography); M4 cited
  "Constraint 1" from an appendix this standalone document does not carry (reworded to
  state the rule directly); and the dependency table's "needs probes" was ambiguous
  between §0.7 flag-probes and N2 probe runs (now explicit).
* **Pass 7** folded, via a quota/tenancy/glossary lens: symbols consumed by the
  normative base but never defined in this document — `test_type ∈ {pp, tg}` row
  model and `suspect_count`/`sample_count` (now glossed in §0.3) and `prompt_offset`
  (now defined at its §0.8 use); the active-roots quota silently counting probes and
  counting comparison members individually (probes now exempt, a group counts once);
  the curve endpoint's tenant scoping unstated (caller-owned rows only, others behind
  the §0.9 floor); and N2 leaving re-probe replacement mechanics implicit (stated:
  replaced through the same UNIQUE key).
* **Pass 10** folded: N3's trigger said comparison models are ones the target worker
  "can hold" without saying how — now pinned to local-cache presence. Mechanical
  sweeps came back clean: all eleven §0 sections referenced at least once, no invalid
  anchors, no doubled words; the apparent missing-exit asymmetry was a false alarm
  (M1 uses a worked-example variant label, M7 is copy-only by design). This pass also
  tried three fresh lenses — anchor/reference integrity, edit-debris typos, and an
  adversarial-user walkthrough — and found nothing beyond the wording pin above. The
  document has converged; further passes are re-reading, not auditing.
* **Pass 9** folded, via a version-skew / ingestion lens: the new worker job types
  (`run_probe`, `measure_quality`) had no capability strings, so an old worker would
  receive an unknown job type — the exact silent-failure class the version-skew gate
  exists to prevent (`probe-v1` / `quality-v1` advertised, dispatch refuses without);
  neither capability had a server-side ingestion path defined (worker-authed,
  idempotent result routes specified for both); and the schema-evolution conventions
  governing where all of this document's DDL lives were never restated standalone
  (now §0.11).
* **Pass 8** folded, via an exit-criteria-completeness lens: Part A amendments M2–M7
  carried no Exit criteria although every Part B item and M1 did (one runnable
  assertion each now closes every amendment; M7 states it is copy-only by design);
  §0.3 never said scoring tuples ignore `n_depth` — without that, reference-depth
  selection has nothing to select across (tuples now group depth-blind, with §0.4's
  other axes retained); and the mockup had drifted from pass 6's trigger story (the
  curve card now shows "Measure missing points").
