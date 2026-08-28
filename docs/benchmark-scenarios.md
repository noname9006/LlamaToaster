# LlamaToaster — Benchmark Scenarios (9 Combinations)

3 goals x 3 workload shapes = 9 scenarios.
Goals decide which profile cards get produced. Workload shapes decide the pp/tg scoring weights.

---

## Scenario matrix

| # | Goal | Workload | Weights (wPP / wTG) | Signature card | What wins |
|---|------|----------|---------------------|----------------|-----------|
| 1 | Balanced | Chat · gen-heavy | 0.25 / 0.75 | Balanced | (0.4 + 0.4·FIT)·S + 0.2·(1 − PRESSURE) |
| 2 | Balanced | Docs · prompt-heavy | 0.70 / 0.30 | Balanced | (0.4 + 0.4·FIT)·S + 0.2·(1 − PRESSURE) |
| 3 | Balanced | Even | 0.50 / 0.50 | Balanced | (0.4 + 0.4·FIT)·S + 0.2·(1 − PRESSURE) |
| 4 | Max tok/s | Chat · gen-heavy | 0.25 / 0.75 | Max Speed | argmax wPP·P̂P + wTG·T̂G |
| 5 | Max tok/s | Docs · prompt-heavy | 0.70 / 0.30 | Max Speed | argmax wPP·P̂P + wTG·T̂G |
| 6 | Max tok/s | Even | 0.50 / 0.50 | Max Speed | argmax wPP·P̂P + wTG·T̂G |
| 7 | Max context | Chat · gen-heavy | 0.25 / 0.75 | Max Context | argmax estimated affordable tokens, above >= speed_floor% of best TG |
| 8 | Max context | Docs · prompt-heavy | 0.70 / 0.30 | Max Context | argmax estimated affordable tokens, above >= speed_floor% of best TG |
| 9 | Max context | Even | 0.50 / 0.50 | Max Context | argmax estimated affordable tokens, above >= speed_floor% of best TG |

---

## Per-goal behaviour

### Balanced (scenarios 1-3)

Produces 3 cards: Max Speed, Balanced, Low Memory. Hides Max Context.

Balanced card formula:
  (0.4 + 0.4·FIT)·S + 0.2·(1 − PRESSURE)

- S = combined(PP̂, TĜ) = wPP·P̂P + wTG·T̂G  (workload-dependent)
- FIT = min(1, maxCtx.tokens / target_ctx); falls back to constant 0.4 when no target_ctx stated
- PRESSURE = max(VRAM_peak/VRAM_total, RAM_peak/RAM_total); null when no memory totals

The 0.4 weight on FIT means: with no stated target, the rule collapses to 0.8·S + 0.2·(1−P) — byte-for-byte the old fixed-weight behaviour. That is what "skippable" means.

### Max tok/s (scenarios 4-6)

Produces 2 cards: Max Speed, Low Memory. Hides Balanced and Max Context.

Max Speed card formula:
  argmax of wPP·P̂P + wTG·T̂G  — "measured speed only"

Context target and FIT play no role. The workload weights are the entire ranking lever.

### Max context (scenarios 7-9)

Produces 2 cards: Max Context, Low Memory. Hides Max Speed and Balanced.

Max Context card formula:
  among configs with TG >= speed_floor_frac · bestTg, argmax maxCtx.tokens

- speed_floor_frac defaults to 0.5 (choices: 0.4 / 0.5 / 0.6)
- maxCtx.tokens comes from M1's inverse VRAM estimate, NOT a measurement at the target
- The card's basis string always says: "ranked on estimated fit, not measured at your target. Verify with a probe run."

Usable-speed floor is mandatory here: without it argmax-context degenerates into "quantize everything and crawl".

---

## Per-workload effect

| Workload | wPP | wTG | Best for |
|----------|-----|-----|----------|
| Chat · gen-heavy (1,4,7) | 0.25 | 0.75 | Conversational apps — generation dominates cost |
| Docs · prompt-heavy (2,5,8) | 0.70 | 0.30 | RAG / summarization — prefill dominates |
| Even (3,6,9) | 0.50 | 0.50 | Balanced mix |

Workload only changes the Max Speed and Balanced card formulas (it multiplies P̂P and T̂G). It does NOT affect the Max Context card (argmax over estimated tokens, gated by a raw TG floor). It does not change what gets measured.

---

## Eligibility gates (identical for all 9 scenarios)

Every tuple must pass all four, or it is tallied and excluded:

| Gate | Rule |
|------|------|
| stability | >= 3 samples AND stddev <= max(10%·mean, 0.5 t/s) |
| suspect_samples | zero suspect readings |
| missing_pp_or_tg | tuple has both a pp row and a tg row |
| caveat_flagged | a tg row exists at (or nearest to) the reference depth, excluding swa rows |

---

## What never changes across any scenario

- The same sweep grid runs on the worker. Goals are intent stored as configuration, re-read at scoring time — never re-measured.
- Normalization uses P90 of per-config medians as the denominator (goal-blind).
- Quality (PPL) is never part of any ranking — a labeled synthetic proxy, excluded by definition.

---

## Quick reference: which scenario for which job

| If you want... | Pick |
|----------------|------|
| Best all-round config for a chatbot | 1 — Balanced + Chat |
| Best all-round config for a doc search | 2 — Balanced + Docs |
| Best all-round config, no strong preference | 3 — Balanced + Even |
| Rawest generation speed for a chatbot | 4 — Max tok/s + Chat |
| Rawest prefill throughput for RAG | 5 — Max tok/s + Docs |
| Rawest overall speed | 6 — Max tok/s + Even |
| Widest usable context for a chatbot | 7 — Max context + Chat |
| Widest usable context for RAG | 8 — Max context + Docs |
| Widest usable context, no strong preference | 9 — Max context + Even |