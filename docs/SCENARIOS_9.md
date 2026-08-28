# 9 Benchmark Scenarios — Goal × Workload Shape Matrix

> **Purpose**: Quick-reference for the 9 combinations of (Goal × Workload Shape).
> Each cell shows: what the scoring optimizes for, which cards you get, and typical use case.

---

## The 3 × 3 Matrix

| | **Gen Heavy** (chat)<br>`wPP=0.25, wTG=0.75` | **Prompt Heavy** (docs)<br>`wPP=0.70, wTG=0.30` | **Even**<br>`wPP=0.50, wTG=0.50` |
|---|---|---|---|
| **Max Speed**<br>`goal: max_speed` | **Scenario 1** | **Scenario 2** | **Scenario 3** |
| **Balanced**<br>`goal: balanced` | **Scenario 4** | **Scenario 5** | **Scenario 6** |
| **Max Context**<br>`goal: max_context` | **Scenario 7** | **Scenario 8** | **Scenario 9** |

---

## Scenario Details

### Scenario 1: Max Speed + Gen Heavy
**Goal**: `max_speed` • **Workload**: `chat` (wPP=0.25, wTG=0.75)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Pure generation throughput (tok/s on TG) |
| **Cards emitted** | `max_speed`, `low_memory` |
| **Cards hidden** | `balanced`, `max_context` |
| **Scoring formula** | `argmax(0.25·P̂P + 0.75·T̂G)` |
| **Target context required?** | No |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | Chatbots, coding assistants, long-form generation where prompt is short |
| **What wins** | Configs with highest TG speed; PP speed barely matters |
| **Watch out** | May pick a config that OOMs on large prompts; no memory pressure consideration |

---

### Scenario 2: Max Speed + Prompt Heavy
**Goal**: `max_speed` • **Workload**: `docs` (wPP=0.70, wTG=0.30)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Prompt processing throughput (tok/s on PP) |
| **Cards emitted** | `max_speed`, `low_memory` |
| **Cards hidden** | `balanced`, `max_context` |
| **Scoring formula** | `argmax(0.70·P̂P + 0.30·T̂G)` |
| **Target context required?** | No |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | RAG ingestion, document summarization, large context loading where generation is short |
| **What wins** | Configs with highest PP speed; TG speed barely matters |
| **Watch out** | May pick a config with slow generation; no memory pressure consideration |

---

### Scenario 3: Max Speed + Even
**Goal**: `max_speed` • **Workload**: `even` (wPP=0.50, wTG=0.50)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Balanced PP + TG throughput (equal weight) |
| **Cards emitted** | `max_speed`, `low_memory` |
| **Cards hidden** | `balanced`, `max_context` |
| **Scoring formula** | `argmax(0.50·P̂P + 0.50·T̂G)` |
| **Target context required?** | No |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | General-purpose "fastest overall" baseline; unknown workload mix |
| **What wins** | Configs with best harmonic mean of PP and TG |
| **Watch out** | Still ignores memory pressure and context fit |

---

### Scenario 4: Balanced + Gen Heavy
**Goal**: `balanced` • **Workload**: `chat` (wPP=0.25, wTG=0.75)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Generation speed + memory pressure + context fit |
| **Cards emitted** | `max_speed`, `balanced`, `low_memory` |
| **Cards hidden** | `max_context` |
| **Scoring formula** | `argmax((0.4 + 0.4×FIT)×S + 0.2×(1−PRESSURE))` where `S = 0.25·P̂P + 0.75·T̂G` |
| **Target context required?** | Optional (enables FIT) |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | Chatbot deployment where you want good TG speed but also care about VRAM headroom and context capacity |
| **What wins** | High TG speed, low memory pressure, fits target context (if stated) |
| **Watch out** | Without `target_ctx`, FIT=null → formula collapses to `0.8×S + 0.2×(1−P)` |

---

### Scenario 5: Balanced + Prompt Heavy
**Goal**: `balanced` • **Workload**: `docs` (wPP=0.70, wTG=0.30)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Prompt speed + memory pressure + context fit |
| **Cards emitted** | `max_speed`, `balanced`, `low_memory` |
| **Cards hidden** | `max_context` |
| **Scoring formula** | `argmax((0.4 + 0.4×FIT)×S + 0.2×(1−PRESSURE))` where `S = 0.70·P̂P + 0.30·T̂G` |
| **Target context required?** | Optional (enables FIT) |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | RAG pipeline where large prompts must process fast, but you also need memory headroom for the index + context |
| **What wins** | High PP speed, low memory pressure, fits target context (if stated) |
| **Watch out** | Without `target_ctx`, FIT=null → formula collapses to `0.8×S + 0.2×(1−P)` |

---

### Scenario 6: Balanced + Even
**Goal**: `balanced` • **Workload**: `even` (wPP=0.50, wTG=0.50)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Balanced speed + memory pressure + context fit |
| **Cards emitted** | `max_speed`, `balanced`, `low_memory` |
| **Cards hidden** | `max_context` |
| **Scoring formula** | `argmax((0.4 + 0.4×FIT)×S + 0.2×(1−PRESSURE))` where `S = 0.50·P̂P + 0.50·T̂G` |
| **Target context required?** | Optional (enables FIT) |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | **Default "safe choice"** — general purpose deployment where workload is mixed or unknown |
| **What wins** | Good PP+TG balance, low memory pressure, fits target context (if stated) |
| **Watch out** | Without `target_ctx`, FIT=null → formula collapses to `0.8×S + 0.2×(1−P)` |

---

### Scenario 7: Max Context + Gen Heavy
**Goal**: `max_context` • **Workload**: `chat` (wPP=0.25, wTG=0.75)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Maximum context at usable generation speed |
| **Cards emitted** | `max_context`, `low_memory` |
| **Cards hidden** | `max_speed`, `balanced` |
| **Speed floor** | `speed_floor_frac` (default 0.50) applied to **TG** (weighted 0.75) |
| **Ranking** | Among configs clearing floor → `argmax(estimated_affordable_tokens)` |
| **Target context required?** | **Yes** (for depth axis + fit estimate) |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | Long-running chat sessions where context grows over time; generation quality matters more than prompt speed |
| **What wins** | Config with highest *estimated* context capacity that still achieves ≥50% of peak TG speed |
| **Watch out** | **Estimated** context, not measured. Verify with **Probe run** (N2). PP weight low so prompt capacity not directly optimized. |

---

### Scenario 8: Max Context + Prompt Heavy
**Goal**: `max_context` • **Workload**: `docs` (wPP=0.70, wTG=0.30)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Maximum context at usable generation speed |
| **Cards emitted** | `max_context`, `low_memory` |
| **Cards hidden** | `max_speed`, `balanced` |
| **Speed floor** | `speed_floor_frac` (default 0.50) applied to **TG** (weighted 0.30) |
| **Ranking** | Among configs clearing floor → `argmax(estimated_affordable_tokens)` |
| **Target context required?** | **Yes** (for depth axis + fit estimate) |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | RAG with huge documents where you need maximum context window, but generation must remain usable |
| **What wins** | Config with highest *estimated* context capacity that still achieves ≥50% of peak TG speed |
| **Watch out** | **Estimated** context, not measured. Verify with **Probe run** (N2). TG weight low → floor easier to clear, but generation may be slower than expected. |

---

### Scenario 9: Max Context + Even
**Goal**: `max_context` • **Workload**: `even` (wPP=0.50, wTG=0.50)

| Aspect | Value |
|--------|-------|
| **Optimizes for** | Maximum context at usable balanced speed |
| **Cards emitted** | `max_context`, `low_memory` |
| **Cards hidden** | `max_speed`, `balanced` |
| **Speed floor** | `speed_floor_frac` (default 0.50) applied to **TG** (weighted 0.50) |
| **Ranking** | Among configs clearing floor → `argmax(estimated_affordable_tokens)` |
| **Target context required?** | **Yes** (for depth axis + fit estimate) |
| **KV tolerance** | Any (default `q4_0_ok`) |
| **Typical use case** | General-purpose "maximum context" baseline when workload mix is unknown |
| **What wins** | Config with highest *estimated* context capacity that still achieves ≥50% of peak TG speed |
| **Watch out** | **Estimated** context, not measured. Verify with **Probe run** (N2). |

---

## Quick Comparison Table

| # | Goal | Workload | Cards You Get | Primary Metric | Needs Target? | Best For |
|---|------|----------|---------------|----------------|---------------|----------|
| 1 | Max Speed | Gen Heavy | Speed, Low Mem | `0.25·P̂P + 0.75·T̂G` | No | Chatbots, code gen |
| 2 | Max Speed | Prompt Heavy | Speed, Low Mem | `0.70·P̂P + 0.30·T̂G` | No | RAG ingestion, summarization |
| 3 | Max Speed | Even | Speed, Low Mem | `0.50·P̂P + 0.50·T̂G` | No | General "fastest" baseline |
| 4 | Balanced | Gen Heavy | Speed, Balanced, Low Mem | `(0.4+0.4·FIT)·S + 0.2·(1−P)` | Optional | Chatbot deployment |
| 5 | Balanced | Prompt Heavy | Speed, Balanced, Low Mem | `(0.4+0.4·FIT)·S + 0.2·(1−P)` | Optional | RAG pipeline |
| 6 | Balanced | Even | Speed, Balanced, Low Mem | `(0.4+0.4·FIT)·S + 0.2·(1−P)` | Optional | **Default safe choice** |
| 7 | Max Context | Gen Heavy | Max Ctx, Low Mem | `max(est_ctx) s.t. TG ≥ floor` | **Yes** | Long chat sessions |
| 8 | Max Context | Prompt Heavy | Max Ctx, Low Mem | `max(est_ctx) s.t. TG ≥ floor` | **Yes** | Huge document RAG |
| 9 | Max Context | Even | Max Ctx, Low Mem | `max(est_ctx) s.t. TG ≥ floor` | **Yes** | General max context |

---

## Key Decision Rules

| If... | Use Scenario |
|-------|--------------|
| "Just give me the fastest generation" | **1** (Max Speed + Gen Heavy) |
| "Fastest prompt ingestion for RAG" | **2** (Max Speed + Prompt Heavy) |
| "Fastest overall, I don't know the workload" | **3** (Max Speed + Even) |
| "Deploy a chatbot, want speed + safety margin" | **4** (Balanced + Gen Heavy) |
| "Deploy RAG, want prompt speed + memory headroom" | **5** (Balanced + Prompt Heavy) |
| "Default balanced choice, mixed/unknown workload" | **6** (Balanced + Even) |
| "Maximum context for long chats, verify with Probe" | **7** (Max Context + Gen Heavy) |
| "Maximum context for huge docs, verify with Probe" | **8** (Max Context + Prompt Heavy) |
| "Maximum context, unknown workload, verify with Probe" | **9** (Max Context + Even) |

---

## Knobs That Apply to ALL 9 Scenarios

| Knob | Effect | Recommended Starting Value |
|------|--------|---------------------------|
| **KV Tolerance** | Prunes KV pairs at grid-build time | `q4_0_ok` (full grid) for research; `q8_0_ok` for production; `f16_only` for quality-only |
| **Repeats** | Statistical stability (min 3 for scoring) | 3 (min), 5 (recommended), 10 (high precision) |
| **Target Context** | Enables depth axis + FIT (Balanced/Max Context) | Set to your actual use-case max (e.g., 8192, 32768) |
| **Speed Floor** (Max Context only) | Minimum usable TG fraction | 0.5 default; 0.4 for aggressive; 0.6 for conservative |

---

## What the Worker Actually Runs (Same for All 9)

> **Critical**: The **sweep grid (Stage C) is identical** across all 9 scenarios for a given KV tolerance + target context. Only **scoring/emission** differs.

```
Stage A: PP batch/ubatch (coarse)          → identical
Stage B: Refine around A's winner          → identical
Stage C: KV × FA × ngl × depth at tuned B  → identical combinations
       |
       v
Worker executes ALL combos → reports PP+TG+mem per combo
       |
       v
Server scores → applies goal+workload logic → emits DIFFERENT CARDS
```

---

## Related Files

| File | Role |
|------|------|
| `shared/goals.ts` | Goal types, `WORKLOAD_WEIGHTS`, KV pruning |
| `shared/scoring.ts` | Card emission logic for all 9 combinations |
| `client/src/pages/Benchmark.tsx` | Stage grid derivation, chain UI |
| `client/src/components/GoalQuestionnaire.tsx` | Goal + workload input UI |

---

*Generated from LlamaToaster v8 codebase — `shared/goals.ts` + `shared/scoring.ts`*