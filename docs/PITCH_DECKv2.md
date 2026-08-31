Local LLMs · your hardware · your data
4 000 000 files. 40 knobs. One right answer.
LlamaToaster — the appliance that finds it.
Local inference shouldn't require becoming a llama.cpp expert.

<!-- Slide 1/8 -->

---

## The Problem
### Nobody can read a 3 000 000-model catalog

~3 000 000 models on Hugging Face
~200 000 of them are GGUF — already ready for llama.cpp
≈4 000 000 files once every quantization is counted
10+ architectures: Llama · Qwen · Gemma · Mistral · Phi · DeepSeek · Kimi · GLM · MoE · multimodal

Every knob interacts with every other across every GPU, every OS, every driver.
The configuration space is effectively infinite.

### Three ways it breaks

❶ One bad knob = silent quality loss, or a hard crash
   Wrong KV cache → degraded tokens. Context too large → OOM.

❷ One good config on GPU X = broken on GPU Y
   Reddit recipes fail across architectures, drivers, OS versions.

❸ The stale oracle
   Ask ChatGPT "best local LLM 2026" → get 2024 advice trained on old data.
   GPT/Claude answer with confident, outdated folklore — their training data is already stale.

### The real bottleneck
The hardware was almost always fine — the choice layer fails.
The first config that doesn't crash wins — not the best one.
Complexity isn't overcome; it filters people out.
Users walk away thinking "local inference isn't for me" — when all that was missing was a tool for choosing.

<!-- Slide 2/8 -->

---

## The Missing Half
### Quality is measured to death. Performance is a black box.

Dozens of leaderboards rank "smartness" — almost nothing shows what you will actually get.
LMArena, Open LLM Leaderboard, MMLU, GPQA, HumanEval, MT-Bench… every release is scored nightly.

But almost nothing answers the other half:
tok/s · latency · genuinely usable context — on YOUR hardware.

Millions of files and mountains of quality scores —
while performance is still folklore and trial-and-error.

<!-- Slide 3/8 -->

---

## Why Now
### Models finally crossed the line — but accessible ≠ obvious

Qwen3.8 27B ≈ previous-generation flagship quality — on an ordinary PC.
Gemma 3, Kimi K2, GLM-4 — every month brings a new "impossible" small model.
Proprietary APIs: outages, refusals, price changes — nothing you control.
Your data stays on your machine, off every provider's logs.

But "accessible" doesn't mean "obvious."
❌ Blindly chasing the biggest model → swap, crash, 0.3 tok/s
❌ Chasing hype → running Llama 4 when Qwen3.8 is better for YOUR task
❌ Following last month's advice → already stale

The missing piece: finding maximum effect for your workload,
not maximum parameters or loudest hype.

<!-- Slide 4/8 -->

---

## The Fix
### An orchestrator that answers, not just measures

One command connects any GPU or CPU box — pull-only, zero open ports, no firewall gymnastics.
You state intent: goal × workload × target context.
The grid builds itself. The sweep runs itself.

Out come profile cards — Max Speed · Balanced · Max Context · Low Memory —
each with a human-readable reason.
Plus verified context ceilings, concurrency knees, and fair model-vs-model comparisons.

Intent in. Decision out.
Set up tonight — five minutes. By morning: "run this exact configuration" — not "go figure it out".
Stop tuning. Start shipping.

<!-- Slide 5/8 -->

---

## Trust the Card
### Every number carries its methods

Eligibility gates — stability, suspect samples, stddev floors —
send timer-bug results like 1e6 tok/s to the trash, not into the average.

Memory is estimated from real per-tensor GGUF placement —
it catches the config that reports "31/31 layers offloaded" while secretly running in system RAM.

Comparisons are fair by construction:
a different machine, build, or GPU rejects the member, not the verdict.

Exports are tamper-evident: a hash per row, methodology versions never mixed.

<!-- Slide 6/8 -->

---

## The Competitors
### They solve "works." We solve "optimal."

|              | Easy to Use | Hard to Use |
|--------------|-------------|-------------|
| **High optimization** | | **← LlamaToaster** |
| **Low optimization** | LM Studio · Ollama · Jan.ai | |

**What they do:**
✓ Pick a model that "probably works"
✓ Abstract complexity
✓ One-click install

**What they don't do:**
✗ Find optimal config for YOUR hardware
✗ Sweep parameter combinations
✗ Detect silent quality degradation
✗ Community grid with k-anonymized data
✗ Tamper-evident benchmark cards

<!-- Slide 7/8 -->

---

## The Ask
### Run your rig tonight

One command connects your machine — no port forwarding, no manual build
(llama.cpp installs from the web UI).

Opt into the k-anonymized community set —
make the whole catalog searchable for everyone.

Star it while it's early.
https://llamatoaster.com/benchmark

<!-- Slide 8/8 -->