Local LLMs · your hardware · your data
≈4 000 000 files. 14 axes. One answer.
LlamaToaster — the appliance that finds it.
Running a model locally shouldn't require becoming a llama.cpp expert.

<!-- Slide 1/10 -->

---

## The Catalog
### Nobody can read a three-million-model index

~3 000 000 models on Hugging Face
~200 000 of them GGUF — already shaped for llama.cpp
≈4 000 000 files once every quantization is counted
10+ architectures: Llama · Qwen · Gemma · Mistral · Phi · DeepSeek · GLM · Kimi · MoE · multimodal

The index grows weekly. Last month's "best" is already stale.
So everyone picks by folklore: a Reddit thread said that one is good.

<!-- Slide 2/10 -->

---

## The Configuration
### Choosing is one haystack. Configuring it is another.

Fourteen axes cross-multiply on every single run:
context · offload layers · KV cache K · KV cache V (9 types each) · flash attention ·
batch · ubatch · threads · prefill depth · concurrency · MoE CPU offload ·
speculative decoding · draft offload · and the llama.cpp build itself.

Every axis interacts with every other, across every GPU, driver and OS.
The space is effectively infinite — and nothing in it is labeled.

### Three ways it breaks

❶ One bad knob — silent quality loss, or a hard crash
   Wrong KV cache → degraded tokens. Context too large → OOM at hour three.

❷ One good config on GPU X — broken on GPU Y
   Copied recipes fail across architectures, drivers, OS versions.

❸ The stale oracle
   Ask a chatbot for the best local setup and you get confident folklore
   from a training set that closed before the model you want existed.

» The first config that doesn't crash wins — not the best one.
» Complexity isn't overcome; it filters people out.

<!-- Slide 3/10 -->

---

## The Missing Half
### Quality is measured to death. Performance is a black box.

Dozens of leaderboards rank "smartness" — every release scored within days.
LMArena · Open LLM Leaderboard · MMLU · GPQA · HumanEval · MT-Bench.

Almost nothing answers the other half:
tok/s · time-to-first-token · genuinely usable context — on YOUR hardware.

Mountains of quality scores. Performance is still trial and error.
The hardware was almost always fine. The choice layer is what fails.

<!-- Slide 4/10 -->

---

## Why Now
### Models crossed the line — but accessible ≠ obvious

Qwen3.8-27B — dense, multimodal, 262 144 tokens of native context —
runs quantized on one consumer GPU.
Every month brings another "impossible" small model.
Proprietary APIs bring outages, refusals, price changes — none of it yours.
Your data stays on your machine, off every provider's logs.

But "it runs here" is not "run it this way."
✗ Chasing the biggest model → swap, crash, 0.3 tok/s
✗ Chasing hype → the loud model instead of the right one for YOUR task
✗ Following last month's advice → already stale

» 262k native context does not mean 262k on your card.
» Nothing in the ecosystem tells you where it actually stops.
The missing piece is maximum effect for your workload — not maximum parameters.

<!-- Slide 5/10 -->

---

## The Fix
### An orchestrator that answers, not just measures

One command connects any GPU or CPU box. Pull-only: the worker long-polls out,
the server never dials in. Nothing to port-forward, nothing to firewall-open.

You state intent — goal × workload × target context.
The grid builds itself. The sweep runs itself, in three linked stages:
tuning → refine → sweep.

Out come four profile cards — Max Speed · Balanced · Max Context · Low Memory —
each with the exact command line and a human-readable reason it won.

Intent in. Decision out.
Set up tonight. By morning: "run this exact configuration" — not "go figure it out."

<!-- Slide 6/10 -->

---

## Already Built
### Not a roadmap — this is what runs today

⚙ Fleet — many machines, one queue; device-flow enrolment, per-user isolation in SQL
⚙ Models — search Hugging Face, download a quant straight onto a chosen machine
⚙ Builds — install, activate and delete llama.cpp releases from the web UI, no compiler
⚙ Probe ladder — 6 modes bisect the real context ceiling instead of guessing it
⚙ Curves — tok/s and TTFT against growing context; the concurrency knee, derived on read
⚙ Speculative decoding — MTP runs benchmarked through llama-server, not just llama-bench
⚙ Quality — llama-perplexity PPL and KLD against a pinned corpus, so degradation is measured
⚙ Assistant — your own hardware, models and results as context, plus opt-in community aggregates
⚙ Exchange — JSON · CSV · Markdown export, and importable bundles that keep their methods

<!-- Slide 7/10 -->

---

## Trust the Card
### Every number carries its methods

✓ Four eligibility gates — stability, suspect samples, missing pairs, caveat flags —
send timer-bug results like 1e6 tok/s to the tally, not into the average. Every rejection
is counted and shown: a silent zero-card outcome would be indistinguishable from a bug.

✓ Memory is estimated per-tensor from the real GGUF, not from file size ÷ layers —
it catches the config that logs "31/31 layers offloaded" while quietly paging into system RAM.

✓ No cell renders a number without naming its source: measured here, derived from
llama-bench, or unavailable. Method versions are never averaged across.

✓ Comparisons are fair by construction — a different machine, build or GPU rejects the
member, not the verdict. Exports carry a hash per row, and community aggregates are
k-anonymised in SQL: never a group under 5 contributors, opt-in, never your own rows.

<!-- Slide 8/10 -->

---

## Where It Sits
### Everyone else solves "it runs." We solve "run it this way."

| Tool | What it gives you | What it never answers |
|---|---|---|
| Ollama · LM Studio · Jan | One-click run, sane defaults | Is this the best config for THIS box? |
| llama-bench | Honest numbers for one command line | Which of 4 000 combinations should I run? |
| Leaderboards | Which model is smartest | What will it do on MY GPU? |
| Ask a chatbot | An instant confident answer | Whether it was ever true |
| **LlamaToaster** | **A ranked decision, with the command line** | **Output quality — until you run a quality pass** |

The gap isn't running models. It's choosing how.

<!-- Slide 9/10 -->

---

## The Ask
### Run your rig tonight

One command connects your machine — no port forwarding, no manual build.
Bring a spare GPU box, an old laptop, or the CPU on your VPS.

Built for anyone who runs models locally: solo GPU owners, mixed-hardware teams
with no shared folklore, and anyone who has to publish reproducible numbers.

Opt into the k-anonymised community set and make the catalog searchable for everyone.
Star it while it's early.

github.com/noname9006/LlamaToaster  ·  llamatoaster.com

<!-- Slide 10/10 -->
