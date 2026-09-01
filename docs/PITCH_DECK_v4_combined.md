Local AI · your hardware · your data
4 000 000 files. 11 knobs. One answer.
LlamaToaster — the appliance that finds it.
Running a model locally shouldn't require becoming a AI expert.

<!-- Slide 1/9 -->

---

## The Catalog
### 4 million files — no one labeled for you

200000+ GGUF repositories on Huggingface
4 000 000 files once every quantization is counted
DeepSeek · Qwen · Llama · Gemma · Kimi · GLM · Mistral · Nemotron

The index grows weekly. Last month's "best" is already stale.
So everyone picks by folklore: a Reddit thread said that one is good.

<!-- Slide 2/9 -->

---

## The Configuration
### Choosing is one haystack. Configuring it is another.

Eleven knobs cross-multiply on every single run:
context length · threads · GPU offload · MoE layers · batch · µbatch ·
K cache · V cache · flash attention · MTP · KV depth.

Every knob interacts with every other, across every GPU, driver and OS.
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

<!-- Slide 3/9 -->

---

## The Missing Half
### Quality is measured to death. Performance is a black box.

Dozens of leaderboards rank "smartness" — every release scored within days.
OpenLLM · MMLU-Pro · IFEval · BBH · MuSR · GPQA · HELM · BIG-Bench · ARC.

Almost nothing answers the other half:
processing speed · generation speed · TTFT · usable context — on YOUR hardware.

Mountains of quality scores. Performance is still trial and error.
The hardware was almost always fine. The choice layer is what fails.

<!-- Slide 4/9 -->

---

## Why Now
### Models crossed the line — but accessible ≠ obvious

A 27B model on one consumer GPU now lands where
last generation's flagship did.
Every month brings another "impossible" small model.
Proprietary APIs bring outages, refusals, price changes — none of it yours.
Your data stays on your machine, off every provider's logs.

But "it runs here" is not "run it this way."
✗ Chasing the biggest model → swap, crash, 0.3 tok/s
✗ Chasing hype → the loud model instead of the right one for YOUR task
✗ Following last month's advice → already stale

» The missing piece is maximum effect for your workload — not maximum parameters, not the loudest release.

<!-- Slide 5/9 -->

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

<!-- Slide 6/9 -->
---

## Already Built
### Not a roadmap — this is what runs today

⚙ Models — search Hugging Face, download a quant straight onto a chosen machine
⚙ Builds — install, activate and delete llama.cpp releases from the web UI, no compiler
⚙ Load test — 6 modes bisect the real context ceiling instead of guessing it
⚙ Curves — tok/s and TTFT against growing context; the concurrency knee, derived on read
⚙ Speculative decoding — MTP runs benchmarked through llama-server, not just llama-bench
⚙ Quality — llama-perplexity PPL and KLD against a pinned corpus, so degradation is measured
⚙ Decentralized network — many machines, one queue; device-flow enrolment, per-user isolation in SQL
⚙ Assistant — your own hardware, models and results as context, plus opt-in community aggregates
⚙ Exchange — JSON · CSV · Markdown export, and importable bundles that keep their methods

<!-- Slide 7/9 -->

---

## Trust the Card · Where It Sits
### Every number carries its methods — and here's where we sit

The gap isn't running models. It's choosing how.

✓ Four eligibility gates — stability, suspect samples, missing pairs, caveat flags —
send timer-bug results like 1e6 tok/s to the tally, not into the average.
✓ Memory is estimated per-tensor from the real GGUF — it catches configs that page into system RAM.
✓ No cell renders a number without naming its source: measured here, derived from llama-bench, or unavailable.
✓ Comparisons are fair by construction — a different machine, build or GPU rejects the member, not the verdict.

| Tool | What it gives you | What it never answers |
|---|---|---|
| Ask a chatbot | An instant confident answer | Whether it was ever true |
| Leaderboards | Which model is smartest | What will it do on MY GPU? |
| Ollama · LM Studio | One-click run, sane defaults | Is this the best config for THIS box? |
| llama-bench | Honest numbers for one command line | Which of 4 000 combinations should I run? |
| **LlamaToaster** | **A ranked decision, with the command line** | **Output quality — until you run a quality pass** |

<!-- Slide 8/9 -->

---

## The Ask
### Run your rig tonight

One command connects your machine — no port forwarding, no manual build.
Bring a spare GPU box, an old laptop, or the CPU on your VPS.

Built for anyone who runs models locally: solo GPU owners, mixed-hardware teams
with no shared folklore, and anyone who has to publish reproducible numbers.

Opt into the k-anonymised community set and make the catalog searchable for everyone.
Star it while it's early.

llamatoaster.com

<!-- Slide 9/9 -->