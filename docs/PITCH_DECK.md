# Local LLMs · your hardware · your data

## 4 000 000 files. 40 knobs. One right answer.

> **LlamaToaster** — the appliance that finds it.

- Local inference shouldn't require becoming a llama.cpp expert.

---

<!-- layout: stats -->
# Pain #1 — choosing

## Nobody can read a 3 000 000-model catalog

> **~3 000 000** models on Hugging Face
> **~200 000** of them are GGUF — already ready for llama.cpp
> **≈4 000 000** files once every quantization is counted

- The catalog grows weekly — today's "best" model is stale by next week.
- So everyone picks by folklore: *"a Reddit thread said that one is good."*

---

# Pain #2 — configuring

## Choosing is one haystack. Configuring it is another.

> One bad knob — silent quality loss, or a hard crash.

- Context size · offload layers · KV cache quantization (9 options) · flash_attn · MTP / draft decoding · batch & ubatch · threads · the llama.cpp build…
- Every knob interacts with every other; no one config survives a different GPU.
- The manual path: copy a stranger's config → tweak at random → retry until it *"mostly works"* — or give up.

---

# The real bottleneck

## The hardware was almost always fine — the choice layer fails.

> The first config that doesn't crash wins — not the best one.

- Complexity isn't overcome; it filters people out.
- Users walk away thinking *"local inference isn't for me"* — when all that was missing was a tool for choosing.

---

# Why now

## Models evolve faster than anyone can keep up — and finally crossed the line

- New generations land every few weeks: Qwen, Llama, Mistral, Gemma… last month's advice is already stale.
- Qwen 3 8B / 27B ≈ previous-generation flagship quality — on an ordinary PC.
- Proprietary APIs: outages, refusals, price changes — nothing you control.
- Your data stays on your machine, off every provider's logs.
- The one piece still missing: deciding what to run.

---

# The missing half

## Quality is measured to death. Performance is a black box.

> Dozens of leaderboards rank "smartness" — almost nothing shows what you will actually get.

- LMArena, Open LLM Leaderboard, MMLU, GPQA, HumanEval, MT-Bench… every release is scored nightly.
- But almost nothing answers the other half: **tok/s, latency, genuinely usable context — on YOUR hardware**.
- Millions of files and mountains of quality scores — while performance is still folklore and trial-and-error.

---

# The datacenter myth

## LLMs don't need a datacenter — they need a right-sized model

> Many people believe that running an LLM means renting a server farm.

- Reality: a small modern quantized model runs even on a modest VPS.
- It won't match a flagship — it doesn't have to. Embedding or speech-to-text solves real everyday tasks in a few GB.
- **You wouldn't hire a PhD to solve a simple equation.** Right tool, right size.
- What keeps people away isn't the hardware — it's the myth.

---

# The fix

## An orchestrator that answers, not just measures

- One command connects any GPU or CPU box — pull-only, zero open ports, no firewall gymnastics.
- You state intent: goal × workload × target context. The grid builds itself, the sweep runs itself.
- Out come profile cards — **Max Speed · Balanced · Max Context · Low Memory** — each with a human-readable reason.
- Plus verified context ceilings, concurrency knees, and fair model-vs-model comparisons.

---

# The toaster promise

## Intent in. Decision out.

> Set up tonight — five minutes. By morning: **"run this exact configuration"** — not *"go figure it out"*.

- Stop tuning. Start shipping.

---

# Why you can trust the card

## Every number carries its methods

- Eligibility gates — stability, suspect samples, stddev floors — send timer-bug results like `1e6 tok/s` to the trash, not into the average.
- Memory is estimated from real per-tensor GGUF placement — it catches the config that reports *"31/31 layers offloaded"* while secretly running in system RAM.
- Comparisons are fair by construction: a different machine, build, or GPU rejects the member, not the verdict.
- Exports are tamper-evident: a hash per row, methodology versions never mixed.

---

# For whom

## Built for anyone who runs models locally

- One GPU in a home PC — hobbyists, creators, researchers.
- Teams with mixed machines and no shared folklore.
- Anyone publishing reproducible local-inference numbers.

---

# The ask

## Run your rig tonight

- One command connects your machine — no port forwarding, no manual build (llama.cpp installs from the web UI).
- Opt into the k-anonymized community set and make the whole catalog searchable for everyone.
- Star it while it's early.

> **https://github.com/noname9006/LlamaToaster**