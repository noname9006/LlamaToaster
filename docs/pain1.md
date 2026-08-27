1. How fast is it really?

Not:

"What's the theoretical TPS?"

But:

"How many tokens/sec will I get on my machine?"

This is arguably the biggest unanswered question today.

Model cards rarely provide reliable numbers.

2. How much memory does it need?

Users constantly ask:

Peak RAM?
Peak VRAM?
KV cache size?
Memory at 8k vs 32k vs 128k?

Current reporting is fragmented and inconsistent.

3. What settings are optimal?

Questions like:

Flash Attention on or off?
Best batch size?
Best ubatch?
Best context size?

Most people currently discover this by trial and error.

4. What is the speed-quality tradeoff?

Users want to know:

Q4 vs Q5
Q5 vs Q6
Q6 vs Q8

without downloading and testing everything themselves.

5. What happens as context grows?

A huge unknown.

Users want answers like:

Context	Prompt TPS
8k	X
16k	Y
32k	Z
64k	A

Most benchmarks don't show this.

6. What is the largest usable configuration?

Not:

"Will it load?"

But:

"What is the largest context/model combination that remains practical?"

There is a big difference between:

Loads successfully

and

Actually usable
7. How reproducible are results?

Today benchmark results are often:

different llama.cpp versions
different flags
different contexts
different GPUs
different quant builds

Making comparisons difficult.

From a benchmarking perspective

The most valuable outputs are probably:

Throughput
prompt processing TPS
generation TPS
Memory
RAM
VRAM
KV cache
Scalability
effect of context size
effect of batch size
Optimal configuration discovery
automatic parameter sweep
best-performing settings
Reproducibility
standardized methodology

Those are the areas where people still spend significant time manually experimenting with llama.cpp today. The pain isn't "how do I ask the model a question?"—it's "what configuration actually performs best on my hardware, and how does it compare to other models under the same conditions?"