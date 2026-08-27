1. Know Your Hardware Limits
Record:

CPU: cores/threads, frequency

RAM: total capacity and bandwidth

GPU: VRAM size, memory bandwidth, compute capability (if applicable)

This establishes hard ceilings for model size, offloading, and context.

2. Build llama.cpp with the Right Backend
Compile from source with your GPU backend:

bash
cmake -B build -DGGML_CUDA=ON   # or -DGGML_METAL=ON, -DGGML_VULKAN=ON
cmake --build build --config Release
Use the latest commit to ensure all benchmarking flags are up to date.

3. Gather a Set of Candidate GGUF Models
Pick models spanning sizes that may fit your VRAM/RAM. Use well‑known quantizations:

Model	Quant	File Size
7B	Q4_K_M, Q5_K_M, Q6_K	~4–6 GB
13B	Q4_K_M, Q5_K_M	~8–10 GB
30B	Q3_K_M, Q4_K_M	~14–18 GB
70B	Q2_K, Q3_K_S	~26–30 GB
Download only what you think might run.

4. Use llama-bench for Rapid Screening
llama-bench runs a standard synthetic workload and reports:

pp = prompt processing tokens per second

tg = text generation tokens per second

total time

Example:

bash
./build/bin/llama-bench -m models/model.gguf -p 512 -n 128
-p sets prompt length in tokens (synthetic)

-n sets number of tokens to generate

Run for each candidate model. Filter out any that:

Fail to load (OOM)

Have generation speed < 5 tokens/s (adjust threshold to your patience)

Record results in a table.

5. Detailed Synthetic Parameter Sweep with llama-cli
Pick the best 1–2 models from step 4 and now vary parameters systematically. Use a fixed synthetic prompt (e.g., a repeated token sequence to avoid any natural language effects). Make it long enough to stress prompt processing (e.g., 1000 tokens) and generate 256 tokens.

Add --verbose-prompt to get detailed timing.

5.1 Sweep GPU Offloading (-ngl)
bash
for ngl in 0 10 20 30 40 50 60 70 80 90 99; do
  ./build/bin/llama-cli -m model.gguf -p "$(cat synthetic_prompt.txt)" -n 256 -c 4096 -ngl $ngl -t 8 --verbose-prompt
done
Monitor VRAM usage with nvidia-smi (or equivalent) during each run. Record:

Prompt tokens/s

Generation tokens/s

Peak VRAM

Total time

Find the highest -ngl that does not exceed VRAM and gives the best speed.

5.2 Sweep Context Length
Using the optimal -ngl found above, vary context size:

bash
for ctx in 1024 2048 4096 8192 16384; do
  ./build/bin/llama-cli -m model.gguf -p "$(cat synthetic_prompt.txt)" -n 128 -c $ctx -ngl <best> -t 8 --verbose-prompt
done
Context length directly affects KV cache memory and prompt processing speed. Record the same metrics.

5.3 Sweep Threads and Batch Size
Still using the optimal -ngl and context:

Threads (-t): Try 2, 4, 8, physical core count.

Batch size (-b): Try 256, 512, 1024, 2048. Larger batch speeds up prompt processing but uses more memory.

Example combined:

bash
for t in 4 8 16; do
  for b in 256 512 1024; do
    ./build/bin/llama-cli -m model.gguf -p "$(cat synthetic_prompt.txt)" -n 128 -c <ctx> -ngl <best> -t $t -b $b --verbose-prompt
  done
done
Record results and look for the combination that gives the highest generation speed without OOM.

5.4 Optional: Flash Attention
If your build supports it, add --flash-attn (or -fa) and rerun the best configuration to see if prompt processing improves, especially at long contexts.

6. Automate and Log
Write a shell script to run all combinations and parse the timing output automatically. For example, extract lines containing llama_print_timings and save to CSV.

A minimal example:

bash
#!/bin/bash
MODEL=$1
for ngl in 0 20 40 60 80 99; do
  for ctx in 2048 4096 8192; do
    output=$(./build/bin/llama-cli -m $MODEL -p "$(cat prompt.txt)" -n 128 -c $ctx -ngl $ngl -t 8 --verbose-prompt 2>&1)
    pp=$(echo "$output" | grep "prompt eval time" | awk '{print $5}')
    tg=$(echo "$output" | grep "eval time" | awk '{print $5}')
    echo "$MODEL,$ngl,$ctx,$pp,$tg" >> results.csv
  done
done
7. (Optional) Synthetic Quality Metric: Perplexity
If you want an objective, synthetic measure of model degradation due to quantization or context, use llama-perplexity:

bash
./build/bin/llama-perplexity -m model.gguf -f test_data.txt -c 2048
Lower perplexity = better fit to the test data. This is still synthetic and can be used to compare quantizations objectively.

8. Final Configuration Table
After the sweeps, create a table like:

Model	Quant	NGL	Context	Threads	Batch	PP t/s	TG t/s	VRAM (GB)
7B	Q4_K_M	99	4096	8	512	150	25	5.2
...	...	...	...	...	...	...	...	...
Choose the row that maximizes generation speed and prompt processing while staying within VRAM and giving a context length you need.

Key Points
llama-bench is your friend for quick model comparisons.

llama-cli --verbose-prompt gives the raw timing numbers for detailed sweeps.

VRAM monitoring is essential to avoid OOM and find the true offloading limit.

Sweep one parameter at a time to isolate effects.

Automate to save time and avoid manual logging errors.

This approach yields the best objective performance configuration for your hardware with llama.cpp. No subjective judgement needed.