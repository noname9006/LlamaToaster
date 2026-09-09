# llama-server verbosity levels

`--verbosity` is a `llama-server` (llama.cpp) flag, not a LlamaToaster setting. This
doc compares what each level prints, and what LlamaToaster itself relies on and
sets (`--verbosity 5`, plus `--metrics` for the `/metrics` endpoint). See
[loadDriver.ts:107](../worker/src/loadDriver.ts:107) and
[serverBench.ts:264](../worker/src/serverBench.ts:264) for where these are set.

## The scale

Per `llama-server --help`, confirmed live against real builds (both at model load
and mid-generation via a real `/completion` request):

| Level | Name | What it adds over the level below |
|---|---|---|
| 0 | generic | Startup banner, listen address, fatal errors only |
| 1 | error | + error conditions (failed loads, bad requests) |
| 2 | warning | + warnings (deprecated flags, recoverable issues) |
| 3 | info (**default**) | + high-level lifecycle: server ready, model load start/finish, request accepted. **No per-tensor detail, no offload summary line.** |
| 4 | trace | + `load_tensors: offloaded X/Y layers to GPU`, a per-backend buffer-size breakdown, slot lifecycle, context checkpoints, `print_timing` | 
| 5 | debug | + per-layer `load_tensors: layer N assigned to device ...` ground truth, and a per-token / per-draft-candidate trace — every generated token and (under MTP) every draft-candidate decision logged individually |

Level 3 (the default) is not enough for LlamaToaster: it prints **none** of the
offload-layer detail that [`parseOffloadLayers`](../worker/src/bench.ts:324)
needs to report `gpu_layers_loaded`/`total_model_layers`
(see [bench.ts:104-110](../worker/src/bench.ts:104)). That detail only appears at
level 4+.

**LlamaToaster always runs `llama-server` at `--verbosity 5`** ("debug"), not 4.
Level 4 was used for a while — it already surfaces the offload-summary line
above — but level 4's `X/Y layers to GPU` claim is computed from `-ngl` before
any allocation happens, so on a build/driver that silently bounces some of
those layers back to system RAM, the claim is fiction. Only level 5 also prints
llama.cpp's own per-layer
[`load_tensors: layer N assigned to device ...`](../worker/src/bench.ts:388)
lines — ground truth for where each individual layer actually landed, counted
into an **exact** resident-on-GPU layer count
([`gpu_layers_exact`](../worker/src/bench.ts:407)) instead of level 4's
byte-ratio estimate. That exact count is what the N2 probe's offload-boundary
verdict and the sweep's VRAM-discrepancy check are built on.

The tradeoff is real: level 5 also emits a per-token/per-draft-candidate trace
for the entire life of the process, not just during load. LlamaToaster doesn't
avoid that cost — it bounds it instead. See "What LlamaToaster does with the
output" below.

### llama-bench's equivalent

`llama-bench` (the non-MTP benchmark path) doesn't have a numeric scale —
just a boolean `-v`/`--verbose` flag, gated behind a runtime probe
(`supportsVerboseFlag`, [bench.ts:113](../worker/src/bench.ts:113) — older
builds don't support it). Off, it prints none of the per-tensor/offload
detail either; on, it prints the same one-line-per-tensor load/repack spam
and the same offload-summary line that `llama-server` also prints from level 4
up (confirmed live — see [bench.ts:549-556](../worker/src/bench.ts:549)).
`llama-bench` has no equivalent of level 5's per-layer device-assignment lines
— its own resident-layer count stays a byte-ratio estimate regardless.

## What LlamaToaster does with the output

The raw stderr is always saved unfiltered to the per-item raw JSON dump,
regardless of level. Two separate safeguards keep level 5's extra volume from
causing damage elsewhere:

- **The live capture is bounded**, not the final string:
  [`appendBoundedOutput`](../worker/src/bench.ts:624) caps the accumulator at
  [`MAX_CAPTURED_PROCESS_OUTPUT_CHARS`](../worker/src/bench.ts:632) (2,000,000
  chars) while the process is still running, keeping the first half (load-time
  diagnostics — device detection, per-layer assignments, buffer sizes) and the
  last half (freshest output, what a mid-run crash needs), so a long-running
  item's per-token trace can't grow the raw JSON dump unbounded.
- **The worker's own text log stays filtered.** Mirroring raw stderr into the
  worker's info-level log unfiltered used to bury the structured
  params/offload/summary lines this app logs for every test under tens of
  thousands of characters of tensor/per-token noise, so
  [`logDiagnosticOutput`](../worker/src/index.ts:1346) mirrors it only at debug
  log level (`LOG_LEVEL=debug`), and any single line over
  [`MAX_LOGGED_LINE_CHARS`](../worker/src/index.ts:1330) (300 chars) is elided.
  That line-length guard exists specifically because of level 5: a
  request/response body — including this app's own synthetic filler-token
  prompts and the model's generated text — could otherwise appear verbatim in
  a per-token trace line, where a genuine llama.cpp diagnostic line is always a
  short, structured, human-authored string.

Separately, tensor-load spam (the `create_tensor: loading tensor ...` /
`repack: repack tensor ...` lines both `llama-bench -v` and `llama-server`
print from level 4 up) is collapsed into count-bearing summary lines by
[`collapseTensorLoadSpam`](../worker/src/bench.ts:571) before a *failed*
item's stderr becomes its stored `error` text — otherwise the one line that
actually explained the failure was buried under a tensor-by-tensor transcript.
Level 5's per-token trace is a different shape of noise (interleaved with
useful output for as long as generation runs, not a fixed line pattern), so it
isn't collapsed the same way — `appendBoundedOutput` above is a size
safeguard for it, not a verified spam filter.

## `--metrics`: the `/metrics` endpoint

Separately from `--verbosity`, `--metrics` turns on a Prometheus-format
`/metrics` HTTP endpoint (not stderr logging at all — it's polled over HTTP
while the server is up). llama-server's own metrics generally include
per-slot/global counters like prompt and predicted token counts,
prompt/predicted processing speed, KV-cache usage, and in-flight/deferred
request counts — the exact set depends on the build.

**LlamaToaster only reads two counters from it**, both spec-decode (MTP)
related, matched by regex in
[serverBench.ts:523-524](../worker/src/serverBench.ts:523):

| Metric | LlamaToaster field | Purpose |
|---|---|---|
| `llamacpp:spec_decode_num_draft_tokens_total` | `spec_drafted` | How many tokens the draft (MTP) model proposed |
| `llamacpp:spec_decode_num_accepted_tokens_total` | `spec_accepted` | How many of those the base model accepted |

Together these confirm the draft model actually contributed to generation,
rather than `--spec-type draft-mtp` silently no-opping (bad/incompatible
`--model-draft` file, draft model failing to load while the base model still
starts fine, etc — see [serverBench.ts:553-559](../worker/src/serverBench.ts:553)).

Per llama-server's own doc comment, these two counters are **absent
entirely** (not zeroed) until the first *completed* speculative request —
LlamaToaster treats "missing from `/metrics`" as a distinct, reportable
condition from "present but zero"
([serverBench.ts:519-522](../worker/src/serverBench.ts:519),
[serverBench.ts:541-550](../worker/src/serverBench.ts:541)): a fetch that
returns no counters at all logs the warning *"MTP: /metrics reported no
speculative-decoding counters — cannot confirm the draft model actually ran"*
([serverBench.ts:563-565](../worker/src/serverBench.ts:563)).

`/metrics` is queried once, right before the server is torn down (it's only
reachable while `llama-server` is still running), and is treated as
diagnostic-only — a failed or unreachable `/metrics` call never fails an
otherwise-good benchmark run
([serverBench.ts:535-538](../worker/src/serverBench.ts:535)).

## Summary: what LlamaToaster actually launches with

```
llama-server ... --metrics --verbosity 5 --no-context-shift   # (last flag only if the build supports it)
```

- `--metrics` → poll `/metrics` once at the end for the two MTP spec-decode counters above.
- `--verbosity 5` → the only level that prints the per-layer device-assignment lines this app needs for an exact resident-GPU-layer count, at the cost of a per-token trace that's bounded (not avoided) at the live-capture stage.
- `--no-context-shift` → see the earlier chat answer / [shared/curves.ts:113](../shared/curves.ts:113); unrelated to verbosity, added conditionally after probing build support.
