# llama-server verbosity levels

`--verbosity` is a `llama-server` (llama.cpp) flag, not a LlamaToaster setting. This
doc compares what each level prints, and what LlamaToaster itself relies on and
sets (`--verbosity 4`, plus `--metrics` for the `/metrics` endpoint). See
[loadDriver.ts:96](../worker/src/loadDriver.ts:96) and
[serverBench.ts:249](../worker/src/serverBench.ts:249) for where these are set.

## The scale

Per `llama-server --help`, confirmed live against the installed b10405 build
(both at model load and mid-generation via a real `/completion` request —
see [serverBench.ts:250](../worker/src/serverBench.ts:250)):

| Level | Name | What it adds over the level below |
|---|---|---|
| 0 | generic | Startup banner, listen address, fatal errors only |
| 1 | error | + error conditions (failed loads, bad requests) |
| 2 | warning | + warnings (deprecated flags, recoverable issues) |
| 3 | info (**default**) | + high-level lifecycle: server ready, model load start/finish, request accepted. **No per-tensor detail, no offload summary line.** |
| 4 | trace | + `load_tensors: offloaded X/Y layers to GPU`, per-layer "layer N assigned to device" lines, slot lifecycle, context checkpoints, `print_timing` | 
| 5 | debug | + per-token / per-draft-candidate trace — every generated token and (under MTP) every draft-candidate decision logged individually |

Level 3 (the default) is not enough for LlamaToaster: it prints **none** of the
offload-layer detail that [index.ts's `parseOffloadLayers`](../worker/src/index.ts)
needs to report `gpu_layers_loaded`/`total_model_layers`
(see [bench.ts:75-78](../worker/src/bench.ts:75)). That detail only appears at
level 4+.

Level 5 was tried first (as `999`, before this flag was tuned) and rejected:
it floods the mirrored worker log with a per-token/per-draft-candidate trace
that nothing in this app reads — see
[serverBench.ts:258-261](../worker/src/serverBench.ts:258).

**LlamaToaster always runs `llama-server` at `--verbosity 4`.** It's the
lowest level that surfaces the offload line and everything else this app
parses, while producing zero of level 5's noise.

### llama-bench's equivalent

`llama-bench` (the non-MTP benchmark path) doesn't have a numeric scale —
just a boolean `-v`/`--verbose` flag, gated behind a runtime probe
(`supportsVerboseFlag`, [bench.ts:81](../worker/src/bench.ts:81) — older
builds don't support it). Off, it prints none of the per-tensor/offload
detail either; on, it prints the same one-line-per-tensor load/repack spam
and the same offload-summary line that llama-server prints at level 4
(confirmed live — see [bench.ts:508-512](../worker/src/bench.ts:508)).

### What LlamaToaster does with the level-4 output

The raw stderr is always saved unfiltered to the per-item raw JSON dump,
regardless of level. The worker's own text log is different: mirroring
level-4 output there unfiltered used to bury the structured params/offload/
summary lines this app logs for every test under tens of thousands of
characters of tensor noise, so `logDiagnosticOutput` now mirrors it only at
debug log level, and any single line over 300 chars is elided (to guard
against a request/response body — including this app's own prompts/generated
text — appearing verbatim in a trace-level line). See
[index.ts:1295-1320](../worker/src/index.ts:1295).

## `--metrics`: the `/metrics` endpoint

Separately from `--verbosity`, `--metrics` turns on a Prometheus-format
`/metrics` HTTP endpoint (not stderr logging at all — it's polled over HTTP
while the server is up). llama-server's own metrics generally include
per-slot/global counters like prompt and predicted token counts,
prompt/predicted processing speed, KV-cache usage, and in-flight/deferred
request counts — the exact set depends on the build.

**LlamaToaster only reads two counters from it**, both spec-decode (MTP)
related, matched by regex in
[serverBench.ts:514-515](../worker/src/serverBench.ts:514):

| Metric | LlamaToaster field | Purpose |
|---|---|---|
| `llamacpp:spec_decode_num_draft_tokens_total` | `spec_drafted` | How many tokens the draft (MTP) model proposed |
| `llamacpp:spec_decode_num_accepted_tokens_total` | `spec_accepted` | How many of those the base model accepted |

Together these confirm the draft model actually contributed to generation,
rather than `--spec-type draft-mtp` silently no-opping (bad/incompatible
`--model-draft` file, draft model failing to load while the base model still
starts fine, etc — see [serverBench.ts:244-248](../worker/src/serverBench.ts:244)).

Per llama-server's own doc comment, these two counters are **absent
entirely** (not zeroed) until the first *completed* speculative request —
LlamaToaster treats "missing from `/metrics`" as a distinct, reportable
condition from "present but zero"
([serverBench.ts:511-513](../worker/src/serverBench.ts:511),
[serverBench.ts:532-535](../worker/src/serverBench.ts:532)): a fetch that
returns no counters at all logs the warning *"MTP: /metrics reported no
speculative-decoding counters — cannot confirm the draft model actually ran"*
([serverBench.ts:554-556](../worker/src/serverBench.ts:554)).

`/metrics` is queried once, right before the server is torn down (it's only
reachable while `llama-server` is still running), and is treated as
diagnostic-only — a failed or unreachable `/metrics` call never fails an
otherwise-good benchmark run
([serverBench.ts:526-528](../worker/src/serverBench.ts:526)).

## Summary: what LlamaToaster actually launches with

```
llama-server ... --metrics --verbosity 4 --no-context-shift   # (last flag only if the build supports it)
```

- `--metrics` → poll `/metrics` once at the end for the two MTP spec-decode counters above.
- `--verbosity 4` → the minimum level that prints the offload-layers line and per-tensor load detail this app parses, without level 5's per-token trace noise.
- `--no-context-shift` → see the earlier chat answer / [shared/curves.ts:113](../shared/curves.ts:113); unrelated to verbosity, added conditionally after probing build support.
