// The synthetic filler prompt every llama-server-driven benchmark sends.
//
// Prompts go out PRE-TOKENIZED (an array of token ids, not a string) because
// these benchmarks are defined by an exact token count -- the same semantics
// llama-bench's own `-p N` has. A text string would make the count tokenizer-
// dependent, so the count is chosen and the ids are supplied directly.
//
// Two separate properties have to hold, and each was expensive to learn.
//
// --- 1. The ids must decode to VALID UTF-8 --------------------------------
//
// llama.cpp's chat-output PEG parser (common/peg-parser.cpp) hard-fails a whole
// response the instant one byte sequence is classified INVALID UTF-8: the "any
// codepoint" primitive underlying rest()/content() -- what even the trivial
// always-succeed Content-only grammar is built from -- returns an unconditional
// FAIL there, without consulting the parser's lenient flag. llama-server runs
// that parser over /completion output too, so the request comes back as
//
//   {"error":{"code":500,"message":"The model produced output that does not
//    match the expected Content-only format"}}
//
// with no timings at all. No request field or server flag makes it tolerant --
// confirmed live against b10793 that --reasoning-format, -rea off,
// --skip-chat-parsing and --reasoning-budget all still hit it (upstream
// ggml-org/llama.cpp#25072).
//
// The filler used to be an arithmetic range of ids starting at 100. In a
// byte-BPE vocabulary (gpt2-style pre-tokenizer: Qwen, Llama 3 and friends) ids
// ~94-255 are the RAW SINGLE BYTE tokens, including the 0x80-0xFF ones that are
// not valid UTF-8 alone -- /detokenize returns nothing but U+FFFD for that
// range. Under greedy decoding the model answers that in kind: reproduced live
// on Qwen3.8-27B, where the first generated token (n_predict=1 suffices) was the
// byte pair 8C 80, a bare continuation sequence, failing the parser before a
// single content frame was sent. Tokenizing real text instead makes the ids
// valid UTF-8 pieces by construction, in whatever vocabulary the model has.
//
// --- 2. The text must READ LIKE REAL, MIXED CONTENT -----------------------
//
// On a mixture-of-experts model with experts on CPU (--n-cpu-moe), prompt
// content changes measured PREFILL speed by up to 58%. Prefill processes a whole
// batch at once, so the batch must read every expert ANY token in it routes to;
// low-entropy input collapses routing onto few experts and reads
// unrealistically fast. Generation decodes one token at a time and always
// touches exactly n_expert_used experts, which is why tg is unaffected.
//
// Measured live on Qwen3.6-35B-A3B (--n-cpu-moe 99, pp768, 5 interleaved rounds,
// llama-bench on the same model and flags = 132.8 tok/s as the reference):
//
//   38-word soup, 20 distinct ids ....... 225.6 tok/s   (+70% vs llama-bench)
//   English prose only .................. 142.7 tok/s   (+7%)
//   these 5 blocks ...................... 130.0 tok/s   (-2%)
//   + chat, logs, CSV (8 blocks) ........ 129.8 tok/s   (-2%)
//   + 4 more scripts, XML/YAML (10) ..... 130.7 tok/s   (-2%)
//
// So it saturates at five registers -- doubling them moves nothing. Distinct
// token count is NOT the variable (34-distinct prose and 329-distinct prose
// measure identically; 765 distinct tokens of synthetic word soup read 14%
// fast); how far the batch's hidden states diverge is.
//
// Deliberately NOT adding more scripts: a model whose vocabulary doesn't cover
// one tokenizes it by byte fallback, putting raw byte fragments back into the
// prompt -- the exact condition described in part 1. Qwen covers Cyrillic and
// Chinese natively; a narrower vocabulary might not, and the extra scripts buy
// nothing measurable.

/** One register each. Changing these changes what the benchmarks measure. */
const FILLER_BLOCKS: readonly string[] = [
  // Narrative prose.
  "The harbour master logged every vessel clearing the northern channel before dawn, " +
    "noting tonnage, draught and destination in a ledger whose columns had not changed " +
    "since the war. Gulls worked the wake of returning trawlers while gantry cranes swung " +
    "containers onto flatbed rail cars bound for the interior provinces.",
  // Source code, two languages plus SQL.
  "export async function fetchWithRetry(url: string, attempts = 3): Promise<Response> {\n" +
    "  for (let i = 0; i < attempts; i++) {\n" +
    "    try { const res = await fetch(url); if (!res.ok) throw new Error(`HTTP ${res.status}`); return res; }\n" +
    "    catch (err) { await new Promise(r => setTimeout(r, 2 ** i * 100)); }\n" +
    "  }\n" +
    "}\n" +
    "SELECT worker_id, COUNT(*) AS n FROM results WHERE test_type = 'pp' GROUP BY worker_id HAVING n >= 5;",
  // Mathematics: symbols, digits, notation.
  "Let f(x) = integral from 0 to x of e^(-t^2) dt, so f'(x) = e^(-x^2). For n >= 1, " +
    "sum k^3 = (n(n+1)/2)^2. sigma^2 = 14.2857 gives sigma = 3.7796 and a 95% CI of " +
    "[12.41, 19.87]. Matrix [[2,-1,0],[-1,2,-1],[0,-1,2]] has eigenvalues 2 - sqrt(2), 2, 2 + sqrt(2).",
  // Structured data.
  '{"worker":"rig-04","gpu":"RX 6600 XT","vram_mib":8176,"results":[{"test":"pp768","tps":132.84}]}\n' +
    "| model | ngl | pp768 | tg16 |\n|---|---|---|---|\n| 35B-A3B | 99 | 132.84 | 14.09 |",
  // Non-Latin scripts, both natively covered by the vocabularies this targets.
  "Портовый смотритель записывал каждое судно, прошедшее северный канал до рассвета.\n" +
    "港务长在黎明前记录了每一艘通过北部航道的船只，注明吨位和目的地。",
];

// Stride between nonces. Prime, so distinct nonces land on distinct rotations
// of any block whose length isn't a multiple of it.
const NONCE_STRIDE = 7919;

// How many consecutive tokens one block contributes before the next block takes
// over. Must stay well below llama-server's ubatch (512) so EVERY prefill batch
// sees every register -- laying the blocks out contiguously instead segregates
// one register per ubatch at large prompt sizes and reads ~3% fast (measured at
// n_prompt 2560, where 5 contiguous blocks fall one per ubatch exactly).
const INTERLEAVE_CHUNK = 64;

const TOKENIZE_TIMEOUT_MS = 30_000;

/** A tokenized filler block per FILLER_BLOCKS entry, in the same order. */
export type FillerBlocks = number[][];

/**
 * Asks the running llama-server to tokenize each filler block, giving ids that
 * are valid UTF-8 pieces IN THAT MODEL'S OWN VOCABULARY.
 *
 * Throws rather than falling back to anything. A silent fallback would emit a
 * number measured against different content, and the measurements above show
 * that number can be 70% off -- an incomparable row in the database is worse
 * than a failed test. The server has already passed its health check by the
 * time this runs, so failure here is genuinely exceptional.
 */
export async function fetchFillerBlocks(port: number): Promise<FillerBlocks> {
  const blocks = await Promise.all(
    FILLER_BLOCKS.map(async (content, i) => {
      let res: Response;
      try {
        res = await fetch(`http://127.0.0.1:${port}/tokenize`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          // No BOS/EOS: those are special tokens, and this is filler, not a turn.
          body: JSON.stringify({ content, add_special: false }),
          signal: AbortSignal.timeout(TOKENIZE_TIMEOUT_MS),
        });
      } catch (err) {
        throw new Error(
          `filler prompt: /tokenize request failed for block ${i}: ${err instanceof Error ? err.message : String(err)}`
        );
      }
      if (!res.ok) throw new Error(`filler prompt: /tokenize returned ${res.status} for block ${i}`);
      const data = (await res.json()) as { tokens?: unknown };
      if (!Array.isArray(data.tokens)) throw new Error(`filler prompt: /tokenize returned no tokens for block ${i}`);
      const tokens = data.tokens.filter((t): t is number => typeof t === "number" && Number.isFinite(t));
      if (tokens.length === 0) throw new Error(`filler prompt: block ${i} tokenized to nothing`);
      return tokens;
    })
  );
  return blocks;
}

/**
 * A synthetic prompt of exactly `tokenCount` tokens.
 *
 * Every block gets an equal share of the total (the remainder spread across the
 * leading blocks), so the register mix is IDENTICAL at 512 tokens and at 32768.
 * That matters because N1 curves measure prefill at increasing depths: if the
 * mix drifted with size, the curve would conflate depth scaling with a changing
 * content mix, and content is worth up to 58% on MoE.
 *
 * `offset` and `nonce` both rotate each block WITHIN its own share, so the mix
 * is preserved while the sequence still differs. `nonce` gives N5's concurrent
 * streams (and N1's warm/discard request) prompts the prefix cache cannot
 * dedupe; `offset` is the retry knob -- a request that tripped the parser bug is
 * retried with a different offset rather than replayed byte-for-byte, which
 * under greedy decoding would fail identically.
 */
export function buildPromptTokens(
  tokenCount: number,
  offset = 0,
  nonce = 0,
  blocks: FillerBlocks
): number[] {
  const count = Math.max(0, Math.trunc(tokenCount));
  if (count === 0) return [];
  const usable = blocks.filter((b) => b.length > 0);
  if (usable.length === 0) throw new Error("filler prompt: no usable token blocks");

  const base = Math.floor(count / usable.length);
  const extra = count % usable.length;
  const remaining = usable.map((_, i) => base + (i < extra ? 1 : 0));
  const start = offset + nonce * NONCE_STRIDE;
  const cursor = usable.map((b) => ((start % b.length) + b.length) % b.length);

  const out: number[] = [];
  while (out.length < count) {
    let progressed = false;
    for (let i = 0; i < usable.length; i++) {
      const take = Math.min(INTERLEAVE_CHUNK, remaining[i]);
      for (let j = 0; j < take; j++) {
        out.push(usable[i][cursor[i]]);
        cursor[i] = (cursor[i] + 1) % usable[i].length;
      }
      remaining[i] -= take;
      if (take > 0) progressed = true;
    }
    // Defensive: shares sum to `count`, so this can only fire if every block
    // is exhausted, which would mean the arithmetic above is wrong.
    if (!progressed) break;
  }
  return out;
}
