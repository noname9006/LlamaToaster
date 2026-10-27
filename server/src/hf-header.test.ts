import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-hf-header-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof import("./db/repo.js")["repo"];
let fetchHfHeader: typeof import("./hf-header.js")["fetchHfHeader"];
let runHfHeaderTick: typeof import("./hf-header.js")["runHfHeaderTick"];

// --- minimal GGUF writer (same format as worker/src/gguf.test.ts's) ---
function u32(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n, 0);
  return b;
}
function u64(n: number): Buffer {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n), 0);
  return b;
}
function str(s: string): Buffer {
  const body = Buffer.from(s, "utf8");
  return Buffer.concat([u64(body.length), body]);
}
function kvString(k: string, v: string): Buffer {
  return Buffer.concat([str(k), u32(8), str(v)]);
}
function kvU32(k: string, v: number): Buffer {
  return Buffer.concat([str(k), u32(4), u32(v)]);
}
// A header padded past several 4 MB range chunks (real tokenizer vocabularies
// make headers this big), so the chunked CDN reading is actually exercised.
function buildGguf(nLayer: number): Buffer {
  const pad = "x".repeat(3 * 1024 * 1024);
  const kvs = [
    kvString("general.architecture", "llama"),
    kvU32("llama.block_count", nLayer),
    kvU32("llama.context_length", 131072),
    kvString("pad.a", pad),
    kvString("pad.b", pad),
    kvString("pad.c", pad),
    kvU32("llama.attention.head_count", 32),
  ];
  return Buffer.concat([u32(0x46554747), u32(3), u64(0), u64(kvs.length), ...kvs]);
}

const SHA = "ab".repeat(32);
const file = buildGguf(16);
// Audit 2026-10-18 A1: a few hundred bytes claiming 300M layers (with one
// tensor and KV geometry, so every per-layer path is reached) used to kill
// the process with an uncatchable heap OOM.
const BOMB_SHA = "b0".repeat(32);
const bomb = Buffer.concat([
  u32(0x46554747), u32(3), u64(1), u64(4),
  kvString("general.architecture", "llama"),
  kvU32("llama.block_count", 300_000_000),
  kvU32("llama.attention.head_count_kv", 8),
  kvU32("llama.attention.key_length", 128),
  Buffer.concat([str("blk.0.attn_q.weight"), u32(1), u64(16), u32(0), u64(0)]),
]);

// Fake HF: resolve/ 302s to a "CDN" path with HF's own headers; the CDN
// honours Range. Behaviour per repo name lets each test pick a scenario.
let server: Server;
let base: string;
let rangeRequests = 0;
let resolveRequests = 0;
beforeAll(async () => {
  ({ repo } = await import("./db/repo.js"));
  ({ fetchHfHeader, runHfHeaderTick } = await import("./hf-header.js"));
  server = createServer((req, res) => {
    const url = req.url ?? "";
    if (url.includes("/resolve/")) {
      resolveRequests++;
      if (url.startsWith("/gated/")) return void res.writeHead(401).end();
      // A renamed repo: same-site redirect to the new name, no file headers yet.
      if (url.startsWith("/moved/")) return void res.writeHead(307, { location: url.replace("/moved/", "/org/") }).end();
      if (url.startsWith("/offsite/")) return void res.writeHead(302, { location: "https://evil.example/f.gguf" }).end();
      if (url.startsWith("/inline/")) return void res.writeHead(200).end("small");
      if (url.startsWith("/bomb/")) {
        res.writeHead(302, { location: "/cdn-bomb", "x-linked-etag": `"${BOMB_SHA}"`, "x-linked-size": String(bomb.length + 64) });
        return void res.end();
      }
      const etag = url.startsWith("/wrongsha/") ? "cd".repeat(32) : url.startsWith("/norange-ef/") ? "ef".repeat(32) : SHA;
      const cdn = url.startsWith("/norange") ? "/cdn-norange" : "/cdn";
      res.writeHead(302, { location: cdn, "x-linked-etag": `"${etag}"`, "x-linked-size": String(file.length) });
      return void res.end();
    }
    if (url === "/cdn") {
      rangeRequests++;
      const m = /bytes=(\d+)-(\d+)/.exec(req.headers.range ?? "");
      if (!m) return void res.writeHead(200).end(file);
      const [s, e] = [Number(m[1]), Number(m[2])];
      res.writeHead(206, { "content-range": `bytes ${s}-${e}/${file.length}` });
      return void res.end(file.subarray(s, e + 1));
    }
    if (url === "/cdn-norange") return void res.writeHead(200).end(file);
    if (url === "/cdn-bomb") {
      // Past the end of the bytes, the "file" is zeros.
      const m = /bytes=(\d+)-(\d+)/.exec(req.headers.range ?? "");
      const padded = Buffer.concat([bomb, Buffer.alloc(64)]);
      const [s, e] = m ? [Number(m[1]), Number(m[2])] : [0, padded.length - 1];
      res.writeHead(206, { "content-range": `bytes ${s}-${e}/${padded.length}` });
      return void res.end(padded.subarray(s, e + 1));
    }
    res.writeHead(404).end();
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("no port");
  base = `http://127.0.0.1:${addr.port}`;
});

afterAll(async () => {
  await new Promise<void>((r) => server.close(() => r()));
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

describe("fetchHfHeader", () => {
  it("reads the header through one resolve request plus a few large CDN range reads", async () => {
    resolveRequests = 0;
    rangeRequests = 0;
    const out = await fetchHfHeader("org/r", "f.gguf", "main", SHA, base);
    expect(out).toEqual({ status: "ok", fields: expect.objectContaining({ n_layer: 16, trained_ctx: 131072 }) });
    expect(resolveRequests).toBe(1);
    expect(rangeRequests).toBeGreaterThan(1); // ~9 MB header, 4 MB chunks
    expect(rangeRequests).toBeLessThan(6);
  });

  it("refuses bytes whose X-Linked-ETag isn't the expected sha256", async () => {
    const out = await fetchHfHeader("wrongsha/r", "f.gguf", "main", SHA, base);
    expect(out).toMatchObject({ status: "unavailable", reason: expect.stringContaining("sha mismatch") });
  });

  it("treats gated/private and non-LFS files as permanently unavailable", async () => {
    expect(await fetchHfHeader("gated/r", "f.gguf", "main", SHA, base)).toMatchObject({ status: "unavailable" });
    expect(await fetchHfHeader("inline/r", "f.gguf", "main", SHA, base)).toMatchObject({ status: "unavailable" });
  });

  it("follows a renamed repo's same-site redirect, but never an off-site one without X-Linked-ETag", async () => {
    expect(await fetchHfHeader("moved/r", "f.gguf", "main", SHA, base)).toMatchObject({ status: "ok" });
    expect(await fetchHfHeader("offsite/r", "f.gguf", "main", SHA, base)).toMatchObject({
      status: "unavailable",
      reason: expect.stringContaining("evil.example"),
    });
  });

  it("a CDN that ignores Range is a retryable failure, never a partial 'ok'", async () => {
    expect(await fetchHfHeader("norange/r", "f.gguf", "main", SHA, base)).toMatchObject({ status: "failed" });
  });

  it("refuses a header claiming an implausible layer count, quickly and without a large allocation", async () => {
    const heapBefore = process.memoryUsage().heapUsed;
    const t0 = Date.now();
    const out = await fetchHfHeader("bomb/r", "f.gguf", "main", BOMB_SHA, base);
    expect(out).toMatchObject({ status: "unavailable", reason: expect.stringContaining("implausible") });
    expect(Date.now() - t0).toBeLessThan(2000);
    expect(process.memoryUsage().heapUsed - heapBefore).toBeLessThan(64 * 1024 * 1024);
  });
});

describe("runHfHeaderTick", () => {
  it("fills the shared record from HF, overriding earlier user reports, and never re-reads an ok file", async () => {
    repo.registerModel({
      id: SHA, filename: "f.gguf", size_bytes: 1, source: "huggingface", hf_repo: "org/r", hf_file: "f.gguf",
      // A liar's earlier report added a field the real file doesn't have.
      metadata: { n_layer: 2, mtp_layers: 3, param_count: 999 },
    });
    repo.recordHeaderReport(SHA, "worker:liar", { n_layer: 2, mtp_layers: 3 });

    await runHfHeaderTick(base);
    expect(repo.getModel(SHA)?.metadata).toMatchObject({ n_layer: 16, trained_ctx: 131072 });
    expect(repo.resolveHeaderFields(SHA).n_layer).toBe(16);
    // Replaced, not merged: the fake key is gone (param_count is only
    // replaced when the HF read carries one -- this fixture has no tensors).
    expect(repo.getModel(SHA)?.metadata.mtp_layers).toBeUndefined();
    expect(repo.getModel(SHA)?.metadata.param_count).toBe(999);

    resolveRequests = 0;
    await runHfHeaderTick(base);
    expect(resolveRequests).toBe(0);
  });

  it("re-reads an ok header from before kv_layout once, and a failed re-read keeps the old header", async () => {
    const old = { n_layer: 16, n_head_kv: 8, head_dim_k: 128, head_dim_v: 128 };
    // Good file: re-read picks up the current reader's fields.
    repo.registerModel({ id: SHA, filename: "f.gguf", size_bytes: 1, source: "huggingface", hf_repo: "org/r", hf_file: "f.gguf" });
    repo.recordHfHeader(SHA, { status: "ok", fields: old });
    expect(repo.listModelsNeedingHfHeader(10, Date.now()).map((m) => m.id)).toContain(SHA);
    // Failing file (CDN ignores Range): the stored header must survive.
    const sha = "12".repeat(32);
    repo.registerModel({ id: sha, filename: "f.gguf", size_bytes: 1, source: "huggingface", hf_repo: "norange/r", hf_file: "f.gguf" });
    repo.recordHfHeader(sha, { status: "ok", fields: old });
    await runHfHeaderTick(base);
    expect(repo.resolveHeaderFields(sha)).toEqual(old);
    // The good file's fixture has no head dims, so it isn't due again.
    expect(repo.listModelsNeedingHfHeader(10, Date.now()).map((m) => m.id)).not.toContain(SHA);
    // The failing one stays due only until the attempt cap.
    for (let i = 0; i < 10; i++) repo.recordHfHeader(sha, { status: "failed", reason: "x" });
    expect(repo.resolveHeaderFields(sha)).toEqual(old);
    expect(repo.listModelsNeedingHfHeader(10, Date.now()).map((m) => m.id)).not.toContain(sha);
  });

  it("a read that never finished (the process died during it) still spends an attempt and backs off", async () => {
    const sha = "cc".repeat(32);
    repo.registerModel({ id: sha, filename: "f.gguf", size_bytes: 1, source: "huggingface", hf_repo: "org/crash", hf_file: "f.gguf" });
    expect(repo.listModelsNeedingHfHeader(50, Date.now()).map((m) => m.id)).toContain(sha);
    // What runHfHeaderTick does before reading -- then the "crash": no outcome is recorded.
    repo.markHfHeaderAttempt(sha);
    expect(repo.listModelsNeedingHfHeader(50, Date.now()).map((m) => m.id)).not.toContain(sha);
    expect(repo.listModelsNeedingHfHeader(50, Date.now() + 11 * 60_000).map((m) => m.id)).toContain(sha);
    // ...and only up to the attempt cap.
    for (let i = 0; i < 10; i++) repo.markHfHeaderAttempt(sha);
    expect(repo.listModelsNeedingHfHeader(50, Date.now() + 365 * 24 * 3600_000).map((m) => m.id)).not.toContain(sha);
  });

  it("counts a finished read once, not twice", async () => {
    const sha = "dd".repeat(32);
    repo.markHfHeaderAttempt(sha);
    repo.recordHfHeader(sha, { status: "failed", reason: "x" }, { attemptCounted: true });
    repo.recordHfHeader(sha, { status: "failed", reason: "y" });
    // Two reads so far: the next one is the third.
    const { getDb } = await import("./db/migrate.js");
    expect((getDb().prepare(`SELECT attempts FROM model_hf_headers WHERE sha256 = ?`).get(sha) as { attempts: number }).attempts).toBe(2);
  });

  it("backs off after a transient failure instead of retrying every tick", async () => {
    const sha = "ef".repeat(32);
    repo.registerModel({
      id: sha, filename: "f.gguf", size_bytes: 1, source: "huggingface", hf_repo: "norange-ef/r", hf_file: "f.gguf",
    });
    await runHfHeaderTick(base);
    resolveRequests = 0;
    await runHfHeaderTick(base);
    expect(resolveRequests).toBe(0);
    expect(repo.listModelsNeedingHfHeader(10, Date.now() + 11 * 60_000).map((m) => m.id)).toContain(sha);
  });
});
