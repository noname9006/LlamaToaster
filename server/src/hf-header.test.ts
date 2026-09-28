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
