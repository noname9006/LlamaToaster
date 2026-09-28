import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import { BUILTIN_ROCM_SUPPORT, isRocmSupportedGpu, type RocmSupportSnapshot } from "../../shared/rocmSupport.js";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-rocm-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let mod: typeof import("./rocm-support.js");
let getDb: (typeof import("./db/migrate.js"))["getDb"];
let app: FastifyInstance;

beforeAll(async () => {
  mod = await import("./rocm-support.js");
  ({ getDb } = await import("./db/migrate.js"));
  const { rocmSupportRoutes } = await import("./routes/rocm-support.js");
  app = Fastify();
  await app.register(rocmSupportRoutes);
  await app.ready();
});

afterAll(async () => {
  await app.close();
  try {
    getDb().close();
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* Windows temp-dir handle */
  }
});

beforeEach(() => {
  getDb().prepare(`DELETE FROM meta WHERE key LIKE 'rocm_support_%'`).run();
});

// Sphinx-shaped tables like the real pages: <tr><td><p>cell</p></td>...
function table(headers: string[], rows: string[][]): string {
  const th = headers.map((h) => `<th class="head"><p>${h}</p></th>`).join(" ");
  const trs = rows.map((r) => `<tr>${r.map((c) => `<td><p>${c}</p></td>`).join(" ")}</tr>`).join("\n");
  return `<table class="table"><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
}

const WIN_HEAD = ["Name", "Architecture", "LLVM target", "Runtime", "HIP SDK", "AMD ROCm Debugger"];
const yes = "✅";
const no = "❌";
const winRow = (name: string, gfx: string, ok: boolean) => [name, "RDNA", gfx, ok ? yes : no, ok ? yes : no, no];

function windowsPage(extraSupported: string[] = []): string {
  const supported = [
    "AMD Radeon RX 9070 XT",
    "AMD RX 9070 GRE",
    "AMD Radeon RX 7900 XTX",
    "AMD Radeon RX 7900 XT",
    "AMD Radeon RX 7800 XT",
    "AMD Radeon PRO W7900",
    ...extraSupported,
  ];
  return [
    // Unrelated table: OS support -- must be ignored.
    table(["Distribution", "Processor architectures", "Validated update"], [["Windows 11", "x86-64", "22H2 (GA)"]]),
    table(WIN_HEAD, [
      ...supported.map((n) => winRow(n, "gfx1100", true)),
      winRow("AMD Radeon PRO W6800", "gfx1030", false),
      winRow("AMD Radeon RX 6600 XT", "gfx1032", false),
    ]),
    // APU table: rows are CPU names -- skipped.
    table(["Name", "Architecture", "LLVM target", "Runtime", "HIP SDK"], [
      ["AMD Ryzen AI Max+ 395", "RDNA3.5", "gfx1151", yes, yes],
    ]),
  ].join("\n");
}

const LIN_HEAD = ["GPU", "Architecture", "LLVM target", "Support"];
function linuxPage(): string {
  return [
    table(["GPU", "Series", "Architecture", "LLVM target", "Support"], [
      ["AMD Instinct MI300X", "MI300", "CDNA3", "gfx942", `${yes} [3]`],
      ["AMD Instinct MI250X", "MI200", "CDNA2", "gfx90a", `${yes} [5]`],
      ["AMD Instinct MI250", "MI200", "CDNA2", "gfx90a", `${yes} [5]`],
      ["AMD Instinct MI50", "MI50", "GCN5.1", "gfx906", no],
    ]),
    table(LIN_HEAD, [
      ["AMD Radeon AI PRO R9700", "RDNA4", "gfx1201", `${yes} [7]`],
      ["AMD Radeon PRO W6800", "RDNA2", "gfx1030", `${yes} [7]`],
      ["AMD Radeon PRO VII", "GCN5.1", "gfx906", no],
    ]),
    table(LIN_HEAD, [
      ["AMD Radeon RX 9070 XT", "RDNA4", "gfx1201", `${yes} [7]`],
      ["AMD Radeon RX 7900 XTX", "RDNA3", "gfx1100", `${yes} [7]`],
      ["AMD Radeon RX 7900 GRE", "RDNA3", "gfx1100", `${yes} [7]`],
      ["AMD Radeon RX 7900 XT", "RDNA3", "gfx1100", `${yes} [7]`],
      ["AMD Radeon RX 7800 XT", "RDNA3", "gfx1101", `${yes} [7]`],
    ]),
    // Unrelated table with a "Support" column but no LLVM target.
    table(["Operating system", "Kernel", "Glibc", "Support"], [["Ubuntu 24.04.4", "6.8", "2.39", yes]]),
  ].join("\n");
}

function fakeFetch(pages: { win?: string | Error; lin?: string | Error }): typeof fetch {
  return (async (url: string) => {
    const page = String(url).includes("install-on-windows") ? pages.win : pages.lin;
    if (page instanceof Error) throw page;
    if (page === undefined) return new Response("nope", { status: 503 });
    return new Response(page, { status: 200, headers: { "content-type": "text/html" } });
  }) as typeof fetch;
}

describe("page parsing", () => {
  it("reads the Windows tables' Runtime column and skips APU and unrelated tables", () => {
    const p = mod.parseWindowsRocmPage(windowsPage());
    expect(p.supported).toEqual(
      ["pro w7900", "rx 7800 xt", "rx 7900 xt", "rx 7900 xtx", "rx 9070 gre", "rx 9070 xt"].sort()
    );
    expect(p.unsupported).toEqual(["pro w6800", "rx 6600 xt"]);
    expect([...p.supported, ...p.unsupported].join(" ")).not.toContain("ryzen");
  });

  it("reads the Linux tables' Support column, incl. footnote markers, and ignores the OS table", () => {
    const p = mod.parseLinuxRocmPage(linuxPage());
    expect(p.supported).toContain("mi300x");
    expect(p.supported).toContain("ai pro r9700");
    expect(p.supported).toContain("pro w6800");
    expect(p.supported).toContain("rx 7900 gre");
    expect(p.unsupported.sort()).toEqual(["mi50", "pro vii"]);
    expect(p.supported.join(" ")).not.toContain("ubuntu");
  });

  it("tolerates entities and nested markup inside cells", () => {
    const html = table(WIN_HEAD, [
      ["<a href='x'>AMD&nbsp;Radeon RX 7900 XTX</a>", "RDNA3", "gfx1100", `<span>${yes}</span>`, yes, no],
    ]);
    expect(mod.parseWindowsRocmPage(html).supported).toEqual(["rx 7900 xtx"]);
  });

  it("returns empty lists for a page with no recognizable tables", () => {
    expect(mod.parseWindowsRocmPage("<html><body>redesigned</body></html>")).toEqual({ supported: [], unsupported: [] });
  });

  it("yields lists that classify the user's RX 6600 XT as unsupported on both platforms", () => {
    const list = { win32: mod.parseWindowsRocmPage(windowsPage()), linux: mod.parseLinuxRocmPage(linuxPage()) };
    expect(isRocmSupportedGpu("AMD Radeon RX 6600 XT", "win32", list)).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon RX 6600 XT", "linux", list)).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon RX 7900 XTX", "win32", list)).toBe(true);
    expect(isRocmSupportedGpu("AMD Radeon PRO W6800", "win32", list)).toBe(false);
    expect(isRocmSupportedGpu("AMD Radeon PRO W6800", "linux", list)).toBe(true);
  });
});

describe("validateRocmList", () => {
  const mk = (win: number, lin: number) => ({
    win32: { supported: Array.from({ length: win }, (_, i) => `w${i}`), unsupported: [] },
    linux: { supported: Array.from({ length: lin }, (_, i) => `l${i}`), unsupported: [] },
  });

  it("accepts a normal list and rejects a nearly-empty parse", () => {
    expect(mod.validateRocmList(mk(19, 24), null)).toBeNull();
    expect(mod.validateRocmList(mk(2, 24), null)).toMatch(/win32.*only 2/);
    expect(mod.validateRocmList(mk(19, 0), null)).toMatch(/linux.*only 0/);
  });

  it("rejects a drop of more than half versus the previous list, allows a modest one", () => {
    expect(mod.validateRocmList(mk(8, 24), mk(19, 24))).toMatch(/win32.*shrank/);
    expect(mod.validateRocmList(mk(15, 20), mk(19, 24))).toBeNull();
  });
});

describe("refreshRocmSupport", () => {
  it("starts on the built-in list, then serves the fetched one", async () => {
    expect(mod.getRocmSupportSnapshot()).toMatchObject({ source: "builtin", fetchedAt: null });
    expect(mod.getRocmSupportSnapshot().list).toEqual(BUILTIN_ROCM_SUPPORT);

    const out = await mod.refreshRocmSupport(fakeFetch({ win: windowsPage(), lin: linuxPage() }));
    expect(out).toEqual({ ok: true, changed: true });

    const snap = mod.getRocmSupportSnapshot();
    expect(snap.source).toBe("live");
    expect(Number.isNaN(Date.parse(snap.fetchedAt ?? ""))).toBe(false);
    expect(snap.list.win32.unsupported).toContain("rx 6600 xt");
  });

  it("reports unchanged when AMD's pages didn't change", async () => {
    const f = fakeFetch({ win: windowsPage(), lin: linuxPage() });
    await mod.refreshRocmSupport(f);
    expect(await mod.refreshRocmSupport(f)).toEqual({ ok: true, changed: false });
  });

  it("keeps the previous list when a fetch fails, times out, or returns an error status", async () => {
    await mod.refreshRocmSupport(fakeFetch({ win: windowsPage(), lin: linuxPage() }));
    const before = mod.getRocmSupportSnapshot();

    const netDown = await mod.refreshRocmSupport(fakeFetch({ win: new Error("ECONNREFUSED"), lin: linuxPage() }));
    expect(netDown).toMatchObject({ ok: false });
    const http503 = await mod.refreshRocmSupport(fakeFetch({ win: windowsPage() }));
    expect(http503).toMatchObject({ ok: false, reason: expect.stringContaining("503") });

    expect(mod.getRocmSupportSnapshot()).toEqual(before);
  });

  it("refuses a redesigned page (no tables parse) instead of wiping the list", async () => {
    await mod.refreshRocmSupport(fakeFetch({ win: windowsPage(), lin: linuxPage() }));
    const before = mod.getRocmSupportSnapshot();

    const out = await mod.refreshRocmSupport(fakeFetch({ win: "<html>new layout</html>", lin: linuxPage() }));
    expect(out).toMatchObject({ ok: false, reason: expect.stringContaining("layout") });
    expect(mod.getRocmSupportSnapshot()).toEqual(before);
  });

  it("falls back to the built-in list if the stored JSON is corrupt", () => {
    getDb().prepare(`INSERT INTO meta (key, value) VALUES ('rocm_support_list', 'not json')`).run();
    expect(mod.getRocmSupportSnapshot().source).toBe("builtin");
  });
});

describe("isRocmRefreshDue", () => {
  const DAY = 24 * 60 * 60 * 1000;
  const HOUR = 60 * 60 * 1000;
  const set = (key: string, value: string) =>
    getDb().prepare(`INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value`).run(key, value);

  it("is due when nothing was ever fetched", () => {
    expect(mod.isRocmRefreshDue()).toBe(true);
  });

  it("is not due within a day of a successful fetch, and is due after", () => {
    const now = Date.now();
    set("rocm_support_fetched_at", new Date(now - 23 * HOUR).toISOString());
    expect(mod.isRocmRefreshDue(now)).toBe(false);
    set("rocm_support_fetched_at", new Date(now - 25 * HOUR).toISOString());
    expect(mod.isRocmRefreshDue(now)).toBe(true);
  });

  it("backs off an hour after a failed attempt even though the list is stale", () => {
    const now = Date.now();
    set("rocm_support_fetched_at", new Date(now - 3 * DAY).toISOString());
    set("rocm_support_attempt_at", String(now - 10 * 60 * 1000));
    expect(mod.isRocmRefreshDue(now)).toBe(false);
    set("rocm_support_attempt_at", String(now - 61 * 60 * 1000));
    expect(mod.isRocmRefreshDue(now)).toBe(true);
  });

  it("records an attempt on every refresh, so a failing source isn't retried until the backoff passes", async () => {
    await mod.refreshRocmSupport(fakeFetch({ win: new Error("down"), lin: new Error("down") }));
    expect(mod.isRocmRefreshDue()).toBe(false);
  });
});

describe("GET /api/rocm-support", () => {
  it("serves the built-in snapshot before any fetch, then the live one", async () => {
    const before = (await app.inject({ method: "GET", url: "/api/rocm-support" })).json<RocmSupportSnapshot>();
    expect(before.source).toBe("builtin");

    await mod.refreshRocmSupport(fakeFetch({ win: windowsPage(), lin: linuxPage() }));
    const res = await app.inject({ method: "GET", url: "/api/rocm-support" });
    expect(res.statusCode).toBe(200);
    expect(res.headers["cache-control"]).toContain("max-age");
    const after = res.json<RocmSupportSnapshot>();
    expect(after.source).toBe("live");
    expect(after.list.linux.supported).toContain("rx 7900 gre");
  });
});
