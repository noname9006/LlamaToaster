import { beforeAll, afterAll, describe, expect, it } from "vitest";
import Fastify, { type FastifyInstance } from "fastify";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { installRoutes } from "./install.js";

// No DB and no auth hook -- these two routes touch neither, and in production
// they're reachable unauthenticated by construction (auth-middleware.ts lets
// every non-/api GET through, since that's how the SPA's own static assets
// are served). This is exactly what a fresh machine with no account sees.
let app: FastifyInstance;
let baseUrl: string;

beforeAll(async () => {
  app = Fastify({ logger: false });
  await app.register(installRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  await app.close();
});

describe("GET /install.ps1 and /install.sh", () => {
  it("redirects to the PowerShell bootstrap script in the public repo", async () => {
    const res = await fetch(`${baseUrl}/install.ps1`, { redirect: "manual" });
    expect(res.status).toBe(302);
    expect(res.headers.get("location")).toBe(
      "https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1"
    );
  });

  it("redirects to the bash bootstrap script in the public repo", async () => {
    const res = await fetch(`${baseUrl}/install.sh`, { redirect: "manual" });
    expect(res.status).toBe(302);
    expect(res.headers.get("location")).toBe(
      "https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh"
    );
  });

  // 302, not 301: the ref (or the file path) has to stay changeable for
  // people who already fetched it once -- see the route's own comment.
  it("uses a non-permanent redirect so the target stays changeable", async () => {
    for (const path of ["/install.ps1", "/install.sh"]) {
      const res = await fetch(`${baseUrl}${path}`, { redirect: "manual" });
      expect(res.status).not.toBe(301);
      expect(res.status).not.toBe(308);
    }
  });

  // The redirect target is only trustworthy if the file is actually there --
  // a typo in the path would send every new machine to a 404 that curl pipes
  // into bash as empty input, silently doing nothing.
  it("points at files that exist in this checkout", () => {
    const repoRoot = join(import.meta.dirname, "..", "..", "..");
    for (const file of ["worker/bootstrap.ps1", "worker/bootstrap.sh"]) {
      expect(readFileSync(join(repoRoot, file), "utf8").length).toBeGreaterThan(0);
    }
  });
});

// The whole point of the short URL is that it works with no arguments, which
// only holds while the scripts themselves default the server URL. If that
// default is ever dropped, `irm .../install.ps1 | iex` starts erroring with
// "-Url is required" for every new user -- fail here instead.
describe("the bootstrap scripts default their server URL", () => {
  const repoRoot = join(import.meta.dirname, "..", "..", "..");

  it("bootstrap.ps1 has a default -Url", () => {
    const ps1 = readFileSync(join(repoRoot, "worker", "bootstrap.ps1"), "utf8");
    expect(ps1).toMatch(/\[string\]\$Url\s*=\s*"https:\/\//);
  });

  it("bootstrap.sh has a default --url", () => {
    const sh = readFileSync(join(repoRoot, "worker", "bootstrap.sh"), "utf8");
    expect(sh).toMatch(/^URL="https:\/\//m);
  });
});
