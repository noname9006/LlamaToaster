import http from "node:http";

// Test-only stand-in for fetch() that can send a real Host header. fetch()
// silently drops one (Host is on the Fetch spec's forbidden list), and the
// admin origin is chosen by the raw Host header alone (routes/admin.ts's
// isAdminHost), so tests that target it go through here. Never follows
// redirects -- callers that pass redirect: "manual" get what they expect.
export function hostFetch(
  input: string,
  init: { method?: string; headers?: Record<string, string>; body?: string; redirect?: "manual" } = {}
): Promise<Response> {
  const url = new URL(input);
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: url.hostname, port: url.port, path: url.pathname + url.search, method: init.method ?? "GET", headers: init.headers },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () => {
          const headers = new Headers();
          for (const [name, value] of Object.entries(res.headers)) {
            for (const v of Array.isArray(value) ? value : value === undefined ? [] : [value]) headers.append(name, v);
          }
          const status = res.statusCode ?? 500;
          const body = [204, 205, 304].includes(status) ? null : Buffer.concat(chunks);
          resolve(new Response(body, { status, headers }));
        });
        res.on("error", reject);
      }
    );
    req.on("error", reject);
    if (init.body !== undefined) req.write(init.body);
    req.end();
  });
}
