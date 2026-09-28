import type { FastifyInstance } from "fastify";
import { getRocmSupportSnapshot } from "../rocm-support.js";

// The ROCm-supported AMD GPU list (see server/src/rocm-support.ts, refreshed
// daily from AMD's docs). Public on purpose: it is AMD's own published data,
// and a worker needs it at startup, before it has any credential, to pick its
// default backend (worker/src/index.ts). The browser reads it for the same
// per-GPU backend check.
export async function rocmSupportRoutes(app: FastifyInstance) {
  app.get("/api/rocm-support", { logLevel: "silent" }, async (_req, reply) => {
    // Changes at most daily -- let browsers/proxies reuse it for an hour.
    reply.header("cache-control", "public, max-age=3600");
    return getRocmSupportSnapshot();
  });
}
