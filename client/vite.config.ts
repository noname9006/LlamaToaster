import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev-only: the Fastify server (`npm run server`) runs separately and isn't
// started by Vite. Point this at wherever it's actually listening if not the
// default -- e.g. `VITE_API_PROXY_TARGET=http://127.0.0.1:4010 npm run dev:client`.
const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:3000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": apiProxyTarget,
      "/health": apiProxyTarget,
      // Multi-user Stage 2 (MULTIUSER_PLAN.md §2.4/§2.5): browser-facing
      // OAuth/session routes the same server also owns in production
      // (same-origin, §0.5) -- without this, vite's own SPA fallback
      // silently swallows /auth/github, /auth/github/callback, and
      // /auth/logout in dev instead of ever reaching the real backend.
      "/auth": apiProxyTarget,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
