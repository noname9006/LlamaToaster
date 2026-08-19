import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev-only: the Fastify server (`npm run server`) runs separately and isn't
// started by Vite -- same pattern as client/vite.config.ts. A distinct
// default port (5174, not client's 5173) so both dev servers can run at
// once without colliding.
const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:3000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174,
    proxy: {
      "/api": apiProxyTarget,
      "/health": apiProxyTarget,
      // The admin origin's own OAuth round trip (MULTIUSER_PLAN.md §5.1) --
      // same reasoning as client/vite.config.ts's own /auth proxy entry.
      "/auth": apiProxyTarget,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
