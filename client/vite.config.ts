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
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
