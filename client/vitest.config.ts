import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

// The client's own Vitest project, referenced from the repo-root config's
// `test.projects` (see ../vitest.config.ts). It exists as a separate project
// -- rather than a second `environment` on the root config -- because the
// backend/worker/shared suites must keep running under Node: giving the whole
// repo a jsdom environment would put a fake `window`/`document` in front of
// server code that legitimately branches on their absence, and would slow
// every one of the 65 existing test files down for no benefit.
//
// Root is this directory, so `react` and @testing-library resolve from
// client/node_modules (they are client devDependencies) rather than the repo
// root, which has no React at all.
//
// `jsdom` is the exception and lives in the ROOT devDependencies: Vitest loads
// the environment package relative to its own installation, not to the
// project root, so a client-local jsdom fails with "Cannot find package
// 'jsdom'" before any test runs. Keep it at the root alongside vitest.
export default defineConfig({
  plugins: [react()],
  test: {
    name: "client",
    environment: "jsdom",
    // Only this package's own tests. `../shared` is compiled into the client
    // bundle but its tests are pure logic and already run in the Node project;
    // including them here would execute every shared test twice.
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: [join(__dirname, "src/test/setup.ts")],
    // Reset call history and spy implementations between tests, so a
    // `vi.spyOn(Storage.prototype, ...)` left by one case cannot change the
    // behaviour of the next. Cross-FILE isolation is separate: Vitest already
    // gives each file its own environment, and the storage reset that depends
    // on lives in src/test/setup.ts's beforeEach.
    restoreMocks: true,
    clearMocks: true,
  },
});
