// PM2 process definition for the server only, as an alternative to
// llamatoaster-server.service (the systemd unit in this same folder). Loads
// deploy/orchestrator.env the same way systemd's EnvironmentFile= does --
// PM2 has no built-in equivalent, so this parses it by hand rather than
// adding a dotenv runtime dependency just for deployment tooling.
//
// Deliberately its OWN file, not a shared apps:[] array with the worker
// (MULTIUSER_PLAN.md §6.0) -- server and worker are independent processes
// that must be deployed, and fail, independently, same as the two separate
// systemd units already do. `pm2 start deploy/ecosystem.server.config.cjs` /
// `pm2 reload ...` / `pm2 delete ...` now touches only this process; a
// server deploy (frequent: every code push, see deploy/push-client.ps1) can
// never bounce the worker mid-benchmark.
const fs = require("fs");
const path = require("path");

function loadEnvFile(relPath) {
  const fullPath = path.join(__dirname, relPath);
  const env = {};
  for (const line of fs.readFileSync(fullPath, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    env[trimmed.slice(0, eq)] = trimmed.slice(eq + 1);
  }
  return env;
}

const repoRoot = path.join(__dirname, "..");

module.exports = {
  apps: [
    {
      name: "llamatoaster-server",
      namespace: "llamatoaster",
      cwd: repoRoot,
      script: "npm",
      args: ["run", "server"],
      interpreter: "none",
      env: loadEnvFile("orchestrator.env"),
      autorestart: true,
      restart_delay: 5000,
    },
  ],
};
