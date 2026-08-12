// PM2 process definitions for the VPS, as an alternative to the systemd
// units in this same folder. Loads deploy/orchestrator.env the same way
// systemd's EnvironmentFile= does -- PM2 has no built-in equivalent, so this
// parses it by hand rather than adding a dotenv runtime dependency just for
// deployment tooling.
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
    {
      name: "llamatoaster-worker",
      namespace: "llamatoaster",
      cwd: repoRoot,
      // `nice -n 10` mirrors the systemd unit's Nice=10 -- there's no PM2
      // equivalent for CPUWeight, so that half of the resource-contention
      // mitigation is not replicated here.
      script: "nice",
      args: ["-n", "10", "npm", "run", "worker"],
      interpreter: "none",
      env: { WORKER_CONFIG: "worker/config.vps.json" },
      autorestart: true,
      restart_delay: 5000,
    },
  ],
};
