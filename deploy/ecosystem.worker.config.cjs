// PM2 process definition for the worker only, as an alternative to
// llamatoaster-worker.service (the systemd unit in this same folder) -- see
// ecosystem.server.config.cjs's own header comment for why this is a
// separate file rather than a shared apps:[] array (MULTIUSER_PLAN.md §6.0).
//
// WORKER_CONFIG: as of MULTIUSER_PLAN.md §6.0, the VPS's own worker is no
// longer a special-cased local/remote pair under the old tailnet model --
// it's just one more machine enrolled under the operator's own account, the
// same as any user's GPU box (§3). That means it needs a REAL config.json
// produced by actually running worker/bootstrap.sh on this VPS and
// approving the resulting device-flow code from the operator's own logged-in
// session (see /device, or "Reconnect" on the machine's card once enrolled)
// -- not a bespoke deploy-time file baked into this repo checkout. There is
// no correct default path to bake in here ahead of time: run bootstrap.sh
// first (it can install anywhere, e.g. ~/llamatoaster-worker), note where it
// wrote config.json, then point WORKER_CONFIG at that exact path below.
const fs = require("fs");
const path = require("path");

const repoRoot = path.join(__dirname, "..");

// CHANGE ME (or set LLAMATOASTER_VPS_WORKER_CONFIG in the shell that runs
// `pm2 start`) after enrolling this VPS's own worker via bootstrap.sh -- an
// absolute path, since this worker's config lives OUTSIDE this repo
// checkout. Matches llamatoaster-worker.service's own placeholder so the PM2
// and systemd paths stay interchangeable.
const WORKER_CONFIG_PATH = process.env.LLAMATOASTER_VPS_WORKER_CONFIG || "/home/ubuntu/.llamatoaster-worker/config.json";

if (!fs.existsSync(WORKER_CONFIG_PATH)) {
  console.error(
    `[ecosystem.worker.config.cjs] ${WORKER_CONFIG_PATH} does not exist -- ` +
      "run worker/bootstrap.sh on this VPS and approve its device-flow code first, " +
      "then set LLAMATOASTER_VPS_WORKER_CONFIG (or edit WORKER_CONFIG_PATH above) to where it wrote config.json."
  );
}

module.exports = {
  apps: [
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
      env: { WORKER_CONFIG: WORKER_CONFIG_PATH },
      autorestart: true,
      restart_delay: 5000,
    },
  ],
};
