import type { FastifyInstance } from "fastify";

// The short, brandable entry point for setting up a worker machine:
//
//   irm https://llamatoaster.com/install.ps1 | iex
//   curl -fsSL https://llamatoaster.com/install.sh | bash
//
// These are 302 REDIRECTS to the pinned raw.githubusercontent.com URL of the
// bootstrap script in this repo -- deliberately not a server-generated or
// server-proxied copy of the file. Piping a URL into a shell is a trust
// decision, and a redirect keeps the bytes that actually execute auditable:
// they come from the public repo, where anyone can read the file, its blame
// and its history, and confirm that what ran is what's committed. A
// dynamically generated payload would be unverifiable by construction, and a
// compromised (or selectively-responding) origin would be undetectable. The
// only thing this origin contributes is the short name.
//
// Because of that, this route CANNOT inject a per-deployment -Url/--url into
// the script. It doesn't need to: both bootstrap scripts default that
// parameter to https://llamatoaster.com, and a self-hoster passes their own
// explicitly, exactly as before (see worker/bootstrap.ps1's header).
//
// Kept in sync with client/src/components/WorkerCard.tsx's
// buildSetupScenarios (the single source of truth for the copy-paste
// commands shown in the UI) and README.md.
const RAW_BASE = "https://raw.githubusercontent.com/noname9006/LlamaToaster";

// The ref the redirect points at. "main" rather than a pinned commit SHA so
// a fix to the installer reaches new machines without a server deploy --
// the same tradeoff the previously-documented raw GitHub one-liner already
// made, since it hardcoded /main/ too.
const REF = "main";

const SCRIPTS: Record<string, { file: string; shell: string }> = {
  "/install.ps1": { file: "worker/bootstrap.ps1", shell: "PowerShell" },
  "/install.sh": { file: "worker/bootstrap.sh", shell: "bash" },
};

export async function installRoutes(app: FastifyInstance): Promise<void> {
  for (const [path, { file }] of Object.entries(SCRIPTS)) {
    app.get(path, async (_req, reply) => {
      // 302, not 301: a permanent redirect gets cached by intermediaries and
      // by curl/PowerShell's own handling, which would make changing the ref
      // (or moving the file) unfixable for anyone who fetched it once.
      return reply.redirect(`${RAW_BASE}/${REF}/${file}`, 302);
    });
  }
}
