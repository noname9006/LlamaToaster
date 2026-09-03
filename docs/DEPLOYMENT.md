# Deployment

See the [README](../README.md) for local/single-user setup. This covers
opening the app up to real accounts on the public Internet, plus the
security model behind that.

## Deploying publicly

The app defaults to a single-user, tailnet-or-loopback-only deployment —
nothing here is required for that. To open it up to real accounts on the
public Internet:

1. Point a domain (and, for the optional admin surface, a subdomain — e.g.
   `supervise.llamatoaster.com`) at the server's public IP.
2. Put nginx or Caddy in front of it and get real TLS certs — see
   [`deploy/nginx.conf`](../deploy/nginx.conf) for a ready-to-adapt config
   (security headers, CSP, both hostnames) and its own header comment for
   the Caddy-equivalent shape. `certbot --nginx -d yourdomain.com -d
   supervise.yourdomain.com` handles the cert.
3. Set `BIND_HOST=127.0.0.1` (the app should not be reachable on any public
   interface directly — confirm with the VPS firewall too, not just this
   value) and `PORT` to whatever the proxy config forwards to.
4. Set `AUTH_ENABLED=true` and the OAuth/`PUBLIC_URL`/`ADMIN_PUBLIC_URL` vars
   from "Accounts & multi-user" in the README.
5. Back up the DB on a schedule — [`scripts/backup-db.sh`](../scripts/backup-db.sh)
   (WAL-safe `sqlite3 .backup`, integrity-checked, retention-pruned) is meant
   to run daily from cron.

## Deploying a CPU worker on the server's own box

Same worker process as any other, just declared `"backend": "cpu"`. As of
the pull-queue architecture, this machine is **not** special-cased at all —
it enrolls exactly the way any user's GPU box does (see "Running a worker"
in the README): run `worker/bootstrap.sh` on the VPS itself, approve the
resulting device-flow code from your own account, done. `worker/config.vps.json.example`
exists as a reference for what the resulting config roughly looks like, but
there's no need to hand-author it — bootstrap.sh does it, including
generating and persisting the session credentials.

1. On the server: clone this repo to e.g. `/home/ubuntu/LlamaToaster`, run
   `npm install`, `npm run build`, `npm run build:admin`, and
   `mkdir -p /home/ubuntu/LlamaToaster/data`. The provided systemd units run
   the services as your regular login user (`User=ubuntu`/`Group=ubuntu`)
   rather than a dedicated system account — simpler for a single-operator
   VPS, at the cost of the app running with the same privileges as your
   login user instead of an isolated one. Adjust `User=`/`Group=`/paths in
   both `deploy/*.service` files if your username or clone path differs.
2. Copy [`deploy/orchestrator.env.example`](../deploy/orchestrator.env.example)
   to `deploy/orchestrator.env` and fill it in (see "Running the server" and
   "Accounts & multi-user" in the README for what each var does).
3. Install the two systemd units:
   ```bash
   sudo cp deploy/llamatoaster-worker.service deploy/llamatoaster-server.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now llamatoaster-server
   ```
   Don't enable `llamatoaster-worker` yet — it needs a real, device-flow-
   enrolled `config.json` first (see step 5), or it'll crash-loop looking
   for one.

   Prefer PM2? [`deploy/ecosystem.server.config.cjs`](../deploy/ecosystem.server.config.cjs)
   and [`deploy/ecosystem.worker.config.cjs`](../deploy/ecosystem.worker.config.cjs)
   are the equivalent, kept as two separate files deliberately — server and
   worker are independent processes that should be deployable, and able to
   fail, independently. `pm2 start deploy/ecosystem.server.config.cjs` for
   one, `pm2 start deploy/ecosystem.worker.config.cjs` for the other (after
   step 5).
4. Once `llamatoaster-server` is up, open the dashboard from your own
   machine and confirm it loads.
5. Enrol the server's own worker exactly like any other machine:
   `./worker/bootstrap.sh --url https://<your-domain> --dir ~/llamatoaster-worker`,
   then approve the printed code from your logged-in session. Note where it
   wrote `config.json`, then point `llamatoaster-worker.service`'s
   `WORKER_CONFIG=` (or `ecosystem.worker.config.cjs`'s
   `LLAMATOASTER_VPS_WORKER_CONFIG`) at that exact path and start the
   worker service/process.
6. Go to **Workers**, confirm the new card loads, and install a build from
   the dropdown — this replaces manually building llama.cpp on the server.
7. Register a small model (search-and-download panel on **Models**, or place
   a `.gguf` in the worker's `model_dir` by hand) and trigger a small run
   against that worker from **New Run** before doing a full sweep.

## Security / Networking

- **Every write is scoped to the caller's own account** in SQL, not filtered
  in the UI — one user's machines/runs/models are invisible to another's API
  calls too, not just hidden from their screen. Enforced whether or not
  `AUTH_ENABLED` is on; with it off, everything belongs to nobody in
  particular (single-tenant, matching the original design).
- **Machine enrolment is worker-initiated** (device flow, RFC 8628-style):
  nothing sensitive ever crosses browser→terminal. The one accepted risk is
  device-code phishing — if a code meant for someone else leaks (shoulder-
  surfing, a shared screen), it can be approved into a different account
  within its 15-minute window. Mitigated, not eliminated, by the confirm
  card showing hostname/GPU/platform, a warning line, and per-user rate
  limiting on the approve endpoint — the blast radius is mis-attributed
  compute, not exposed secrets or data.
- **The AI assistant's system prompt is server-owned** — a client-supplied
  `system` message is discarded outright, so `curl`ing the chat endpoint
  directly can't override its instructions or its per-user data scoping.
- **Community benchmark aggregates are k-anonymised in SQL**, not just
  stripped of a username column: any group describing fewer than 5 distinct
  contributors is never returned, opt-in (Settings), and never includes the
  querying user's own data.
- **The superadmin surface lives on a separate hostname** (`ADMIN_PUBLIC_URL`)
  and is unreachable from the main site even with a valid superadmin
  session — a request with the wrong `Host` header 404s before any auth
  check runs, so it doesn't even reveal that the surface exists.
- **Put the app behind nginx/Caddy + TLS for any public deployment** — see
  "Deploying publicly" above. A tailnet-only deployment (bind to your VPS's
  Tailscale IP, no reverse proxy, `AUTH_ENABLED` left off) remains a valid,
  simpler alternative if you don't need multiple real-world accounts —
  Tailscale's own WireGuard encryption covers that topology. The worker
  installer refuses a plain `http://` `--url` other than
  localhost/127.0.0.1 by default (a real bearer credential shouldn't go out
  in the clear) — pass `--allow-insecure-url`/`-AllowInsecureUrl` for this
  mode specifically, since Tailscale's own tunnel is the encryption there,
  not TLS.
- No app-level secret is stored anywhere reversible: sessions are looked up
  by `sha256(token)`, GitHub OAuth issues no long-lived shared secret this
  app has to protect beyond the client secret itself.
