# Deploying LlamaToaster to production

A start-to-finish runbook for taking a fresh VPS to a live, public, multi-user
LlamaToaster deployment. For how the app works day to day, see
[`README.md`](./README.md); for the architecture/security reasoning behind
each step here, see [`MULTIUSER_PLAN.md`](./MULTIUSER_PLAN.md) (particularly
its own "Security checklist" near the end — treat that as the final go-live
gate, not just this document).

This assumes: a Linux VPS you control (examples below use Ubuntu/`apt` +
`systemd`), a domain you control, and a GitHub account to register an OAuth
App under. Skip straight to step 8 if you only want a tailnet-only,
single-user deployment (no public accounts) — everything about OAuth/nginx/
TLS is for going *public*, not a hard requirement of the app itself.

## 0. Before you start

- [ ] Node.js 22+ on the VPS (`node --version`)
- [ ] `sqlite3` CLI installed (`apt install sqlite3`) — used by the backup
      script, not by the app itself (which uses `better-sqlite3`, a native
      module bundled via npm)
- [ ] A domain pointed at the VPS's public IP (an A/AAAA record). If you want
      the admin surface (§5.1 — optional), a second record for its subdomain
      too, e.g. `supervise.yourdomain.com`
- [ ] nginx or Caddy installed (`apt install nginx` / see caddyserver.com) —
      this runbook uses nginx; see [`deploy/nginx.conf`](deploy/nginx.conf)'s
      own header comment for the Caddy-equivalent shape
- [ ] `certbot` installed (`apt install certbot python3-certbot-nginx`)

## 1. Clone and build

```bash
git clone https://github.com/noname9006/LlamaToaster.git /home/ubuntu/LlamaToaster
cd /home/ubuntu/LlamaToaster
npm install                 # root deps, then client's and admin's own (postinstall)
npm run build                # client/dist
npm run build:admin          # admin/dist -- skip only if you're not using the admin surface
mkdir -p data
```

`npm install`'s own `postinstall` step needs a working C++ toolchain to build
`better-sqlite3` (`build-essential` on Ubuntu: `apt install build-essential
python3`). If that's already failed once, `npm install` again after
installing it — it won't re-run automatically.

## 2. Register a GitHub OAuth App

Skip this step (and the `AUTH_ENABLED`/`GITHUB_*` vars below) for a
single-user, no-login deployment.

1. GitHub → Settings → Developer settings → OAuth Apps → New OAuth App.
2. **Homepage URL**: `https://yourdomain.com`
3. **Authorization callback URL**: `https://yourdomain.com/auth/github/callback`
   — must match exactly, including scheme.
4. Save the generated **Client ID** and **Client Secret** — you'll put both
   in `orchestrator.env` in step 4.

## 3. Reverse proxy + TLS

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/llamatoaster
sudo ln -s /etc/nginx/sites-available/llamatoaster /etc/nginx/sites-enabled/
```

Edit `/etc/nginx/sites-available/llamatoaster` first:
- Replace `llamatoaster.com` / `supervise.llamatoaster.com` with your real
  domain(s) — these are placeholders (see the file's own header comment).
  Drop the `supervise.*` server blocks entirely if you're not using the
  admin surface.
- Match `proxy_pass http://127.0.0.1:<port>` to whatever `PORT` you set in
  step 4 (default `3000`).

Get real certs (nginx config must already be in place, unmodified paths, for
certbot's `--nginx` plugin to find and patch it):

```bash
sudo certbot --nginx -d yourdomain.com -d supervise.yourdomain.com
sudo systemctl status certbot.timer   # confirm auto-renewal is actually scheduled
sudo nginx -t && sudo systemctl reload nginx
```

Firewall — only nginx should be reachable from the public Internet, never
the app port directly:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow OpenSSH
sudo ufw enable
```

## 4. Configure the environment

```bash
cp deploy/orchestrator.env.example deploy/orchestrator.env
```

Edit `deploy/orchestrator.env` (gitignored — never commit this file):

| Var | Value |
|---|---|
| `PORT` | e.g. `3000` — must match nginx's `proxy_pass` from step 3 |
| `BIND_HOST` | `127.0.0.1` — nginx is the only thing that should reach this process |
| `DB_PATH` | `/home/ubuntu/LlamaToaster/data/llamatoaster.db` |
| `AUTH_ENABLED` | `true` (omit entirely for a no-login deployment) |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | from step 2 |
| `SUPERADMIN_IDENTITIES` | `github:<your-numeric-github-id>` — find yours at `https://api.github.com/users/<your-username>` (the `id` field). Leave unset if you don't want an admin surface at all |
| `PUBLIC_URL` | `https://yourdomain.com` |
| `ADMIN_PUBLIC_URL` | `https://supervise.yourdomain.com` (omit if not using the admin surface — every `/api/admin/*` route 404s with nothing here) |

Every other var (AI assistant, `GITHUB_TOKEN`, `LOG_LEVEL`, ...) is optional
— see the file's own comments. **No `SESSION_SECRET`** exists or is needed;
if you find one referenced anywhere, that's stale documentation, not this
app.

## 5. Install and start the server

```bash
sudo cp deploy/llamatoaster-server.service /etc/systemd/system/
```

Edit the copied unit if your username or clone path differs from
`ubuntu`/`/home/ubuntu/LlamaToaster`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llamatoaster-server
sudo systemctl status llamatoaster-server   # should be active (running)
curl -s http://127.0.0.1:$PORT/health       # {"ok":true,"db":true}
curl -sI https://yourdomain.com/            # 200, real TLS cert
```

Prefer PM2 over systemd? Use
[`deploy/ecosystem.server.config.cjs`](deploy/ecosystem.server.config.cjs)
instead (`pm2 start deploy/ecosystem.server.config.cjs`) — functionally
equivalent, your call which process manager is the daily driver. Either way,
**do not** use the old combined `ecosystem.config.cjs` if you have a copy
from before this file was split (§6.0) — it no longer exists in this repo;
server and worker are separate PM2 apps/systemd units on purpose, so a
frequent server deploy can never bounce a worker mid-benchmark.

At this point: visit `https://yourdomain.com` in a browser. With
`AUTH_ENABLED=true` you should land on `/login` and be able to sign in with
GitHub. Sign in with the account whose id you put in `SUPERADMIN_IDENTITIES`
to confirm superadmin status (visit `https://supervise.yourdomain.com` if
configured — should show the admin dashboard, not a 403/404).

## 6. Enrol the server's own worker (optional CPU worker)

The server's own box can also run benchmarks — as of the pull-queue
architecture it's enrolled exactly like any user's GPU machine, not a
special-cased config file. Skip this whole step if the server is
orchestration-only and all benchmarking happens on separate GPU boxes.

```bash
cd /home/ubuntu
./LlamaToaster/worker/bootstrap.sh --vps-url https://yourdomain.com --dir /home/ubuntu/llamatoaster-worker
```

Watch the output for a short code and a `/device` link — open that link
signed in as your own (superadmin) account and approve it. Note the path
`bootstrap.sh` printed for the resulting `config.json` (inside
`--dir`/`worker/config.json`), then:

```bash
sudo cp deploy/llamatoaster-worker.service /etc/systemd/system/
```

Edit the copied unit's `Environment=WORKER_CONFIG=...` line to point at that
exact path (it's an absolute path outside the server's own repo checkout —
see the unit file's own comment for why). Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llamatoaster-worker
```

Confirm in the browser: **Workers** should show this machine's card. Install
a llama.cpp build from the dropdown, register a small model, and trigger a
short run against it before trusting it with a real sweep.

(PM2 equivalent:
[`deploy/ecosystem.worker.config.cjs`](deploy/ecosystem.worker.config.cjs),
set `LLAMATOASTER_VPS_WORKER_CONFIG` to the same path instead of editing the
file.)

## 7. Onboard real users' workers

Nothing left to configure server-side — every other user's GPU box follows
the exact same public path, with their own account. The app doesn't serve
the installer itself (no `/install.sh` route today — it's fetched straight
from GitHub):

```bash
curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --vps-url https://yourdomain.com
```

(Windows: `worker/bootstrap.ps1` — see README's "Running a worker" section
for the exact one-liners.) Point people at the README's own instructions
rather than re-explaining the flow here.

## 8. Backups

```bash
mkdir -p /home/ubuntu/LlamaToaster/backups   # cron's own log redirect below needs this to exist first
sudo -u ubuntu crontab -e
```

Add (adjust paths to match `DB_PATH`/wherever you want backups kept):

```cron
0 3 * * * DB_PATH=/home/ubuntu/LlamaToaster/data/llamatoaster.db BACKUP_DIR=/home/ubuntu/LlamaToaster/backups bash /home/ubuntu/LlamaToaster/scripts/backup-db.sh >> /home/ubuntu/LlamaToaster/backups/backup.log 2>&1
```

(`bash script.sh`, not `./script.sh` — this repo doesn't rely on the git
executable bit for any of its shell scripts, same as `worker/bootstrap.sh`.)

This runs a WAL-safe `sqlite3 .backup` (safe against the live, actively-
written DB — no need to stop the server), integrity-checks the result, and
prunes backups older than 14 days (`RETENTION_DAYS`, override via the same
cron line if you want longer). **Actually test a restore once** —
`sqlite3 backup-file.db "PRAGMA integrity_check;"` should say `ok`, and
`cp backup-file.db data/llamatoaster.db` (server stopped) should boot clean.

## 9. Deploying updates

```powershell
.\deploy\push-client.ps1                          # over Tailscale (operator access), default
.\deploy\push-client.ps1 -PublicHost yourdomain.com   # over the public hostname instead
```

Builds `client/dist` + `admin/dist` locally, `scp`s both to the VPS, and
`pm2 restart llamatoaster-server` (edit the script if you're on systemd
instead — swap the last `ssh ... "pm2 restart ..."` line for
`sudo systemctl restart llamatoaster-server`). For a **server code** change
(not just client/admin), you still need to `git pull` on the VPS yourself
and restart — this script only ever pushes the built frontend bundles.

## 10. Go-live checklist

Before calling this production-ready for real users, walk
[`MULTIUSER_PLAN.md`](./MULTIUSER_PLAN.md)'s own "Security checklist"
section top to bottom — it's the authoritative, itemized list this whole
plan was built against. At minimum, confirm:

- [ ] `https://yourdomain.com` and `https://supervise.yourdomain.com` (if
      used) both serve real, valid TLS certs, and `certbot.timer` is active
- [ ] `curl http://<vps-public-ip>:$PORT/health` from **outside** the VPS
      (not via nginx) times out/refuses — the app port itself must not be
      reachable directly
- [ ] A GitHub sign-in round-trip actually works (this is the one thing no
      automated test in this repo can confirm — it needs the real,
      registered OAuth App from step 2)
- [ ] The superadmin account signs into `supervise.yourdomain.com`
      successfully; a non-superadmin account gets a clear "not authorized"
      screen there, not an error
- [ ] `deploy/orchestrator.env` is not world-readable (`chmod 600
      deploy/orchestrator.env`) and is genuinely excluded from any backup or
      log destination that isn't itself access-controlled the same way
- [ ] A cron-driven backup has actually run once and its integrity check
      passed (check `backups/backup.log`)
- [ ] `AI_GLOBAL_DAILY_CAP` is set to something you're comfortable paying
      for if the AI assistant is enabled — this is the one endpoint that
      spends real money per request

## Troubleshooting

- **502 from nginx**: `systemctl status llamatoaster-server` — is it even
  running? `curl 127.0.0.1:$PORT/health` directly on the VPS to rule out
  nginx vs. the app itself.
- **OAuth callback fails / redirects to `/login?error=oauth_state`**: the
  `oauth_state` cookie didn't round-trip — almost always `PUBLIC_URL`
  mismatching the actual browser-visible origin (scheme included), or the
  GitHub OAuth App's callback URL not matching exactly.
- **A worker enrols but never shows as online**: check its own
  `worker-YYYY-MM-DD.log` (next to wherever it's installed) for the actual
  error — a wrong `vps_url`, or the server unreachable from that machine's
  network, are the two most common causes.
- **`/api/admin/*` 404s even with the right hostname**: `ADMIN_PUBLIC_URL`
  isn't set (or nginx's `server_name` for that block doesn't match it
  exactly) — the whole admin surface is gated on that env var alone.
