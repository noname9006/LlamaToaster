# Contributing to LlamaToaster

This is a solo-maintained project — response times vary, but contributions
are welcome. This doc covers the mechanics; for what the app actually does
and why, see [README.md](README.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project layout

```
server/   Fastify API + SQLite (better-sqlite3), serves client/dist and admin/dist
client/   React + Vite + Tailwind SPA (the main dashboard)
admin/    React + Vite + Tailwind SPA (superadmin-only, separate origin)
worker/   runs on each benchmark box: long-polls the server, spawns llama-bench/llama-server
shared/   TypeScript types shared by server, worker, and client
```

## Setup

Requires Node.js 22+.

```bash
npm install          # root deps; postinstall also installs client/ and admin/
npm run build        # builds client/dist
npm run build:admin  # builds admin/dist
```

Run the server:

```bash
PORT=3000 npm run server
```

For frontend work, run the Vite dev server(s) alongside the server above
instead of rebuilding on every change:

```bash
npm run dev:client   # port 5173
npm run dev:admin    # port 5174, needs ADMIN_PUBLIC_URL/AUTH_ENABLED to be useful
```

Run a worker against a local server: see "Running a worker" in the README.

## Before opening a PR

```bash
npm test                              # vitest — server/worker/shared tests
npx tsc --noEmit                      # typecheck server/worker/shared
npm run typecheck --prefix client     # typecheck client
npm run typecheck --prefix admin      # typecheck admin
```

CI (`.github/workflows/test.yml`) runs `npm test` on every push and PR to
`main` — it must pass before merge. Typechecking isn't wired into CI yet,
so please run it locally; a PR that fails it will just get sent back.

Tests live next to the code they cover (`server/src/*.test.ts`, etc.). Add
or update tests for behavior you change, especially anything touching auth,
multi-tenancy (`AUTH_ENABLED`), or the worker job protocol — these are the
areas most likely to silently break.

## Making changes

- Keep PRs scoped to one thing. Unrelated reformatting or drive-by
  refactors in the same diff make review slower, not faster.
- Match the existing code style in the file you're editing — no linter is
  enforced yet, so consistency is by convention.
- Commit messages: short, imperative, explain the *why* over the *what*
  where it's not obvious (`git log` has plenty of examples).
- If your change affects deployment, env vars, or the worker protocol,
  update the relevant section of `README.md` / `docs/ARCHITECTURE.md` in
  the same PR. The specs under `docs/plans/` are historical — don't edit
  them to describe new behavior.

## Reporting bugs / requesting features

Open a GitHub issue. For anything security-sensitive, see
[SECURITY.md](SECURITY.md) instead — don't file it as a public issue.

## License

By contributing, you agree your contribution is licensed under this
project's [MIT license](LICENSE). No CLA is required.
