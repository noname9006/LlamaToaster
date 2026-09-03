# Security Policy

LlamaToaster is open-source and self-hostable — you can run the server and
worker(s) yourself, on your own infrastructure. The maintainer also runs a
public multi-user instance at **llamatoaster.com** (GitHub sign-in,
`AUTH_ENABLED=true`), which holds real accounts and their machine/run data.

Anything that could expose or modify another account's data on
**llamatoaster.com** specifically — not just in a self-hosted lab setup — is
treated as the highest priority category below.

## Reporting a vulnerability

Please **don't open a public GitHub issue** for security reports.

Preferred: use GitHub's private reporting —
[Security → Report a vulnerability](https://github.com/noname9006/LlamaToaster/security/advisories/new)
on this repo. It opens a private advisory only the maintainer can see until
it's resolved.

Fallback: email **noname9006@gmail.com** with a description, reproduction
steps, and affected version/commit.

You should get an acknowledgement within a few days. This is maintained by
a single developer rather than a dedicated security team, so there's no
formal SLA — but confirmed vulnerabilities are treated as top priority and
fixed ahead of everything else in progress.

## Scope

In scope — bugs in code that ships in this repo (`server/`, `worker/`,
`client/`, `admin/`, `shared/`):

- **Auth / multi-tenancy** (`AUTH_ENABLED=true`): anything that lets one
  account read or modify another account's machines, runs, or results;
  session/token handling; the superadmin origin gate.
- **Worker execution**: anything that lets a downloaded model file, a
  crafted API response, or a malicious job payload cause the worker to
  execute unintended code or write outside its working directories —
  beyond the inherent risk of running `llama-bench`/`llama-server` itself.
- **Server API**: injection (SQL, command), path traversal, SSRF, auth
  bypass, or anything that lets an unauthenticated caller reach
  unauthenticated-shouldn't-be-reachable data.
- **Client/admin SPA**: XSS or CSRF that a normal deployment (following the
  README) would still be exposed to.

Out of scope — please report these upstream / it's expected behavior:

- Vulnerabilities in `llama.cpp`/`llama-bench`/`llama-server` themselves —
  report to the [llama.cpp project](https://github.com/ggml-org/llama.cpp).
- Malicious content inside a model file or binary you chose to download
  and run — LlamaToaster fetches what you point it at (Hugging Face, GitHub
  releases); it doesn't vet third-party model/binary contents. Only report
  here if LlamaToaster itself mishandles the download (e.g. path traversal
  from a crafted filename).
- Issues caused by ignoring the README's deployment guidance — e.g. binding
  the server to `0.0.0.0` directly instead of behind a reverse proxy, or
  running with `AUTH_ENABLED` off on a network you don't trust.
- Denial-of-service via resource exhaustion on a self-hosted box you
  control (e.g. queuing many runs) — not a security boundary there. A DoS
  that affects **llamatoaster.com** or other users' workers is in scope.

## Supported versions

Pre-1.0, single rolling release: only the latest commit on `main` gets
fixes. There are no maintained older branches.
