// Loads a repo-root .env into process.env. Imported first by index.ts, because
// several modules read env vars as they load (GITHUB_CLIENT_ID in routes/auth.ts,
// ADMIN_PUBLIC_URL in routes/admin.ts, SUPERADMIN_IDENTITIES in superadmin.ts):
// loaded any later, a value that's only in .env is silently ignored for those
// (audit 2026-09-27 L1). A missing .env is the common case, not an error. Values
// already in the environment (systemd, pm2) win. process.loadEnvFile is a
// built-in Node 20.12+/22 API, no dotenv dependency needed.
try {
  process.loadEnvFile();
} catch {
  /* no .env file at the repo root */
}
