// Multi-user Stage 2/5: superadmin is evaluated PER REQUEST from env, never a
// stored/cached DB column (MULTIUSER_PLAN.md §5.1) -- removing an entry from
// SUPERADMIN_IDENTITIES + restarting the process is instant, complete
// revocation, including every live session, unlike a `users.is_superadmin`
// flag that would persist until that session's next login.
//
// Keyed on the SAME (provider, provider_user_id) pair identities.provider/
// provider_user_id uses (§2.1) -- provider-qualified stable ids, never
// logins (a login/handle is mutable and reused across accounts over time).
const SUPERADMIN_IDENTITIES = new Set(
  (process.env.SUPERADMIN_IDENTITIES ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
  // e.g. SUPERADMIN_IDENTITIES=github:123456,google:104852001937446291234
);

export function isSuperadminIdentity(provider: string, providerUserId: string): boolean {
  return SUPERADMIN_IDENTITIES.has(`${provider}:${providerUserId}`);
}
