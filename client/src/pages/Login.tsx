// MULTIUSER_PLAN.md §2.8: sign-up and log-in are the same flow -- there is
// no separate registration screen. One button per configured provider (just
// GitHub today); clicking it is a full-page navigation, not a fetch, since
// the server needs to set the oauth_state cookie and redirect to the
// provider's own consent screen.
const PROVIDERS: { id: string; label: string }[] = [{ id: "github", label: "Continue with GitHub" }];

export function Login() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-4">
      <div className="w-full max-w-sm rounded-2xl border border-border bg-surface p-8 text-center">
        <img src="/logo.png" alt="LlamaToaster" className="mx-auto w-16" />
        <h1 className="mt-4 text-xl font-semibold text-fg">LlamaToaster</h1>
        <p className="mt-1 text-sm text-muted">Sign in to manage your machines and benchmarks.</p>

        <div className="mt-6 flex flex-col gap-2">
          {PROVIDERS.map((p) => (
            <a
              key={p.id}
              href={`/auth/${p.id}`}
              className="flex items-center justify-center rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-accent-fg hover:opacity-90"
            >
              {p.label}
            </a>
          ))}
        </div>

        <p className="mt-6 text-xs text-muted">
          Nothing to type — no username, no password. Already have an account? Signing in with the
          same provider signs you straight back into it.
        </p>
        <p className="mt-2 text-xs text-muted">
          By default, your benchmark results contribute to anonymised community averages shown to
          other users (never your name or machine — see Settings to turn this off anytime).
        </p>
      </div>
    </div>
  );
}
