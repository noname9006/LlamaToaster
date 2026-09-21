// The unauthenticated screen, shared by every page: a 403 (a real session that
// isn't superadmin-listed) and a 404 (the admin surface answering "not found"
// -- shouldn't normally happen since this bundle is only ever served FROM the
// admin hostname) both read as "not signed in here yet", and the fix is the
// same sign-in link either way.
export function SignIn() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <h1 className="text-xl font-semibold text-fg">LlamaToaster Admin</h1>
      <p className="max-w-sm text-sm text-muted">
        This is an operator-only surface. Sign in with a superadmin-listed GitHub account to continue.
      </p>
      <a
        href="/auth/github"
        className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-fg hover:bg-accent/90"
      >
        Sign in with GitHub
      </a>
    </div>
  );
}
