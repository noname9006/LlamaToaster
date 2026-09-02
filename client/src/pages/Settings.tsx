import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SessionInfo, IdentityInfo, AppSettings } from "../types";
import { formatRelativeTime } from "../utils";
import { IconTrash } from "../components/icons";

export function Settings() {
  // Reachable directly by URL even when the Sidebar hides its own link (see
  // Sidebar.tsx's authEnabled gate) -- these routes are only ever registered
  // server-side when AUTH_ENABLED, so this page needs to know that too
  // rather than attempting fetches against routes that don't exist.
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null);
  const [identities, setIdentities] = useState<IdentityInfo[] | null>(null);
  const [shareBenchmarks, setShareBenchmarksState] = useState<boolean | null>(null);
  // Operator-controlled gates from the supervise dashboard (shared/types.ts's
  // AppSettings) -- null until the initial getAuthStatus() resolves, same
  // "haven't loaded yet" convention as the other nullable state here.
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleting, setDeleting] = useState(false);

  async function load() {
    try {
      const [s, i] = await Promise.all([api.listSessions(), api.listIdentities()]);
      setSessions(s);
      setIdentities(i);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    api
      .getAuthStatus()
      .then((status) => {
        setAuthEnabled(status.authEnabled);
        setAppSettings(status.appSettings);
        if (status.authEnabled) {
          setShareBenchmarksState(status.user?.shareBenchmarks ?? null);
          void load();
        }
      })
      .catch(() => setAuthEnabled(false));
  }, []);

  // Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- optimistic toggle, reverted
  // on failure. This is the one thing on the account that gates what the AI
  // assistant's community tools (routes/ai.ts) are allowed to include.
  async function handleToggleShareBenchmarks() {
    if (shareBenchmarks === null) return;
    const next = !shareBenchmarks;
    setShareBenchmarksState(next);
    try {
      const result = await api.setShareBenchmarks(next);
      setShareBenchmarksState(result.shareBenchmarks);
    } catch (err) {
      setShareBenchmarksState(!next);
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRevoke(id: string) {
    setBusy(true);
    try {
      await api.revokeSession(id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleRevokeAll() {
    if (!window.confirm("Sign out every other session? This won't affect the one you're using right now.")) return;
    setBusy(true);
    try {
      await api.revokeAllOtherSessions();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  // Security checklist's own "Account deletion ships in v1" item -- a
  // type-to-confirm text field, not just a window.confirm(), since this is
  // the one action here with no undo (unlike revoking a session, which a
  // fresh sign-in immediately reverses). Full navigation on success, not a
  // client-side route change: the account (and the session backing this
  // page) no longer exists, so there's nothing left for the SPA's own state
  // to usefully re-render.
  async function handleDeleteAccount() {
    setDeleting(true);
    setError(null);
    try {
      await api.deleteAccount();
      window.location.href = "/login";
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setDeleting(false);
    }
  }

  const linkedProviders = new Set(identities?.map((i) => i.provider) ?? []);
  const otherSessionCount = sessions?.filter((s) => !s.current).length ?? 0;

  if (authEnabled === null) {
    return <p className="text-sm text-muted">Loading…</p>;
  }
  if (!authEnabled) {
    return (
      <div>
        <h1 className="text-2xl font-semibold text-fg">Settings</h1>
        <p className="mt-4 text-sm text-muted">
          Account settings aren't available on this deployment — it isn't running with user
          accounts enabled.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-fg">Settings</h1>
        <a
          href="/auth/logout"
          className="rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-danger/40 hover:text-danger"
        >
          Sign out
        </a>
      </div>

      {error && (
        <p className="mt-4 rounded-lg border border-danger/30 bg-danger-bg px-3 py-2 text-sm text-danger">{error}</p>
      )}

      <section className="mt-8">
        <h2 className="text-sm font-semibold text-fg">Connected accounts</h2>
        <p className="mt-1 text-sm text-muted">
          Sign-in providers linked to this account. Linking a new one attaches it here rather than
          creating a separate account.
        </p>
        <div className="mt-3 flex flex-col gap-2">
          {identities === null ? (
            <p className="text-sm text-muted">Loading…</p>
          ) : (
            identities.map((i) => (
              <div
                key={`${i.provider}:${i.createdAt}`}
                className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5"
              >
                <div>
                  <div className="text-sm font-medium capitalize text-fg">{i.provider}</div>
                  <div className="text-xs text-muted">{i.providerLogin ?? "—"}</div>
                </div>
                <span className="text-xs text-muted">
                  Connected {formatRelativeTime(new Date(i.createdAt).toISOString()) ?? "recently"}
                </span>
              </div>
            ))
          )}
          {!linkedProviders.has("github") && (
            <a
              href="/auth/github?link=1"
              className="w-fit rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              + Connect GitHub
            </a>
          )}
        </div>
      </section>

      {appSettings?.communitySharingAllowed && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold text-fg">Community benchmark data</h2>
          <p className="mt-1 text-sm text-muted">
            All benchmarks run through this platform are automatically saved to our database, including
            hardware configurations, OS details, models, and test results. Your privacy is strictly
            protected: all shared data is fully anonymized, ensuring no other user can ever associate your
            account or machines with your tests.
          </p>
          <div className="mt-3 flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
            <span className="text-sm font-medium text-fg">Contribute to the community benchmark database</span>
            <button
              type="button"
              role="switch"
              aria-checked={shareBenchmarks ?? false}
              onClick={() => void handleToggleShareBenchmarks()}
              disabled={shareBenchmarks === null}
              className={`relative h-6 w-11 flex-none rounded-full transition-colors disabled:opacity-50 ${
                shareBenchmarks ? "bg-accent" : "bg-white/10"
              }`}
            >
              <span
                className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                  shareBenchmarks ? "translate-x-[22px]" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </section>
      )}

      <section className="mt-8">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-fg">Active sessions</h2>
          {otherSessionCount > 0 && (
            <button
              type="button"
              onClick={() => void handleRevokeAll()}
              disabled={busy}
              className="text-xs font-semibold text-muted hover:text-danger disabled:opacity-50"
            >
              Sign out everywhere else
            </button>
          )}
        </div>
        <p className="mt-1 text-sm text-muted">Every browser or machine currently signed into this account.</p>
        <div className="mt-3 flex flex-col gap-2">
          {sessions === null ? (
            <p className="text-sm text-muted">Loading…</p>
          ) : sessions.length === 0 ? (
            <p className="text-sm text-muted">No active sessions.</p>
          ) : (
            sessions.map((s) => (
              <div key={s.id} className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
                <div>
                  <div className="flex items-center gap-2 text-sm font-medium text-fg">
                    {s.label ?? (s.isWorker ? "Worker" : "Unknown device")}
                    {s.current && (
                      <span className="rounded-full bg-accent/15 px-1.5 py-0.5 text-[10px] font-semibold text-accent">
                        This session
                      </span>
                    )}
                    {s.isWorker && (
                      <span className="rounded-full bg-white/5 px-1.5 py-0.5 text-[10px] font-semibold text-muted">
                        Worker
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted">
                    Last active {formatRelativeTime(s.lastSeenAt ? new Date(s.lastSeenAt).toISOString() : null) ?? "just now"} ·
                    Created {formatRelativeTime(new Date(s.createdAt).toISOString()) ?? "recently"}
                  </div>
                </div>
                {!s.current && (
                  <button
                    type="button"
                    onClick={() => void handleRevoke(s.id)}
                    disabled={busy}
                    className="rounded-md p-1.5 text-muted hover:bg-white/5 hover:text-danger disabled:opacity-40"
                    aria-label="Sign out this session"
                    title="Sign out this session"
                  >
                    <IconTrash width={15} height={15} />
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      </section>

      {appSettings?.accountDeletionAllowed && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold text-danger">Danger zone</h2>
          <p className="mt-1 text-sm text-muted">
            Permanently deletes your account: every linked sign-in, active session, machine, test, and
            result. Models you registered stay in the shared catalog for other users, just no longer
            attributed to you. This cannot be undone —{" "}
            <a href="/api/results/export?format=csv" className="text-accent hover:underline">
              export your results first
            </a>{" "}
            if you want a copy.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-danger/30 bg-danger-bg px-3 py-2.5">
            <label className="flex-1 text-xs text-muted" htmlFor="delete-account-confirm">
              Type <span className="font-mono font-semibold text-fg">DELETE</span> to confirm
            </label>
            <input
              id="delete-account-confirm"
              type="text"
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              disabled={deleting}
              className="w-32 rounded-md border border-border bg-surface px-2 py-1 text-sm text-fg"
              autoComplete="off"
            />
            <button
              type="button"
              onClick={() => void handleDeleteAccount()}
              disabled={deleteConfirmText !== "DELETE" || deleting}
              className="rounded-lg border border-danger/40 px-3 py-1.5 text-sm font-semibold text-danger hover:bg-danger-bg disabled:cursor-not-allowed disabled:opacity-40"
            >
              {deleting ? "Deleting…" : "Delete my account"}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
