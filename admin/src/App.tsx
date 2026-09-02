import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "./api";
import type { AdminStats, AdminTestSummary, AppSettings } from "./types";

type Phase = "loading" | "unauthorized" | "ready" | "error";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex-1 rounded-xl border border-border bg-surface px-6 py-4 text-center">
      <div className="text-3xl font-bold text-accent">{value}</div>
      <div className="mt-1 text-sm text-muted">{label}</div>
    </div>
  );
}

// Same visual pattern as the main app's own Settings.tsx toggle -- kept as a
// separate copy rather than a shared component since admin/ and client/ are
// two independently-built SPAs with no shared component module between them.
function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      disabled={disabled}
      className={`relative h-6 w-11 flex-none rounded-full transition-colors disabled:opacity-50 ${
        checked ? "bg-accent" : "bg-white/10"
      }`}
    >
      <span
        className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
          checked ? "translate-x-[22px]" : "translate-x-0"
        }`}
      />
    </button>
  );
}

function formatDate(ms: number): string {
  return new Date(ms).toLocaleString();
}

function FilterSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-fg"
    >
      <option value="">{label}: all</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): the whole admin surface is
// one page, deliberately -- a stats row, the filterable global runs table,
// an export button. No routing, no chat panel, no run-triggering UI;
// nothing a normal user's app needs and nothing this one does either.
export default function App() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [runs, setTests] = useState<AdminTestSummary[]>([]);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const [userFilter, setUserFilter] = useState("");
  const [workerFilter, setWorkerFilter] = useState("");
  const [backendFilter, setBackendFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");

  useEffect(() => {
    Promise.all([api.getStats(), api.listTests(), api.getSettings()])
      .then(([s, r, settingsResult]) => {
        setStats(s);
        setTests(r.runs);
        setSettings(settingsResult);
        setPhase("ready");
      })
      .catch((err) => {
        // 403 (real session, not superadmin-listed) and 404 (misconfigured
        // ADMIN_HOSTNAME -- shouldn't normally happen since this bundle is
        // only ever served FROM that hostname) both read as "not signed in
        // here yet" to this page; either way, the fix is the same sign-in
        // link.
        if (err instanceof ApiError && (err.status === 403 || err.status === 404)) {
          setPhase("unauthorized");
        } else {
          setErrorMessage(err instanceof Error ? err.message : String(err));
          setPhase("error");
        }
      });
  }, []);

  // Client-side filtering over the one fetched batch (capped at 500 rows
  // server-side, see repo.ts's adminRepo.listTests) -- feels instant, and
  // avoids a round trip per filter change for a table this size. The
  // server's own ?userId=/?workerId=/?backend=/?status= querystring support
  // (routes/admin.ts, exercised by admin.test.ts) stays available for any
  // other caller that wants a narrower fetch than "everything."
  const users = useMemo(
    () => [...new Set(runs.map((r) => r.userDisplayName).filter((v): v is string => Boolean(v)))].sort(),
    [runs]
  );
  const machines = useMemo(
    () => [...new Set(runs.map((r) => r.workerDisplayName).filter((v): v is string => Boolean(v)))].sort(),
    [runs]
  );
  const backends = useMemo(() => [...new Set(runs.map((r) => r.llamaCppBackend))].sort(), [runs]);
  const statuses = useMemo(() => [...new Set(runs.map((r) => r.status))].sort(), [runs]);

  const filteredTests = useMemo(
    () =>
      runs.filter(
        (r) =>
          (!userFilter || r.userDisplayName === userFilter) &&
          (!workerFilter || r.workerDisplayName === workerFilter) &&
          (!backendFilter || r.llamaCppBackend === backendFilter) &&
          (!statusFilter || r.status === statusFilter)
      ),
    [runs, userFilter, workerFilter, backendFilter, statusFilter]
  );

  // Optimistic toggle, reverted on failure -- same pattern as the main app's
  // own Settings.tsx handleToggleShareBenchmarks.
  async function handleToggleSetting(key: keyof AppSettings) {
    if (settings === null) return;
    const next = { ...settings, [key]: !settings[key] };
    setSettings(next);
    setSettingsError(null);
    try {
      const result = await api.updateSettings({ [key]: next[key] });
      setSettings(result);
    } catch (err) {
      setSettings(settings);
      setSettingsError(err instanceof Error ? err.message : String(err));
    }
  }

  // Enum-valued sibling of handleToggleSetting above -- same optimistic-apply-
  // revert-on-failure shape, but sends a specific value instead of boolean-
  // flipping (the policy select below).
  async function handleSetVramPolicy(value: AppSettings["workerVramDiscrepancyPolicy"]) {
    if (settings === null) return;
    const prev = settings;
    setSettings({ ...settings, workerVramDiscrepancyPolicy: value });
    setSettingsError(null);
    try {
      const result = await api.updateSettings({ workerVramDiscrepancyPolicy: value });
      setSettings(result);
    } catch (err) {
      setSettings(prev);
      setSettingsError(err instanceof Error ? err.message : String(err));
    }
  }

  // Numeric-valued sibling of handleToggleSetting/handleSetVramPolicy above --
  // same optimistic-apply-revert-on-failure shape, sending a specific integer
  // instead of boolean-flipping or picking from an enum (the number input
  // below). The server independently validates/clamps to [1, 200], so an
  // out-of-range value here just comes back as a 400 that reverts the
  // optimistic update.
  async function handleSetProbeMaxLoads(value: number) {
    if (settings === null) return;
    const prev = settings;
    setSettings({ ...settings, probeMaxLoads: value });
    setSettingsError(null);
    try {
      const result = await api.updateSettings({ probeMaxLoads: value });
      setSettings(result);
    } catch (err) {
      setSettings(prev);
      setSettingsError(err instanceof Error ? err.message : String(err));
    }
  }

  if (phase === "loading") {
    return <div className="flex min-h-screen items-center justify-center text-sm text-muted">Loading…</div>;
  }

  if (phase === "unauthorized") {
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

  if (phase === "error") {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 text-center text-sm text-danger">
        Could not load admin data: {errorMessage}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <h1 className="text-2xl font-semibold text-fg">LlamaToaster Admin</h1>
      <p className="mt-1 text-sm text-muted">Cross-tenant read view — every user's machines and runs.</p>

      <section className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Users" value={stats!.users} />
        <StatCard label="Machines" value={stats!.machines} />
        <StatCard label="Models tested" value={stats!.modelsTested} />
        <StatCard label="Quants options" value={stats!.quants} />
        <StatCard label="Tests performed" value={stats!.tests} />
        <StatCard label="Total runs" value={stats!.runs} />
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-fg">Platform settings</h2>
        <p className="mt-1 text-sm text-muted">
          Feature gates for every user on this deployment (shared/types.ts's AppSettings).
        </p>
        {settingsError && (
          <p className="mt-2 rounded-lg border border-danger/30 bg-danger-bg px-3 py-2 text-sm text-danger">
            {settingsError}
          </p>
        )}
        <div className="mt-3 flex flex-col gap-2">
          <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
            <div>
              <div className="text-sm font-medium text-fg">Allow community benchmark sharing</div>
              <div className="text-xs text-muted">
                When off, every user's "Contribute to the community benchmark database" toggle in Settings
                stays greyed out, and the AI assistant's community tools refuse to run.
              </div>
            </div>
            <Toggle
              checked={settings?.communitySharingAllowed ?? false}
              onChange={() => void handleToggleSetting("communitySharingAllowed")}
              disabled={settings === null}
            />
          </div>
          <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
            <div>
              <div className="text-sm font-medium text-fg">Allow self-service account deletion</div>
              <div className="text-xs text-muted">
                When off, the Danger zone section (and "Delete my account") disappears from every user's
                Settings page entirely, and the delete endpoint refuses server-side too.
              </div>
            </div>
            <Toggle
              checked={settings?.accountDeletionAllowed ?? false}
              onChange={() => void handleToggleSetting("accountDeletionAllowed")}
              disabled={settings === null}
            />
          </div>
          <div className="flex items-center justify-between gap-4 rounded-lg border border-border px-3 py-2.5">
            <div>
              <div className="text-sm font-medium text-fg">Worker VRAM fallback policy</div>
              <div className="text-xs text-muted">
                What machines do when a benchmark claims full GPU offload but provably ran from system RAM
                (claimed X/Y layers, ~0 actually resident). warn: record results with a warning (default).
                retry_once_then_fail: one silent re-run to rule out transient VRAM contention, then fail —
                later items in the same run skip straight to failed. fail: never record such results.
                Delivered to workers via their ~10s heartbeat; applies from the next item onward.
              </div>
            </div>
            <select
              value={settings?.workerVramDiscrepancyPolicy ?? "warn"}
              onChange={(e) => void handleSetVramPolicy(e.target.value as AppSettings["workerVramDiscrepancyPolicy"])}
              disabled={settings === null}
              className="flex-none rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-fg"
            >
              <option value="warn">warn</option>
              <option value="retry_once_then_fail">retry_once_then_fail</option>
              <option value="fail">fail</option>
            </select>
          </div>
          <div className="flex items-center justify-between gap-4 rounded-lg border border-border px-3 py-2.5">
            <div>
              <div className="text-sm font-medium text-fg">Probe max loads</div>
              <div className="text-xs text-muted">
                Hard cap on how many real model loads a single N2 probe run may perform, across every search
                phase. Default 24. Delivered to workers via their ~10s heartbeat; also enforced server-side
                when a worker reports probe results, so a stale or misbehaving worker can't exceed it.
              </div>
            </div>
            <input
              type="number"
              min={1}
              max={200}
              step={1}
              value={settings?.probeMaxLoads ?? 24}
              onChange={(e) => {
                const parsed = Number.parseInt(e.target.value, 10);
                if (Number.isFinite(parsed)) void handleSetProbeMaxLoads(parsed);
              }}
              disabled={settings === null}
              className="w-24 flex-none rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-fg"
            />
          </div>
        </div>
      </section>

      <section className="mt-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-fg">Tests</h2>
          <div className="flex flex-wrap gap-2">
            <a
              href={api.exportUrl("csv")}
              className="rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              Export CSV
            </a>
            <a
              href={api.exportUrl("json")}
              className="rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent"
            >
              Export JSON
            </a>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <FilterSelect label="User" value={userFilter} options={users} onChange={setUserFilter} />
          <FilterSelect label="Machine" value={workerFilter} options={machines} onChange={setWorkerFilter} />
          <FilterSelect label="Backend" value={backendFilter} options={backends} onChange={setBackendFilter} />
          <FilterSelect label="Status" value={statusFilter} options={statuses} onChange={setStatusFilter} />
        </div>

        <div className="mt-3 overflow-x-auto rounded-lg border border-border">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="bg-surface-raised text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-3 py-2">User</th>
                <th className="px-3 py-2">Machine</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Backend</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Tests</th>
                <th className="px-3 py-2">Started</th>
              </tr>
            </thead>
            <tbody>
              {filteredTests.map((r) => (
                <tr key={r.id} className="border-t border-border">
                  <td className="px-3 py-2 text-fg">{r.userDisplayName ?? "—"}</td>
                  <td className="px-3 py-2 text-fg">{r.workerDisplayName ?? "—"}</td>
                  <td className="px-3 py-2 text-muted">{r.modelFilename ?? r.modelId}</td>
                  <td className="px-3 py-2 text-muted">{r.llamaCppBackend}</td>
                  <td className="px-3 py-2 text-muted">{r.status}</td>
                  <td className="px-3 py-2 text-muted">
                    {r.itemsDone}/{r.itemsTotal}
                  </td>
                  <td className="px-3 py-2 text-muted">{formatDate(r.startedAt)}</td>
                </tr>
              ))}
              {filteredTests.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-3 py-6 text-center text-muted">
                    No runs match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
