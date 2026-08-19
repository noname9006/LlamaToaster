import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "./api";
import type { AdminStats, AdminRunSummary } from "./types";

type Phase = "loading" | "unauthorized" | "ready" | "error";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex-1 rounded-xl border border-border bg-surface px-6 py-4 text-center">
      <div className="text-3xl font-bold text-accent">{value}</div>
      <div className="mt-1 text-sm text-muted">{label}</div>
    </div>
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
  const [runs, setRuns] = useState<AdminRunSummary[]>([]);

  const [userFilter, setUserFilter] = useState("");
  const [workerFilter, setWorkerFilter] = useState("");
  const [backendFilter, setBackendFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");

  useEffect(() => {
    Promise.all([api.getStats(), api.listRuns()])
      .then(([s, r]) => {
        setStats(s);
        setRuns(r.runs);
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
  // server-side, see repo.ts's adminRepo.listRuns) -- feels instant, and
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

  const filteredRuns = useMemo(
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

      <section className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-5">
        <StatCard label="Users" value={stats!.users} />
        <StatCard label="Machines" value={stats!.machines} />
        <StatCard label="Runs" value={stats!.runs} />
        <StatCard label="Tests" value={stats!.tests} />
        <StatCard label="Models" value={stats!.models} />
      </section>

      <section className="mt-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-fg">Runs</h2>
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
              {filteredRuns.map((r) => (
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
              {filteredRuns.length === 0 && (
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
