import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DeviceStatusResponse } from "../types";
import { CopyCommandButton, SETUP_OS_LABELS, FRESH_INSTALL_CMD, type SetupOS } from "../components/WorkerCard";
import { IconServer, IconInfo, IconCheck } from "../components/icons";

// Matches server/src/session.ts's generateUserCode (4 chars, a dash, 4 more,
// drawn from Crockford base32 minus ambiguous characters) -- used both to
// know when the field is "full enough to poll" and to reject obviously
// unfinished input before it ever reaches the server.
const CODE_RE = /^[A-Z0-9]{4}-[A-Z0-9]{4}$/;

function normalizeCodeInput(raw: string): string {
  const upper = raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, 8);
  return upper.length > 4 ? `${upper.slice(0, 4)}-${upper.slice(4)}` : upper;
}

const POLL_INTERVAL_MS = 2000;

// MULTIUSER_PLAN.md §3.1's "Add machine" screen, reachable both as a normal
// in-app page (a button on Workers) and at this same bookmarkable /device URL
// for the case where enrolment starts somewhere the browser wasn't already
// open (a headless box set up over SSH, a code relayed from someone else) --
// it's the SAME component either way, not two separate screens. Requires a
// logged-in session (App.tsx redirects an unauthenticated visitor to /login
// before this ever renders, same as every other route once AUTH_ENABLED).
export function Device() {
  // Reachable directly by URL even when nothing links to it on a
  // Stage-1-only (no AUTH_ENABLED) deployment -- deviceApprovalRoutes are
  // only ever registered server-side when AUTH_ENABLED (server/src/index.ts),
  // so this page needs to know that too rather than attempting fetches
  // against routes that don't exist (same pattern as Settings.tsx).
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null);
  const [os, setOs] = useState<SetupOS>("linux");
  const [codeInput, setCodeInput] = useState("");
  const [status, setStatus] = useState<DeviceStatusResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approvedHost, setApprovedHost] = useState<string | null>(null);

  useEffect(() => {
    api
      .getAuthStatus()
      .then((s) => setAuthEnabled(s.authEnabled))
      .catch(() => setAuthEnabled(false));
  }, []);

  const validCode = CODE_RE.test(codeInput);

  // Polls GET /api/device/status while a full-length code is present (§3.1
  // step 4) -- stops as soon as it resolves to "approved" (whether from this
  // tab's own Approve click below, or a second browser tab/device that
  // approved it first) or the input changes to something no longer a full
  // code.
  useEffect(() => {
    if (!authEnabled || !validCode) return;
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    async function poll() {
      try {
        const s = await api.getDeviceStatus(codeInput);
        if (cancelled) return;
        setStatus(s);
        setError(null);
        if (s.state === "approved" && timer) {
          clearInterval(timer);
          timer = null;
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    }
    void poll();
    timer = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [authEnabled, validCode, codeInput]);

  function handleCodeChange(raw: string) {
    setCodeInput(normalizeCodeInput(raw));
    setStatus(null);
    setError(null);
  }

  async function handleApprove() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.approveDevice(codeInput);
      setApprovedHost(res.machine.hostname ?? "This machine");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  // No dedicated deny endpoint exists (or needs to) -- a code nobody approves
  // just expires on its own in 15 minutes (server/src/routes/device.ts's
  // ENROLMENT_TTL_MS). This just walks the user away from the confirm card.
  function handleDeny() {
    setCodeInput("");
    setStatus(null);
    setError(null);
  }

  if (authEnabled === null) {
    return <p className="text-sm text-muted">Loading…</p>;
  }
  if (!authEnabled) {
    return (
      <div>
        <h1 className="text-2xl font-semibold text-fg">Add a machine</h1>
        <p className="mt-4 max-w-lg text-sm text-muted">
          Device enrolment needs user accounts to be enabled on this deployment. Set up a new
          worker with the shared worker token instead — see the setup commands on the Workers
          page.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-semibold text-fg">Add a machine</h1>
      <p className="mt-2 text-sm text-muted">
        Run this on the GPU box you want to connect — it's the same command every time, for every
        machine. It prints a one-time code; enter that code below once it appears.
      </p>

      <div className="mt-6 flex gap-1.5">
        {SETUP_OS_LABELS.map(({ key, label, badgeClass }) => (
          <button
            key={key}
            type="button"
            onClick={() => setOs(key)}
            className={`rounded-md px-2.5 py-1 text-[11px] font-semibold tracking-wide transition-colors ${
              os === key ? badgeClass : "text-muted hover:bg-white/5"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="mt-2 flex items-start gap-3 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
        <code className="flex-1 whitespace-pre-wrap break-all font-mono text-xs text-fg">
          {FRESH_INSTALL_CMD[os]}
        </code>
        <CopyCommandButton text={FRESH_INSTALL_CMD[os]} />
      </div>

      {approvedHost ? (
        <div className="mt-6 flex items-center gap-3 rounded-lg border border-success/30 bg-success-bg px-4 py-3">
          <IconCheck width={18} height={18} className="flex-none text-success" />
          <div>
            <div className="text-sm font-semibold text-fg">Connected</div>
            <div className="text-xs text-muted">
              {approvedHost} is approved — it'll show up on the Workers page shortly.
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6">
          <label htmlFor="device-code" className="text-sm font-semibold text-fg">
            Have a code?
          </label>
          <div>
            <input
              id="device-code"
              type="text"
              value={codeInput}
              onChange={(e) => handleCodeChange(e.target.value)}
              placeholder="ABCD-EFGH"
              maxLength={9}
              className="mt-1.5 w-40 rounded-lg border border-border bg-surface px-3 py-2 text-center font-mono text-sm uppercase tracking-widest text-fg outline-none focus:border-accent/50"
            />
          </div>

          {error && <p className="mt-2 text-sm text-danger">{error}</p>}

          {!error && validCode && status?.state === "not_found" && (
            <p className="mt-2 text-sm text-muted">
              No pending code found for that — check it was typed correctly, or it may have
              expired (codes last 15 minutes).
            </p>
          )}

          {!error && status?.state === "approved" && (
            <p className="mt-2 text-sm text-muted">This code has already been approved.</p>
          )}

          {status?.state === "pending" && (
            <div className="mt-4 rounded-lg border border-border bg-surface-raised p-4">
              <div className="flex items-center gap-2.5">
                <IconServer width={20} height={20} className="text-muted" />
                <span className="text-base font-semibold text-fg">
                  {status.machine.hostname ?? "unknown host"}
                </span>
              </div>
              <div className="mt-1 text-xs text-muted">
                {[status.machine.platform, status.machine.arch, status.machine.gpu]
                  .filter(Boolean)
                  .join(" · ") || "no machine details reported"}
              </div>
              <div className="mt-3 flex items-start gap-2 rounded-md bg-warning-bg px-2.5 py-2 text-xs text-warning">
                <IconInfo width={14} height={14} className="mt-0.5 flex-none" />
                Only approve a code you generated, on a machine you trust.
              </div>
              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  onClick={() => void handleApprove()}
                  disabled={busy}
                  className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
                >
                  Approve
                </button>
                <button
                  type="button"
                  onClick={handleDeny}
                  disabled={busy}
                  className="rounded-lg border border-border px-4 py-1.5 text-sm font-semibold text-fg hover:border-danger/40 hover:text-danger disabled:opacity-50"
                >
                  Deny
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
