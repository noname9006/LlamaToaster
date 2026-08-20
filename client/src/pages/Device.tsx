import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { DeviceStatusResponse } from "../types";
import { CopyCommandButton, SETUP_OS_LABELS, buildSetupScenarios, detectOS, type SetupOS } from "../components/WorkerCard";
import { IconServer, IconInfo, IconCheck } from "../components/icons";
import { formatRelativeTime } from "../utils";

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
  // Pre-selected from the browser's own OS (§1 of the Settings/Device
  // rework) so the right setup command is already showing on first paint --
  // still just the tab's initial value, so a click on WIN/MACOS/LINUX above
  // freely overrides it same as before.
  const [os, setOs] = useState<SetupOS>(() => detectOS());
  const [codeInput, setCodeInput] = useState("");
  const [status, setStatus] = useState<DeviceStatusResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approved, setApproved] = useState<{ hostname: string; merged: boolean } | null>(null);

  // This page's own origin IS the server's public URL (see
  // WorkerCard.tsx's buildSetupScenarios doc comment) -- stable for the
  // life of the page, so computed once rather than on every render.
  const setupScenarios = useMemo(() => buildSetupScenarios(window.location.origin), []);

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

  async function handleApprove(mergeInto?: string) {
    setBusy(true);
    setError(null);
    try {
      // confirm_duplicate reflects what the last status poll already showed
      // -- see server/src/routes/device.ts's POST /api/device/approve and
      // workerRepo.findPossibleDuplicate's doc comment for why this exists:
      // deleting a worker's install folder wipes its persisted machine_id,
      // so re-running setup looks like a brand-new machine to the server,
      // and without this check would silently create an indistinguishable
      // duplicate of a machine the user already has. mergeInto instead asks
      // the server to re-attach this connection to that existing machine.
      const confirmDuplicate = !mergeInto && status?.state === "pending" && status.possibleDuplicate != null;
      const res = await api.approveDevice(codeInput, confirmDuplicate, mergeInto);
      if (!res.ok) {
        // Race: the duplicate was only detected server-side just now (this
        // poll hadn't caught up yet) -- the next poll tick will pick up
        // possibleDuplicate and relabel the buttons; nothing was approved.
        setError('This machine looks like one you already have -- pick "Merge" or "Add as new machine" once it appears below.');
        return;
      }
      setApproved({ hostname: res.machine.hostname ?? "This machine", merged: res.merged === true });
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
        Run whichever of these matches the GPU box you want to connect. Setting it up for the
        first time (or reconnecting a machine whose session was revoked) prints a one-time code —
        enter that code below once it appears. A plain restart doesn't need this page at all.
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
      {/* First-run command has this page's own origin baked in as -VpsUrl/
          --vps-url (see WorkerCard.tsx's buildSetupScenarios) -- copy-paste
          runs as-is instead of erroring "-VpsUrl is required". Already-
          installed/restart need no URL: config.json has it saved by then. */}
      <div className="mt-2 flex flex-col gap-3">
        {setupScenarios.map((scenario, i) => (
          <div key={scenario.title} className="overflow-hidden rounded-lg border border-border">
            <div className="bg-surface-raised px-3 py-2">
              <div className="flex items-center gap-2">
                <span className="flex h-5 w-5 flex-none items-center justify-center rounded-md bg-accent/15 font-mono text-[11px] font-bold text-accent">
                  {i + 1}
                </span>
                <span className="text-sm font-semibold text-fg">{scenario.title}</span>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-muted">{scenario.desc}</p>
            </div>
            <div className="flex items-start gap-3 border-t border-border bg-bg px-3 py-2.5">
              <code className="flex-1 whitespace-pre-wrap break-all font-mono text-xs text-fg">
                {scenario.cmd[os]}
              </code>
              <CopyCommandButton text={scenario.cmd[os]} />
            </div>
          </div>
        ))}
      </div>

      {approved ? (
        <div className="mt-6 flex items-center gap-3 rounded-lg border border-success/30 bg-success-bg px-4 py-3">
          <IconCheck width={18} height={18} className="flex-none text-success" />
          <div>
            <div className="text-sm font-semibold text-fg">{approved.merged ? "Merged" : "Connected"}</div>
            <div className="text-xs text-muted">
              {approved.merged
                ? `${approved.hostname} reconnected to its existing worker -- its history and settings are unchanged.`
                : `${approved.hostname} is approved — it'll show up on the Workers page shortly.`}
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
              {status.possibleDuplicate ? (
                <div className="mt-3 flex items-start gap-2 rounded-md bg-warning-bg px-2.5 py-2 text-xs text-warning">
                  <IconInfo width={14} height={14} className="mt-0.5 flex-none" />
                  <span>
                    This looks like a machine you already have -- "{status.possibleDuplicate.displayName}" (
                    {status.possibleDuplicate.hostnameMatch && status.possibleDuplicate.hardwareMatch
                      ? "same name and hardware"
                      : status.possibleDuplicate.hostnameMatch
                        ? "same name"
                        : "looks like the same hardware, different name"}
                    ){status.possibleDuplicate.lastHeartbeatAt != null
                      ? `, last seen ${formatRelativeTime(new Date(status.possibleDuplicate.lastHeartbeatAt).toISOString())}`
                      : ", never connected"}
                    . Merge to reconnect it to that same worker and keep its history, or add it as a new,
                    separate one.
                  </span>
                </div>
              ) : (
                <div className="mt-3 flex items-start gap-2 rounded-md bg-warning-bg px-2.5 py-2 text-xs text-warning">
                  <IconInfo width={14} height={14} className="mt-0.5 flex-none" />
                  Only approve a code you generated, on a machine you trust.
                </div>
              )}
              <div className="mt-3 flex gap-2">
                {status.possibleDuplicate ? (
                  <button
                    type="button"
                    onClick={() => void handleApprove(status.possibleDuplicate!.id)}
                    disabled={busy}
                    className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
                  >
                    Merge into "{status.possibleDuplicate.displayName}"
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => void handleApprove()}
                    disabled={busy}
                    className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
                  >
                    Approve
                  </button>
                )}
                {status.possibleDuplicate && (
                  <button
                    type="button"
                    onClick={() => void handleApprove()}
                    disabled={busy}
                    className="rounded-lg border border-border px-4 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
                  >
                    Add as new machine
                  </button>
                )}
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
