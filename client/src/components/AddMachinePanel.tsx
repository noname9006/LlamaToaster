import { useMemo, useState } from "react";
import { useDeviceEnrolment } from "../api/useDeviceEnrolment";
import {
  CopyCommandButton,
  PowerShellNotice,
  SETUP_OS_LABELS,
  buildSetupScenarios,
  detectOS,
  type SetupOS,
} from "./WorkerCard";
import { IconServer, IconInfo, IconCheck } from "./icons";
import { formatRelativeTime } from "../utils";

// MULTIUSER_PLAN.md §3.1's "Add machine" screen, extracted from pages/
// Device.tsx so it can be embedded directly on the Workers page (top of the
// page) AND still rendered by the bookmarkable /device route -- it's the
// SAME component either way, not two separate screens. Assumes a logged-in
// session with AUTH_ENABLED (the /device route checks that itself before
// rendering this).
//
// Enrolment-code state (Workers.tsx also shows a compact code box next to
// the collapsed "Add a machine" summary, wired to this same state) can be
// passed in via `enrolment`; when omitted (the /device route's standalone
// use) this component creates its own.
export function AddMachinePanel({
  enrolment,
}: {
  enrolment?: ReturnType<typeof useDeviceEnrolment>;
}) {
  // Pre-selected from the browser's own OS (§1 of the Settings/Device
  // rework) so the right setup command is already showing on first paint --
  // still just the tab's initial value, so a click on WIN/MACOS/LINUX above
  // freely overrides it same as before.
  const [os, setOs] = useState<SetupOS>(() => detectOS());
  const ownEnrolment = useDeviceEnrolment();
  const { codeInput, handleCodeChange, validCode, status, busy, error, approved, handleApprove, handleDeny } =
    enrolment ?? ownEnrolment;

  // This page's origin IS the server's public URL (see
  // WorkerCard.tsx's buildSetupScenarios doc comment) -- stable for the
  // life of the page, so computed once rather than on every render.
  const setupScenarios = useMemo(() => buildSetupScenarios(window.location.origin), []);

  return (
    <div className="max-w-2xl">
      <p className="text-sm text-muted">
        Run whichever of these matches the GPU box you want to connect. Setting it up for the
        first time (or reconnecting a machine whose session was revoked) prints a one-time code —
        enter that code below once it appears. A plain restart doesn't need this page at all.
      </p>

      <div className="mt-6 flex items-center gap-1.5">
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
        {os === "windows" && <PowerShellNotice />}
      </div>
      {/* First-run command has this page's own origin baked in as -Url/
          --url (see WorkerCard.tsx's buildSetupScenarios) -- copy-paste
          runs as-is instead of erroring "-Url is required". Already-
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
          {/* When embedded on the Workers page (`enrolment` passed in), the
              same code box already sits next to the "Add a machine" summary
              above -- showing it again here would just be a second box bound
              to the same state. Standalone (/device route) still needs it. */}
          {!enrolment && (
            <>
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
            </>
          )}

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