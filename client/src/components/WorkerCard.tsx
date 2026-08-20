import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Worker, Run, LlamaCppRelease } from "../types";
import { StatusPill, WorkerStatusPill, ElapsedSince } from "./StatusPill";
import { IconCheck, IconChevronDown, IconDownload, IconPencil, IconTrash } from "./icons";
import { Tooltip } from "./Tooltip";
import { copyToClipboard, formatBytes, formatDate, formatGpuLabel } from "../utils";

export type SetupOS = "windows" | "macos" | "linux";

export const SETUP_OS_LABELS: Array<{ key: SetupOS; label: string; badgeClass: string }> = [
  { key: "windows", label: "WIN", badgeClass: "text-[#8fc6e8] bg-[#8fc6e8]/15" },
  { key: "macos", label: "MACOS", badgeClass: "text-[#c9a6e8] bg-[#c9a6e8]/15" },
  { key: "linux", label: "LINUX", badgeClass: "text-[#f0b86e] bg-[#f0b86e]/15" },
];

// Every Windows command below is PowerShell syntax (iex "& {...}", .\x.ps1
// invocation) -- pasting it into cmd.exe just errors. Deliberately quiet
// (not the warning-colored treatment used elsewhere on this page) since it's
// a heads-up, not something wrong.
export function PowerShellNotice() {
  return (
    <Tooltip text="Written for PowerShell -- paste into a PowerShell window, not Command Prompt (cmd.exe).">
      <span className="text-[10px] font-normal normal-case tracking-normal text-muted/70">PowerShell</span>
    </Tooltip>
  );
}

// Best-effort guess at the OS of the machine viewing the page -- used to
// pre-select client/src/pages/Device.tsx's OS tab so the right setup command
// is already showing on first paint instead of defaulting to "linux"
// regardless of who's looking. navigator.userAgentData (Chromium) is
// preferred where present since it's a plain platform string rather than a
// UA string to pattern-match; every other browser falls back to
// navigator.userAgent. Genuinely ambiguous/unrecognized cases fall back to
// "linux" -- the same default this always had, just now only reached when
// detection can't tell.
export function detectOS(): SetupOS {
  const uaData = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData;
  const platform = (uaData?.platform || navigator.userAgent || "").toLowerCase();
  if (platform.includes("win")) return "windows";
  if (platform.includes("mac")) return "macos";
  return "linux";
}

// worker.platform is Node's raw os.platform() value (see worker/src/
// hardware.ts), not the SetupOS union above -- maps one to the other so the
// Workers-list setup panel can show just THAT machine's own command instead
// of every OS's. Returns null for a worker that's never heartbeated hardware
// info yet (or an unrecognized platform string), so the caller can fall back
// to the browser-viewer-guessing detectOS() + a manual OS switcher instead.
export function platformToSetupOS(platform: string | null | undefined): SetupOS | null {
  if (platform === "win32") return "windows";
  if (platform === "darwin") return "macos";
  if (platform === "linux") return "linux";
  return null;
}

export interface SetupScenario {
  title: string;
  desc: string;
  cmd: Record<SetupOS, string>;
}

// vpsUrl is this server's own public origin (pass window.location.origin --
// the page showing this command IS served from PUBLIC_URL, see
// deploy/orchestrator.env.example) -- baked into the fresh-install command
// so it's copy-paste-runnable as-is instead of erroring with "-VpsUrl is
// required" (bootstrap.ps1/bootstrap.sh both require it on first setup).
// Already-cloned/restart need no URL: by then config.json has it saved.
//
// Exported for client/src/pages/Device.tsx's own install-command display
// (MULTIUSER_PLAN.md §3.1) -- single source of truth so an existing worker's
// restart/reinstall reference panel here and the "Add machine" onboarding
// screen there never drift apart. Kept in sync with worker/bootstrap.ps1,
// worker/bootstrap.sh, worker/setup-worker.ps1, worker/setup-worker.sh, and
// README.md's "Running the worker (GPU box)" section.
export function buildSetupScenarios(vpsUrl: string): SetupScenario[] {
  return [
    {
      title: "Fresh install",
      desc: "Brand-new machine, nothing downloaded yet (no repo, no config, no llama.cpp) -- one command fetches the repo, installs dependencies, and starts the worker. It'll ask which drive/volume to use (showing free space -- models are often tens of GB each) and a folder name, then create it.",
      cmd: {
        windows: `iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) } -VpsUrl ${vpsUrl}"`,
        macos: `curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --vps-url ${vpsUrl}`,
        linux: `curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --vps-url ${vpsUrl}`,
      },
    },
    {
      title: "Already installed",
      desc: "Already have the repo checked out -- same command for first setup (still asks where, unless you pass a folder) and every restart after. Run from the repo root.",
      cmd: {
        windows: ".\\worker\\setup-worker.ps1",
        macos: "bash worker/setup-worker.sh",
        linux: "bash worker/setup-worker.sh",
      },
    },
    {
      title: "Restart",
      desc: "Once it's already set up and running, stop it (Ctrl+C, or close the window) and run this to start it again -- reuses the saved config.json as-is, no prompts.",
      cmd: {
        windows: "worker\\start.bat",
        macos: "npm run worker",
        linux: "npm run worker",
      },
    },
  ];
}

export function CopyCommandButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      onClick={async () => {
        if (await copyToClipboard(text)) {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }
      }}
      className="flex flex-none items-center gap-1 self-start rounded-md border border-border px-2 py-1 text-[11px] font-medium text-muted hover:border-accent/40 hover:text-accent"
    >
      {copied ? (
        <>
          <IconCheck width={12} height={12} /> Copied
        </>
      ) : (
        "Copy"
      )}
    </button>
  );
}

export function WorkerCard({ worker, onRefresh }: { worker: Worker; onRefresh: () => void }) {
  const [busyTag, setBusyTag] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const [available, setAvailable] = useState<LlamaCppRelease[]>([]);
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [activeRun, setActiveRun] = useState<Run | undefined>(undefined);
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState(worker.displayName);
  // Manual OS override, only ever consulted when this worker's own reported
  // platform is unknown (e.g. it's never heartbeated hardware info yet) --
  // seeded from the viewing browser's own OS as a starting guess, same as
  // Device.tsx's add-machine screen.
  const [manualOS, setManualOS] = useState<SetupOS>(() => detectOS());

  const inaccessible = worker.status === "offline";

  // Which releases this machine could install -- a live GitHub lookup
  // (cached server-side), not something the worker reports about itself, so
  // it's its own small fetch rather than inline on the Worker object (see
  // server/src/routes/workers.ts's GET /api/workers/:id/available-builds).
  useEffect(() => {
    let cancelled = false;
    api
      .getAvailableBuilds(worker.id)
      .then((d) => {
        if (cancelled) return;
        setAvailable(d.available);
        setUpdateAvailable(d.update_available);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [worker.id]);

  // Worker.activeRunId only names the run (MULTIUSER_PLAN.md §1.16's "no
  // outbound HTTP" also means no inline model_filename/started_at without
  // this second small fetch) -- resolved separately so the running-banner
  // below can show what the old worker-pushed WorkerCurrentRun did.
  useEffect(() => {
    if (!worker.activeRunId) {
      setActiveRun(undefined);
      return;
    }
    let cancelled = false;
    api
      .getRun(worker.activeRunId)
      .then((d) => {
        if (!cancelled) setActiveRun(d.run);
      })
      .catch(() => {
        if (!cancelled) setActiveRun(undefined);
      });
    return () => {
      cancelled = true;
    };
  }, [worker.activeRunId]);

  async function handleShutdown() {
    const busyNote =
      worker.status === "busy"
        ? " It's currently busy -- the shutdown will happen once the current job finishes, not immediately."
        : "";
    if (
      !window.confirm(
        `Shut down the LlamaToaster worker process on ${worker.displayName}?${busyNote} You'll need to start it again yourself (console access, or however it's set up to run).`
      )
    ) {
      return;
    }
    await withBusy("shutdown", () => api.shutdownWorker(worker.id).then(() => undefined), "Shutdown queued");
  }

  async function handleRemove() {
    if (
      !window.confirm(
        `Remove ${worker.displayName}? This deletes it from the Workers list -- its run history is kept, but if this machine reconnects later it'll need to be re-approved as if it were new.`
      )
    ) {
      return;
    }
    await withBusy("remove", () => api.deleteWorker(worker.id).then(() => undefined), "Removed");
  }

  async function commitRename() {
    const trimmed = nameDraft.trim();
    if (!trimmed || trimmed === worker.displayName) {
      setNameDraft(worker.displayName);
      setRenaming(false);
      return;
    }
    setRenaming(false);
    await withBusy("rename", () => api.renameWorker(worker.id, trimmed).then(() => undefined), "Renamed");
  }

  async function withBusy(tag: string, action: () => Promise<void>, doneMsg: string) {
    setBusyTag(tag);
    setMsg("");
    try {
      await action();
      setMsg(doneMsg);
    } catch (err) {
      setMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusyTag(null);
      onRefresh();
    }
  }

  const installedTags = new Set(worker.installedBuilds.map((b) => b.tag));
  const availableToInstall = available.filter((rel) => !installedTags.has(rel.tag) && rel.assets.length > 0);

  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {renaming ? (
          <input
            autoFocus
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            onBlur={() => void commitRename()}
            onKeyDown={(e) => {
              if (e.key === "Enter") void commitRename();
              else if (e.key === "Escape") {
                setNameDraft(worker.displayName);
                setRenaming(false);
              }
            }}
            maxLength={100}
            className="rounded-lg border border-border bg-surface px-2 py-1 text-lg font-semibold text-fg outline-none focus:border-accent/50"
          />
        ) : (
          <button
            type="button"
            onClick={() => {
              setNameDraft(worker.displayName);
              setRenaming(true);
            }}
            className="group/rename flex items-center gap-1.5 text-left"
            title="Rename this machine"
          >
            <h3 className="text-lg font-semibold text-fg">{worker.displayName}</h3>
            <IconPencil
              width={13}
              height={13}
              className="flex-none text-muted opacity-0 transition-opacity group-hover/rename:opacity-100"
            />
          </button>
        )}
        <div className="flex items-center gap-2">
          {worker.backend && <StatusPill label={worker.backend} tone="muted" />}
          <WorkerStatusPill inaccessible={inaccessible} />
          {!inaccessible && (
            <button
              type="button"
              onClick={() => void handleShutdown()}
              disabled={busyTag !== null}
              className="flex-none text-xs font-semibold text-muted hover:text-danger disabled:opacity-50"
              title="Queue a shutdown -- the worker process exits once it's free (or right away if idle)"
            >
              {busyTag === "shutdown" ? "Queuing…" : "Shut down"}
            </button>
          )}
        </div>
      </div>

      {inaccessible && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm text-muted">
            This machine hasn't checked in recently. It may be offline, asleep, or disconnected.
          </p>
          <button
            type="button"
            onClick={() => void handleRemove()}
            disabled={busyTag !== null}
            className="flex-none text-xs font-semibold text-muted hover:text-danger disabled:opacity-50"
          >
            {busyTag === "remove" ? "Removing…" : "Remove worker"}
          </button>
        </div>
      )}
      <details className="group mt-3 rounded-lg border border-border bg-surface-raised">
        <summary className="flex cursor-pointer items-center justify-between px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
          Setup / restart commands
          <IconChevronDown width={14} height={14} className="transition-transform group-open:rotate-180" />
        </summary>
        <div className="flex flex-col gap-3 border-t border-border p-3">
          {(() => {
            // Prefer this machine's own reported platform over guessing --
            // only fall back to a manual switcher (seeded from the viewing
            // browser's OS) when it's never heartbeated hardware info yet.
            const knownOS = platformToSetupOS(worker.platform);
            const effectiveOS = knownOS ?? manualOS;
            const effectiveLabel = SETUP_OS_LABELS.find((o) => o.key === effectiveOS)!;
            return (
              <>
                <div className="flex items-center gap-1.5">
                  {knownOS ? (
                    <span
                      className={`rounded-md px-2.5 py-1 text-[11px] font-semibold tracking-wide ${effectiveLabel.badgeClass}`}
                    >
                      {effectiveLabel.label}
                    </span>
                  ) : (
                    SETUP_OS_LABELS.map(({ key, label, badgeClass }) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => setManualOS(key)}
                        className={`rounded-md px-2.5 py-1 text-[11px] font-semibold tracking-wide transition-colors ${
                          effectiveOS === key ? badgeClass : "text-muted hover:bg-white/5"
                        }`}
                      >
                        {label}
                      </button>
                    ))
                  )}
                  {effectiveOS === "windows" && <PowerShellNotice />}
                </div>
                {buildSetupScenarios(window.location.origin).map((scenario, i) => (
                  <div key={scenario.title} className="overflow-hidden rounded-lg border border-border">
                    <div className="bg-surface-raised px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="flex h-5 w-5 flex-none items-center justify-center rounded-md bg-accent/15 font-mono text-[11px] font-bold text-accent">
                          {i + 1}
                        </span>
                        <span className="text-sm font-semibold text-fg">{scenario.title}</span>
                      </div>
                      <p className="mt-1.5 text-xs leading-relaxed text-muted">{scenario.desc}</p>
                    </div>
                    <div className="flex items-start gap-3 border-t border-border bg-bg px-3 py-2.5">
                      <code className="flex-1 whitespace-pre-wrap break-all font-mono text-xs text-fg">
                        {scenario.cmd[effectiveOS]}
                      </code>
                      <CopyCommandButton text={scenario.cmd[effectiveOS]} />
                    </div>
                  </div>
                ))}
              </>
            );
          })()}
        </div>
      </details>
      {!inaccessible && worker.status === "busy" && (
        <p className="mt-3 text-sm text-fg">
          {activeRun ? (
            <>
              Running <span className="font-medium">{activeRun.model_filename ?? activeRun.model_id}</span>{" "}
              <span className="text-muted">
                — <ElapsedSince startedAt={activeRun.started_at} />
              </span>
            </>
          ) : worker.activeJobProgress ? (
            <>
              {worker.activeJobProgress.phase}
              {worker.activeJobProgress.detail ? <span className="text-muted"> — {worker.activeJobProgress.detail}</span> : null}
            </>
          ) : (
            <span className="text-muted">Busy</span>
          )}
        </p>
      )}

      {!inaccessible && (
        <>
          <div className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">Hardware &amp; OS</h4>
            <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
              <dt className="text-muted">OS</dt>
              <dd className="text-fg">
                {worker.platform} ({worker.arch})
              </dd>
              <dt className="text-muted">CPU</dt>
              <dd className="text-fg">
                {worker.hardware?.cpu.brand || worker.hardware?.cpu.manufacturer || "unknown"}
                {worker.hardware?.cpu.cores ? (
                  <span className="text-muted"> · {worker.hardware.cpu.cores} threads</span>
                ) : null}
              </dd>
              <dt className="text-muted">RAM</dt>
              <dd className="text-fg">
                {worker.hardware?.mem_total_bytes ? (
                  formatBytes(worker.hardware.mem_total_bytes)
                ) : (
                  <span className="text-muted">unknown</span>
                )}
              </dd>
              <dt className="text-muted">GPU</dt>
              <dd className="text-fg">
                {worker.hardware ? (
                  worker.hardware.gpu.length > 0 ? (
                    // systeminformation's GPU model strings already include the
                    // vendor name (e.g. "AMD Radeon RX 6600 XT") -- prefixing
                    // g.vendor too would just repeat it.
                    worker.hardware.gpu.map((g) => formatGpuLabel(g)).join(", ")
                  ) : (
                    <span className="text-muted">none detected</span>
                  )
                ) : (
                  <span className="text-muted">unknown</span>
                )}
              </dd>
            </dl>
          </div>
          {updateAvailable && (
            <div className="mt-2">
              <StatusPill label="update available" tone="warning" />
            </div>
          )}

          <div className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">Downloaded</h4>
            {worker.installedBuilds.length === 0 ? (
              <p className="mt-1.5 text-sm text-muted">No builds installed yet.</p>
            ) : (
              <ul className="mt-1.5 flex flex-col gap-1.5">
                {worker.installedBuilds.map((b) => (
                  <li
                    key={b.tag}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2"
                  >
                    <div className="flex items-center gap-2 text-sm">
                      {b.active ? (
                        <span className="flex items-center gap-1 text-success">
                          <IconCheck width={14} height={14} /> {b.tag}
                        </span>
                      ) : (
                        <span className="text-fg">{b.tag}</span>
                      )}
                      <span className="text-xs text-muted">
                        {b.asset_name} · installed {formatDate(b.installed_at)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {b.active ? (
                        <StatusPill label="active" tone="accent" />
                      ) : (
                        <>
                          <button
                            type="button"
                            disabled={busyTag === b.tag}
                            onClick={() =>
                              withBusy(
                                b.tag,
                                () => api.activateBuild(worker.id, b.tag).then(() => undefined),
                                `Queued: activate ${b.tag}`
                              )
                            }
                            className="rounded-md bg-accent px-2.5 py-1 text-xs font-semibold text-accent-fg disabled:opacity-50"
                          >
                            Activate
                          </button>
                          <button
                            type="button"
                            disabled={busyTag === b.tag}
                            onClick={() => {
                              if (!window.confirm(`Delete build ${b.tag} from ${worker.displayName}?`)) return;
                              void withBusy(
                                b.tag,
                                () => api.deleteBuild(worker.id, b.tag).then(() => undefined),
                                `Queued: delete ${b.tag}`
                              );
                            }}
                            className="rounded-md border border-border p-1.5 text-muted hover:border-danger/40 hover:text-danger disabled:opacity-50"
                            aria-label={`Delete ${b.tag}`}
                          >
                            <IconTrash width={14} height={14} />
                          </button>
                        </>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">Available to install</h4>
            {availableToInstall.length === 0 ? (
              <p className="mt-1.5 text-sm text-muted">Nothing new for this platform/backend.</p>
            ) : (
              <ul className="mt-1.5 flex max-h-64 flex-col gap-1.5 overflow-y-auto">
                {availableToInstall.map((rel) => {
                  const asset = rel.assets[0];
                  return (
                    <li
                      key={rel.tag}
                      className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border px-3 py-2"
                    >
                      <div className="text-sm">
                        <span className="text-fg">{rel.tag}</span>{" "}
                        <span className="text-xs text-muted">
                          {asset.name} · {formatBytes(asset.size_bytes)}
                        </span>
                      </div>
                      <button
                        type="button"
                        disabled={busyTag === rel.tag}
                        onClick={() =>
                          withBusy(
                            rel.tag,
                            () => api.installBuild(worker.id, rel.tag, asset.name).then(() => undefined),
                            `Queued: install ${rel.tag}`
                          )
                        }
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
                      >
                        <IconDownload width={14} height={14} />
                        {busyTag === rel.tag ? "Queuing…" : "Install"}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </>
      )}

      {msg && <p className="mt-3 text-sm text-muted">{msg}</p>}
    </div>
  );
}
