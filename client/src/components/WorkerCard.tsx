import { useState } from "react";
import { api, type WorkerListEntry } from "../api/client";
import type { WorkerStatus } from "../api/useWorkerStatus";
import { StatusPill, WorkerStatusPill, ElapsedSince } from "./StatusPill";
import { IconCheck, IconChevronDown, IconDownload, IconTrash } from "./icons";
import { formatBytes, formatDate } from "../utils";

export function WorkerCard({
  worker,
  status,
  onRefresh,
}: {
  worker: WorkerListEntry;
  status?: WorkerStatus;
  onRefresh: () => void;
}) {
  const [busyTag, setBusyTag] = useState<string | null>(null);
  const [msg, setMsg] = useState("");

  const info = status?.info;
  const loading = status?.loading ?? true;
  const inaccessible = status?.inaccessible ?? false;

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

  const installedTags = new Set(info?.installed.map((b) => b.tag) ?? []);
  const availableToInstall = (info?.available ?? []).filter(
    (rel) => !installedTags.has(rel.tag) && rel.assets.length > 0
  );

  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-lg font-semibold text-fg">{worker.name}</h3>
        <div className="flex items-center gap-2">
          {info && <StatusPill label={info.backend} tone="muted" />}
          <WorkerStatusPill inaccessible={inaccessible} loading={loading} />
        </div>
      </div>

      {inaccessible && (
        <p className="mt-3 text-sm text-muted">
          This worker can't be reached right now. It may be offline, asleep, or not connected to
          the tailnet.
        </p>
      )}
      {worker.name === "Local" && (
        <details open={inaccessible} className="group mt-3 rounded-lg border border-border bg-surface-raised">
          <summary className="flex cursor-pointer items-center justify-between px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
            Setup / restart commands
            <IconChevronDown width={14} height={14} className="transition-transform group-open:rotate-180" />
          </summary>
          <div className="border-t border-border px-2.5 py-2">
            <p className="text-xs text-muted">
              Brand-new machine, nothing downloaded yet (no repo, no config, no llama.cpp) --
              one command fetches the repo, installs dependencies, and starts the worker. It'll
              ask which drive/volume to use (showing free space -- models are often tens of GB
              each) and a folder name, then create it:
            </p>
            <p className="mt-1 text-xs text-muted">Windows:</p>
            <code className="block whitespace-pre-wrap break-all font-mono text-xs text-fg">
              {'iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) }"'}
            </code>
            <p className="mt-1 text-xs text-muted">macOS:</p>
            <code className="block whitespace-pre-wrap break-all font-mono text-xs text-fg">
              curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash
            </code>
            <p className="mt-1 text-xs text-muted">Linux:</p>
            <code className="block whitespace-pre-wrap break-all font-mono text-xs text-fg">
              curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash
            </code>
            <p className="mt-2 text-xs text-muted">
              Already have the repo checked out -- same command for first setup (still asks
              where, unless you pass a folder) and every restart after, run from the repo root:
            </p>
            <p className="mt-1 text-xs text-muted">Windows:</p>
            <code className="block font-mono text-xs text-fg">.\worker\setup-worker.ps1</code>
            <p className="mt-1 text-xs text-muted">macOS:</p>
            <code className="block font-mono text-xs text-fg">bash worker/setup-worker.sh</code>
            <p className="mt-1 text-xs text-muted">Linux:</p>
            <code className="block font-mono text-xs text-fg">bash worker/setup-worker.sh</code>
            <p className="mt-2 text-xs text-muted">Or, once it's already set up, a plain restart:</p>
            <p className="mt-1 text-xs text-muted">Windows:</p>
            <code className="block font-mono text-xs text-fg">worker\start.bat</code>
            <p className="mt-1 text-xs text-muted">macOS:</p>
            <code className="block font-mono text-xs text-fg">npm run worker</code>
            <p className="mt-1 text-xs text-muted">Linux:</p>
            <code className="block font-mono text-xs text-fg">npm run worker</code>
          </div>
        </details>
      )}
      {!inaccessible && status?.error && <p className="mt-3 text-sm text-danger">{status.error}</p>}
      {!inaccessible && info?.current_run && (
        <p className="mt-3 text-sm text-fg">
          Running <span className="font-medium">{info.current_run.model_filename}</span>{" "}
          <span className="text-muted">— <ElapsedSince startedAt={info.current_run.started_at} /></span>
        </p>
      )}

      {info && !inaccessible && (
        <>
          <div className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">Hardware &amp; OS</h4>
            <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
              <dt className="text-muted">OS</dt>
              <dd className="text-fg">
                {info.platform} ({info.arch})
              </dd>
              <dt className="text-muted">CPU</dt>
              <dd className="text-fg">
                {info.hardware?.cpu.brand || info.hardware?.cpu.manufacturer || "unknown"}
                {info.hardware?.cpu.cores ? (
                  <span className="text-muted"> · {info.hardware.cpu.cores} threads</span>
                ) : null}
              </dd>
              <dt className="text-muted">GPU</dt>
              <dd className="text-fg">
                {info.hardware ? (
                  info.hardware.gpu.length > 0 ? (
                    // systeminformation's GPU model strings already include the
                    // vendor name (e.g. "AMD Radeon RX 6600 XT") -- prefixing
                    // g.vendor too would just repeat it.
                    info.hardware.gpu.map((g) => g.model).join(", ")
                  ) : (
                    <span className="text-muted">none detected</span>
                  )
                ) : (
                  <span className="text-muted">unknown</span>
                )}
              </dd>
            </dl>
          </div>
          {info.update_available && (
            <div className="mt-2">
              <StatusPill label="update available" tone="warning" />
            </div>
          )}

          <div className="mt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-muted">Downloaded</h4>
            {info.installed.length === 0 ? (
              <p className="mt-1.5 text-sm text-muted">No builds installed yet.</p>
            ) : (
              <ul className="mt-1.5 flex flex-col gap-1.5">
                {info.installed.map((b) => (
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
                                () => api.activateBuild(worker.name, b.tag).then(() => undefined),
                                `Active: ${b.tag}`
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
                              if (!window.confirm(`Delete build ${b.tag} from ${worker.name}?`)) return;
                              void withBusy(
                                b.tag,
                                () => api.deleteBuild(worker.name, b.tag).then(() => undefined),
                                `Deleted ${b.tag}`
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
                            () => api.installBuild(worker.name, rel.tag, asset.name).then(() => undefined),
                            `Installed ${rel.tag}`
                          )
                        }
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
                      >
                        <IconDownload width={14} height={14} />
                        {busyTag === rel.tag ? "Installing…" : "Install"}
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
