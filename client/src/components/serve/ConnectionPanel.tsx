import { useEffect, useState } from "react";
import { api, ApiError } from "../../api/client";
import { ALL_INTERFACES_HOST, isLoopbackHost, LOOPBACK_HOST, type GatewaySettings } from "../../../../shared/daemonSettings";
import type { DaemonApiKeyInfo, DaemonStatus } from "../../../../shared/daemonTypes";
import { LtIcon } from "../ltIcons";
import { Btn, Kicker, Panel, StepTitle } from "../optimize/ui";
import { baseUrl } from "./serveData";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Btn
      kind="ghost"
      aria-label={label}
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      <LtIcon name={copied ? "ok" : "copy"} />
      {copied ? "Copied" : "Copy"}
    </Btn>
  );
}

function CodeLine({ text, label }: { text: string; label: string }) {
  return (
    <div className="flex items-start gap-2 border border-border bg-well py-1 pl-3 pr-1">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-pre py-1.5 font-mono text-[13px] text-fg">{text}</code>
      <CopyButton text={text} label={label} />
    </div>
  );
}

export function ConnectionPanel({
  workerId,
  status,
  gateway,
  keys,
  alias,
  onChanged,
}: {
  workerId: string;
  status: DaemonStatus | null;
  gateway: GatewaySettings;
  keys: DaemonApiKeyInfo[];
  alias: string | null;
  onChanged: () => void;
}) {
  const [draft, setDraft] = useState<GatewaySettings>(gateway);
  const [originsText, setOriginsText] = useState(gateway.cors_origins.join(", "));
  const [error, setError] = useState<string | null>(null);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [keyName, setKeyName] = useState("");
  const gatewayKey = JSON.stringify(gateway);

  // Follow the server's value, including when revoking the last key closes the endpoint.
  useEffect(() => {
    setDraft(gateway);
    setOriginsText(gateway.cors_origins.join(", "));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gatewayKey]);

  const active = keys.filter((k) => k.revoked_at == null);
  const dirty = JSON.stringify({ ...draft, cors_origins: parseOrigins(originsText) }) !== gatewayKey;
  const url = baseUrl(gateway.host, gateway.port, status);
  const exposed = !isLoopbackHost(gateway.host);
  const addresses = status?.interfaces ?? [];
  const customHost = draft.host !== LOOPBACK_HOST && draft.host !== ALL_INTERFACES_HOST && !addresses.some((a) => a.address === draft.host);
  const run = (fn: () => Promise<unknown>) => {
    setError(null);
    fn().then(onChanged, (err: unknown) => setError(err instanceof ApiError ? err.message : String(err)));
  };

  const curl = [
    `curl ${url}/chat/completions \\`,
    `  -H "Content-Type: application/json" \\`,
    ...(active.length > 0 ? [`  -H "Authorization: Bearer $LLAMATOASTER_KEY" \\`] : []),
    `  -d '{"model":"${alias ?? "MODEL"}","messages":[{"role":"user","content":"Hello"}]}'`,
  ].join("\n");

  return (
    <Panel>
      <StepTitle>Connect an agent</StepTitle>
      <div className="flex flex-col gap-3">
        <div>
          <Kicker>OpenAI base URL</Kicker>
          <div className="mt-1">
            <CodeLine text={url} label="Copy the base URL" />
          </div>
          {status?.gateway.error ? (
            <p className="m-0 mt-1 flex items-center gap-1.5 text-sm text-danger">
              <LtIcon name="warn" /> {status.gateway.error}
            </p>
          ) : (
            <p className="m-0 mt-1 text-xs text-muted">
              {status?.gateway.listening
                ? exposed
                  ? "Answers on the network over plain HTTP. Keep it on a network you trust, or put TLS in front of it."
                  : "Answers on that machine only."
                : "Starts listening when you load a model."}{" "}
              Also speaks /v1/messages for Anthropic clients.
            </p>
          )}
        </div>
        <details>
          <summary className="cursor-pointer text-sm text-fg-2">Example request</summary>
          <div className="mt-1">
            <CodeLine text={curl} label="Copy the example request" />
          </div>
        </details>

        <div>
          <Kicker>API keys</Kicker>
          {newKey && (
            <div className="mt-1 border border-warning bg-warning-bg p-2">
              <p className="m-0 mb-1 text-sm text-warning">Copy this key now. It is only stored hashed, so it cannot be shown again.</p>
              <CodeLine text={newKey} label="Copy the new API key" />
            </div>
          )}
          <ul className="m-0 mt-1 flex list-none flex-col p-0">
            {active.map((k) => (
              <li key={k.id} className="flex items-center justify-between gap-2 border-t border-border py-1 text-sm first:border-t-0">
                <span className="min-w-0 truncate text-fg">
                  {k.name} <span className="font-mono text-xs text-muted">{k.prefix}…</span>
                </span>
                <Btn kind="ghost" onClick={() => run(() => api.revokeDaemonKey(workerId, k.id))}>
                  Revoke
                </Btn>
              </li>
            ))}
          </ul>
          {active.length === 0 && <p className="m-0 mt-1 text-xs text-muted">None. Without a key, anything running on that machine can use the endpoint, and it cannot be opened to the network.</p>}
          <form
            className="mt-2 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              run(async () => {
                const created = await api.createDaemonKey(workerId, keyName);
                setNewKey(created.key);
                setKeyName("");
              });
            }}
          >
            <input type="text" aria-label="New key name" value={keyName} maxLength={64} placeholder="Key name" onChange={(e) => setKeyName(e.target.value)} className="min-h-10 min-w-0 flex-1 border border-border-strong bg-well px-2 text-sm text-fg" />
            <Btn type="submit">Create key</Btn>
          </form>
        </div>

        <details>
          <summary className="cursor-pointer text-sm text-fg-2">Endpoint settings</summary>
          <div className="mt-2 flex flex-col gap-2">
            <label className="flex flex-col gap-1 text-sm text-fg-2">
              Listen on
              <select
                value={draft.host}
                onChange={(e) => setDraft({ ...draft, host: e.target.value })}
                className="min-h-10 border border-border-strong bg-well px-2 text-sm text-fg"
              >
                <option value={LOOPBACK_HOST}>This machine only</option>
                {addresses.map((a) => (
                  <option key={a.address} value={a.address} disabled={active.length === 0}>
                    {a.address} ({a.name})
                  </option>
                ))}
                <option value={ALL_INTERFACES_HOST} disabled={active.length === 0}>
                  All network addresses
                </option>
                {customHost && <option value={draft.host}>{draft.host}</option>}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-fg-2">
              Port
              <input type="number" min={1024} max={65535} value={draft.port} onChange={(e) => setDraft({ ...draft, port: Number(e.target.value) })} className="min-h-10 border border-border-strong bg-well px-2 font-mono text-sm text-fg" />
            </label>
            <label className="flex flex-col gap-1 text-sm text-fg-2">
              Allowed browser origins
              <input type="text" value={originsText} placeholder="None. Agents outside a browser need none." onChange={(e) => setOriginsText(e.target.value)} className="min-h-10 border border-border-strong bg-well px-2 font-mono text-sm text-fg" />
            </label>
            <label className="flex min-h-10 items-center gap-2 text-sm text-fg-2">
              <input type="checkbox" checked={draft.autostart} onChange={(e) => setDraft({ ...draft, autostart: e.target.checked })} />
              Load the last model when the worker starts
            </label>
            <div>
              <Btn disabled={!dirty} onClick={() => run(() => api.saveDaemonGateway(workerId, { ...draft, cors_origins: parseOrigins(originsText) }))}>
                Save
              </Btn>
            </div>
          </div>
        </details>
        {error && <p className="m-0 text-sm text-danger">{error}</p>}
        <p className="m-0 text-xs text-muted">Prompts and answers stay on that machine. Only token counts and speeds reach the LlamaToaster server.</p>
      </div>
    </Panel>
  );
}

function parseOrigins(text: string): string[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}
