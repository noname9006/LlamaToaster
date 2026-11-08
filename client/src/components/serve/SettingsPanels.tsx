import { useEffect, useState, type ReactNode } from "react";
import {
  LOAD_GROUP_LABELS,
  LOAD_SETTINGS,
  SAMPLING_SETTINGS,
  type LoadSettingDef,
  type LoadSettingGroup,
  type SamplingDef,
  type SamplingSettings,
  type SettingValue,
  type SettingValues,
} from "../../../../shared/daemonSettings";
import { Kicker, Panel, StepTitle } from "../optimize/ui";
import { Tooltip } from "../Tooltip";

const INPUT = "min-h-9 w-full min-w-0 border border-border-strong bg-well px-2 font-mono text-sm text-fg disabled:opacity-50";

function Row({ label, help, flag, children, onReset, note }: { label: string; help: string; flag?: string; children: ReactNode; onReset?: () => void; note?: string }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_28px] items-center gap-x-2 gap-y-0.5 border-t border-border py-1.5 first:border-t-0">
      <div className="min-w-0">
        <Tooltip text={help}>
          <span className="cursor-help text-sm text-fg-2 underline decoration-border-strong decoration-dotted underline-offset-2">{label}</span>
        </Tooltip>
        {flag && <div className="truncate font-mono text-[11px] text-muted">{flag}</div>}
        {note && <div className="text-[11px] text-warning">{note}</div>}
      </div>
      <div className="min-w-0">{children}</div>
      <div>
        {onReset && (
          <button type="button" onClick={onReset} aria-label={`Reset ${label} to the default`} title="Back to the default" className="h-7 w-7 border-0 bg-transparent text-muted hover:text-fg">
            ×
          </button>
        )}
      </div>
    </div>
  );
}

// Context sizes are chosen in doublings, so a linear slider would be useless for
// everything below a million tokens.
const LOG_MIN = 10;
const LOG_MAX = 20;

function NumberControl({ def, value, onChange, placeholder, sliderMax, disabled }: { def: { label: string; min: number; max: number; step?: number; kind: string; slider?: boolean; logScale?: boolean }; value: number | undefined; onChange: (v: number | undefined) => void; placeholder: string; sliderMax?: number; disabled?: boolean }) {
  const number = (
    <input
      type="number"
      aria-label={def.label}
      min={def.min}
      max={def.max}
      step={def.step ?? (def.kind === "int" ? 1 : "any")}
      value={value ?? ""}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
      className={INPUT}
    />
  );
  if (!def.slider) return number;
  const log = def.logScale === true;
  const min = log ? LOG_MIN : def.min;
  const max = log ? LOG_MAX : sliderMax ?? def.max;
  // Unset rests on the build's own default, where the page knows it, not on the
  // far left, which would read as that value.
  const shown = value ?? (Number.isFinite(Number(placeholder)) ? Number(placeholder) : null);
  const position = shown == null ? min : log ? Math.log2(Math.max(1, shown)) : shown;
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        aria-label={`${def.label} slider`}
        min={min}
        max={max}
        step={log ? 1 : def.step ?? 1}
        value={Math.min(max, Math.max(min, position))}
        disabled={disabled}
        onChange={(e) => onChange(log ? 2 ** Number(e.target.value) : Number(e.target.value))}
        className="min-w-0 flex-[1_1_60px]"
      />
      <div className="w-[88px] shrink-0">{number}</div>
    </div>
  );
}

function LoadControl({ def, value, onChange, disabled, sliderMax }: { def: LoadSettingDef; value: SettingValue | undefined; onChange: (v: SettingValue | undefined) => void; disabled: boolean; sliderMax?: number }) {
  switch (def.kind) {
    case "int":
    case "float":
      return <NumberControl def={def} value={value as number | undefined} onChange={onChange} placeholder="default" disabled={disabled} sliderMax={sliderMax} />;
    case "enum":
      return (
        <select aria-label={def.label} value={(value as string | undefined) ?? ""} disabled={disabled} onChange={(e) => onChange(e.target.value || undefined)} className={INPUT}>
          <option value="">Default</option>
          {def.options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      );
    case "toggle":
      return (
        <select
          aria-label={def.label}
          value={value === undefined ? "" : value ? "on" : "off"}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value === "on")}
          className={INPUT}
        >
          <option value="">Default</option>
          <option value="on">On</option>
          {def.offFlag && <option value="off">Off</option>}
        </select>
      );
    case "text":
      return <input type="text" aria-label={def.label} maxLength={def.maxLength} value={(value as string | undefined) ?? ""} placeholder={def.placeholder ?? "default"} disabled={disabled} onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)} className={INPUT} />;
  }
}

function AdvancedToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm text-fg-2">
      <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} />
      Show advanced
    </label>
  );
}

export interface LoadExtras {
  alias: string;
  mmproj: string;
  draftModelId: string;
}

export function LoadSettingsPanel({
  values,
  onChange,
  extras,
  onExtras,
  defaultAlias,
  supported,
  buildTag,
  projectors,
  draftModels,
  cores,
  footer,
}: {
  values: SettingValues;
  onChange: (next: SettingValues) => void;
  extras: LoadExtras;
  onExtras: (next: LoadExtras) => void;
  defaultAlias: string;
  /** Empty until the worker reports: everything stays enabled. */
  supported: string[];
  buildTag: string | null;
  projectors: string[];
  draftModels: { id: string; label: string }[];
  cores: number | null;
  footer: ReactNode;
}) {
  const [advanced, setAdvanced] = useState(false);
  const groups = Object.keys(LOAD_GROUP_LABELS) as LoadSettingGroup[];
  const isSupported = (key: string) => supported.length === 0 || supported.includes(key);
  const set = (key: string, value: SettingValue | undefined) => {
    const next = { ...values };
    if (value === undefined) delete next[key];
    else next[key] = value;
    onChange(next);
  };
  const sliderMaxFor = (key: string) => (key === "threads" ? cores ?? undefined : key === "parallel" ? 16 : undefined);

  return (
    <Panel>
      <StepTitle right={<AdvancedToggle on={advanced} onChange={setAdvanced} />}>Load settings</StepTitle>
      <p className="m-0 mb-3 text-sm text-muted">Set when the model loads. Changing them needs a reload. Anything left on Default uses what the llama.cpp build chooses.</p>
      <div className="flex flex-col gap-4">
        {groups.map((group) => {
          const defs = LOAD_SETTINGS.filter((d) => d.group === group && (advanced || !d.advanced || values[d.key] !== undefined));
          const special = group === "serving" || group === "vision" || group === "speculative";
          if (defs.length === 0 && !special) return null;
          return (
            <div key={group}>
              <Kicker>{LOAD_GROUP_LABELS[group]}</Kicker>
              <div className="mt-1">
                {group === "serving" && (
                  <Row label="API identifier" flag="--alias" help="The model name agents send. Requests naming anything else are refused." onReset={extras.alias ? () => onExtras({ ...extras, alias: "" }) : undefined}>
                    <input type="text" aria-label="API identifier" maxLength={128} value={extras.alias} placeholder={defaultAlias} onChange={(e) => onExtras({ ...extras, alias: e.target.value })} className={INPUT} />
                  </Row>
                )}
                {group === "vision" && (
                  <Row label="Projector file" flag="--mmproj" help="A multimodal projector on this machine, for vision models.">
                    <select aria-label="Projector file" value={extras.mmproj} onChange={(e) => onExtras({ ...extras, mmproj: e.target.value })} className={INPUT}>
                      <option value="">None</option>
                      {projectors.map((p) => (
                        <option key={p} value={p}>
                          {p.split(/[\\/]/).pop()}
                        </option>
                      ))}
                    </select>
                  </Row>
                )}
                {group === "speculative" && (
                  <Row label="Draft model" flag="--model-draft" help="A smaller model on this machine that drafts tokens for the main one.">
                    <select aria-label="Draft model" value={extras.draftModelId} onChange={(e) => onExtras({ ...extras, draftModelId: e.target.value })} className={INPUT}>
                      <option value="">None</option>
                      {draftModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </Row>
                )}
                {defs.map((def) => {
                  const ok = isSupported(def.key);
                  return (
                    <Row
                      key={def.key}
                      label={def.label}
                      flag={def.flag}
                      help={def.help}
                      note={ok ? undefined : `Not in ${buildTag ?? "this build"}`}
                      onReset={values[def.key] !== undefined ? () => set(def.key, undefined) : undefined}
                    >
                      <LoadControl def={def} value={values[def.key]} disabled={!ok} sliderMax={sliderMaxFor(def.key)} onChange={(v) => set(def.key, v)} />
                    </Row>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      {footer && <div className="mt-4 border-t border-border pt-3">{footer}</div>}
    </Panel>
  );
}

function SamplingRow({ def, value, placeholder, onChange }: { def: SamplingDef; value: number | undefined; placeholder: string; onChange: (v: number | undefined) => void }) {
  const hasSlider = def.kind === "float" && def.max <= 5;
  return (
    <Row label={def.label} help={def.help} onReset={value !== undefined ? () => onChange(undefined) : undefined}>
      <NumberControl def={{ ...def, slider: hasSlider }} value={value} onChange={onChange} placeholder={placeholder} />
    </Row>
  );
}

function parseStopStrings(text: string): string[] | undefined {
  const stop = text.split("\n").filter((line) => line !== "");
  return stop.length > 0 ? stop : undefined;
}

// Holds what is being typed apart from the parsed list. Rendering from the list would
// erase a line break the moment it is typed, before there is anything after it.
function StopStrings({ value, onChange }: { value: string[] | undefined; onChange: (stop: string[] | undefined) => void }) {
  const [text, setText] = useState((value ?? []).join("\n"));
  const parsedKey = JSON.stringify(parseStopStrings(text) ?? []);
  const valueKey = JSON.stringify(value ?? []);
  // The list changed from outside: another model's settings, or a reset.
  useEffect(() => {
    if (valueKey !== parsedKey) setText((value ?? []).join("\n"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [valueKey]);
  return (
    <textarea
      aria-label="Stop strings"
      rows={2}
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        onChange(parseStopStrings(e.target.value));
      }}
      placeholder="None"
      className="min-h-9 w-full min-w-0 border border-border-strong bg-well px-2 py-1 font-mono text-sm text-fg"
    />
  );
}

export function SamplingPanel({
  value,
  onChange,
  defaults,
  footer,
}: {
  value: SamplingSettings;
  onChange: (next: SamplingSettings) => void;
  /** The build's own defaults, read from the running server. */
  defaults: Record<string, number> | null;
  footer: ReactNode;
}) {
  const [advanced, setAdvanced] = useState(false);
  const setValue = (key: string, v: number | undefined) => {
    const values = { ...value.values };
    if (v === undefined) delete values[key];
    else values[key] = v;
    onChange({ ...value, values });
  };
  const placeholderFor = (def: SamplingDef) => {
    const d = defaults?.[def.key];
    if (d == null) return "default";
    if (def.key === "max_tokens" && d < 0) return "no limit";
    // The build reports its "random" seed as -1 or as 2^32-1, depending on the path.
    if (def.key === "seed" && (d < 0 || d > def.max)) return "random";
    return String(Math.round(d * 1000) / 1000);
  };
  return (
    <Panel>
      <StepTitle right={<AdvancedToggle on={advanced} onChange={setAdvanced} />}>Sampling</StepTitle>
      <p className="m-0 mb-3 text-sm text-muted">Applied on the next request, with no reload. Fills in what a request leaves unset, so an agent's own values win unless you override them.</p>
      <div>
        <label className="flex flex-col gap-1 pb-2">
          <Tooltip text="Used for requests that send no system message of their own.">
            <span className="cursor-help text-sm text-fg-2 underline decoration-border-strong decoration-dotted underline-offset-2">System prompt</span>
          </Tooltip>
          <textarea
            rows={3}
            maxLength={65536}
            value={value.system_prompt ?? ""}
            onChange={(e) => onChange({ ...value, system_prompt: e.target.value || undefined })}
            placeholder="None"
            className="w-full border border-border-strong bg-well px-2 py-1.5 text-sm text-fg"
          />
        </label>
        {SAMPLING_SETTINGS.filter((d) => advanced || !d.advanced || value.values[d.key] !== undefined).map((def) => (
          <SamplingRow key={def.key} def={def} value={value.values[def.key]} placeholder={placeholderFor(def)} onChange={(v) => setValue(def.key, v)} />
        ))}
        <Row label="Stop strings" help="One per line. Generation stops at any of them." onReset={value.stop ? () => onChange({ ...value, stop: undefined }) : undefined}>
          <StopStrings value={value.stop} onChange={(stop) => onChange({ ...value, stop })} />
        </Row>
        <label className="mt-2 flex min-h-10 items-center gap-2 border-t border-border pt-2 text-sm text-fg-2">
          <input type="checkbox" checked={value.override_client === true} onChange={(e) => onChange({ ...value, override_client: e.target.checked || undefined })} />
          Override what clients send
        </label>
      </div>
      {footer && <div className="mt-3 border-t border-border pt-3">{footer}</div>}
    </Panel>
  );
}
