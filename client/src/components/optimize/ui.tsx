import { useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { LtIcon } from "../ltIcons";

// Small building blocks in the v2 style (docs/plans/app-v2.dc.html): square
// corners, 1px rules, condensed headings, mono kickers.

export function PageHeader({ title, sub, actions, crumbs }: { title: string; sub?: ReactNode; actions?: ReactNode; crumbs?: ReactNode }) {
  return (
    <header className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {crumbs && <nav aria-label="Breadcrumb" className="mb-2 text-sm text-muted">{crumbs}</nav>}
        <h1 className="m-0 font-display text-[32px] font-semibold leading-tight">{title}</h1>
        {sub && <p className="mt-1 text-fg-2">{sub}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </header>
  );
}

export function Kicker({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`font-mono text-xs uppercase tracking-[0.06em] text-muted ${className}`}>{children}</div>;
}

export function Panel({ children, className = "", accent = false, id }: { children: ReactNode; className?: string; accent?: boolean; id?: string }) {
  return (
    <section id={id} className={`border bg-surface p-4 sm:px-[18px] ${accent ? "border-accent" : "border-border"} ${className}`}>
      {children}
    </section>
  );
}

export function StepTitle({ n, children, right }: { n?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-2">
      <h2 className="m-0 font-display text-lg font-semibold">
        {n && <span className="mr-1.5 font-mono text-sm font-normal text-muted">{n} ·</span>}
        {children}
      </h2>
      {right}
    </div>
  );
}

type BtnKind = "primary" | "secondary" | "ghost";

export function Btn({
  kind = "secondary",
  big = false,
  className = "",
  children,
  ...rest
}: { kind?: BtnKind; big?: boolean } & ButtonHTMLAttributes<HTMLButtonElement>) {
  const base = "inline-flex items-center justify-center gap-2 border px-3.5 text-sm font-medium transition-colors disabled:opacity-45";
  const size = big ? "min-h-12 px-5 font-display text-lg font-semibold" : "min-h-10";
  const look =
    kind === "primary"
      ? "border-accent bg-accent text-accent-fg hover:bg-accent-hover font-display text-base font-semibold"
      : kind === "ghost"
        ? "border-transparent bg-transparent text-fg-2 hover:border-border-strong hover:text-fg"
        : "border-border-strong bg-transparent text-fg hover:border-accent hover:text-accent";
  return (
    <button type="button" className={`${base} ${size} ${look} ${className}`} {...rest}>
      {children}
    </button>
  );
}

export function Seg<T extends string | number>({
  options,
  value,
  onChange,
  label,
  disabled,
}: {
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <div role="group" aria-label={label} className="flex w-fit max-w-full flex-wrap border border-border-strong">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={String(o.value)}
            type="button"
            aria-pressed={on}
            title={o.title}
            disabled={disabled || o.disabled}
            onClick={() => onChange(o.value)}
            className={`min-h-10 flex-[1_1_auto] border-r border-border px-3 text-sm font-medium last:border-r-0 disabled:opacity-45 ${
              on ? "bg-accent-tint text-accent" : "bg-transparent text-fg-2 hover:bg-surface-raised"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function StatCard({ label, value, sub, wide = false }: { label: string; value: ReactNode; sub?: ReactNode; wide?: boolean }) {
  return (
    <div className={`flex min-w-0 flex-col gap-0.5 border border-border bg-surface-raised px-3 py-2.5 ${wide ? "sm:col-span-2" : ""}`}>
      <Kicker>{label}</Kicker>
      <span className="font-display text-[22px] font-semibold leading-tight [overflow-wrap:anywhere]">{value}</span>
      {sub && <span className="text-xs text-fg-2">{sub}</span>}
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warn" | "fail"; children: ReactNode }) {
  const cls =
    tone === "fail"
      ? "border-danger/60 bg-danger-bg text-fg"
      : tone === "warn"
        ? "border-accent/50 bg-accent-tint text-fg"
        : "border-border bg-surface-raised text-fg-2";
  return (
    <div role={tone === "fail" ? "alert" : undefined} className={`flex items-start gap-2.5 border px-3 py-2.5 text-sm ${cls}`}>
      <LtIcon name={tone === "info" ? "chip" : "warn"} className={`mt-0.5 flex-none ${tone === "fail" ? "text-danger" : tone === "warn" ? "text-accent" : "text-muted"}`} size={16} />
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export function ProgressBar({ pct, live = false, label }: { pct: number; live?: boolean; label?: string }) {
  const v = Math.max(0, Math.min(100, Math.round(pct)));
  return (
    <span
      role="progressbar"
      aria-label={label ?? "Progress"}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={v}
      className="block h-2 w-full border border-border-strong bg-well"
    >
      <span className={`block h-full ${live ? "lt-progress-fill" : "bg-accent"}`} style={{ width: `${v}%` }} />
    </span>
  );
}

export function CopyCommand({ command }: { command: string }) {
  return (
    <div className="flex flex-wrap items-center gap-2.5">
      <code className="block min-w-0 flex-[1_1_320px] border border-border bg-well px-3 py-2.5 font-mono text-[13px] text-fg [overflow-wrap:anywhere]">
        {command}
      </code>
      <CopyButton text={command} />
    </div>
  );
}


export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Btn
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard blocked -- the text is selectable */
        }
      }}
    >
      <LtIcon name="copy" size={16} />
      <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
    </Btn>
  );
}

export function fmtGb(mib: number | null | undefined, digits = 1): string {
  return mib == null ? "—" : `${(mib / 1024).toFixed(digits)} GB`;
}

export function fmtTps(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v >= 100 ? Math.round(v).toLocaleString("en-US") : v.toFixed(1);
}

export function fmtDuration(seconds: number): string {
  const m = Math.max(1, Math.round(seconds / 60));
  return m < 60 ? `≈ ${m} min` : `≈ ${Math.floor(m / 60)} h ${m % 60} min`;
}
