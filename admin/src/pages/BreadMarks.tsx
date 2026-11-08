import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api";
import type { AppSettings } from "../types";
import { SignIn } from "../components/SignIn";
import { Toggle } from "./Overview";
import {
  BREAD_FALLOFF_RANGE,
  BREAD_STEP_RANGE,
  DEFAULT_BREAD_MARK_STYLE,
  gradeSvg,
  uncheckedSvg,
  type BreadMarkStyle,
} from "../../../shared/breadMarks";
import { fitGradeLegendLabel, fitGrades, type FitScale } from "../../../shared/fitGrade";

// The bread fit marks' look (shared/breadMarks.ts), tuned here and stored in
// AppSettings.breadMarkStyle. The Models page reads it on load; the Chrome
// extension ships with DEFAULT_BREAD_MARK_STYLE. The preview draws the same
// SVG the site does, on the three backgrounds a mark lands on.

type Phase = "loading" | "ready" | "unauthorized" | "error";

const BACKDROPS = [
  { key: "app", label: "Models page", bg: "#1b1611", chip: "#13100c", border: "#3d3329", fg: "#f4efe6", dark: true },
  { key: "hf-light", label: "Hugging Face, light", bg: "#ffffff", chip: "#ffffff", border: "#e5e7eb", fg: "#374151", dark: false },
  { key: "hf-dark", label: "Hugging Face, dark", bg: "#0b0f19", chip: "#111827", border: "#374151", fg: "#d1d5db", dark: true },
] as const;

const SCALES: { scale: FitScale; title: string }[] = [
  { scale: "gpu", title: "GPU with its own VRAM, and fast unified memory" },
  { scale: "ram", title: "CPU only, and integrated GPU on ordinary DDR" },
];

const QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

function Svg({ html }: { html: string }) {
  // Built from constants in shared/breadMarks.ts, never from user input.
  return <span className="inline-flex shrink-0 items-center" dangerouslySetInnerHTML={{ __html: html }} />;
}

function sameStyle(a: BreadMarkStyle, b: BreadMarkStyle): boolean {
  return a.golden === b.golden && a.darkFix === b.darkFix && a.step === b.step && a.falloff === b.falloff && a.direction === b.direction;
}

export function BreadMarks() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [draft, setDraft] = useState<BreadMarkStyle>(DEFAULT_BREAD_MARK_STYLE);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSettings()
      .then((s) => {
        setSettings(s);
        setDraft(s.breadMarkStyle);
        setPhase("ready");
      })
      .catch((err) => {
        if (err instanceof ApiError && (err.status === 403 || err.status === 404)) setPhase("unauthorized");
        else {
          setErrorMessage(err instanceof Error ? err.message : String(err));
          setPhase("error");
        }
      });
  }, []);

  const saved = settings?.breadMarkStyle ?? DEFAULT_BREAD_MARK_STYLE;
  const dirty = !sameStyle(draft, saved);
  const isDefault = sameStyle(draft, DEFAULT_BREAD_MARK_STYLE);

  async function save() {
    setSaving(true);
    setSaveError(null);
    try {
      setSettings(await api.updateSettings({ breadMarkStyle: draft }));
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function toggleEnabled() {
    if (!settings) return;
    const prev = settings;
    setSettings({ ...settings, breadMarksEnabled: !settings.breadMarksEnabled });
    try {
      setSettings(await api.updateSettings({ breadMarksEnabled: !prev.breadMarksEnabled }));
    } catch (err) {
      setSettings(prev);
      setSaveError(err instanceof Error ? err.message : String(err));
    }
  }

  const set = <K extends keyof BreadMarkStyle>(key: K, value: BreadMarkStyle[K]) => setDraft((d) => ({ ...d, [key]: value }));

  const previews = useMemo(
    () =>
      BACKDROPS.map((b) => ({
        ...b,
        scales: SCALES.map(({ scale, title }) => ({
          scale,
          title,
          rows: fitGrades(scale).map((g, i) => ({
            grade: g,
            label: fitGradeLegendLabel(scale, g, "shorter"),
            quant: QUANTS[i],
            small: gradeSvg(scale, g, { px: 12, style: draft, dark: b.dark }),
            mid: gradeSvg(scale, g, { px: 16, style: draft, dark: b.dark }),
          })),
        })),
        big: [
          gradeSvg("gpu", 5, { px: 48, style: draft, dark: b.dark }),
          gradeSvg("ram", 2, { px: 48, style: draft, dark: b.dark }),
          gradeSvg("gpu", 0, { px: 48, style: draft, dark: b.dark }),
        ],
      })),
    [draft]
  );

  if (phase === "loading") return <div className="flex min-h-screen items-center justify-center text-sm text-muted">Loading…</div>;
  if (phase === "unauthorized") return <SignIn />;
  if (phase === "error") {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 text-center text-sm text-danger">
        Could not load settings: {errorMessage}
      </div>
    );
  }

  const selectClass = "rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-fg";
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <Link to="/" className="text-sm text-accent hover:underline">
        ← Overview
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-fg">Bread marks</h1>
      <p className="mt-1 max-w-3xl text-sm text-muted">
        How "Will it fit?" answers are drawn on the Models page. The preview uses the same drawing code as the site. The
        Chrome extension ships with the default style and does not read this setting.
      </p>

      <div className="mt-5 flex items-center justify-between gap-4 rounded-lg border border-border px-3 py-2.5">
        <div>
          <div className="text-sm font-medium text-fg">Show bread marks on the Models page</div>
          <div className="text-xs text-muted">When off, the Models page shows the older check marks.</div>
        </div>
        <Toggle checked={settings?.breadMarksEnabled ?? true} onChange={() => void toggleEnabled()} disabled={settings === null} />
      </div>

      <section className="mt-5 grid gap-3 rounded-lg border border-border p-4 sm:grid-cols-2 lg:grid-cols-3">
        <label className="flex flex-col gap-1 text-sm text-fg">
          Golden slice
          <select className={selectClass} value={draft.golden} onChange={(e) => set("golden", e.target.value as BreadMarkStyle["golden"])}>
            <option value="shine">Vivid + shine</option>
            <option value="butter">Vivid + butter pat</option>
            <option value="plain">Plain</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-fg">
          Dark-background fix
          <select className={selectClass} value={draft.darkFix} onChange={(e) => set("darkFix", e.target.value as BreadMarkStyle["darkFix"])}>
            <option value="shrink">Shrink pale slices</option>
            <option value="crust">Darker crust on pale slices</option>
            <option value="both">Both</option>
            <option value="none">None</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-fg">
          Stack direction
          <select className={selectClass} value={draft.direction} onChange={(e) => set("direction", e.target.value as BreadMarkStyle["direction"])}>
            <option value="ltr">Pale to golden, left to right</option>
            <option value="rtl">Pale to golden, right to left</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-fg">
          <span>
            Gap between slices <span className="font-mono text-muted">{draft.step}</span>
          </span>
          <input
            type="range"
            min={BREAD_STEP_RANGE.min}
            max={BREAD_STEP_RANGE.max}
            step={0.5}
            value={draft.step}
            onChange={(e) => set("step", Number(e.target.value))}
            className="accent-accent"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-fg">
          <span>
            Gap falloff <span className="font-mono text-muted">{draft.falloff === 1 ? "1 (even)" : draft.falloff}</span>
          </span>
          <input
            type="range"
            min={BREAD_FALLOFF_RANGE.min}
            max={BREAD_FALLOFF_RANGE.max}
            step={0.05}
            value={draft.falloff}
            onChange={(e) => set("falloff", Math.round(Number(e.target.value) * 100) / 100)}
            className="accent-accent"
          />
        </label>
        <div className="flex flex-wrap items-end gap-2">
          <button
            type="button"
            disabled={!dirty || saving}
            onClick={() => void save()}
            className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg hover:bg-accent/90 disabled:opacity-40"
          >
            {saving ? "Saving…" : dirty ? "Save" : "Saved"}
          </button>
          <button
            type="button"
            disabled={!dirty}
            onClick={() => setDraft(saved)}
            className="rounded-lg border border-border px-3 py-1.5 text-sm text-fg hover:border-accent disabled:opacity-40"
          >
            Revert
          </button>
          <button
            type="button"
            disabled={isDefault}
            onClick={() => setDraft({ ...DEFAULT_BREAD_MARK_STYLE })}
            className="rounded-lg border border-border px-3 py-1.5 text-sm text-fg hover:border-accent disabled:opacity-40"
          >
            Shipped default
          </button>
        </div>
        {saveError && <p className="text-sm text-danger sm:col-span-2 lg:col-span-3">{saveError}</p>}
      </section>

      <section className="mt-6 grid gap-4 lg:grid-cols-3">
        {previews.map((b) => (
          <div key={b.key} className="rounded-lg p-4" style={{ background: b.bg, color: b.fg, border: `1px solid ${b.border}` }}>
            <h2 className="text-sm font-semibold">{b.label}</h2>
            {b.scales.map((sc) => (
              <div key={sc.scale} className="mt-3">
                <div className="text-[11px] uppercase tracking-wide opacity-60">{sc.title}</div>
                <div className="mt-1 flex flex-col gap-1.5">
                  {sc.rows.map((r) => (
                    <div key={r.grade} className="flex items-center gap-2" title={r.label}>
                      <span className="w-3 text-right font-mono text-[11px] opacity-60">{r.grade}</span>
                      <span
                        className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold"
                        style={{ background: b.chip, border: `1px solid ${b.border}` }}
                      >
                        <Svg html={r.small} />
                        {r.quant}
                      </span>
                      <Svg html={r.mid} />
                      <span className="min-w-0 truncate text-xs opacity-70">{r.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            <div className="mt-3 flex items-center gap-2 text-xs opacity-70">
              <Svg html={uncheckedSvg(16, b.dark ? "#ab9f8d" : "#9ca3af")} /> Not checked yet
            </div>
            <div className="mt-3 flex flex-wrap items-end gap-4">
              {b.big.map((html, i) => (
                <Svg key={i} html={html} />
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
