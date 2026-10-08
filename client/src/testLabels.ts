import type { LtIconName } from "./components/ltIcons";
import type { Test } from "./types";
import type { FitMapSpec } from "../../shared/optimizeFlow.js";
import type { SpeedRunSpec, ThreadSweepSpec } from "../../shared/speedRun.js";

// Plain-language labels for a test row, shared by the Dashboard, the Tests
// list and the status bar.

const STEP_LABEL: Record<string, string> = {
  baseline: "Baseline",
  fit: "Fit map",
  real_load: "Real-load check",
  threads: "Thread sweep",
  confirm: "Confirm",
};

export function kindLabel(t: Test): string {
  const step = (t.config as { flow_step?: string }).flow_step;
  if (step && STEP_LABEL[step]) return STEP_LABEL[step];
  switch (t.kind) {
    case "fit":
      return "Fit map";
    case "probe":
      return "Context test";
    case "quality":
      return "Quality";
    case "runtime":
      return "Speed test";
    case "tuning":
    case "refine":
    case "sweep":
      return "Chain";
    default:
      return "Custom test";
  }
}

export function testStatusView(t: Pick<Test, "status">): { label: string; tone: string; icon: LtIconName | null } {
  switch (t.status) {
    case "running":
      return { label: "RUNNING", tone: "text-accent", icon: "run" };
    case "scheduled":
      return { label: "QUEUED", tone: "text-fg-2", icon: "queue" };
    case "done":
      return { label: "DONE", tone: "text-fg-2", icon: "ok" };
    case "partial":
      return { label: "PARTLY DONE", tone: "text-accent", icon: "warn" };
    case "failed":
      return { label: "FAILED", tone: "text-danger", icon: "oom" };
    case "cancelled":
      return { label: "STOPPED", tone: "text-muted", icon: "stop" };
  }
}

/** One-line summary of what an optimization-flow run measures; null for
 * every other kind (they keep their own params line). */
export function flowParams(t: Test): string | null {
  const c = t.config as { fit_map?: FitMapSpec; speed_run?: SpeedRunSpec; thread_sweep?: ThreadSweepSpec };
  if (c.fit_map) {
    const s = c.fit_map;
    return `${s.machine === "dgpu" ? "ctx × layers" : "max context"} · ${s.ctx_stops.length} contexts · margin ${s.margin_mib} MiB${s.kv_fit ? " · KV caches" : ""}`;
  }
  if (c.speed_run) {
    const s = c.speed_run;
    const n = s.candidates.length;
    return `${n} config${n === 1 ? "" : "s"} · ${Math.round(s.target_ctx / 1024)}k ctx · ${s.prompt_tokens}+${s.n_gen} tokens · ${s.repeats}×`;
  }
  if (c.thread_sweep) {
    const s = c.thread_sweep;
    return `${s.items.length} thread setting${s.items.length === 1 ? "" : "s"} · -ngl ${s.placement.ngl} · ${s.repeats}×`;
  }
  return null;
}
