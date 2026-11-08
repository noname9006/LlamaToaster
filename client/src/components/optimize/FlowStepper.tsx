import { STEP_IDS, STEP_LABEL, type StepId, type StepState } from "./steps";

/** The seven steps across the top of New test. Done steps carry a check,
 * the open one an accent underline; a locked step says why on hover/focus. */
export function FlowStepper({ states, current, onPick }: { states: StepState[]; current: StepId; onPick: (id: StepId) => void }) {
  return (
    <nav aria-label="Steps" className="mb-6">
      <ol className="m-0 flex list-none flex-wrap gap-x-5 gap-y-1 p-0">
        {STEP_IDS.map((id, i) => {
          const s = states.find((x) => x.id === id)!;
          const now = id === current;
          return (
            <li key={id}>
              <button
                type="button"
                onClick={() => onPick(id)}
                disabled={!!s.locked}
                title={s.locked ?? undefined}
                aria-current={now ? "step" : undefined}
                className={`inline-flex min-h-10 items-center gap-1.5 border-b-2 bg-transparent px-0 text-sm ${
                  now
                    ? "border-accent font-semibold text-fg"
                    : s.locked
                      ? "border-transparent text-muted opacity-60"
                      : s.done
                        ? "border-transparent text-fg-2 hover:text-fg"
                        : "border-transparent text-muted hover:text-fg"
                }`}
              >
                <span aria-hidden="true" className="font-mono text-xs">{s.done && !now ? "✓" : i + 1}</span>
                {STEP_LABEL[id]}
                {s.done && !now && <span className="sr-only"> (done)</span>}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
