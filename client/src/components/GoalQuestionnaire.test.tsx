// N2 UI collapse: the six-mode card grid ("Max GPU speed", "Max context",
// "Fixed context", "Balanced", "Fixed offload", "Custom") became two
// scenario cards, Wizard and Targets, with Targets' three-way pin choice
// (Context / Offload / Both) picking which underlying ProbeMode it
// dispatches. This is a real render of the component (not a
// reimplementation) against a crafted placement, the cheapest way to prove
// the click/selection wiring survived the collapse without a live
// model+worker pairing.
//
// fireEvent rather than @testing-library/user-event -- the latter isn't a
// project dependency, and every interaction here is a single synchronous
// click, so fireEvent's lower-fidelity simulation loses nothing.

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { GoalQuestionnaire, type GoalQuestionnaireProps } from "./GoalQuestionnaire";
import { defaultGoals } from "../goals";

function basePlacement(): NonNullable<GoalQuestionnaireProps["placement"]> {
  return {
    ngl: 17,
    onNglChange: vi.fn(),
    nglMax: 41,
    kvLayerCount: 40,
    modelSizeBytes: 17205 * 1024 * 1024,
    tensorBreakdown: null,
    locked: null,
    vram: { totalMib: 8176, freeMib: 7600 },
    ram: { totalMib: 32768, freeMib: 26000 },
    unifiedPool: false,
    noGpu: false,
    poolHaircutFrac: 0,
    onReset: vi.fn(),
    verifyResults: {},
    onRunModes: vi.fn().mockResolvedValue(undefined),
    heldModes: new Set(),
  };
}

function renderWithPlacement(overrides: Partial<GoalQuestionnaireProps> = {}) {
  const onChange = vi.fn();
  const placement = overrides.placement ?? basePlacement();
  render(
    <MemoryRouter>
      <GoalQuestionnaire goals={defaultGoals()} onChange={onChange} {...overrides} placement={placement} />
    </MemoryRouter>
  );
  return { onChange, placement };
}

describe("GoalQuestionnaire's Tested-configurations section", () => {
  it("renders exactly two scenario cards, not the old six", () => {
    renderWithPlacement();
    expect(screen.getByRole("button", { name: /^Wizard/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Targets/ })).toBeInTheDocument();
    // The old six-card labels must not survive as scenario cards. "Max
    // context" and "Balanced" are excluded from this check -- both are ALSO
    // GoalKind labels in the unrelated "What are you optimizing for?" chips
    // rendered elsewhere in this same component (confirmed by this test
    // itself: asserting "Balanced" absent here originally failed against
    // that chip, not against a leftover mode card), so asserting their
    // absence would be asserting a regression in a feature this change
    // never touched.
    for (const retired of ["Max GPU speed", "Fixed context", "Fixed offload"]) {
      expect(screen.queryByText(retired)).not.toBeInTheDocument();
    }
  });

  // A frontier probe measures a boundary per context stop. Summarising it by
  // the single stored ceiling names the largest context that loaded, which on
  // a machine where context is expensive is routinely the rung with NOTHING on
  // the GPU -- true, useless, and worse as something to apply to the sliders.
  describe("a Wizard card holding a measured curve", () => {
    const curveResult = {
      ngl: 41,
      ctx: 1024,
      testId: "t-curve",
      status: "verified" as const,
      mode: "frontier" as const,
      // The ceiling that loaded is 262,144 -- with zero layers on the GPU.
      verifiedCtxTokens: 262_144,
      measuredNgl: 0,
      curve: [
        { ctx: 1024, ngl: 40 },
        { ctx: 16_384, ngl: 40 },
        { ctx: 65_536, ngl: 33 },
        { ctx: 262_144, ngl: 0 },
      ],
    };
    const withCurve = () => {
      const placement = basePlacement();
      placement.verifyResults = { frontier: curveResult };
      return renderWithPlacement({ placement });
    };

    it("names both ends of the curve rather than the zero-layer ceiling", () => {
      withCurve();
      expect(screen.getByText(/40 layers @ 1k → 0 @ 256k/)).toBeInTheDocument();
      expect(screen.queryByText(/262,144 tokens · 0 layers/)).not.toBeInTheDocument();
    });

    it("applies the point at the context the user is aiming at, not the curve's top end", () => {
      const { placement } = withCurve();
      fireEvent.click(screen.getByRole("button", { name: /^Wizard/ }));
      // Whatever the slider's current target is, the applied layer count is a
      // real point on the curve -- and never the 0-layer top end while a
      // smaller context is being targeted.
      const appliedNgl = (placement.onNglChange as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0];
      expect(curveResult.curve.map((p) => p.ngl)).toContain(appliedNgl);
      expect(appliedNgl).toBeGreaterThan(0);
    });
  });

  // A probe with a --list-devices reading answers twice: the most offload with
  // no spill, and the most offload whose llama.cpp claim fits free VRAM. The
  // card names both and lets the user pick which one goes on the sliders.
  describe("a card holding both targets", () => {
    const wizardResult = {
      ngl: 41,
      ctx: 1024,
      testId: "t-two",
      status: "verified" as const,
      mode: "frontier" as const,
      verifiedCtxTokens: 131_072,
      measuredNgl: 2,
      curve: [
        { ctx: 1024, ngl: 6 },
        { ctx: 131_072, ngl: 2 },
      ],
      fitCurve: [
        { ctx: 1024, ngl: 17 },
        { ctx: 262_144, ngl: 14 },
      ],
    };

    it("names both answers", () => {
      const placement = basePlacement();
      placement.verifyResults = { frontier: wizardResult };
      renderWithPlacement({ placement });
      expect(screen.getByText(/no spill: 6 layers @ 1k → 2 @ 128k/)).toBeInTheDocument();
      expect(screen.getByText(/fits VRAM: 17 layers @ 1k → 14 @ 256k/)).toBeInTheDocument();
    });

    it("applies the chosen answer without selecting the card", () => {
      const placement = basePlacement();
      placement.verifyResults = { frontier: wizardResult };
      renderWithPlacement({ placement });
      fireEvent.click(screen.getByText(/fits VRAM:/));
      const appliedNgl = (placement.onNglChange as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0];
      expect(wizardResult.fitCurve.map((p) => p.ngl)).toContain(appliedNgl);
      expect(screen.getByRole("button", { name: /^Wizard/ })).toHaveAttribute("aria-pressed", "false");

      fireEvent.click(screen.getByText(/no spill:/));
      const cleanNgl = (placement.onNglChange as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0];
      expect(wizardResult.curve.map((p) => p.ngl)).toContain(cleanNgl);
    });

    it("shows target 2 on a Targets probe that found no no-spill answer, instead of a bare 'didn't fit'", () => {
      const placement = basePlacement();
      placement.verifyResults = {
        keep_context: { ngl: 11, ctx: 262_144, testId: "t-f", status: "failed", measuredNgl: null, fit: { ctx: 262_144, ngl: 11 } },
      };
      renderWithPlacement({ placement });
      expect(screen.getByText("✓ fits VRAM: 11 layers @ 256k")).toBeInTheDocument();
      expect(screen.getByText("✗ no spill: none")).toBeInTheDocument();
      expect(screen.queryByText("✗ didn’t fit")).not.toBeInTheDocument();
      fireEvent.click(screen.getByText(/fits VRAM:/));
      expect((placement.onNglChange as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0]).toBe(11);
    });

    it("says where the Wizard's budget ran out", () => {
      const placement = basePlacement();
      placement.verifyResults = { frontier: { ...wizardResult, unfinishedFrom: 262_144 } };
      renderWithPlacement({ placement });
      expect(screen.getByText(/budget ran out at 262 k/)).toBeInTheDocument();
    });

    it("names both answers on a Targets card as single placements", () => {
      const placement = basePlacement();
      placement.verifyResults = {
        keep_context: {
          ngl: 17, ctx: 1024, testId: "t-k", status: "verified", verifiedCtxTokens: 1024, measuredNgl: 6,
          fit: { ctx: 1024, ngl: 17 },
        },
      };
      renderWithPlacement({ placement });
      expect(screen.getByText("✓ no spill: 6 layers @ 1k")).toBeInTheDocument();
      expect(screen.getByText("✓ fits VRAM: 17 layers @ 1k")).toBeInTheDocument();
    });
  });

  // The fine setting is gone: the search has no finer grid any more.
  it("offers no granularity choice", () => {
    renderWithPlacement();
    expect(screen.queryByRole("radiogroup", { name: "Test granularity" })).not.toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: "Fine tune" })).not.toBeInTheDocument();
  });

  it("labels the Wizard card with its real dispatch mode for assistive tech", () => {
    renderWithPlacement();
    expect(screen.getByRole("button", { name: /Wizard — runs as frontier/ })).toBeInTheDocument();
  });

  it("defaults Targets to a context pin, dispatching keep_context", () => {
    renderWithPlacement();
    expect(screen.getByRole("button", { name: /Targets — runs as keep_context/ })).toBeInTheDocument();
  });

  it("switches Targets to fixed_offload when Offload is chosen", () => {
    renderWithPlacement();
    fireEvent.click(screen.getByRole("radio", { name: "Offload" }));
    expect(screen.getByRole("button", { name: /Targets — runs as fixed_offload/ })).toBeInTheDocument();
  });

  it("switches Targets to custom when Both is chosen", () => {
    renderWithPlacement();
    fireEvent.click(screen.getByRole("radio", { name: "Both" }));
    expect(screen.getByRole("button", { name: /Targets — runs as custom/ })).toBeInTheDocument();
  });

  it("only one pin choice is checked at a time", () => {
    renderWithPlacement();
    expect(screen.getByRole("radio", { name: "Context" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: "Offload" })).toHaveAttribute("aria-checked", "false");
    expect(screen.getByRole("radio", { name: "Both" })).toHaveAttribute("aria-checked", "false");
    fireEvent.click(screen.getByRole("radio", { name: "Offload" }));
    expect(screen.getByRole("radio", { name: "Context" })).toHaveAttribute("aria-checked", "false");
    expect(screen.getByRole("radio", { name: "Offload" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: "Both" })).toHaveAttribute("aria-checked", "false");
  });

  it("choosing a pin does not select or deselect the Targets card itself", () => {
    renderWithPlacement();
    const targetsCard = screen.getByRole("button", { name: /^Targets/ });
    expect(targetsCard).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(screen.getByRole("radio", { name: "Offload" }));
    expect(targetsCard).toHaveAttribute("aria-pressed", "false");
  });

  it("selects Wizard on click, applies its starting placement, and enables Run test", () => {
    const { placement } = renderWithPlacement();
    const wizardCard = screen.getByRole("button", { name: /^Wizard/ });
    fireEvent.click(wizardCard);
    expect(wizardCard).toHaveAttribute("aria-pressed", "true");
    // Selecting also applies the card's own starting placement to the
    // sliders (via placement.onNglChange, PlacementMatrix's own
    // onApplyConfig closure) -- the same behavior the old six-card grid had.
    expect(placement.onNglChange).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /Run test/ })).toBeEnabled();
  });

  it("fires both scenarios together as one batch when both are selected", async () => {
    const onRunModes = vi.fn().mockResolvedValue(undefined);
    renderWithPlacement({ placement: { ...basePlacement(), onRunModes } });
    fireEvent.click(screen.getByRole("button", { name: /^Wizard/ }));
    fireEvent.click(screen.getByRole("button", { name: /^Targets/ }));
    fireEvent.click(screen.getByRole("button", { name: /Run test/ }));
    await waitFor(() => expect(onRunModes).toHaveBeenCalledTimes(1));
    const [modes] = onRunModes.mock.calls[0] as [string[], unknown, unknown];
    expect(modes).toEqual(["frontier", "keep_context"]);
  });

  it("re-resolves the pinned mode at click time, not at selection time", async () => {
    const onRunModes = vi.fn().mockResolvedValue(undefined);
    renderWithPlacement({ placement: { ...basePlacement(), onRunModes } });
    fireEvent.click(screen.getByRole("button", { name: /^Targets/ })); // selects while pinned to context (default)
    fireEvent.click(screen.getByRole("radio", { name: "Both" })); // now both pinned -> custom
    fireEvent.click(screen.getByRole("button", { name: /Run test/ }));
    await waitFor(() => expect(onRunModes).toHaveBeenCalledTimes(1));
    const [modes] = onRunModes.mock.calls[0] as [string[], unknown, unknown];
    expect(modes).toEqual(["custom"]);
  });

  it("locks the pin choice while a probe fired under a DIFFERENT pin combination is still pending", () => {
    // The scenario this guards against: Targets fires as keep_context, then
    // -- while that probe is still "pending" server-side -- the user
    // chooses a different pin. Without the guard, the display remaps to a
    // clean fixed_offload key (no result yet under that key) and the choice
    // goes live again, so a second probe could fire while the first is
    // still loading the model under keep_context's own key.
    renderWithPlacement({
      placement: {
        ...basePlacement(),
        verifyResults: { keep_context: { ngl: 17, ctx: 262144, testId: "t1", status: "pending" } },
      },
    });
    expect(screen.getByRole("radio", { name: "Context" })).toBeDisabled();
    expect(screen.getByRole("radio", { name: "Offload" })).toBeDisabled();
    expect(screen.getByRole("radio", { name: "Both" })).toBeDisabled();
    fireEvent.click(screen.getByRole("radio", { name: "Offload" }));
    // Still keep_context -- the click was refused, not silently accepted.
    expect(screen.getByRole("button", { name: /Targets — runs as keep_context/ })).toBeInTheDocument();
  });

  it("re-enables the pin choice once the in-flight probe resolves", () => {
    renderWithPlacement({
      placement: {
        ...basePlacement(),
        verifyResults: { keep_context: { ngl: 17, ctx: 262144, testId: "t1", status: "verified", verifiedCtxTokens: 262144 } },
      },
    });
    expect(screen.getByRole("radio", { name: "Offload" })).toBeEnabled();
    fireEvent.click(screen.getByRole("radio", { name: "Offload" }));
    expect(screen.getByRole("button", { name: /Targets — runs as fixed_offload/ })).toBeInTheDocument();
  });

  it("locks the pin choice while a sibling combination is held behind a precheck, even with no result yet", () => {
    renderWithPlacement({
      placement: { ...basePlacement(), heldModes: new Set(["fixed_offload"]) },
    });
    expect(screen.getByRole("radio", { name: "Offload" })).toBeDisabled();
  });

  it("leaves the unrelated GoalKind 'Max context' chip untouched", () => {
    renderWithPlacement();
    // Confirms the M2 goal selector (a different feature, sharing a label by
    // coincidence with the retired ProbeMode) was not touched by this change.
    expect(screen.getByText("Max context")).toBeInTheDocument();
  });
});
