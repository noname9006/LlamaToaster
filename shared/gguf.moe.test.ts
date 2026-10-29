import { describe, expect, it } from "vitest";
import { buildTensorLayerBreakdown } from "./gguf.js";
import { detectMoe } from "./moePlacement.js";

const t = (name: string, size: number) => ({ name, size });

describe("buildTensorLayerBreakdown: routed-expert tensor names", () => {
  it("counts the fused ffn_gate_up_exps stack (gemma-4 26B-A4B) as expert bytes", () => {
    const b = buildTensorLayerBreakdown(
      [
        t("blk.0.attn_q.weight", 100),
        t("blk.0.ffn_gate_inp.weight", 5),
        t("blk.0.ffn_gate_up_exps.weight", 300),
        t("blk.0.ffn_down_exps.weight", 150),
        t("blk.1.ffn_gate_up_exps.weight", 300),
        t("blk.1.ffn_down_exps.weight", 150),
      ],
      2
    );
    expect(b.moe).toEqual([450, 450]);
    expect(b.dense).toEqual([105, 0]);
    const moe = detectMoe(b);
    expect(moe.isMoe).toBe(true);
    expect(moe.moeLayers).toEqual([0, 1]);
    expect(moe.expertBytesPerLayer).toBe(450);
  });

  it("counts the split gate / up / down stacks and the _chexps variant", () => {
    const b = buildTensorLayerBreakdown(
      [
        t("blk.0.ffn_gate_exps.weight", 10),
        t("blk.0.ffn_up_exps.weight", 20),
        t("blk.0.ffn_down_exps.weight", 30),
        t("blk.1.ffn_gate_chexps.weight", 40),
      ],
      2
    );
    expect(b.moe).toEqual([60, 40]);
    expect(b.dense).toEqual([0, 0]);
  });

  it("counts the older per-expert tensors (ffn_gate.3.weight)", () => {
    const b = buildTensorLayerBreakdown([t("blk.0.ffn_gate.3.weight", 7), t("blk.0.ffn_up.0.weight", 8)], 1);
    expect(b.moe).toEqual([15]);
  });

  it("keeps the router, shared experts and plain dense FFN tensors dense", () => {
    const b = buildTensorLayerBreakdown(
      [
        t("blk.0.ffn_gate_inp.weight", 5),
        t("blk.0.ffn_gate_shexp.weight", 7),
        t("blk.0.ffn_up_shexp.weight", 8),
        t("blk.0.ffn_down_shexp.weight", 9),
        t("blk.1.ffn_gate.weight", 11),
        t("blk.1.ffn_up.weight", 12),
        t("blk.1.ffn_down.weight", 13),
        t("blk.1.ffn_gate_up.weight", 14),
      ],
      2
    );
    expect(b.moe).toEqual([0, 0]);
    expect(b.dense).toEqual([29, 50]);
    expect(detectMoe(b).isMoe).toBe(false);
  });

  it("detects leading dense blocks followed by MoE blocks with fused experts", () => {
    const b = buildTensorLayerBreakdown(
      [
        t("blk.0.ffn_gate.weight", 11),
        t("blk.1.ffn_gate_up_exps.weight", 200),
        t("blk.2.ffn_gate_up_exps.weight", 200),
      ],
      3
    );
    expect(detectMoe(b).moeLayers).toEqual([1, 2]);
  });
});
