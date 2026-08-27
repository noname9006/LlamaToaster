// BENCHMARKING_PLAN_V8.md N1 + N5 read paths. Both compute on read over
// ordinary `results` rows; neither materializes anything. The curve endpoint
// serves the caller's OWN rows -- other tenants' contributions surface only
// behind the §0.9 k-anonymity floor, which lives in the community-aggregate
// path, not here.

import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { resolveAuthUser } from "../auth-middleware.js";
import { buildCurve, buildLadder, deriveKnee, type CurveSourceRow } from "../../../shared/curves.js";
import { CURVE_METHOD_VERSION, type ResultRow, type RunConfig } from "../../../shared/types.js";
import { maxAffordableContext, residentWeightsMibFromPeak } from "../../../shared/vramEstimate.js";
import { priceMatrix, priceRate, type RateCandidate } from "../../../shared/pricing.js";

function toCurveRow(row: ResultRow): CurveSourceRow {
  return {
    id: row.id,
    run_id: row.run_id,
    idx: row.idx,
    test_type: row.test_type,
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_depth: row.n_depth ?? 0,
    avg_tps: row.avg_tps,
    stddev_tps: row.stddev_tps,
    vram_peak_mib: row.vram_peak_mib,
    ram_peak_mib: row.ram_peak_mib,
    ttft_ms_p50: row.ttft_ms_p50 ?? null,
    ttft_ms_p95: row.ttft_ms_p95 ?? null,
    ttft_n: row.ttft_n ?? null,
    e2e_ms_mean: row.e2e_ms_mean ?? null,
    method_version: row.method_version ?? null,
    caveat_flags: row.caveat_flags ?? [],
    concurrency: row.concurrency ?? null,
    engine: row.engine ?? null,
    created_at: row.created_at,
  };
}

export async function curveRoutes(app: FastifyInstance): Promise<void> {
  app.get<{
    Params: { id: string };
    Querystring: { worker?: string; build?: string; engine?: string; method_version?: string };
  }>("/api/models/:id/curve", async (request, reply) => {
    const authed = resolveAuthUser(request);
    const userId = authed?.user.id;
    const model = repo.getModel(request.params.id);
    if (!model) return reply.code(404).send({ error: "model not found" });

    const engine = request.query.engine;
    if (engine !== undefined && engine !== "bench" && engine !== "server") {
      return reply.code(400).send({ error: 'engine must be "bench" or "server"' });
    }
    const methodVersionRaw = request.query.method_version;
    let methodVersion: number | null = null;
    if (methodVersionRaw !== undefined) {
      const parsed = Number(methodVersionRaw);
      if (!Number.isInteger(parsed)) {
        return reply.code(400).send({ error: "method_version must be an integer" });
      }
      methodVersion = parsed;
    } else if (engine === "server") {
      // Choreographed server-path points stamp METHOD_VERSION 2 (§0.1's
      // never-average rule then segregates them by construction), which is
      // what keeps ordinary runtime rows' warm-biased TTFT out of curves.
      methodVersion = CURVE_METHOD_VERSION;
    }

    const rows = repo.listCurveRows(userId, model.id, {
      workerId: request.query.worker,
      build: request.query.build,
      engine,
      methodVersion,
    });
    const points = buildCurve(rows.map(toCurveRow));

    // M1's affordability ceiling for the widest measured placement -- the
    // ladder renders anything past it as "unavailable — reason", never as a
    // silent gap.
    let affordableTokens: number | null = null;
    const widest = rows
      .filter((r) => r.vram_peak_mib != null && r.gpu_memory_total_mb != null)
      .sort((a, b) => (b.n_depth ?? 0) + b.n_prompt - ((a.n_depth ?? 0) + a.n_prompt))[0];
    if (widest) {
      const geometry = {
        nLayer: model.metadata.n_layer ?? 0,
        nHeadKv: model.metadata.n_head_kv ?? 0,
        headDimK: model.metadata.head_dim_k,
        headDimV: model.metadata.head_dim_v,
        nEmbd: model.metadata.n_embd,
        nHead: model.metadata.n_head,
        cacheTypeK: widest.cache_type_k,
        cacheTypeV: widest.cache_type_v,
        slidingWindow: model.metadata.sliding_window,
      };
      const estimate = maxAffordableContext({
        ...geometry,
        totalMib: widest.gpu_memory_total_mb!,
        weightsMib: residentWeightsMibFromPeak({
          ...geometry,
          vramPeakMib: widest.vram_peak_mib,
          contextTokens: (widest.n_depth ?? 0) + widest.n_prompt + widest.n_gen,
        }),
        parallelSlots: widest.concurrency ?? 1,
      });
      if (estimate.confidence !== "unknown") affordableTokens = estimate.tokens;
    }

    const ladder = buildLadder(
      points.filter((p) => !p.superseded && !p.excluded).map((p) => p.effectiveCtx),
      { trainedCtx: model.metadata.trained_ctx ?? null, affordableTokens }
    );

    request.log.info(
      { curve_served: true, points: points.length, cache_hit: false, model_id: model.id },
      "curve_served"
    );
    return {
      model_id: model.id,
      engine: engine ?? null,
      method_version: methodVersion,
      points,
      ladder,
      /** True when no ladder cell is still uncovered -- "Measure missing points" renders aria-disabled with this reason. */
      all_points_covered: ladder.every((cell) => !cell.available || cell.measured),
    };
  });

  // §0.6 -- the rate table every ETA prices from. The ORDER is the rule:
  // a {server, spec:"off"} rate first, then a llama-bench rate carrying the
  // mandatory "derived from llama-bench" label, then nothing at all. No cell
  // renders a number without naming its source, so the provenance travels
  // with the rate rather than being reconstructed client-side.
  app.get<{ Params: { id: string }; Querystring: { worker?: string; build?: string } }>(
    "/api/models/:id/rates",
    async (request, reply) => {
      const authed = resolveAuthUser(request);
      const userId = authed?.user.id;
      const model = repo.getModel(request.params.id);
      if (!model) return reply.code(404).send({ error: "model not found" });

      const rows = repo.listCurveRows(userId, model.id, {
        workerId: request.query.worker,
        build: request.query.build,
      });
      const candidates: RateCandidate[] = rows
        .filter((r) => r.test_type === "pp" || r.test_type === "tg")
        .map((r) => ({
          tps: r.avg_tps,
          engine: r.engine ?? (r.mtp === "on" ? "server" : "bench"),
          // A speculative row is not a {server, off} baseline (§0.2's legal
          // pairs), so it can never price an ordinary cell.
          spec: r.mtp === "on" ? "mtp" : "off",
          test_type: r.test_type as "pp" | "tg",
          model_id: model.id,
          worker_id: r.worker_id ?? null,
          llama_cpp_build: r.llama_cpp_build ?? null,
          created_at: r.created_at,
        }));
      const query = {
        model_id: model.id,
        worker_id: request.query.worker ?? null,
        llama_cpp_build: request.query.build ?? null,
      };
      return {
        model_id: model.id,
        pp: priceRate(candidates, query, "pp"),
        tg: priceRate(candidates, query, "tg"),
      };
    }
  );

  // N6 -- the throttle-aware follow-up. > 1/3 of a run's items flagged
  // (denominator: ALL non-cancelled items, skipped included) offers a priced
  // re-run of just those items after cooldown. NEVER automatic: this route
  // reports, the user decides.
  app.get<{ Params: { id: string } }>("/api/runs/:id/sustained", async (request, reply) => {
    const authed = resolveAuthUser(request);
    const userId = authed?.user.id;
    const run = repo.getRun(userId, request.params.id);
    if (!run) return reply.code(404).send({ error: "run not found" });

    const items = repo.getRunItems(run.id);
    const { ratio, denominator } = repo.thermallyFlaggedRatio(run.id);
    const flaggedIdx = new Set(
      repo
        .getResultsForRun(run.id)
        .filter((r) => (r.caveat_flags ?? []).includes("thermally_throttled"))
        .map((r) => r.idx)
    );

    // §0.12's minimum event set fires once per completed run, from the item-
    // terminal write that actually finalizes it (routes/runs.ts) -- not from
    // this read-only GET, which the UI polls repeatedly and would otherwise
    // multiply the same event by every page view.

    const config = run.config as RunConfig;
    const repeats = config?.sweep?.repeats ?? 1;
    const rows = repo.listCurveRows(userId, run.model_id, { workerId: run.worker_id ?? undefined });
    const candidates: RateCandidate[] = rows
      .filter((r) => r.test_type === "pp" || r.test_type === "tg")
      .map((r) => ({
        tps: r.avg_tps,
        engine: r.engine ?? (r.mtp === "on" ? "server" : "bench"),
        spec: r.mtp === "on" ? "mtp" : "off",
        test_type: r.test_type as "pp" | "tg",
        model_id: run.model_id,
        worker_id: r.worker_id ?? null,
        llama_cpp_build: r.llama_cpp_build ?? null,
        created_at: r.created_at,
      }));
    const rateQuery = { model_id: run.model_id, worker_id: run.worker_id ?? null };
    const ppRate = priceRate(candidates, rateQuery, "pp");
    const tgRate = priceRate(candidates, rateQuery, "tg");
    const rerun = priceMatrix(
      items
        .filter((i) => flaggedIdx.has(i.idx))
        .map((i) => ({ nPrompt: i.n_prompt, nGen: i.n_gen, repeats, ppRate, tgRate }))
    );

    // N6's steady-state option is only OFFERED where post-discard n still
    // satisfies the n >= 3 gate; otherwise the control is disabled with that
    // exact reason rather than silently missing.
    const discardFirst = config?.discard_first_repeats ?? 0;
    const discardAvailable = repeats >= 1 + 3;
    return {
      run_id: run.id,
      flagged_items: [...flaggedIdx].sort((a, b) => a - b),
      denominator,
      ratio,
      // Offered, never scheduled.
      offer_rerun: ratio > 1 / 3 && flaggedIdx.size > 0,
      rerun_estimate: rerun.display,
      rerun_seconds: rerun.seconds,
      steady_state: {
        discard_first_repeats: discardFirst,
        available: discardAvailable,
        reason: discardAvailable
          ? null
          : `discarding from ${repeats} repeats would leave n = ${repeats - 1} and break the stability gate's n >= 3 floor`,
      },
    };
  });

  // N5 -- the knee is a DERIVED read, not a stored verdict.
  app.get<{ Params: { id: string } }>("/api/runs/:id/knee", async (request, reply) => {
    const authed = resolveAuthUser(request);
    const userId = authed?.user.id;
    const run = repo.getRun(userId, request.params.id);
    if (!run) return reply.code(404).send({ error: "run not found" });
    const rootRunId = run.root_run_id ?? run.id;
    const rows: ResultRow[] = [];
    for (const member of repo.listRunsUnderRoot(userId, rootRunId)) {
      rows.push(...repo.getResultsForRun(member.id));
    }
    const knee = deriveKnee(rows.map(toCurveRow));
    return {
      run_id: run.id,
      root_run_id: rootRunId,
      spec: (run.config as RunConfig).knee ?? null,
      ...knee,
    };
  });
}
