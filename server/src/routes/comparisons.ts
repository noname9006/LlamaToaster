// BENCHMARKING_PLAN_V8.md N3 -- the model-vs-model comparison view, plus the
// per-member fairness re-check. Storage is zero beyond runs.comparison_id:
// every measurement is an ordinary `results` row on its own model_id.

import type { FastifyInstance, FastifyBaseLogger } from "fastify";
import { repo } from "../db/repo.js";
import { resolveAuthUser } from "../auth-middleware.js";
import type { Run, RunConfig } from "../../../shared/types.js";
import {
  checkComparisonFairness,
  gridSignature,
  paretoFrontier,
  type ComparisonFairnessFacts,
  type ComparisonMemberRow,
  type FairnessViolation,
} from "../../../shared/comparison.js";

export function factsForRun(run: Run): ComparisonFairnessFacts {
  const config = run.config as RunConfig;
  return {
    worker_id: run.worker_id ?? null,
    llama_cpp_build: run.llama_cpp_build ?? null,
    llama_cpp_backend: run.llama_cpp_backend ?? null,
    backend_device_name: run.backend_device_name ?? null,
    repeats: config?.sweep?.repeats ?? null,
    // Only known once rows land; the per-member re-check below fills it in.
    method_version: null,
    grid_signature: gridSignature(config?.sweep as unknown as Record<string, unknown>),
  };
}

// Re-checked PER MEMBER, not only at trigger: a build swapped mid-group is
// exactly the silent confound this exists to catch, and it can only be seen
// once the member has actually run.
export function recheckComparisonMember(
  comparisonId: string,
  run: Run,
  log?: FastifyBaseLogger
): FairnessViolation[] {
  const members = repo.listComparisonMembers(comparisonId);
  const reference = members.find((m) => m.id !== run.id);
  if (!reference) return [];
  const referenceFacts = { ...factsForRun(reference), method_version: methodVersionOf(reference.id) };
  const candidateFacts = { ...factsForRun(run), method_version: methodVersionOf(run.id) };
  const violations = checkComparisonFairness(referenceFacts, candidateFacts);
  for (const violation of violations) {
    log?.warn(
      {
        comparison_member_failed: true,
        member_run_id: run.id,
        reason: violation.field,
        expected: violation.expected,
        found: violation.found,
      },
      "comparison_member_failed"
    );
  }
  return violations;
}

function methodVersionOf(runId: string): number | null {
  const rows = repo.getResultsForRun(runId);
  const versions = new Set(rows.map((r) => r.method_version ?? null).filter((v): v is number => v != null));
  return versions.size === 1 ? [...versions][0] : null;
}

export async function comparisonRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Params: { id: string } }>("/api/comparisons/:id", async (request, reply) => {
    const authed = resolveAuthUser(request);
    const userId = authed?.user.id;
    const members = repo.listComparisonMembers(request.params.id, userId);
    if (members.length === 0) return reply.code(404).send({ error: "comparison not found" });

    const reference = members[0];
    const rows: ComparisonMemberRow[] = members.map((run) => {
      const violations =
        run.id === reference.id
          ? []
          : checkComparisonFairness(
              { ...factsForRun(reference), method_version: methodVersionOf(reference.id) },
              { ...factsForRun(run), method_version: methodVersionOf(run.id) }
            );
      const results = repo.getResultsForRun(run.id);
      // Per-model BEST config row: the fastest tg, with its own pp partner.
      const tgRows = results.filter((r) => r.test_type === "tg");
      const best = tgRows.sort((a, b) => b.avg_tps - a.avg_tps)[0];
      const pp = best ? results.find((r) => r.test_type === "pp" && r.idx === best.idx) : undefined;
      const model = repo.getModel(run.model_id);
      const quality = repo.qualityRepo.listForModel(run.model_id)[0];
      const verified = run.worker_id
        ? repo.limitsRepo.listForModelAndWorker(run.model_id, run.worker_id)[0]
        : undefined;
      return {
        run_id: run.id,
        model_id: run.model_id,
        model_filename: model?.filename ?? run.model_id,
        // Parsed from GGUF metadata, never from the filename.
        quant_label: model?.metadata.quant ?? null,
        // models.id IS the file's sha256, so identical labels on different
        // files are already distinct rows here: labels lie, hashes don't.
        file_sha256: run.model_id,
        status: violations.length > 0 ? "drifted" : run.status,
        pp: pp?.avg_tps ?? null,
        tg: best?.avg_tps ?? null,
        vram_peak_mib: best?.vram_peak_mib ?? null,
        ram_peak_mib: best?.ram_peak_mib ?? null,
        // N4's column, when it exists. Never a score, never ranked on.
        ppl: quality?.ppl ?? null,
        kld_vs_baseline: quality?.kld_vs_baseline ?? null,
        dataset_hash: quality?.dataset_hash ?? null,
        verified_ctx_tokens: verified?.verified_ctx_tokens ?? null,
        violations,
      };
    });

    for (const row of rows) {
      for (const violation of row.violations) {
        request.log.warn(
          {
            comparison_member_failed: true,
            member_run_id: row.run_id,
            reason: violation.field,
          },
          "comparison_member_failed"
        );
      }
    }

    return {
      comparison_id: request.params.id,
      members: rows,
      pareto: paretoFrontier(rows),
      // Aborting the group keeps completed members comparable: the ones that
      // finished cleanly are still shown, the drifted ones are marked.
      drifted_members: rows.filter((r) => r.violations.length > 0).map((r) => r.run_id),
      quality_disclaimer:
        "Perplexity and KLD are a labeled synthetic proxy against a pinned corpus — clearly not your workload, and never inside any ranking.",
    };
  });
}
