import type {
  AdminStats,
  AdminTestSummary,
  AdminTestFilters,
  AdminUserSummary,
  AppSettings,
  Model,
  Worker,
} from "./types";
import type { TestViewApi } from "../../client/src/api/testView";
// The main site's own query-string builders -- the console must send the same
// parameters the owner's browser would, so it reuses them rather than
// re-deriving them.
import { curveQuery, profilesQuery } from "../../client/src/api/client";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function extractErrorMessage(data: unknown, status: number): string {
  if (data && typeof data === "object" && "error" in data) {
    const err = (data as { error?: unknown }).error;
    if (typeof err === "string") return err;
  }
  return `request failed: ${status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (err) {
    throw new ApiError(err instanceof Error ? err.message : String(err), 0);
  }
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) throw new ApiError(extractErrorMessage(data, res.status), res.status);
  return data as T;
}

function filtersToQuery(filters: AdminTestFilters): string {
  const params = new URLSearchParams();
  if (filters.userId) params.set("userId", filters.userId);
  if (filters.workerId) params.set("workerId", filters.workerId);
  if (filters.backend) params.set("backend", filters.backend);
  if (filters.status) params.set("status", filters.status);
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

const seg = encodeURIComponent;

// What the shared TestDetail page reads, pointed at the console's read-only,
// cross-tenant mirrors (server/src/routes/admin.ts). Same payload shapes as the
// main site's own routes -- the server builds them with the same handlers.
//
// `ownerUserId` is the test's owner: the curve is keyed by MODEL, not by test,
// so without it the panel would fold every tenant's measurements of that model
// together instead of showing the owner's (see admin.ts's curveOwnerScope).
export function createAdminTestViewApi(ownerUserId: string | null): TestViewApi {
  return {
    getTest: (id) => request(`/api/admin/tests/${seg(id)}`),
    getBatchMembers: (id) => request(`/api/admin/tests/${seg(id)}/batch-members`),
    getProbeAttempts: (id) => request(`/api/admin/tests/${seg(id)}/probe-attempts`),
    getProfiles: (id, goals) => request(`/api/admin/tests/${seg(id)}/profiles${profilesQuery(goals)}`),
    getCurve: (modelId, opts) =>
      request(`/api/admin/models/${seg(modelId)}/curve${curveQuery(opts, ownerUserId ? { user: ownerUserId } : {})}`),
    getKnee: (id) => request(`/api/admin/tests/${seg(id)}/knee`),
    listWorkers: () => request<{ workers: Worker[] }>("/api/admin/workers").then((d) => d.workers),
    listModels: () => request<{ models: Model[] }>("/api/admin/models").then((d) => d.models),
    csvExportUrl: (id) => `/api/admin/results/export?format=csv&tests=${seg(id)}`,
    bundleExportUrl: (id, scope = "test") => `/api/admin/tests/${seg(id)}/export?scope=${scope}`,
    testLogUrl: (id) => `/api/admin/tests/${seg(id)}/log`,
  };
}

export const api = {
  getStats: (): Promise<AdminStats> => request("/api/admin/stats"),

  listTests: (filters: AdminTestFilters = {}): Promise<{ runs: AdminTestSummary[] }> =>
    request(`/api/admin/tests${filtersToQuery(filters)}`),

  // Owner / machine / model for one test -- the per-test page's header, and
  // (a 200 vs 403/404) its own sign-in / not-found check, since TestDetail
  // itself doesn't surface a failed load.
  getTestSummary: (id: string): Promise<AdminTestSummary> => request(`/api/admin/tests/${seg(id)}/summary`),

  listUsers: (): Promise<{ users: AdminUserSummary[] }> => request("/api/admin/users"),

  exportUrl: (format: "json" | "csv" | "md"): string => `/api/admin/results/export?format=${format}`,

  getSettings: (): Promise<AppSettings> => request("/api/admin/settings"),

  updateSettings: (patch: Partial<AppSettings>): Promise<AppSettings> =>
    request("/api/admin/settings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(patch),
    }),
};
