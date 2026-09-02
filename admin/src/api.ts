import type { AdminStats, AdminTestSummary, AdminTestFilters, AdminUserSummary, AppSettings } from "./types";

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

export const api = {
  getStats: (): Promise<AdminStats> => request("/api/admin/stats"),

  listTests: (filters: AdminTestFilters = {}): Promise<{ runs: AdminTestSummary[] }> =>
    request(`/api/admin/tests${filtersToQuery(filters)}`),

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
