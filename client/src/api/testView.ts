// The seam that lets ONE test-detail page render in two apps.
//
// TestDetail (and the panels it embeds) is shared, source-for-source, with the
// supervise console (admin/), so an operator sees a user's test exactly as its
// owner does -- there is no second copy to drift. The two apps differ in only
// three things, and this context carries all of them:
//
//   * where the data comes from -- the main site reads its own tenant-scoped
//     /api/tests/... routes; the console reads read-only /api/admin/... mirrors
//     of them (server/src/routes/admin.ts) that serve the same payloads for any
//     tenant;
//   * whether the viewer may change anything -- the console only observes, so
//     every control that pauses, stops, triggers or deletes is hidden;
//   * where a sibling test's page lives (`/tests/:id` in both today, but the
//     admin app owns its own router).
//
// The default value is the main site's, so a component rendered with no
// provider (the main app, every existing test) behaves exactly as before.

import { createContext, useContext } from "react";
import { api } from "./client";

type Api = typeof api;

/**
 * Exactly the reads a test's page makes -- deliberately narrow, and typed off
 * the main client's own signatures so the two implementations can't drift.
 * Writes (pause/stop, trigger, delete a verified limit) are NOT here: a
 * read-only viewer has no way to reach them, and the main site keeps importing
 * them from `api` directly.
 */
export interface TestViewApi {
  getTest: Api["getTest"];
  getBatchMembers: Api["getBatchMembers"];
  getProbeAttempts: Api["getProbeAttempts"];
  getProfiles: Api["getProfiles"];
  getCurve: Api["getCurve"];
  getKnee: Api["getKnee"];
  /** Resolves the machine's hardware line by run.worker_id. */
  listWorkers: Api["listWorkers"];
  /** Resolves an MTP draft model's filename by id. */
  listModels: Api["listModels"];
  /** This one test's results as CSV. */
  csvExportUrl: (testId: string) => string;
  bundleExportUrl: Api["bundleExportUrl"];
  testLogUrl: Api["testLogUrl"];
}

export interface TestViewContextValue {
  api: TestViewApi;
  /** Observing, not operating: hides every control that would change a test. */
  readOnly: boolean;
  /** Where another test's page lives in the running app. */
  testPath: (testId: string) => string;
}

// Thin wrappers rather than `api.getTest` captured by reference: they resolve
// the method at call time, so a module mock installed after this file loads
// (the component tests do exactly that) is still what gets called.
export const mainTestViewApi: TestViewApi = {
  getTest: (id) => api.getTest(id),
  getBatchMembers: (id) => api.getBatchMembers(id),
  getProbeAttempts: (id) => api.getProbeAttempts(id),
  getProfiles: (id, goals) => api.getProfiles(id, goals),
  getCurve: (modelId, opts) => api.getCurve(modelId, opts),
  getKnee: (id) => api.getKnee(id),
  listWorkers: () => api.listWorkers(),
  listModels: () => api.listModels(),
  csvExportUrl: (id) => api.exportUrl("csv", [id]),
  bundleExportUrl: (id, scope) => api.bundleExportUrl(id, scope),
  testLogUrl: (id) => api.testLogUrl(id),
};

const TestViewContext = createContext<TestViewContextValue>({
  api: mainTestViewApi,
  readOnly: false,
  testPath: (id) => `/tests/${id}`,
});

export const TestViewProvider = TestViewContext.Provider;

export function useTestView(): TestViewContextValue {
  return useContext(TestViewContext);
}
