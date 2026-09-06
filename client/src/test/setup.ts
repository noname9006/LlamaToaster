import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach } from "vitest";

// Unmounts anything a test rendered. Without this, a component whose effect
// schedules a timer (every polling hook in this app does) keeps running after
// its test finishes and reports state into a torn-down test, which surfaces
// as an unrelated later test failing.
afterEach(() => {
  cleanup();
});

// jsdom gives every test file one shared Storage instance for the whole file.
// The modules under test here (api/aiStorage.ts, utils.ts's
// migrateLegacyStorageKeys) read and write real keys, so without an explicit
// reset a test would inherit whatever the previous one left behind and pass
// or fail depending on declaration order.
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});
