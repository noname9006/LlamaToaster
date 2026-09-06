import { defineConfig, configDefaults } from "vitest/config";

const exclude = [
  ...configDefaults.exclude,
  // Claude Code checks worktrees out under .claude/worktrees/ when more
  // than one session is working in this folder. Those are full copies of
  // the tree, so without this every test file in the repo runs twice (or
  // more) and unrelated in-progress work in another worktree shows up as
  // a failure here.
  "**/.claude/**",
  "client/dist/**",
  "admin/dist/**",
];

// Two projects rather than one suite, because the two halves of this repo
// need different environments: server/, worker/ and shared/ are Node code and
// must keep running under the Node environment (several of them branch on
// `typeof window === "undefined"` or touch the filesystem), while client/
// needs a DOM. Splitting them also keeps the 65 existing Node test files free
// of jsdom's startup cost.
//
// `npm test` at the repo root runs both. To run one: `npm test -- --project node`
// or `--project client`.
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "node",
          environment: "node",
          // Explicit roots rather than a bare default, so a stray test file
          // added outside these three packages fails loudly (unmatched) rather
          // than silently inheriting the wrong environment.
          include: ["server/**/*.test.ts", "worker/**/*.test.ts", "shared/**/*.test.ts"],
          exclude,
        },
      },
      // Loads client/vitest.config.ts, whose root is client/ so React, jsdom
      // and @testing-library resolve from client/node_modules.
      "./client",
    ],
  },
});
