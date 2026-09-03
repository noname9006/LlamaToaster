import { defineConfig, configDefaults } from "vitest/config";

export default defineConfig({
  test: {
    exclude: [
      ...configDefaults.exclude,
      // Claude Code checks worktrees out under .claude/worktrees/ when more
      // than one session is working in this folder. Those are full copies of
      // the tree, so without this every test file in the repo runs twice (or
      // more) and unrelated in-progress work in another worktree shows up as
      // a failure here.
      "**/.claude/**",
      "client/dist/**",
      "admin/dist/**",
    ],
  },
});
