import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// A worker install runs `npm install --prefix worker` against
// worker/package.json alone -- the root manifest is never installed there.
// So any package imported by worker or shared code that is missing from
// worker/package.json works in a dev checkout (root node_modules) and then
// crashes on every real worker. These checks catch that before it ships.

const workerDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(workerDir, "..");

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      if (name !== "node_modules" && name !== "__fixtures__") out.push(...sourceFiles(p));
    } else if (name.endsWith(".ts") && !name.endsWith(".test.ts")) {
      out.push(p);
    }
  }
  return out;
}

// Anchored to statement starts so prose in comments/strings ("... from
// \"failed\"") isn't mistaken for an import.
const STATIC_IMPORT = /^\s*(?:import|export)\b[^;]*?\bfrom\s*["']([^"']+)["']/gm;
const SIDE_EFFECT_IMPORT = /^\s*import\s*["']([^"']+)["']/gm;
const DYNAMIC_IMPORT = /\bimport\(\s*["']([^"']+)["']\s*\)/g;

function packageName(spec: string): string | null {
  if (spec.startsWith(".") || spec.startsWith("/") || spec.startsWith("node:")) return null;
  const parts = spec.split("/");
  return spec.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
}

function importedPackages(): Map<string, string> {
  const found = new Map<string, string>();
  for (const file of [...sourceFiles(join(workerDir, "src")), ...sourceFiles(join(repoRoot, "shared"))]) {
    const text = readFileSync(file, "utf8");
    for (const re of [STATIC_IMPORT, SIDE_EFFECT_IMPORT, DYNAMIC_IMPORT]) {
      for (const m of text.matchAll(re)) {
        const pkg = packageName(m[1]);
        if (pkg && !found.has(pkg)) found.set(pkg, file);
      }
    }
  }
  return found;
}

const workerPkg = JSON.parse(readFileSync(join(workerDir, "package.json"), "utf8"));
const rootPkg = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8"));

describe("worker/package.json", () => {
  it("declares every package worker and shared code imports", () => {
    const declared = new Set(Object.keys(workerPkg.dependencies ?? {}));
    const missing = [...importedPackages()].filter(([pkg]) => !declared.has(pkg));
    expect(missing.map(([pkg, file]) => `${pkg} (imported by ${file})`)).toEqual([]);
  });

  it("declares nothing the worker doesn't import, except the tsx runner", () => {
    const imported = importedPackages();
    const unused = Object.keys(workerPkg.dependencies ?? {}).filter((d) => d !== "tsx" && !imported.has(d));
    expect(unused).toEqual([]);
  });

  it("uses the same version ranges as the root package.json", () => {
    const rootDeps = { ...rootPkg.dependencies, ...rootPkg.devDependencies };
    for (const [name, range] of Object.entries(workerPkg.dependencies ?? {})) {
      expect(rootDeps[name], name).toBe(range);
    }
  });

  it("keeps the same engines and module type as the root", () => {
    expect(workerPkg.engines).toEqual(rootPkg.engines);
    expect(workerPkg.type).toBe(rootPkg.type);
  });
});
