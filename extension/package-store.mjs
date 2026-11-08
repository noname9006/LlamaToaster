// Packs the extension for upload to the Chrome Web Store (also accepted by
// Microsoft Edge Add-ons).
//
//   npm run package     ->  extension/store/build/llamatoaster-<version>.zip
//
// What differs from a plain `dist/` zip, and why:
//   - manifest.json loses "key": the store assigns the ID itself (the dev key
//     only pins the ID of the unpacked build, see store/LISTING.md, "Extension ID").
//   - no *.map files and no "//# sourceMappingURL" comments (dead weight in the
//     package, and they point at files that are not shipped).
//   - entries use forward slashes and the manifest sits at the archive root.
//     (PowerShell's Compress-Archive writes backslashes; the store rejects or
//     mis-extracts such packages.)
//   - every path the manifest and the HTML pages refer to is checked to exist.
//
// The zip is written here with node:zlib so the output is reproducible and the
// extension keeps its dev dependencies at "type definitions only".

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { crc32, deflateRawSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
const buildDir = join(here, "store", "build");

// 1. Fresh build.
execFileSync(process.execPath, [join(here, "build.mjs")], { stdio: "inherit", cwd: here });

// 2. Collect files, transforming as we go.
function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]));
}
const manifest = JSON.parse(readFileSync(join(dist, "manifest.json"), "utf8"));
delete manifest.key;
const files = new Map(); // archive path -> Buffer
for (const abs of walk(dist)) {
  const name = relative(dist, abs).split(sep).join("/");
  if (name.endsWith(".map")) continue;
  let data = readFileSync(abs);
  if (name === "manifest.json") data = Buffer.from(JSON.stringify(manifest, null, 2) + "\n");
  else if (name.endsWith(".js")) data = Buffer.from(data.toString("utf8").replace(/\n?\/\/# sourceMappingURL=.*\s*$/, "\n"));
  files.set(name, data);
}

// 3. Sanity checks: everything referenced is present, nothing secret is.
const need = new Set();
const add = (p) => p && need.add(p);
Object.values(manifest.icons ?? {}).forEach(add);
Object.values(manifest.action?.default_icon ?? {}).forEach(add);
add(manifest.action?.default_popup);
add(manifest.side_panel?.default_path);
add(manifest.background?.service_worker);
for (const cs of manifest.content_scripts ?? []) [...(cs.js ?? []), ...(cs.css ?? [])].forEach(add);
for (const page of ["popup.html", "panel.html"]) {
  const html = files.get(page)?.toString("utf8") ?? "";
  for (const m of html.matchAll(/(?:src|href)="([^"#?]+)"/g)) if (!/^(https?:|data:)/.test(m[1])) add(m[1]);
}
const missing = [...need].filter((p) => !files.has(p));
if (missing.length) throw new Error(`Package is missing files the extension refers to: ${missing.join(", ")}`);
if ("key" in JSON.parse(files.get("manifest.json").toString())) throw new Error('manifest.json still has "key"');
const bad = [...files.keys()].filter((n) => /\.(pem|crx|map)$/.test(n) || n.includes("\\"));
if (bad.length) throw new Error(`Files that must not ship: ${bad.join(", ")}`);
for (const n of ["background.js", "content.js", "popup.js"]) if (/sourceMappingURL/.test(files.get(n).toString())) throw new Error(`${n} still has a sourceMappingURL`);

// 4. Write the zip (store the manifest first, then the rest sorted; fixed timestamp).
const DOS_TIME = 0; // 00:00:00
const DOS_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1; // reproducible: 2026-01-01
const order = ["manifest.json", ...[...files.keys()].filter((n) => n !== "manifest.json").sort()];
const locals = [];
const centrals = [];
let offset = 0;
for (const name of order) {
  const raw = files.get(name);
  const deflated = deflateRawSync(raw, { level: 9 });
  const useDeflate = deflated.length < raw.length;
  const body = useDeflate ? deflated : raw;
  const method = useDeflate ? 8 : 0;
  const nameBuf = Buffer.from(name, "utf8");
  const crc = crc32(raw) >>> 0;
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0);
  local.writeUInt16LE(20, 4); // version needed
  local.writeUInt16LE(0x0800, 6); // UTF-8 names
  local.writeUInt16LE(method, 8);
  local.writeUInt16LE(DOS_TIME, 10);
  local.writeUInt16LE(DOS_DATE, 12);
  local.writeUInt32LE(crc, 14);
  local.writeUInt32LE(body.length, 18);
  local.writeUInt32LE(raw.length, 22);
  local.writeUInt16LE(nameBuf.length, 26);
  local.writeUInt16LE(0, 28);
  locals.push(local, nameBuf, body);
  const central = Buffer.alloc(46);
  central.writeUInt32LE(0x02014b50, 0);
  central.writeUInt16LE(20, 4); // version made by
  central.writeUInt16LE(20, 6); // version needed
  central.writeUInt16LE(0x0800, 8);
  central.writeUInt16LE(method, 10);
  central.writeUInt16LE(DOS_TIME, 12);
  central.writeUInt16LE(DOS_DATE, 14);
  central.writeUInt32LE(crc, 16);
  central.writeUInt32LE(body.length, 20);
  central.writeUInt32LE(raw.length, 24);
  central.writeUInt16LE(nameBuf.length, 28);
  central.writeUInt32LE(offset, 42);
  centrals.push(central, nameBuf);
  offset += local.length + nameBuf.length + body.length;
}
const centralSize = centrals.reduce((n, b) => n + b.length, 0);
const end = Buffer.alloc(22);
end.writeUInt32LE(0x06054b50, 0);
end.writeUInt16LE(order.length, 8);
end.writeUInt16LE(order.length, 10);
end.writeUInt32LE(centralSize, 12);
end.writeUInt32LE(offset, 16);

rmSync(buildDir, { recursive: true, force: true });
mkdirSync(buildDir, { recursive: true });
const zipPath = join(buildDir, `llamatoaster-${manifest.version}.zip`);
writeFileSync(zipPath, Buffer.concat([...locals, ...centrals, end]));

const total = [...files.values()].reduce((n, b) => n + b.length, 0);
console.log(`\n${relative(join(here, ".."), zipPath)}  ${(statSync(zipPath).size / 1024).toFixed(0)} KB zipped, ${(total / 1024).toFixed(0)} KB unpacked, ${order.length} files, v${manifest.version}`);
for (const n of order) console.log("  " + n);
