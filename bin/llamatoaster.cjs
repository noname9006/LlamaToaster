#!/usr/bin/env node
'use strict';

const { spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

const repoRoot = path.join(__dirname, '..');

// ── tsx resolution ──────────────────────────────────────────────────────────
// Prefer the local node_modules/.bin symlink (works for a cloned checkout);
// fall back to a globally-installed tsx (works after `npm install -g .`).
function resolveTsc() {
  const isWindows = process.platform === 'win32';
  const ext = isWindows ? '.cmd' : '';
  const localBin = path.join(repoRoot, 'node_modules', '.bin', 'tsx' + ext);
  if (fs.existsSync(localBin)) {
    // .cmd files must be invoked through cmd.exe on Windows. Use an absolute
    // path — cmd.exe /c with a relative path doesn't resolve correctly.
    if (isWindows) {
      const absBin = path.resolve(repoRoot, 'node_modules', '.bin', 'tsx.cmd');
      return { cmd: 'cmd.exe', args: ['/c', absBin] };
    }
    return { cmd: localBin, args: [] };
  }

  // Quick probe: is tsx on PATH?
  const probe = spawnSync('tsx', ['--version'], { stdio: 'pipe' });
  if (probe.status === 0) return { cmd: 'tsx', args: [] };

  return null;
}

function ensureTsc() {
  const tsx = resolveTsc();
  if (!tsx) {
    console.error(
      'tsx not found — run "npm install" first (it installs tsx as a dependency).'
    );
    process.exit(1);
  }
  return tsx;
}

// ── commands ────────────────────────────────────────────────────────────────

function runServer() {
  const { cmd, args } = ensureTsc();
  const result = spawnSync(cmd, [...args, 'server/src/index.ts'], {
    cwd: repoRoot,
    stdio: 'inherit',
  });
  process.exit(result.status ?? 1);
}

function runWorker() {
  const { cmd, args } = ensureTsc();
  const result = spawnSync(cmd, [...args, 'worker/src/index.ts'], {
    cwd: repoRoot,
    stdio: 'inherit',
  });
  process.exit(result.status ?? 1);
}

function runSetup() {
  const isWindows = process.platform === 'win32';
  const scriptName = isWindows ? 'setup-worker.ps1' : 'setup-worker.sh';
  const scriptPath = path.join(repoRoot, 'worker', scriptName);

  if (!fs.existsSync(scriptPath)) {
    console.error(`Setup script not found: ${scriptPath}`);
    process.exit(1);
  }

  // Forward every arg after the "setup" subcommand itself.
  const scriptArgs = process.argv.slice(3);

  if (isWindows) {
    const result = spawnSync(
      'powershell.exe',
      ['-ExecutionPolicy', 'Bypass', '-File', scriptPath, ...scriptArgs],
      { stdio: 'inherit' }
    );
    process.exit(result.status ?? 1);
  } else {
    const result = spawnSync('bash', [scriptPath, ...scriptArgs], {
      stdio: 'inherit',
    });
    process.exit(result.status ?? 1);
  }
}

function showHelp() {
  console.log(`Usage: llamatoaster [command] [options]

Commands:
  server    Start the LlamaToaster server (dashboard + API)   [default]
  worker    Start a worker process (needs an existing config.json)
  setup     Run the platform-appropriate setup-worker script

Options:
  -h, --help   Show this help message

Examples:
  llamatoaster                  # Start the server
  llamatoaster server           # Same as above
  llamatoaster worker           # Start a worker
  llamatoaster setup --vps-url https://llamatoaster.com
`);
}

// ── dispatch ────────────────────────────────────────────────────────────────

const command = process.argv[2] ?? 'server';

switch (command) {
  case 'server':
  case undefined:
    runServer();
    break;
  case 'worker':
    runWorker();
    break;
  case 'setup':
    runSetup();
    break;
  case '-h':
  case '--help':
    showHelp();
    break;
  default:
    console.error(`Unknown command: ${command}`);
    showHelp();
    process.exit(1);
}