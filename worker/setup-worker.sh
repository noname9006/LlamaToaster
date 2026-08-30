#!/usr/bin/env bash
# One-shot bootstrap for a worker that has nothing downloaded yet: no
# config.json, no llama.cpp binary, no models. You only supply one base
# folder -- llama.cpp builds and any future GGUF downloads get fetched
# automatically later (llama.cpp on first triggered run, models whenever you
# download one from the Models page) into <dir>/llama and <dir>/models. This
# script itself never downloads llama.cpp or a model.
#
# Usage (from anywhere). Safe to run every time, first setup or a plain
# restart -- if config.json already exists, setup (and the --dir prompt) is
# skipped entirely and this just starts the worker:
#   ./worker/setup-worker.sh --url https://llamatoaster.com
#
# On first run (no config.json yet), omitting --dir shows `df -h` (models
# are often tens of GB each) and asks for a base folder and a name. Pass
# --dir to skip that prompt, e.g. for unattended/scripted use:
#   ./worker/setup-worker.sh --url https://llamatoaster.com --dir ~/LlamaToaster
#
# No IP or port to configure at all -- the worker has no inbound listener
# (it long-polls the server, MULTIUSER_PLAN.md §1). The FIRST time it runs
# with no session configured yet, it prints a short code and a URL
# (https://<url>/device) to approve it from your account -- do that once,
# in a browser, and it's connected. The saved session survives ordinary
# restarts and never asks again on its own -- if it needs to reconnect (its
# session was revoked from Settings, or its refresh token expired), pass
# --reconnect (see below) rather than re-running plain setup, which is a
# total no-op once config.json already exists.
#
# Other optional overrides:
#   --worker-name <name>      default "Local"
#   --backend <value>         any backend llama.cpp's releases use (cpu,
#                             vulkan, cuda, rocm, sycl, opencl-adreno,
#                             openvino, or anything a future release adds) --
#                             not validated against a fixed list. Default:
#                             left unset, so the worker auto-detects it live
#                             from this machine's actual GPU on startup. Pass
#                             this only to pin/override that, e.g. for
#                             hardware the auto-detect heuristic doesn't
#                             specifically know about.
#   --reconnect                clear this machine's saved session (keeping
#                             machine_id and every other setting) so it goes
#                             through device-flow approval again on the same
#                             machine identity -- the server recognizes it as
#                             the SAME machine (same history/display_name),
#                             unlike --force below. Requires config.json to
#                             already exist.
#   --force                   overwrite an existing worker/config.json from
#                             scratch, INCLUDING machine_id -- the server
#                             will treat this as a brand-new machine with no
#                             history. Use --reconnect instead unless that's
#                             actually what you want.
#   --allow-insecure-url       allow a plain http:// --url other than
#                             localhost/127.0.0.1 (see below). Only for a
#                             deployment you've secured another way -- e.g.
#                             this app's own tailnet-only mode, where
#                             Tailscale's own WireGuard tunnel is the
#                             encryption, not TLS.
#
# Safe by default: refuses to overwrite an existing worker/config.json unless
# --force (or --reconnect, which only ever touches the session fields) is
# passed -- no git in this repo checkout, so an overwritten config.json has
# no recovery path. Idempotent otherwise: if config.json already exists and
# neither flag is given, setup is skipped and this just (re)starts the
# worker with it as-is -- so the exact same command works both for
# first-time setup and every restart after.

set -euo pipefail

DIR=""
WORKER_NAME="Local"
BACKEND=""
URL=""
FORCE=0
RECONNECT=0
ALLOW_INSECURE_URL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --worker-name) WORKER_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --reconnect) RECONNECT=1; shift ;;
    --allow-insecure-url) ALLOW_INSECURE_URL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ "$FORCE" -eq 1 ] && [ "$RECONNECT" -eq 1 ]; then
  echo "--force and --reconnect are mutually exclusive (--force wipes machine_id, --reconnect preserves it)." >&2
  exit 1
fi

# This script writes plain POSIX paths (e.g. /f/LlamaToaster) straight into
# config.json as literal strings. That's correct on real macOS/Linux, but on
# Windows the worker itself runs as a native Node process, which does NOT
# understand Git-Bash/MSYS drive syntax -- it resolves a leading "/f/..." as
# "root of the current drive, folder named f", silently writing everything
# to e.g. F:\f\LlamaToaster instead of F:\LlamaToaster. Refuse to run under
# Git Bash/MSYS/Cygwin rather than risk that again -- use setup-worker.ps1
# on Windows instead.
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "This looks like Git Bash/MSYS on Windows. Paths written by this script" >&2
    echo "are POSIX-style and get silently misinterpreted by the worker's" >&2
    echo "Windows Node process (e.g. /f/LlamaToaster becomes F:\\f\\LlamaToaster)." >&2
    echo "Use worker\\setup-worker.ps1 in PowerShell on Windows instead." >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$REPO_ROOT/worker/config.json"

select_install_dir() {
  if { exec 3</dev/tty; } 2>/dev/null; then
    exec 3<&-
  else
    echo "No terminal available to prompt for an install folder (running non-interactively?). Pass --dir explicitly, e.g. --dir ~/LlamaToaster." >&2
    exit 1
  fi

  echo "" >&2
  echo "Where should LlamaToaster live?" >&2
  echo "Downloaded models are often tens of GB each -- pick a volume with room to spare." >&2
  echo "" >&2
  df -h >&2
  echo "" >&2
  local base_dir folder_name
  read -r -p "Base folder to install into [default: $HOME]: " base_dir < /dev/tty
  base_dir="${base_dir:-$HOME}"
  read -r -p "Folder name to create inside it [default: LlamaToaster]: " folder_name < /dev/tty
  folder_name="${folder_name:-LlamaToaster}"
  echo "" >&2
  echo "${base_dir%/}/$folder_name"
}

if [ "$RECONNECT" -eq 1 ]; then
  if [ ! -f "$CONFIG_PATH" ]; then
    echo "--reconnect needs an existing $CONFIG_PATH -- nothing to reconnect. Run without --reconnect first." >&2
    exit 1
  fi
  # node, not sed/jq -- Node is already a hard requirement for this whole
  # script, unlike jq, and this needs real JSON parsing (session_token's
  # value can contain characters a naive sed delete-the-line approach would
  # mishandle). Every other field -- machine_id above all -- passes through
  # untouched; only these two are ever removed.
  node -e '
    const fs = require("fs");
    const p = process.argv[1];
    const cfg = JSON.parse(fs.readFileSync(p, "utf8"));
    delete cfg.session_token;
    delete cfg.refresh_token;
    fs.writeFileSync(p, JSON.stringify(cfg, null, 2) + "\n");
  ' "$CONFIG_PATH"
  echo "Cleared the saved session in $CONFIG_PATH -- this machine will go through device-flow approval again on startup, same identity."
fi

SKIPPED_SETUP=0
if [ -f "$CONFIG_PATH" ] && [ "$FORCE" -ne 1 ] && [ "$RECONNECT" -ne 1 ]; then
  SKIPPED_SETUP=1
  echo "worker/config.json already exists -- skipping setup, starting the worker with it as-is."
  echo "(Re-run with --force to redo setup from scratch, or --reconnect to just re-approve this machine.)"
elif [ "$RECONNECT" -eq 1 ]; then
  SKIPPED_SETUP=1
else
  if [ -z "$URL" ]; then
    echo "--url is required on first setup, e.g. --url https://llamatoaster.com" >&2
    exit 1
  fi
  # This ends up in config.json and gets a real 90-day bearer token sent to
  # it on every heartbeat/queue-poll (MULTIUSER_PLAN.md §3.5's own "the
  # installer refuses http://") -- plaintext HTTP would put that credential
  # on the wire in the clear. localhost/127.0.0.1/::1 are exempted, same
  # convention browsers use for "secure context" -- needed for local
  # dev/testing against a server that isn't fronted by TLS yet. Anything else
  # over http:// needs --allow-insecure-url (e.g. this app's own tailnet-only
  # mode, where Tailscale's own tunnel is the encryption, not TLS).
  case "$URL" in
    https://*) ;;
    http://localhost*|http://127.0.0.1*|http://\[::1\]*) ;;
    http://*)
      if [ "$ALLOW_INSECURE_URL" -ne 1 ]; then
        echo "--url is http:// ($URL) -- that sends a real bearer credential in the clear." >&2
        echo "Use https://, or pass --allow-insecure-url if this connection is secured another way (e.g. a Tailscale-only deployment)." >&2
        exit 1
      fi
      ;;
    *)
      echo "--url must start with https:// or http://, got: $URL" >&2
      exit 1
      ;;
  esac

  if [ -z "$DIR" ]; then
    DIR="$(select_install_dir)"
    echo "Using $DIR"
  fi

  LLAMA_DIR="$DIR/llama"
  MODELS_DIR="$DIR/models"
  RAW_DIR="$MODELS_DIR/raw"
  mkdir -p "$LLAMA_DIR" "$MODELS_DIR" "$RAW_DIR"

  # backend is left out entirely when not given (not even a null/empty value)
  # so the worker's own startup logic auto-detects it from live hardware.
  cat > "$CONFIG_PATH" <<EOF
{
  "worker_name": "$WORKER_NAME",
$( [ -n "$BACKEND" ] && printf '  "backend": "%s",\n' "$BACKEND" )
  "llama_cpp_build": "none",
  "llama_bench_path": "$LLAMA_DIR/llama-bench",
  "model_dir": "$MODELS_DIR",
  "url": "$URL",
  "raw_json_dir": "$RAW_DIR",
  "llama_cpp_builds_dir": "$LLAMA_DIR"
}
EOF
  echo "Wrote $CONFIG_PATH"
fi

cd "$REPO_ROOT"
if [ ! -e node_modules/.bin/tsx ]; then
  echo "Installing dependencies (npm install)..."
  # --ignore-scripts: skips better-sqlite3's install step, which always
  # compiles from source via node-gyp (it has no prebuilt-binary fallback)
  # and needs a real C++ toolchain (Xcode Command Line Tools on macOS,
  # build-essential on Linux). The worker never imports better-sqlite3
  # (server-only), so there's nothing to build here.
  npm install --ignore-scripts
fi

if [ "$RECONNECT" -eq 1 ]; then
  echo "Starting worker -- watch the console for a short code and a link to re-approve this machine."
elif [ "$SKIPPED_SETUP" -eq 1 ]; then
  echo "Starting worker with existing $CONFIG_PATH"
else
  BACKEND_LABEL="${BACKEND:-auto-detected from hardware}"
  echo "Starting worker '$WORKER_NAME' ($BACKEND_LABEL) -- llama.cpp builds will install to $LLAMA_DIR on first run, models download to $MODELS_DIR from the Models page."
  echo "First run only: watch the console for a short code and a link to approve this machine."
fi
npm run worker
