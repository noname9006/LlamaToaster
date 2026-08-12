#!/usr/bin/env bash
# One-shot bootstrap for a worker that has nothing downloaded yet: no
# config.json, no llama.cpp binary, no models. You only supply one base
# folder -- llama.cpp builds and any future GGUF downloads get fetched
# automatically later (llama.cpp on first triggered run, models whenever you
# download one from the Models page) into <dir>/llama and <dir>/models. This
# script itself never downloads llama.cpp or a model.
#
# Usage (from anywhere), defaults to ~/LlamaToaster if --dir is omitted:
#   ./worker/setup-worker.sh
#   ./worker/setup-worker.sh --dir ~/LlamaToaster
#
# No IP needed either: bind_host auto-detects via `tailscale ip -4` unless
# you pass --bind-host explicitly.
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
#   --vps-url <url>           default the orchestrator's known Tailscale address
#   --port <n>                default 8080
#   --force                   overwrite an existing worker/config.json
#
# Safe by default: refuses to touch an existing worker/config.json unless
# --force is passed (no git in this repo checkout, so an overwritten
# config.json has no recovery path).

set -euo pipefail

DIR="$HOME/LlamaToaster"
WORKER_NAME="Local"
BACKEND=""
VPS_URL="http://100.122.1.111:4010"
BIND_HOST=""
PORT=8080
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --worker-name) WORKER_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --vps-url) VPS_URL="$2"; shift 2 ;;
    --bind-host) BIND_HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

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

if [ -f "$CONFIG_PATH" ] && [ "$FORCE" -ne 1 ]; then
  echo "worker/config.json already exists -- not touching it. Current contents:"
  cat "$CONFIG_PATH"
  echo
  echo "Re-run with --force to overwrite it."
  exit 1
fi

if [ -z "$BIND_HOST" ]; then
  echo "No --bind-host given, trying 'tailscale ip -4'..."
  BIND_HOST="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [ -z "$BIND_HOST" ]; then
    echo "Could not auto-detect a Tailscale IP. Pass --bind-host <this machine's tailnet IP> explicitly." >&2
    exit 1
  fi
  echo "Using detected Tailscale IP: $BIND_HOST"
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
  "vps_url": "$VPS_URL",
  "bind_host": "$BIND_HOST",
  "port": $PORT,
  "raw_json_dir": "$RAW_DIR",
  "llama_cpp_builds_dir": "$LLAMA_DIR"
}
EOF
echo "Wrote $CONFIG_PATH"

cd "$REPO_ROOT"
if [ ! -e node_modules/.bin/tsx ]; then
  echo "Installing dependencies (npm install)..."
  npm install
fi

BACKEND_LABEL="${BACKEND:-auto-detected from hardware}"
echo "Starting worker '$WORKER_NAME' ($BACKEND_LABEL) -- llama.cpp builds will install to $LLAMA_DIR on first run, models download to $MODELS_DIR from the Models page."
npm run worker
