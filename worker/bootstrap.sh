#!/usr/bin/env bash
# One-liner bootstrap for a macOS/Linux machine that doesn't have the repo
# yet -- no LlamaToaster checkout, no config.json, no llama.cpp, no models.
# You only supply a base folder: it becomes both the code checkout and (via
# setup-worker.sh) the home for the "llama" and "models" subfolders.
#
# Usage from a totally fresh machine (only bash + Node.js 22+ needed; git is
# used if present, otherwise falls back to a plain tarball download). Omit
# --dir and it'll ask -- shows `df -h` (models are often tens of GB each) and
# asks for a base folder and a name:
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash
#
# Pass --dir to skip the prompts (e.g. for unattended/scripted use):
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --dir ~/LlamaToaster
#
# Safe to re-run: if --dir already has a checkout, the download step is
# skipped; setup-worker.sh underneath is itself idempotent (skips writing
# config.json if one already exists). So the same command works for first
# setup, picking up code updates, and every plain restart after.
#
# All setup-worker.sh overrides are forwarded -- see that script's own
# header for what each one does: --worker-name --backend --vps-url
# --bind-host --port --force. Plus one bootstrap-only option:
#   --branch <name>   git branch/ref to fetch, default "main"

set -euo pipefail

REPO_URL="https://github.com/noname9006/LlamaToaster.git"
REPO_OWNER_SLASH="noname9006/LlamaToaster"
DIR=""
BRANCH="main"
WORKER_NAME="Local"
BACKEND=""
VPS_URL="http://100.122.1.111:4010"
BIND_HOST=""
PORT=8080
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --worker-name) WORKER_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --vps-url) VPS_URL="$2"; shift 2 ;;
    --bind-host) BIND_HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Same Git Bash/MSYS/Cygwin footgun setup-worker.sh guards against -- paths
# written on Windows under those shells get silently misinterpreted by the
# worker's native Windows Node process.
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "This looks like Git Bash/MSYS on Windows. Use worker\\bootstrap.ps1 in PowerShell instead." >&2
    exit 1
    ;;
esac

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found on PATH. Install Node.js 22+ (e.g. via nvm or your package manager) first, then re-run this script." >&2
  exit 1
fi

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

if [ -z "$DIR" ]; then
  DIR="$(select_install_dir)"
  echo "Using $DIR"
fi

if [ ! -f "$DIR/package.json" ]; then
  echo "$DIR has no LlamaToaster checkout yet -- downloading it (branch: $BRANCH)..."
  mkdir -p "$DIR"
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$DIR"
  else
    echo "git not found -- downloading a tarball of the repo instead."
    TMP_TAR="$(mktemp -t llamatoaster-XXXXXX).tar.gz"
    curl -fsSL "https://github.com/$REPO_OWNER_SLASH/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP_TAR"
    # GitHub's archive tarball has one top-level folder (e.g.
    # LlamaToaster-main/) wrapping everything -- strip it on extract.
    tar -xzf "$TMP_TAR" -C "$DIR" --strip-components=1
    rm -f "$TMP_TAR"
  fi
  echo "Downloaded to $DIR"
else
  echo "$DIR already has a LlamaToaster checkout -- skipping download."
fi

cd "$DIR"
echo "Installing dependencies (npm install)..."
if ! npm install; then
  echo "npm install failed. If this is a native-module build error for better-sqlite3, make sure build tools are installed (Xcode Command Line Tools on macOS, build-essential/python3 on Linux) or use a Node.js version with a published prebuilt binary (Node 22 LTS is a safe bet), then re-run this script." >&2
  exit 1
fi

SETUP_ARGS=(--dir "$DIR" --worker-name "$WORKER_NAME" --vps-url "$VPS_URL" --port "$PORT")
[ -n "$BACKEND" ] && SETUP_ARGS+=(--backend "$BACKEND")
[ -n "$BIND_HOST" ] && SETUP_ARGS+=(--bind-host "$BIND_HOST")
[ "$FORCE" -eq 1 ] && SETUP_ARGS+=(--force)

exec "$DIR/worker/setup-worker.sh" "${SETUP_ARGS[@]}"
