#!/usr/bin/env bash
# One-liner bootstrap for a macOS/Linux machine that doesn't have the repo
# yet -- no LlamaToaster checkout, no config.json, no llama.cpp, no models.
# You only supply a base folder: it becomes both the code checkout and (via
# setup-worker.sh) the home for the "llama" and "models" subfolders.
#
# Usage from a totally fresh machine (only bash + Node.js 22+ needed; git is
# used if present, otherwise falls back to a plain tarball download). --vps-url
# is required. Omit --dir and it'll ask -- shows `df -h` (models are often
# tens of GB each) and asks for a base folder and a name:
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --vps-url https://llamatoaster.com
#
# Pass --dir to skip the prompts (e.g. for unattended/scripted use):
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --vps-url https://llamatoaster.com --dir ~/LlamaToaster
#
# Safe to re-run: if --dir already has a checkout, the download step is
# skipped; setup-worker.sh underneath is itself idempotent (skips writing
# config.json if one already exists). So the same command works for first
# setup, picking up code updates, and every plain restart after. To
# re-approve a machine whose session was revoked, add --reconnect (needs an
# existing --dir checkout with a config.json already in it).
#
# All setup-worker.sh overrides are forwarded -- see that script's own
# header for what each one does: --worker-name --backend --vps-url
# --reconnect --force --allow-insecure-url. Plus one bootstrap-only option:
#   --branch <name>   git branch/ref to fetch, default "main"

set -euo pipefail

REPO_URL="https://github.com/noname9006/LlamaToaster.git"
REPO_OWNER_SLASH="noname9006/LlamaToaster"
DIR=""
BRANCH="main"
WORKER_NAME="Local"
BACKEND=""
VPS_URL=""
FORCE=0
RECONNECT=0
ALLOW_INSECURE_URL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --worker-name) WORKER_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --vps-url) VPS_URL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --reconnect) RECONNECT=1; shift ;;
    --allow-insecure-url) ALLOW_INSECURE_URL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Not required for --reconnect: it only clears the session fields in an
# EXISTING config.json, which already has its own vps_url saved.
if [ -z "$VPS_URL" ] && [ "$RECONNECT" -ne 1 ]; then
  echo "--vps-url is required, e.g. --vps-url https://llamatoaster.com" >&2
  exit 1
fi

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
# --ignore-scripts: skips better-sqlite3's install step, which always compiles
# from source via node-gyp (it has no prebuilt-binary fallback) and needs a
# real C++ toolchain (Xcode Command Line Tools on macOS, build-essential on
# Linux). The worker never imports better-sqlite3 (server-only), so there's
# nothing to build here.
if ! npm install --ignore-scripts; then
  echo "npm install failed." >&2
  exit 1
fi

SETUP_ARGS=(--dir "$DIR" --worker-name "$WORKER_NAME")
[ -n "$VPS_URL" ] && SETUP_ARGS+=(--vps-url "$VPS_URL")
[ -n "$BACKEND" ] && SETUP_ARGS+=(--backend "$BACKEND")
[ "$FORCE" -eq 1 ] && SETUP_ARGS+=(--force)
[ "$RECONNECT" -eq 1 ] && SETUP_ARGS+=(--reconnect)
[ "$ALLOW_INSECURE_URL" -eq 1 ] && SETUP_ARGS+=(--allow-insecure-url)

exec "$DIR/worker/setup-worker.sh" "${SETUP_ARGS[@]}"
