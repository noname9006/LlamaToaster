#!/usr/bin/env bash
# One-liner bootstrap for a macOS/Linux machine that doesn't have the repo
# yet -- no LlamaToaster checkout, no config.json, no llama.cpp, no models.
# You only supply a base folder: it becomes both the code checkout and (via
# setup-worker.sh) the home for the "llama" and "models" subfolders.
#
# Usage from a totally fresh machine (only bash + Node.js 22+ needed; git is
# used if present, otherwise falls back to a plain tarball download). --url
# is required. Omit --dir and it'll ask -- shows `df -h` (models are often
# tens of GB each) and asks for a base folder and a name:
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --url https://llamatoaster.com
#
# Pass --dir to skip the prompts (e.g. for unattended/scripted use):
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --url https://llamatoaster.com --dir ~/LlamaToaster
#
# Safe to re-run: if --dir already has a LlamaToaster install, it is updated
# IN PLACE -- via git fetch + reset --hard when it is a git checkout, or a
# fresh-tarball sync over just the same-named files otherwise. Everything
# user-created survives untouched: config.json, models, .db files, logs,
# node_modules, llama.cpp builds. setup-worker.sh underneath is itself
# idempotent (skips writing config.json if one already exists). So the same
# command works for first setup, picking up code updates, and every plain
# restart after. To re-approve a machine whose session was revoked, add
# --reconnect (needs an existing --dir install with a config.json in it).
#
# All setup-worker.sh overrides are forwarded -- see that script's own
# header for what each one does: --worker-name --backend --url
# --reconnect --force --allow-insecure-url. Plus one bootstrap-only option:
#   --branch <name>   git branch/ref to fetch, default "main"

set -euo pipefail

REPO_URL="https://github.com/noname9006/LlamaToaster.git"
REPO_OWNER_SLASH="noname9006/LlamaToaster"
DIR=""
BRANCH="main"
WORKER_NAME="Local"
BACKEND=""
URL=""
FORCE=0
RECONNECT=0
ALLOW_INSECURE_URL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --worker-name) WORKER_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --reconnect) RECONNECT=1; shift ;;
    --allow-insecure-url) ALLOW_INSECURE_URL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Not required for --reconnect: it only clears the session fields in an
# EXISTING config.json, which already has its own url saved.
if [ -z "$URL" ] && [ "$RECONNECT" -ne 1 ]; then
  echo "--url is required, e.g. --url https://llamatoaster.com" >&2
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
elif [ -d "$DIR/.git" ] && command -v git >/dev/null 2>&1; then
  # Re-run against an existing checkout -- hard-update tracked files to latest
  # $BRANCH (same treatment bootstrap.ps1 gives). reset --hard only rewrites
  # TRACKED files; everything this setup treats as user data lives outside
  # git's view (.gitignore'd worker/config.json, *.db, logs/,
  # mtp-offsets.json, or plain untracked dirs like models/, data/, node_modules),
  # so none of it can be touched here.
  echo "$DIR is already a git checkout -- updating to latest $BRANCH..."
  cd "$DIR"
  if ! git fetch origin "$BRANCH"; then
    echo "git fetch failed -- check network access and the branch name ($BRANCH)." >&2
    exit 1
  fi
  if ! git reset --hard FETCH_HEAD; then
    echo "git reset --hard failed -- the checkout may be corrupt. Delete $DIR and re-run." >&2
    exit 1
  fi
  echo "Updated $DIR to latest $BRANCH (config.json, models, and other local files kept)."
else
  # LlamaToaster files present but NOT a usable git checkout (originally
  # installed from a tarball, or git was removed since): sync a fresh tarball
  # over the folder instead. Only repo-shipped files get overwritten --
  # config.json, models/, logs and DBs stay exactly as they are.
  echo ""
  echo "$DIR already has LlamaToaster files, but is not a git checkout." >&2
  CONFIRM=""
  if { exec 3</dev/tty; } 2>/dev/null; then
    printf 'Extract fresh repo files over it? Same-named files are replaced; config.json, models, logs and DBs are kept. [y/N]: ' >&2
    read -r CONFIRM < /dev/tty
    exec 3<&-
  fi
  case "$CONFIRM" in
    y|Y|yes|Yes|YES)
      ;;
    *)
      echo "Aborted. (Delete $DIR by hand and re-run this command for a truly fresh install.)" >&2
      exit 1
      ;;
  esac
  echo "Syncing latest $BRANCH over $DIR..."
  TMP_TAR="$(mktemp -t llamatoaster-XXXXXX).tar.gz"
  TMP_EXTRACT="$(mktemp -d)"
  curl -fsSL "https://github.com/$REPO_OWNER_SLASH/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP_TAR"
  tar -xzf "$TMP_TAR" -C "$TMP_EXTRACT" --strip-components=1
  if ! cp -R "$TMP_EXTRACT"/. "$DIR"/; then
    echo "Failed copying new files over $DIR -- check permissions and retry." >&2
    exit 1
  fi
  rm -rf "$TMP_TAR" "$TMP_EXTRACT"
  echo "Synced $DIR to latest $BRANCH."
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
[ -n "$URL" ] && SETUP_ARGS+=(--url "$URL")
[ -n "$BACKEND" ] && SETUP_ARGS+=(--backend "$BACKEND")
[ "$FORCE" -eq 1 ] && SETUP_ARGS+=(--force)
[ "$RECONNECT" -eq 1 ] && SETUP_ARGS+=(--reconnect)
[ "$ALLOW_INSECURE_URL" -eq 1 ] && SETUP_ARGS+=(--allow-insecure-url)

exec "$DIR/worker/setup-worker.sh" "${SETUP_ARGS[@]}"
