#!/usr/bin/env bash
# One-liner bootstrap for a macOS/Linux machine that doesn't have the repo
# yet -- no LlamaToaster checkout, no config.json, no llama.cpp, no models.
# You only supply a base folder: it becomes both the code checkout and (via
# setup-worker.sh) the home for the "llama" and "models" subfolders.
#
# Usage from a totally fresh machine (only bash + curl needed; Node.js 22+ is
# downloaded into the install folder if missing, and git is used if present,
# otherwise falls back to a plain tarball download):
#
#   curl -fsSL https://llamatoaster.com/install.sh | bash
#
# That short URL is a 302 redirect to THIS file's raw.githubusercontent.com
# address (server/src/routes/install.ts) -- the bytes that run still come
# from the public repo, where they can be read and diffed, the domain just
# supplies the short name. The raw URL keeps working directly if you prefer
# to see exactly where it points, or the origin is unreachable:
#
#   curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash
#
# It'll ask where to install. To pass any option (including --dir, to skip
# the prompts for unattended/scripted use), forward arguments through bash:
#
#   curl -fsSL https://llamatoaster.com/install.sh | bash -s -- --dir ~/LlamaToaster
#
# --url defaults to the public instance (https://llamatoaster.com).
# Self-hosted deployments pass their own, e.g. --url https://toaster.example.com.
#
# When setup finishes, a "toaster" command is installed for this user so
# every later start/update/restart is just `toaster` from any folder -- see
# setup-worker.sh's install_toaster_shim for exactly what that writes and how
# to remove it (`toaster uninstall`).
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
# Defaults to the public instance so `curl ... | bash` -- which passes no
# arguments at all -- works as-is. Self-hosters pass their own origin.
URL="https://llamatoaster.com"
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

# $URL has a default (see above) so the zero-argument pipe form works, but
# an explicitly-passed empty string would still reach setup-worker.sh and
# fail there with a much less obvious message.
if [ -z "$URL" ] && [ "$RECONNECT" -ne 1 ]; then
  echo "--url was passed but empty. Omit it to use https://llamatoaster.com, or give a real origin." >&2
  exit 1
fi

# Said out loud before anything happens. This script is normally reached by
# piping a URL straight into a shell, which by nature hides what it does
# until it's already doing it -- printing the plan first costs nothing and is
# the difference between "some script ran" and an informed install.
echo ""
echo "LlamaToaster worker setup"
echo "  1. download this repo (no sudo, nothing installed system-wide)"
echo "  2. install Node.js 22 into the install folder if it's missing, then npm dependencies"
echo "  3. ask where to keep code, llama.cpp builds and models"
echo "  4. install a 'toaster' command for your user (undo: toaster uninstall)"
echo "  5. start the worker -- it prints a code to approve this machine at $URL/device"
echo "The worker opens no inbound ports; it polls $URL for work."
echo ""

# Same Git Bash/MSYS/Cygwin footgun setup-worker.sh guards against -- paths
# written on Windows under those shells get silently misinterpreted by the
# worker's native Windows Node process.
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "This looks like Git Bash/MSYS on Windows. Use worker\\bootstrap.ps1 in PowerShell instead." >&2
    exit 1
    ;;
esac

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
fi

# Absolutize $DIR now, before anything below can `cd` into it. $DIR is
# referenced again, as the same string, after this script does `cd "$DIR"`
# further down (and again when forwarding --dir to setup-worker.sh) -- a
# relative --dir, or a relative answer to select_install_dir's prompt (e.g.
# "some/folder" instead of "~/some/folder" or "/abs/some/folder"), would
# otherwise get re-resolved against the new cwd and doubled onto itself.
mkdir -p "$DIR"
DIR="$(cd "$DIR" && pwd)"
echo "Using $DIR"

# Node.js 22+ is required (the worker runs on it, and so does npm below). If
# it's missing or too old, download the official prebuilt tarball from
# nodejs.org into $DIR/.node -- no sudo, no package manager, nothing
# system-wide, same spirit as bootstrap.ps1's winget step. The tarball is
# checked against nodejs.org's published SHASUMS256.txt before it's unpacked.
# setup-worker.sh and the generated `toaster` command put $DIR/.node/bin first
# on PATH whenever that folder exists, so the private copy is what later runs
# use; a system Node that is already 22+ is left alone and preferred.
node_ok() {
  command -v node >/dev/null 2>&1 && node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)' >/dev/null 2>&1
}

ensure_node() {
  local dir="$1"
  if [ -x "$dir/.node/bin/node" ]; then
    export PATH="$dir/.node/bin:$PATH"
  fi
  if node_ok; then
    return
  fi
  if command -v node >/dev/null 2>&1; then
    echo "Found Node.js $(node --version 2>/dev/null), but 22+ is required."
  else
    echo "Node.js not found."
  fi

  local os arch
  case "$(uname -s)" in
    Darwin) os="darwin" ;;
    Linux)  os="linux" ;;
    *) echo "Can't auto-install Node.js on $(uname -s). Install Node.js 22+ from https://nodejs.org and re-run this script." >&2; exit 1 ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) arch="x64" ;;
    arm64|aarch64) arch="arm64" ;;
    *) echo "Can't auto-install Node.js for CPU type $(uname -m). Install Node.js 22+ from https://nodejs.org and re-run this script." >&2; exit 1 ;;
  esac
  # nodejs.org's Linux builds need glibc; Alpine and other musl distros can't
  # run them, and unpacking one would just fail later with a cryptic error.
  if [ "$os" = "linux" ] && ls /lib/ld-musl-* >/dev/null 2>&1; then
    echo "This looks like a musl-based Linux (e.g. Alpine); nodejs.org's builds won't run here. Install Node.js 22+ with your package manager (apk add nodejs npm) and re-run." >&2
    exit 1
  fi

  local base="https://nodejs.org/dist/latest-v22.x"
  echo "Downloading Node.js 22 (LTS line) from nodejs.org into $dir/.node ..."
  local sums file want
  if ! sums="$(curl -fsSL "$base/SHASUMS256.txt")"; then
    echo "Couldn't reach $base -- check your network, or install Node.js 22+ yourself from https://nodejs.org and re-run." >&2
    exit 1
  fi
  file="$(printf '%s\n' "$sums" | awk -v suffix="-$os-$arch.tar.gz" 'length($2) > length(suffix) && substr($2, length($2) - length(suffix) + 1) == suffix && $2 ~ /^node-v[0-9.]+-/ {print $2; exit}')"
  want="$(printf '%s\n' "$sums" | awk -v f="$file" '$2 == f {print $1; exit}')"
  if [ -z "$file" ] || [ -z "$want" ]; then
    echo "No Node.js $os-$arch build listed in $base/SHASUMS256.txt. Install Node.js 22+ yourself from https://nodejs.org and re-run." >&2
    exit 1
  fi

  local tmp got
  tmp="$(mktemp -t llamatoaster-node-XXXXXX)"
  if ! curl -fSL --progress-bar "$base/$file" -o "$tmp"; then
    rm -f "$tmp"
    echo "Node.js download failed." >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$tmp" | awk '{print $1}')"
  else
    got="$(shasum -a 256 "$tmp" | awk '{print $1}')"
  fi
  if [ "$got" != "$want" ]; then
    rm -f "$tmp"
    echo "Node.js download failed its checksum (expected $want, got $got). Not installing it." >&2
    exit 1
  fi
  rm -rf "$dir/.node"
  mkdir -p "$dir/.node"
  tar -xzf "$tmp" -C "$dir/.node" --strip-components=1
  rm -f "$tmp"
  export PATH="$dir/.node/bin:$PATH"
  if ! node_ok; then
    echo "Node.js was unpacked to $dir/.node but doesn't run here. Install Node.js 22+ yourself from https://nodejs.org and re-run." >&2
    exit 1
  fi
  echo "Node.js $(node --version) installed to $dir/.node (private to LlamaToaster; your system is untouched)."
}

ensure_node "$DIR"

# Trims a git checkout down to worker/ + shared/ -- everything directly in
# the repo root (package.json, README, ...) is kept automatically by git's
# cone mode. A worker never runs the server or its admin/client UIs, and
# never needs docs, deploy configs or CI files, so there's no reason an
# install folder should show any of it. Uses git's own sparse-checkout
# rather than a one-off delete so it keeps applying after every future
# `git fetch`/`reset --hard` -- from `toaster update`, a re-run of this
# script, or a plain `git pull` -- not just the moment this runs.
# Best-effort: a git too old for cone-mode sparse-checkout (pre-2.25) just
# keeps the full checkout, which is harmless since this is purely cosmetic.
prune_checkout() {
  local dir="$1"
  if git -C "$dir" sparse-checkout init --cone >/dev/null 2>&1; then
    git -C "$dir" sparse-checkout set worker shared >/dev/null 2>&1 || true
  fi
}

# Same pruning, for the no-git tarball path where sparse-checkout doesn't
# apply -- just remove the folders a worker doesn't need after extraction.
prune_extracted_tarball() {
  local dir="$1"
  rm -rf "$dir/admin" "$dir/client" "$dir/server" "$dir/docs" "$dir/deploy" "$dir/scripts" "$dir/assets" "$dir/bin" "$dir/.github"
}

if [ ! -f "$DIR/package.json" ]; then
  echo "$DIR has no LlamaToaster checkout yet -- downloading it (branch: $BRANCH)..."
  mkdir -p "$DIR"
  if command -v git >/dev/null 2>&1; then
    # Not `git clone`: ensure_node above may already have put Node.js in
    # $DIR/.node, and clone refuses any non-empty target. init + fetch +
    # checkout works in a folder that already has files, and leaves them be.
    git -C "$DIR" init --quiet
    git -C "$DIR" remote add origin "$REPO_URL"
    if ! git -C "$DIR" fetch --depth 1 origin "$BRANCH"; then
      echo "git fetch failed -- check network access and the branch name ($BRANCH)." >&2
      exit 1
    fi
    # Before the checkout, so the pruned folders are never written at all.
    prune_checkout "$DIR"
    git -C "$DIR" checkout --quiet -B "$BRANCH" FETCH_HEAD
  else
    echo "git not found -- downloading a tarball of the repo instead."
    TMP_TAR="$(mktemp -t llamatoaster-XXXXXX).tar.gz"
    curl -fsSL "https://github.com/$REPO_OWNER_SLASH/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP_TAR"
    # GitHub's archive tarball has one top-level folder (e.g.
    # LlamaToaster-main/) wrapping everything -- strip it on extract.
    tar -xzf "$TMP_TAR" -C "$DIR" --strip-components=1
    rm -f "$TMP_TAR"
    prune_extracted_tarball "$DIR"
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
  # Converts a pre-existing full checkout (installed before pruning existed)
  # to sparse right here, so the very next `git fetch`/`reset --hard` below
  # -- and every one after, including from `toaster update` -- keeps the
  # working tree trimmed instead of only ever pruning brand-new installs.
  prune_checkout "$DIR"
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
  prune_extracted_tarball "$TMP_EXTRACT"
  if ! cp -R "$TMP_EXTRACT"/. "$DIR"/; then
    echo "Failed copying new files over $DIR -- check permissions and retry." >&2
    exit 1
  fi
  rm -rf "$TMP_TAR" "$TMP_EXTRACT"
  echo "Synced $DIR to latest $BRANCH."
fi

cd "$DIR"
echo "Installing worker dependencies (npm install --prefix worker)..."
# Only worker/package.json's runtime deps -- see setup-worker.sh. The root
# manifest (server, UI builds, test tooling) is never installed on a worker.
if ! npm install --prefix worker --omit=dev --ignore-scripts; then
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
