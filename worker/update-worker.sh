#!/usr/bin/env bash
# Brings this install's code and dependencies up to date. Called by
# `toaster update` (the shim setup-worker.sh writes), which only starts the
# worker if this exits 0 -- so every failure below exits non-zero with the
# real reason, rather than falling through to an old copy of the code that
# looks, from the outside, like an update that did nothing.
#
# Handles three shapes of install:
#   - a git checkout (the normal case): fetch + reset --hard to latest branch
#   - a folder with no .git (installed from the tarball fallback on a machine
#     that had no git at the time): turned into a checkout in place, same
#     sparse worker/+shared/ layout bootstrap.sh uses. Untracked files --
#     worker/config.json, llama/, models/ -- are never touched by git here.
#   - no git on this machine at all: stops and says how to get it.
#
# Usage: ./worker/update-worker.sh [--branch main]
#
# Everything lives inside main(), called on the last line: `git reset --hard`
# below can rewrite THIS file mid-run, and bash reads scripts incrementally --
# a function body is parsed in full before any of it executes, so the old
# copy runs to completion no matter what lands on disk.

main() {
  local branch="main"
  while [ $# -gt 0 ]; do
    case "$1" in
      --branch) branch="$2"; shift 2 ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
  done

  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  local repo_url="https://github.com/noname9006/LlamaToaster.git"

  fail() {
    echo "" >&2
    echo "$1" >&2
    echo "Nothing was started. Fix the problem above, then run: toaster update" >&2
    exit 1
  }

  # Runs git, and on failure prints its output plus a specific fix for the
  # known causes before failing.
  run_git() {
    local what="$1"; shift
    local out code=0
    out="$(git -C "$repo_root" "$@" 2>&1)" || code=$?
    if [ "$code" -eq 0 ]; then
      [ -z "$out" ] || printf '%s\n' "$out"
      return 0
    fi
    printf '%s\n' "$out" >&2
    case "$out" in
      *"dubious ownership"*)
        echo "" >&2
        echo "git refuses to use $repo_root because it's owned by a different user. Trust it with:" >&2
        echo "  git config --global --add safe.directory \"$repo_root\"" >&2 ;;
      *index.lock*)
        echo "" >&2
        echo "A previous git command was interrupted and left a lock file. If no other git command is running, delete:" >&2
        echo "  $repo_root/.git/index.lock" >&2 ;;
      *"Could not resolve host"*|*"unable to access"*|*"Failed to connect"*|*"timed out"*)
        echo "" >&2
        echo "Could not reach GitHub -- check this machine's internet connection." >&2 ;;
    esac
    fail "Update failed while trying to $what (git exit $code)."
  }

  # A worker still running from this folder keeps running the old code. On
  # Linux, /proc gives each process's working directory; `npm run worker`
  # always runs from the repo root. No /proc (macOS): skip the check.
  if [ -d /proc/self ]; then
    local pids="" p
    for p in /proc/[0-9]*; do
      if [ "$(readlink "$p/cwd" 2>/dev/null)" = "$repo_root" ] \
        && tr '\0' ' ' < "$p/cmdline" 2>/dev/null | grep -q 'tsx.*worker/src/index.ts'; then
        pids="$pids ${p#/proc/}"
      fi
    done
    if [ -n "$pids" ]; then
      fail "A worker from $repo_root is still running (PID$pids). Stop it with Ctrl+C in its window first."
    fi
  fi

  command -v git >/dev/null 2>&1 \
    || fail "git is not installed, so this install can't be updated in place. Install git with your package manager, then run toaster update again."

  local before=""
  if [ -d "$repo_root/.git" ]; then
    before="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || true)"
  else
    echo "$repo_root is not a git checkout yet (it was installed from a tarball) -- converting it so it can be updated."
    run_git "initialise a git repository" init --quiet
    run_git "add the GitHub remote" remote add origin "$repo_url"
    # Same trimmed layout as a fresh git install. Best-effort, like
    # bootstrap.sh: an old git without cone mode just gets the full tree.
    if git -C "$repo_root" sparse-checkout init --cone >/dev/null 2>&1; then
      git -C "$repo_root" sparse-checkout set worker shared >/dev/null 2>&1 || true
    fi
  fi

  echo "Fetching latest $branch..."
  run_git "download the latest code" fetch --quiet origin "$branch"
  run_git "switch this install to the latest code" reset --hard --quiet FETCH_HEAD
  local after
  after="$(git -C "$repo_root" rev-parse --short HEAD)"

  if [ -z "$before" ]; then
    echo "Converted to a git checkout at $after."
  elif [ "$before" = "$after" ]; then
    echo "Already up to date ($after)."
  else
    echo "Updated $before -> $after:"
    git -C "$repo_root" log --oneline --no-decorate "$before..$after"
  fi

  echo ""
  echo "Installing dependencies (npm install)..."
  # --ignore-scripts: see setup-worker.sh -- better-sqlite3 is server-only.
  (cd "$repo_root" && npm install --ignore-scripts) \
    || fail "npm install failed. The code is updated but its dependencies may not be."
}

main "$@"
exit 0
