# Brings this install's code and dependencies up to date. Called by
# `toaster update` (the shim setup-worker.ps1 writes), which only starts the
# worker if this exits 0 -- so every failure below exits non-zero with the
# real reason, rather than falling through to an old copy of the code that
# looks, from the outside, like an update that did nothing.
#
# Handles three shapes of install:
#   - a git checkout (the normal case): fetch + reset --hard to latest $Branch
#   - a folder with no .git (installed from the zip fallback on a machine
#     that had no git at the time): turned into a checkout in place, same
#     sparse worker/+shared/ layout bootstrap.ps1 uses. Untracked files --
#     worker\config.json, llama\, models\ -- are never touched by git here.
#   - no git on this machine at all: stops and says how to get it.
#
# Usage: .\worker\update-worker.ps1 [-Branch main]

param(
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RepoUrl = "https://github.com/noname9006/LlamaToaster.git"

function Fail {
    param([string]$Message)
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    Write-Host "Nothing was started. Fix the problem above, then run: toaster update" -ForegroundColor Red
    exit 1
}

# git writes routine progress to stderr, which $ErrorActionPreference=Stop
# would turn into a terminating error -- same wrapper as bootstrap.ps1. Output
# is kept (not just printed) so a failure can be matched against the known
# causes below and given a specific fix instead of a generic one.
function Invoke-Git {
    param([string[]]$GitArgs)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    # "$_" turns each stderr ErrorRecord back into git's plain message line --
    # Out-String on the record itself prints PowerShell's NativeCommandError
    # boilerplate around it.
    $output = & git -C $RepoRoot @GitArgs 2>&1 | ForEach-Object { "$_" } | Out-String
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    return [pscustomobject]@{ Exit = $exitCode; Output = $output.Trim() }
}

function Assert-Git {
    param([string[]]$GitArgs, [string]$What)
    $r = Invoke-Git $GitArgs
    if ($r.Exit -eq 0) { return $r }
    Write-Host $r.Output
    $hint = ""
    if ($r.Output -match 'dubious ownership') {
        $hint = "git refuses to use $RepoRoot because the folder is owned by a different Windows account. Trust it with:`n  git config --global --add safe.directory `"$($RepoRoot -replace '\\','/')`""
    } elseif ($r.Output -match 'index\.lock') {
        $hint = "A previous git command was interrupted and left a lock file. If no other git command is running, delete:`n  $RepoRoot\.git\index.lock"
    } elseif ($r.Output -match 'Could not resolve host|unable to access|Failed to connect|timed out') {
        $hint = "Could not reach GitHub -- check this machine's internet connection."
    }
    if ($hint) { Write-Host ""; Write-Host $hint -ForegroundColor Yellow }
    Fail "Update failed while trying to $What (git exit $($r.Exit))."
}

# A worker already running from this folder holds node_modules files open
# (tsx's esbuild.exe above all), which makes `npm install` fail with
# EBUSY/EPERM -- and even when it doesn't, that worker keeps running the old
# code. tsx's own command line always carries the absolute node_modules path,
# so that's what identifies "a worker from THIS install" rather than any
# node process.
$tsxMarker = (Join-Path $RepoRoot "node_modules\tsx").ToLower()
$running = @(Get-CimInstance Win32_Process -Filter "Name='node.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine.ToLower().Contains($tsxMarker) })
if ($running.Count -gt 0) {
    Fail "A worker from $RepoRoot is still running (PID $(($running.ProcessId) -join ', ')). Stop it with Ctrl+C in its window first."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not installed, so this install can't be updated in place. Install it (winget install --id Git.Git -e), open a new terminal, and run toaster update again."
}

if (Test-Path (Join-Path $RepoRoot ".git")) {
    $before = (Invoke-Git @('rev-parse', '--short', 'HEAD')).Output
} else {
    Write-Host "$RepoRoot is not a git checkout yet (it was installed from a zip) -- converting it so it can be updated."
    Assert-Git @('init', '--quiet') "initialise a git repository" | Out-Null
    Assert-Git @('remote', 'add', 'origin', $RepoUrl) "add the GitHub remote" | Out-Null
    # Same trimmed layout as a fresh git install (bootstrap.ps1's
    # Set-SparseCheckout). Best-effort, like there: an old git without cone
    # mode just gets the full tree.
    if ((Invoke-Git @('sparse-checkout', 'init', '--cone')).Exit -eq 0) {
        Invoke-Git @('sparse-checkout', 'set', 'worker', 'shared') | Out-Null
    }
    $before = $null
}

Write-Host "Fetching latest $Branch..."
Assert-Git @('fetch', 'origin', $Branch) "download the latest code" | Out-Null
Assert-Git @('reset', '--hard', 'FETCH_HEAD') "switch this install to the latest code" | Out-Null
$after = (Invoke-Git @('rev-parse', '--short', 'HEAD')).Output

if (-not $before) {
    Write-Host "Converted to a git checkout at $after." -ForegroundColor Green
} elseif ($before -eq $after) {
    Write-Host "Already up to date ($after)." -ForegroundColor Green
} else {
    Write-Host "Updated ${before} -> ${after}:" -ForegroundColor Green
    Write-Host (Invoke-Git @('log', '--oneline', '--no-decorate', "$before..$after")).Output
}

Write-Host ""
Write-Host "Installing dependencies (npm install)..."
Push-Location $RepoRoot
# --ignore-scripts: see setup-worker.ps1 -- better-sqlite3 is server-only and
# would otherwise need the Visual Studio C++ Build Tools to compile.
npm install --ignore-scripts
$npmExit = $LASTEXITCODE
Pop-Location
if ($npmExit -ne 0) {
    Fail "npm install failed (exit $npmExit). The code is updated but its dependencies may not be. If the error mentions EBUSY/EPERM, close every worker window and any editor open in $RepoRoot, then retry."
}
exit 0
