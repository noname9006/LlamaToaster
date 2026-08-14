# One-liner bootstrap for a Windows machine that doesn't have the repo yet --
# no LlamaToaster checkout, no config.json, no llama.cpp, no models. You only
# supply a base folder: it becomes both the code checkout and (via
# setup-worker.ps1) the home for the "llama" and "models" subfolders.
#
# Usage from a totally fresh machine (only PowerShell needed -- Node.js is
# installed automatically via winget if missing; git is used if present,
# otherwise falls back to a plain zip download).
# Omit -Dir and it'll ask -- lists your local drives with free space (models
# are often tens of GB each) and asks which one, then a folder name:
#
#   iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) }"
#
# Pass -Dir to skip the prompts (e.g. for unattended/scripted use):
#
#   iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) } -Dir F:\LlamaToaster"
#
# Or download it first and run locally:
#
#   irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1 -OutFile bootstrap.ps1
#   .\bootstrap.ps1
#
# Safe to re-run: if $Dir is already a git checkout, it's updated in place
# (git fetch + reset --hard to latest $Branch) instead of re-cloning;
# setup-worker.ps1 underneath is itself idempotent (skips writing config.json
# if one already exists). So the same command works for first setup, picking
# up code updates, and every plain restart after. If $Dir exists with
# unrelated files in it (no .git), you'll be asked to confirm before repo
# files get synced in on top.
#
# All setup-worker.ps1 overrides are forwarded -- see that script's own
# header for what each one does: -WorkerName -Backend -VpsUrl -BindHost
# -Port -Force. Plus one bootstrap-only option:
#   -Branch <name>   git branch/ref to fetch, default "main"

param(
    [string]$Dir,
    [string]$Branch = "main",
    [string]$WorkerName = "Local",
    [string]$Backend,
    [string]$VpsUrl = "http://100.122.1.111:4010",
    [string]$BindHost,
    [int]$Port = 8080,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/noname9006/LlamaToaster.git"
$RepoOwnerSlash = "noname9006/LlamaToaster"

function Select-InstallDir {
    Write-Host ""
    Write-Host "Which drive should LlamaToaster use?" -ForegroundColor Cyan
    Write-Host "Downloaded models are often tens of GB each -- pick a drive with room to spare."
    Write-Host ""
    $drives = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DriveType=3" | Sort-Object DeviceID
    if (-not $drives) {
        Write-Error "Could not enumerate local drives. Pass -Dir explicitly, e.g. -Dir F:\LlamaToaster."
        exit 1
    }
    foreach ($d in $drives) {
        $freeGB = [math]::Round($d.FreeSpace / 1GB, 1)
        $totalGB = [math]::Round($d.Size / 1GB, 1)
        $label = if ($d.VolumeName) { " ($($d.VolumeName))" } else { "" }
        Write-Host ("  {0}   {1,8:N1} GB free of {2,-8:N1} GB{3}" -f $d.DeviceID, $freeGB, $totalGB, $label)
    }
    Write-Host ""
    $validLetters = $drives | ForEach-Object { $_.DeviceID.TrimEnd(':') }
    do {
        $driveLetter = (Read-Host "Drive letter").Trim().TrimEnd(':').ToUpper()
        if ($driveLetter -notin $validLetters) {
            Write-Host "Not one of the drives listed above -- try again." -ForegroundColor Yellow
        }
    } while ($driveLetter -notin $validLetters)

    $folderName = Read-Host "Folder name to create on ${driveLetter}: (default: LlamaToaster)"
    if (-not $folderName) { $folderName = "LlamaToaster" }
    $resolved = "${driveLetter}:\$folderName"
    Write-Host "Using $resolved" -ForegroundColor Cyan
    Write-Host ""
    return $resolved
}

if (-not $Dir) {
    $Dir = Select-InstallDir
}

# $ErrorActionPreference = "Stop" (set above) makes PowerShell treat *any*
# stderr line from a native command as a terminating error -- and git writes
# routine progress ("Cloning into ...", "Receiving objects...") to stderr, so
# a plain `git ...` call here would abort on git's normal chatter, not just
# real failures, and swallow the actual error text in the process. Run git
# through this wrapper instead: it drops $ErrorActionPreference to 'Continue'
# just for the call, captures combined output, and only prints/returns it so
# real errors are visible instead of silently lost.
function Invoke-GitCommand {
    param([string[]]$GitArgs)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & git @GitArgs 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($exitCode -ne 0) {
        Write-Host ($output | Out-String)
    }
    return $exitCode
}

# Fetches the repo (git clone if available, else a zip download) into a
# scratch folder, then merges it into $TargetDir with Copy-Item -Force.
# Cloning to scratch first -- rather than straight into $TargetDir -- means
# this works whether $TargetDir is empty, already a checkout, or has
# unrelated files sitting in it (e.g. a stray .claude folder): git never
# sees a non-empty clone target, and the merge only touches files the repo
# actually ships.
function Sync-LlamaToasterCode {
    param([string]$TargetDir, [string]$RefBranch)

    $scratch = Join-Path $env:TEMP "llamatoaster-sync-$([guid]::NewGuid())"
    New-Item -ItemType Directory -Force -Path $scratch | Out-Null

    if (Get-Command git -ErrorAction SilentlyContinue) {
        $cloneExit = Invoke-GitCommand @('clone', '--depth', '1', '--branch', $RefBranch, $RepoUrl, $scratch)
        if ($cloneExit -ne 0) {
            Remove-Item $scratch -Recurse -Force -ErrorAction SilentlyContinue
            Write-Error "git clone failed (exit $cloneExit) -- see the git output above for the actual reason."
            exit 1
        }
    } else {
        Write-Host "git not found -- downloading a zip of the repo instead."
        $zipPath = Join-Path $env:TEMP "llamatoaster-$RefBranch-$([guid]::NewGuid()).zip"
        $extractDir = Join-Path $env:TEMP "llamatoaster-extract-$([guid]::NewGuid())"
        Invoke-WebRequest -Uri "https://github.com/$RepoOwnerSlash/archive/refs/heads/$RefBranch.zip" -OutFile $zipPath
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        # GitHub's archive zip has one top-level folder (e.g. LlamaToaster-main/)
        # wrapping everything -- copy its contents, not the wrapper, into scratch.
        $extractedRoot = Get-ChildItem $extractDir | Select-Object -First 1
        Get-ChildItem $extractedRoot.FullName | Move-Item -Destination $scratch -Force
        Remove-Item $zipPath -Force
        Remove-Item $extractDir -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Copy-Item -Path (Join-Path $scratch "*") -Destination $TargetDir -Recurse -Force
    Remove-Item $scratch -Recurse -Force
}

$GitDir = Join-Path $Dir ".git"
$dirHasContent = (Test-Path $Dir) -and ((Get-ChildItem -Path $Dir -Force -ErrorAction SilentlyContinue | Select-Object -First 1) -ne $null)

if (Test-Path $GitDir) {
    Write-Host "$Dir is already a git checkout -- updating to latest $Branch..."
    Push-Location $Dir
    $fetchExit = Invoke-GitCommand @('fetch', 'origin', $Branch)
    $resetExit = 0
    if ($fetchExit -eq 0) {
        $resetExit = Invoke-GitCommand @('reset', '--hard', 'FETCH_HEAD')
    }
    Pop-Location
    if ($fetchExit -ne 0 -or $resetExit -ne 0) {
        Write-Error "Failed to update the existing checkout at $Dir (git fetch/reset). Fix it manually or delete $Dir and re-run."
        exit 1
    }
    Write-Host "Updated $Dir to latest $Branch."
} elseif ($dirHasContent) {
    Write-Host ""
    Write-Host "$Dir already exists and has files in it, but isn't a git checkout." -ForegroundColor Yellow
    $proceed = Read-Host "Set up LlamaToaster here anyway? Repo files will overwrite same-named files; anything else in the folder is left alone. [y/N]"
    if ($proceed -notmatch '^(y|yes)$') {
        Write-Host "Aborted."
        exit 1
    }
    Sync-LlamaToasterCode -TargetDir $Dir -RefBranch $Branch
    Write-Host "Synced $Dir to latest $Branch."
} else {
    Write-Host "$Dir has no LlamaToaster checkout yet -- downloading it (branch: $Branch)..."
    Sync-LlamaToasterCode -TargetDir $Dir -RefBranch $Branch
    Write-Host "Downloaded to $Dir"
}

# npm needs Node.js on PATH. If it's missing, install it via winget rather
# than just erroring out -- this is meant to work on a totally fresh machine
# where Node was never set up. winget writes PATH to the registry but this
# process's own $env:Path doesn't pick that up, so rebuild it from both
# scopes afterward -- otherwise `npm install` below would still fail even
# right after a successful install.
function Ensure-NodeJs {
    if (Get-Command node -ErrorAction SilentlyContinue) { return }

    Write-Host "Node.js not found -- installing via winget..." -ForegroundColor Cyan
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error "Node.js is required but not installed, and winget isn't available to install it automatically. Install Node.js 22+ from https://nodejs.org and re-run this script."
        exit 1
    }

    winget install --id OpenJS.NodeJS.LTS -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Node.js install via winget failed (exit $LASTEXITCODE). Install it manually from https://nodejs.org and re-run this script."
        exit 1
    }

    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"

    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Write-Error "Node.js was installed but isn't on PATH in this session. Close and reopen PowerShell, then re-run this script."
        exit 1
    }
    Write-Host "Node.js installed: $(node --version)" -ForegroundColor Green
}

Ensure-NodeJs

Push-Location $Dir
Write-Host "Installing dependencies (npm install)..."
# --ignore-scripts: skips better-sqlite3's install step, which always compiles
# from source via node-gyp (it has no prebuilt-binary fallback) and needs the
# full Visual Studio C++ Build Tools workload. The worker never imports
# better-sqlite3 (server-only), so there's nothing to build here.
npm install --ignore-scripts
$installExit = $LASTEXITCODE
Pop-Location
if ($installExit -ne 0) {
    Write-Error "npm install failed (exit $installExit)."
    exit 1
}

$setupArgs = @{ Dir = $Dir; WorkerName = $WorkerName; VpsUrl = $VpsUrl; Port = $Port }
if ($Backend) { $setupArgs.Backend = $Backend }
if ($BindHost) { $setupArgs.BindHost = $BindHost }
if ($Force) { $setupArgs.Force = $true }

& (Join-Path $Dir "worker\setup-worker.ps1") @setupArgs
