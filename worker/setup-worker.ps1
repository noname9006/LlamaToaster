# One-shot bootstrap for a worker that has nothing downloaded yet: no
# config.json, no llama.cpp binary, no models. You only supply one base
# folder -- llama.cpp builds and any future GGUF downloads get fetched
# automatically later (llama.cpp on first triggered run, models whenever you
# download one from the Models page) into <Dir>\llama and <Dir>\models. This
# script itself never downloads llama.cpp or a model.
#
# Usage (from anywhere). Safe to run every time, first setup or a plain
# restart -- if config.json already exists, setup (and the -Dir prompt) is
# skipped entirely and this just starts the worker:
#   .\worker\setup-worker.ps1 -Url https://llamatoaster.com
#
# On first run (no config.json yet), omitting -Dir asks which drive to use
# (with free-space info -- models are often tens of GB each) and a folder
# name. Pass -Dir to skip that prompt, e.g. for unattended/scripted use:
#   .\worker\setup-worker.ps1 -Url https://llamatoaster.com -Dir F:\LlamaToaster
#
# No IP or port to configure at all -- the worker has no inbound listener
# (it long-polls the server, MULTIUSER_PLAN.md §1). The FIRST time it runs
# with no session configured yet, it prints a short code and a URL
# (https://<url>/device) to approve it from your account -- do that once,
# in a browser, and it's connected. The saved session survives ordinary
# restarts and never asks again on its own -- if it needs to reconnect (its
# session was revoked from Settings, or its refresh token expired), pass
# -Reconnect (see below) rather than re-running plain setup, which is a
# total no-op once config.json already exists.
#
# Other optional overrides:
#   -WorkerName <name>   default "Local"
#   -Backend <value>     any backend llama.cpp's releases use (cpu, vulkan,
#                        cuda, rocm, sycl, opencl-adreno, openvino, or
#                        anything a future release adds) -- not validated
#                        against a fixed list. Default: left unset, so the
#                        worker auto-detects it live from this machine's
#                        actual GPU on startup. Pass this only to pin/override
#                        that, e.g. for hardware the auto-detect heuristic
#                        doesn't specifically know about.
#   -InstallLhm          fetch LibreHardwareMonitor onto this machine -- the
#                        sensor_lhm clock/temperature source worker/src/
#                        sensors.ts reads on Windows AMD/Intel GPUs (an NVIDIA
#                        box needs only nvidia-smi, which ships with its own
#                        driver -- nothing to install). Tries winget first,
#                        falls back to extracting the latest GitHub release
#                        zip into <Dir>\LibreHardwareMonitor. One step stays
#                        manual no matter what: LHM must be RUNNING AS
#                        ADMINISTRATOR while benchmarks execute, or its GPU
#                        sensor readings are simply not there to be read.
#   -Reconnect            clear this machine's saved session (keeping
#                        machine_id and every other setting) so it goes
#                        through device-flow approval again on the same
#                        machine identity -- the server recognizes it as the
#                        SAME machine (same history/display_name), unlike
#                        -Force below. Requires config.json to already exist.
#   -Force               overwrite an existing worker/config.json from
#                        scratch, INCLUDING machine_id -- the server will
#                        treat this as a brand-new machine with no history.
#                        Use -Reconnect instead unless that's actually what
#                        you want.
#   -AllowInsecureUrl     allow a plain http:// -Url other than
#                        localhost/127.0.0.1 (see below). Only for a
#                        deployment you've secured another way -- e.g. this
#                        app's own tailnet-only mode, where Tailscale's own
#                        WireGuard tunnel is the encryption, not TLS.
#
# Safe by default: refuses to overwrite an existing worker/config.json unless
# -Force (or -Reconnect, which only ever touches the session fields) is
# passed -- no git in this repo checkout, so an overwritten config.json has
# no recovery path. Idempotent otherwise: if config.json already exists and
# neither switch is given, setup is skipped and this just (re)starts the
# worker with it as-is -- so the exact same command works both for
# first-time setup and every restart after.

param(
    [string]$Dir,
    [string]$WorkerName = "Local",
    [string]$Backend,
    [string]$Url,
    [switch]$Reconnect,
    [switch]$Force,
    [switch]$AllowInsecureUrl,
    [switch]$InstallLhm
)

$ErrorActionPreference = "Stop"

if ($Force -and $Reconnect) {
    Write-Error "-Force and -Reconnect are mutually exclusive (-Force wipes machine_id, -Reconnect preserves it)."
    exit 1
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $RepoRoot "worker\config.json"

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

# -InstallLhm worker -- see the header doc above for why this exists. LHM is
# NOT an npm dependency and cannot become one: it's a .NET desktop app that a
# human runs elevated. This function only automates getting it ONTO the disk,
# via whatever channel the machine already has (winget, else GitHub releases).
# Asset NAMES drift between builds ("...net472.zip", "...NET.10.zip"), but
# every release ships exactly one .zip, so name-match loosely.
function Install-LibreHardwareMonitor {
    $installedViaWinget = $false
    try {
        $listed = @(winget list --id LibreHardwareMonitor.LibreHardwareMonitor 2>$null)
        if (($listed -join "`n") -match "LibreHardwareMonitor") {
            Write-Host "LibreHardwareMonitor already present via winget -- keeping it."
            $installedViaWinget = $true
        }
    } catch {
        # No winget, or it failed to enumerate -- the zip fallback covers it.
    }

    if (-not $installedViaWinget -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Installing LibreHardwareMonitor via winget..."
        winget install -e --id LibreHardwareMonitor.LibreHardwareMonitor `
            --accept-source-agreements --accept-package-agreements --silent | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $installedViaWinget = $true
        } else {
            Write-Host "winget exited $LASTEXITCODE -- falling back to the GitHub release zip." -ForegroundColor Yellow
        }
    }

    if (-not $installedViaWinget) {
        $extractDir = Join-Path $Dir "LibreHardwareMonitor"
        if (Test-Path (Join-Path $extractDir "LibreHardwareMonitor.exe")) {
            Write-Host "Found existing $extractDir\LibreHardwareMonitor.exe -- reusing it."
        } else {
            Write-Host "Downloading the latest LibreHardwareMonitor release..."
            $release = Invoke-RestMethod -Headers @{ "User-Agent" = "llamatoaster-worker-setup" } `
                "https://api.github.com/repos/LibreHardwareMonitor/LibreHardwareMonitor/releases/latest"
            $zipAsset = @($release.assets) | Where-Object { $_.name -match "\.zip$" } | Select-Object -First 1
            if (-not $zipAsset) {
                Write-Error "No .zip asset found in the latest LibreHardwareMonitor release."
                exit 1
            }
            $zipPath = Join-Path $env:TEMP $zipAsset.name
            Invoke-WebRequest -Uri $zipAsset.browser_download_url -OutFile $zipPath -UseBasicParsing
            Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
            Remove-Item $zipPath -Force
            $exe = Get-ChildItem $extractDir -Recurse -Filter "LibreHardwareMonitor.exe" | Select-Object -First 1
            if (-not $exe) {
                Write-Error "Downloaded $($zipAsset.name) but no LibreHardwareMonitor.exe inside after extraction."
                exit 1
            }
            Write-Host "Extracted to $($exe.FullName)"
        }
    }

    # The part every deployment shares, stated out loud because no automation
    # can do it -- elevation is consent the OS exists to extract from a human.
    Write-Host ""
    Write-Host "One step stays manual (UAC cannot be scripted away):" -ForegroundColor Cyan
    Write-Host "  1. Run LibreHardwareMonitor AS ADMINISTRATOR and keep it open during benchmarks." -ForegroundColor Cyan
    Write-Host "     Every sensor read dies with it -- HTTP feed AND WMI both stop populating." -ForegroundColor Cyan
    Write-Host "  2. Options -> tick 'Remote Web Server' so the worker reads http://127.0.0.1:8085/data.json" -ForegroundColor Cyan
    Write-Host "     (override the port with the LHM_HTTP_PORT env var; without the web server," -ForegroundColor Cyan
    Write-Host "     the worker falls back to LHM's WMI namespace at the cost of a slower read)." -ForegroundColor Cyan
    Write-Host "  3. Auto-start elevated each boot: Task Scheduler -> Create Task -> 'Run with highest privileges'," -ForegroundColor Cyan
    Write-Host "     logon trigger, action = LibreHardwareMonitor.exe." -ForegroundColor Cyan
    Write-Host "  4. A worker started while LHM was down picks availability up within ~10 minutes;" -ForegroundColor Cyan
    Write-Host "     starting the worker AFTER LHM is running lands instantly." -ForegroundColor Cyan
}


if ($Reconnect) {
    if (-not (Test-Path $ConfigPath)) {
        Write-Error "-Reconnect needs an existing $ConfigPath -- nothing to reconnect. Run without -Reconnect first."
        exit 1
    }
    # ConvertFrom-Json / ConvertTo-Json, not a text edit -- real JSON parsing,
    # same reasoning as setup-worker.sh's node one-liner. Every other field
    # -- machine_id above all -- passes through untouched; only these two
    # are ever removed.
    $existing = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $existing.PSObject.Properties.Remove('session_token')
    $existing.PSObject.Properties.Remove('refresh_token')
    $json = $existing | ConvertTo-Json
    [System.IO.File]::WriteAllText($ConfigPath, $json, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "Cleared the saved session in $ConfigPath -- this machine will go through device-flow approval again on startup, same identity." -ForegroundColor Yellow
}

$SkippedSetup = ((Test-Path $ConfigPath) -and -not $Force -and -not $Reconnect) -or $Reconnect
if ($Reconnect) {
    # config.json already handled above -- nothing else to do before starting.
} elseif ($SkippedSetup) {
    Write-Host "worker\config.json already exists -- skipping setup, starting the worker with it as-is." -ForegroundColor Yellow
    Write-Host "(Re-run with -Force to redo setup from scratch, or -Reconnect to just re-approve this machine.)" -ForegroundColor Yellow
} else {
    if (-not $Url) {
        Write-Error "-Url is required on first setup, e.g. -Url https://llamatoaster.com"
        exit 1
    }
    # This ends up in config.json and gets a real 90-day bearer token sent to
    # it on every heartbeat/queue-poll (MULTIUSER_PLAN.md §3.5's own "the
    # installer refuses http://") -- plaintext HTTP would put that credential
    # on the wire in the clear. localhost/127.0.0.1/::1 are exempted, same
    # convention browsers use for "secure context" -- needed for local
    # dev/testing against a server that isn't fronted by TLS yet. Anything
    # else over http:// needs -AllowInsecureUrl (e.g. this app's own
    # tailnet-only mode, where Tailscale's own tunnel is the encryption, not
    # TLS).
    $isLocalHttp = $Url -match '^http://(localhost|127\.0\.0\.1|\[::1\])'
    if (-not $Url.StartsWith("https://") -and -not $isLocalHttp) {
        if ($Url.StartsWith("http://") -and $AllowInsecureUrl) {
            # explicitly accepted
        } elseif ($Url.StartsWith("http://")) {
            Write-Error "-Url is http:// ($Url) -- that sends a real bearer credential in the clear. Use https://, or pass -AllowInsecureUrl if this connection is secured another way (e.g. a Tailscale-only deployment)."
            exit 1
        } else {
            Write-Error "-Url must start with https:// or http://, got: $Url"
            exit 1
        }
    }

    if (-not $Dir) {
        $Dir = Select-InstallDir
    }
    $LlamaCppDir = Join-Path $Dir "llama"
    $ModelsDir = Join-Path $Dir "models"

    foreach ($dir in @($LlamaCppDir, $ModelsDir)) {
        if (-not (Test-Path $dir)) {
            Write-Host "Creating $dir"
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
    }
    $RawJsonDir = Join-Path $ModelsDir "raw"
    if (-not (Test-Path $RawJsonDir)) {
        New-Item -ItemType Directory -Force -Path $RawJsonDir | Out-Null
    }

    $config = [ordered]@{
        worker_name          = $WorkerName
        llama_cpp_build      = "none"
        llama_bench_path     = (Join-Path $LlamaCppDir "llama-bench.exe")
        model_dir            = $ModelsDir
        url                  = $Url
        raw_json_dir         = $RawJsonDir
        llama_cpp_builds_dir = $LlamaCppDir
    }
    # Left out entirely (not even a null/empty value) when not given, so the
    # worker's own startup logic auto-detects it from live hardware instead.
    if ($Backend) { $config.backend = $Backend }
    $json = $config | ConvertTo-Json
    # Set-Content -Encoding utf8 writes a BOM on Windows PowerShell 5.1, which
    # Node's JSON.parse (used by loadConfig in worker/src/index.ts) chokes on --
    # write plain UTF-8 without one instead.
    [System.IO.File]::WriteAllText($ConfigPath, $json, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "Wrote $ConfigPath"
}

if ($InstallLhm) {
    # Usable on plain restarts too -- config.json exists then, so setup was
    # skipped and $Dir never got assigned above. Prompt once rather than
    # guessing where the fallback zip lands.
    if (-not $Dir) {
        $Dir = Select-InstallDir
    }
    Install-LibreHardwareMonitor
}


Push-Location $RepoRoot
if (-not (Test-Path "node_modules\.bin\tsx.cmd") -and -not (Test-Path "node_modules\.bin\tsx")) {
    Write-Host "Installing dependencies (npm install)..."
    # --ignore-scripts: skips better-sqlite3's install step, which always
    # compiles from source via node-gyp (it has no prebuilt-binary fallback)
    # and needs the full Visual Studio C++ Build Tools workload. The worker
    # never imports better-sqlite3 (server-only), so there's nothing to
    # build here.
    npm install --ignore-scripts
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "npm install failed (exit $LASTEXITCODE)."
        exit 1
    }
}

if ($Reconnect) {
    Write-Host "Starting worker -- watch the console for a short code and a link to re-approve this machine."
} elseif ($SkippedSetup) {
    Write-Host "Starting worker with existing $ConfigPath"
} else {
    $backendLabel = if ($Backend) { $Backend } else { "auto-detected from hardware" }
    Write-Host "Starting worker '$WorkerName' ($backendLabel) -- llama.cpp builds will install to $LlamaCppDir on first run, models download to $ModelsDir from the Models page."
    Write-Host "First run only: watch the console for a short code and a link to approve this machine."
}
npm run worker
Pop-Location
