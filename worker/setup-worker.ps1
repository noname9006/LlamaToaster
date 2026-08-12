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
#   .\worker\setup-worker.ps1
#
# On first run (no config.json yet), omitting -Dir asks which drive to use
# (with free-space info -- models are often tens of GB each) and a folder
# name. Pass -Dir to skip that prompt, e.g. for unattended/scripted use:
#   .\worker\setup-worker.ps1 -Dir F:\LlamaToaster
#
# No IP needed either: bind_host auto-detects via `tailscale ip -4` unless
# you pass -BindHost explicitly.
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
#   -VpsUrl <url>        default the orchestrator's known Tailscale address
#   -Port <n>            default 8080
#   -Force               overwrite an existing worker/config.json
#
# Safe by default: refuses to overwrite an existing worker/config.json unless
# -Force is passed (no git in this repo checkout, so an overwritten config.json
# has no recovery path). Idempotent otherwise: if config.json already exists,
# setup is skipped and this just (re)starts the worker with it as-is -- so the
# exact same command works both for first-time setup and every restart after.

param(
    [string]$Dir,
    [string]$WorkerName = "Local",
    [string]$Backend,
    [string]$VpsUrl = "http://100.122.1.111:4010",
    [string]$BindHost,
    [int]$Port = 8080,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
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

$SkippedSetup = (Test-Path $ConfigPath) -and -not $Force
if ($SkippedSetup) {
    Write-Host "worker\config.json already exists -- skipping setup, starting the worker with it as-is." -ForegroundColor Yellow
    Write-Host "(Re-run with -Force to redo setup, e.g. after changing -Dir or -BindHost.)" -ForegroundColor Yellow
} else {
    if (-not $Dir) {
        $Dir = Select-InstallDir
    }
    $LlamaCppDir = Join-Path $Dir "llama"
    $ModelsDir = Join-Path $Dir "models"

    if (-not $BindHost) {
        Write-Host "No -BindHost given, trying 'tailscale ip -4'..."
        try {
            $BindHost = (& tailscale ip -4 2>$null | Select-Object -First 1).Trim()
        } catch {
            $BindHost = $null
        }
        if (-not $BindHost) {
            Write-Error "Could not auto-detect a Tailscale IP. Pass -BindHost <this machine's tailnet IP> explicitly."
            exit 1
        }
        Write-Host "Using detected Tailscale IP: $BindHost"
    }

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
        vps_url              = $VpsUrl
        bind_host            = $BindHost
        port                 = $Port
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

Push-Location $RepoRoot
if (-not (Test-Path "node_modules\.bin\tsx.cmd") -and -not (Test-Path "node_modules\.bin\tsx")) {
    Write-Host "Installing dependencies (npm install)..."
    npm install
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "npm install failed (exit $LASTEXITCODE)."
        exit 1
    }
}

if ($SkippedSetup) {
    Write-Host "Starting worker with existing $ConfigPath"
} else {
    $backendLabel = if ($Backend) { $Backend } else { "auto-detected from hardware" }
    Write-Host "Starting worker '$WorkerName' ($backendLabel) -- llama.cpp builds will install to $LlamaCppDir on first run, models download to $ModelsDir from the Models page."
}
npm run worker
Pop-Location
