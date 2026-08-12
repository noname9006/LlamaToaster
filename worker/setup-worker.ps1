# One-shot bootstrap for a worker that has nothing downloaded yet: no
# config.json, no llama.cpp binary, no models. You only supply one base
# folder -- llama.cpp builds and any future GGUF downloads get fetched
# automatically later (llama.cpp on first triggered run, models whenever you
# download one from the Models page) into <Dir>\llama and <Dir>\models. This
# script itself never downloads llama.cpp or a model.
#
# Usage (from anywhere), defaults to F:\LlamaToaster if -Dir is omitted:
#   .\worker\setup-worker.ps1
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
# Safe by default: refuses to touch an existing worker/config.json unless
# -Force is passed (no git in this repo checkout, so an overwritten config.json
# has no recovery path).

param(
    [string]$Dir = "F:\LlamaToaster",
    [string]$WorkerName = "Local",
    [string]$Backend,
    [string]$VpsUrl = "http://100.122.1.111:4010",
    [string]$BindHost,
    [int]$Port = 8080,
    [switch]$Force
)

$LlamaCppDir = Join-Path $Dir "llama"
$ModelsDir = Join-Path $Dir "models"

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $RepoRoot "worker\config.json"

if ((Test-Path $ConfigPath) -and -not $Force) {
    Write-Host "worker\config.json already exists -- not touching it. Current contents:" -ForegroundColor Yellow
    Get-Content $ConfigPath | Write-Host
    Write-Host "`nRe-run with -Force to overwrite it." -ForegroundColor Yellow
    exit 1
}

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

$backendLabel = if ($Backend) { $Backend } else { "auto-detected from hardware" }
Write-Host "Starting worker '$WorkerName' ($backendLabel) -- llama.cpp builds will install to $LlamaCppDir on first run, models download to $ModelsDir from the Models page."
npm run worker
Pop-Location
