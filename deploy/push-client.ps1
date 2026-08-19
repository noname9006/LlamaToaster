# Rebuilds the client + admin SPAs and copies client/dist/ and admin/dist/ to
# the VPS, then restarts the server via pm2. Run this from anywhere -- it
# resolves paths off its own location, not the caller's working directory.
#
# Connects over the VPS's Tailscale address by default (password auth --
# ssh/scp will prompt interactively, since no -i identity file is
# configured). Once the app is public (MULTIUSER_PLAN.md §6.1), the VPS is
# still reachable over the tailnet for operator access, so that stays the
# default -- but pass -PublicHost to deploy over the public hostname instead
# (e.g. if you're somewhere without a working Tailscale connection):
#
# Usage:
#   .\deploy\push-client.ps1
#   .\deploy\push-client.ps1 -PublicHost llamatoaster.com
#   .\deploy\push-client.ps1 -VpsHost 203.0.113.10 -VpsUser deploy

param(
    [string]$VpsHost = "100.122.1.111",
    [string]$PublicHost,
    [string]$VpsUser = "ubuntu",
    [string]$PmApp = "llamatoaster-server"
)

if ($PublicHost) { $VpsHost = $PublicHost }

$RemoteClientDir = "~/LlamaToaster/client/"
$RemoteAdminDir = "~/LlamaToaster/admin/"

$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Building client + admin..."
Push-Location $RepoRoot
npm run build
$clientBuildExit = $LASTEXITCODE
if ($clientBuildExit -eq 0) {
    npm run build:admin
    $adminBuildExit = $LASTEXITCODE
} else {
    $adminBuildExit = 1
}
Pop-Location
if ($clientBuildExit -ne 0 -or $adminBuildExit -ne 0) {
    Write-Error "Build failed (client exit $clientBuildExit, admin exit $adminBuildExit) -- not copying anything."
    exit 1
}

Write-Host "Copying client/dist to $VpsHost (you'll be prompted for the password)..."
scp -r "$RepoRoot\client\dist" "${VpsUser}@${VpsHost}:${RemoteClientDir}"
if ($LASTEXITCODE -ne 0) {
    Write-Error "scp (client) failed (exit $LASTEXITCODE)."
    exit 1
}

Write-Host "Copying admin/dist to $VpsHost (you'll be prompted for the password again)..."
scp -r "$RepoRoot\admin\dist" "${VpsUser}@${VpsHost}:${RemoteAdminDir}"
if ($LASTEXITCODE -ne 0) {
    Write-Error "scp (admin) failed (exit $LASTEXITCODE)."
    exit 1
}

Write-Host "Restarting $PmApp via pm2 (you'll be prompted for the password again)..."
ssh "${VpsUser}@${VpsHost}" "pm2 restart $PmApp"

Write-Host "Done. Hard-refresh the browser to see the change."
