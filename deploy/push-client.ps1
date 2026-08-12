# Rebuilds the client and copies client/dist/ to the VPS, then restarts the
# server via pm2. Run this from anywhere -- it resolves paths off its own
# location, not the caller's working directory.
#
# Connects over the VPS's Tailscale address (not its public IP) using
# password auth -- ssh/scp will prompt for the password interactively when
# you run this, since no -i identity file is configured.
#
# Usage:
#   .\deploy\push-client.ps1

$VpsUser = "ubuntu"
$VpsHost = "100.122.1.111"
$RemoteClientDir = "~/LlamaToaster/client/"
$PmApp = "llamatoaster-server"

$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Building client..."
Push-Location $RepoRoot
npm run build
$buildExit = $LASTEXITCODE
Pop-Location
if ($buildExit -ne 0) {
    Write-Error "Build failed (exit $buildExit) -- not copying anything."
    exit 1
}

Write-Host "Copying client/dist to VPS (you'll be prompted for the password)..."
scp -r "$RepoRoot\client\dist" "${VpsUser}@${VpsHost}:${RemoteClientDir}"
if ($LASTEXITCODE -ne 0) {
    Write-Error "scp failed (exit $LASTEXITCODE)."
    exit 1
}

Write-Host "Restarting $PmApp via pm2 (you'll be prompted for the password again)..."
ssh "${VpsUser}@${VpsHost}" "pm2 restart $PmApp"

Write-Host "Done. Hard-refresh the browser to see the change."
