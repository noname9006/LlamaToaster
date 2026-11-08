# Registers this worker's VRAM host with Chrome, Edge, Brave and Chromium so the
# LlamaToaster Chrome extension can read VRAM -- installed before or after the
# worker, either order works (Chrome reads the registration on every call).
# Current user only, no admin. Called by setup-worker.ps1 and update-worker.ps1;
# safe to re-run. -Uninstall removes it.
param([switch]$Uninstall)
$ErrorActionPreference = 'Stop'
$name = 'com.llamatoaster.vram'
# Every LlamaToaster extension ID allowed to call the host: the unpacked build
# (fixed by the "key" in manifest.json of the LlamaToaster/extension repo). Add the Chrome Web Store ID
# here if it is ever published under a different key.
$extensionIds = @('odghfahbjacleniojhemnojbiabfebon')
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Join-Path $dir "$name.json"
$browsers = 'Google\Chrome', 'Microsoft\Edge', 'BraveSoftware\Brave-Browser', 'Chromium'

if ($Uninstall) {
  foreach ($b in $browsers) { Remove-Item "HKCU:\Software\$b\NativeMessagingHosts\$name" -ErrorAction SilentlyContinue }
  Remove-Item $manifest -ErrorAction SilentlyContinue
  exit 0
}

[ordered]@{
  name            = $name
  description     = 'LlamaToaster VRAM host'
  path            = (Join-Path $dir 'vram-host.bat')
  type            = 'stdio'
  allowed_origins = @($extensionIds | ForEach-Object { "chrome-extension://$_/" })
} | ConvertTo-Json | Set-Content -Encoding ascii $manifest

foreach ($b in $browsers) {
  $key = "HKCU:\Software\$b\NativeMessagingHosts\$name"
  New-Item -Path $key -Force | Out-Null
  Set-ItemProperty -Path $key -Name '(default)' -Value $manifest
}
