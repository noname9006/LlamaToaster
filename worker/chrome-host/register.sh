#!/bin/sh
# Registers this worker's VRAM host with Chrome, Chromium, Brave and Edge so the
# LlamaToaster Chrome extension can read VRAM -- installed before or after the
# worker, either order works. Current user only. Called by setup-worker.sh and
# update-worker.sh; safe to re-run. --uninstall removes it.
set -e
NAME=com.llamatoaster.vram
# Every LlamaToaster extension ID allowed to call the host (see register.ps1).
ORIGINS='"chrome-extension://odghfahbjacleniojhemnojbiabfebon/"'
DIR=$(cd "$(dirname "$0")" && pwd)
UNINSTALL=0
[ "${1:-}" = --uninstall ] && UNINSTALL=1

if [ "$(uname)" = Darwin ]; then
  BASE="$HOME/Library/Application Support"
  set -- "Google/Chrome" "Chromium" "BraveSoftware/Brave-Browser" "Microsoft Edge"
else
  BASE="$HOME/.config"
  set -- "google-chrome" "chromium" "BraveSoftware/Brave-Browser" "microsoft-edge"
fi

if [ "$UNINSTALL" = 0 ]; then
  # Chrome starts the host with a minimal PATH (no nvm/fnm shims), so pin the
  # node this install runs with.
  NODE=$(command -v node)
  [ -n "$NODE" ] || { echo "register.sh: node not found" >&2; exit 1; }
  printf '#!/bin/sh\nexec "%s" "%s/vram-host.mjs" "$@"\n' "$NODE" "$DIR" > "$DIR/vram-host.sh"
  chmod +x "$DIR/vram-host.sh"
fi

for b in "$@"; do
  T="$BASE/$b/NativeMessagingHosts"
  if [ "$UNINSTALL" = 1 ]; then rm -f "$T/$NAME.json"; continue; fi
  mkdir -p "$T"
  printf '{"name":"%s","description":"LlamaToaster VRAM host","path":"%s","type":"stdio","allowed_origins":[%s]}\n' \
    "$NAME" "$DIR/vram-host.sh" "$ORIGINS" > "$T/$NAME.json"
done
