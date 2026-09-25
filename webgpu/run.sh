#!/usr/bin/env bash
# Serve this directory and open the benchmark in Chrome with WebGPU enabled.
# Works on Linux, macOS and Windows (Git Bash). Override with PORT= or CHROME=.
set -euo pipefail

PORT="${PORT:-8000}"
cd "$(dirname "$0")"

find_chrome() {
  if [[ -n "${CHROME:-}" ]]; then
    [[ -x "$CHROME" ]] || { echo "CHROME is set but not executable: $CHROME" >&2; return 1; }
    printf '%s' "$CHROME"; return 0
  fi
  local c
  for c in \
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary" \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "$HOME/.cache/chrome-for-testing/chrome-linux64/chrome" \
    "/c/Users/${USERNAME:-}/AppData/Local/Google/Chrome SxS/Application/chrome.exe" \
    "/c/Program Files/Google/Chrome/Application/chrome.exe"
  do
    [[ -x "$c" ]] && { printf '%s' "$c"; return 0; }
  done
  for c in google-chrome-canary google-chrome google-chrome-stable chromium chrome; do
    command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return 0; }
  done
  return 1
}

if ! CHROME_BIN="$(find_chrome)"; then
  echo "No Chrome found. Install Chrome 144+ and re-run, or point at it:" >&2
  echo "  CHROME=/path/to/chrome ./run.sh" >&2
  exit 1
fi

python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
SERVER_PID=$!
# The original per-OS scripts leaked this server; a trap cleans it up on any exit.
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 1

FLAGS=(
  --ignore-gpu-blocklist
  --enable-gpu-rasterization
  --disable-software-rasterizer
  --disable-gpu-driver-bug-workarounds
  --enable-unsafe-webgpu
  # unquantized GPU timestamps, otherwise the timing is rounded into uselessness
  --disable-dawn-features=timestamp_quantization
)
# Linux reaches the discrete GPU through the Vulkan backend.
[[ "$(uname -s)" == "Linux" ]] && FLAGS+=(--use-angle=vulkan --enable-features=Vulkan)

echo "serving  http://localhost:$PORT"
echo "chrome   $CHROME_BIN"
echo "(results also print to the browser console; close Chrome to stop the server)"

# A fresh profile forces a new instance, so this blocks until that window closes
# instead of handing off to an already-running Chrome and returning immediately.
"$CHROME_BIN" --user-data-dir="$(mktemp -d)" --no-first-run --no-default-browser-check \
  "${FLAGS[@]}" "http://localhost:$PORT/index.html"
