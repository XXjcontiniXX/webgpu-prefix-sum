#!/usr/bin/env bash

set -e

# Start Python HTTP server in background
echo "Starting Python HTTP server on port 8007.."
python3 -m http.server 8007 &
SERVER_PID=$!

# Give the server a moment to start
sleep 1

# Chrome Canary path (Windows via Git Bash)
CHROME_CANARY="/c/Users/$USERNAME/AppData/Local/Google/Chrome SxS/Application/chrome.exe"

if [[ ! -f "$CHROME_CANARY" ]]; then
  echo "Chrome Canary not found at:"
  echo "   $CHROME_CANARY"
  echo "Install Chrome Canary first."
  kill $SERVER_PID
  exit 1
fi

echo "Launching Chrome Canary with WebGPU flags..."
"$CHROME_CANARY" \
  --no-sandbox \
  --ignore-gpu-blocklist \
  --enable-gpu-rasterization \
  --disable-software-rasterizer \
  --disable-gpu-driver-bug-workarounds \
  --enable-unsafe-webgpu \
  --enable-features=WebGPU \
  --disable-dawn-features=timestamp_quantization \
  http://localhost:8007 &

# Optional: wait for Chrome to exit, then clean up server
wait
echo "Stopping Python HTTP server..."
kill $SERVER_PID