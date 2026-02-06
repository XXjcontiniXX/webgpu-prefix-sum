#!/bin/bash

# Start Python HTTP server in background
echo "Starting Python HTTP server on port 8101..."
python3 -m http.server 8101 &
SERVER_PID=$!

# Wait briefly to ensure the server starts
sleep 1

# Launch Google Chrome Canary with WebGPU flags on macOS
echo "Launching Google Chrome Canary to http://localhost:8101..."

open -a "Google Chrome Canary" --args \
  --ignore-gpu-blocklist \
  --enable-gpu-rasterization \
  --disable-software-rasterizer \
  --disable-gpu-driver-bug-workarounds \
  --enable-unsafe-webgpu \
  --enable-features=WebGPU \
  --disable-dawn-features=timestamp_quantization \
  http://localhost:8101

# Wait for Chrome to exit
wait
echo "Stopping Python HTTP server..."
kill $SERVER_PID