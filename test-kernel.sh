#!/bin/bash

# Start Python HTTP server in background
echo "Starting Python HTTP server on port 8000..."
python3 -m http.server 8000 &
SERVER_PID=$!

# Wait briefly to ensure the server starts
sleep 1

# Launch Google Chrome with specified flags
echo "Launching Google Chrome to http://localhost:8000..."
google-chrome \
  --no-sandbox \
  --ignore-gpu-blocklist \
  --enable-gpu-rasterization \
  --disable-software-rasterizer \
  --disable-gpu-driver-bug-workarounds \
  --enable-unsafe-webgpu \
  --enable-features=WebGPU \
  --disable-dawn-features=timestamp_quantization \
  http://localhost:8000