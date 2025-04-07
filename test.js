async function runWebGPU() {
    if (!navigator.gpu) {
      console.error("WebGPU is not supported on this browser.");
      return;
    }
  
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    console.log("WebGPU device created:", device);
  
    // Example: Create a simple GPU buffer
    const buffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
    });
    console.log("GPU buffer created:", buffer);
  }
  
  runWebGPU();