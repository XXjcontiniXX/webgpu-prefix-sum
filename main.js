// WebGPU bindings in JavaScript
const workgroupSize = 256;
const numWorkgroups = 32768;
const BATCH_SIZE = 1;
let deviceID = 0;
let alt = 1;
let checkResults = false;
let vec_size;
let debug_size = 2;
let par_lookback = 1;

// Declare buffers globally (or at the top of your script)
let ABuffer, BBuffer, CBuffer, CReadBuffer, DBuffer, debugBuffer;
let debugReadBuffer, TimestampResolveBuffer, TimestampReadBuffer;

async function deviceLostCallback(reason, message) {
  console.log(`Device lost: reason ${reason}`);
  if (message) console.log(` (message: ${message})`);
}

async function loadShader(device, path) {
  const response = await fetch(path);
  const shaderSource = await response.text();

  let fullShaderSource = `enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n`;
  fullShaderSource += shaderSource.replace("const BATCH_SIZE = 4;", `const BATCH_SIZE = ${BATCH_SIZE};`);

  const shaderModule = device.createShaderModule({ code: fullShaderSource });
  return shaderModule;
}

async function initBindGroupLayout(device) {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  return bindGroupLayout;
}

async function initBindGroup(device, bindGroupLayout) {
  const entries = [];

  // AEntry
  if (ABuffer) {
    const AEntry = {
      binding: 0,
      resource: {
        buffer: ABuffer,
        offset: 0,
        size: vec_size * 4, // 4 bytes per int
      },
    };
    entries.push(AEntry);
  }

  // BEntry
  if (BBuffer) {
    const BEntry = {
      binding: 1,
      resource: {
        buffer: BBuffer,
        offset: 0,
        size: numWorkgroups * 4, // 4 bytes per int
      },
    };
    entries.push(BEntry);
  }

  // CEntry
  if (CBuffer) {
    const CEntry = {
      binding: 2,
      resource: {
        buffer: CBuffer,
        offset: 0,
        size: vec_size * 4, // 4 bytes per int
      },
    };
    entries.push(CEntry);
  }

  // DEntry
  if (DBuffer) {
    const DEntry = {
      binding: 3,
      resource: {
        buffer: DBuffer,
        offset: 0,
        size: 4, // 4 bytes per int
      },
    };
    entries.push(DEntry);
  }

  // DebugEntry
  if (debugBuffer) {
    const debugEntry = {
      binding: 4,
      resource: {
        buffer: debugBuffer,
        offset: 0,
        size: 4 * debug_size, // 4 bytes per int
      },
    };
    entries.push(debugEntry);
  }

  // Check that there are entries in the bind group
  if (entries.length === 0) {
    throw new Error("No valid entries in the bind group");
  }


  // BindGroupDescriptor
  const bindGroupDesc = {
    layout: bindGroupLayout,
    entries: entries,
  };
  const bindGroup = device.createBindGroup(bindGroupDesc);
  return bindGroup
}

async function initComputePipeline(device, bindGroupLayout) {
  const shaderModule = await loadShader(device, 'prefix-sum.wgsl');

  if (!Number.isFinite(workgroupSize)) {
    throw new Error(`Invalid workgroupSize: ${workgroupSize}`);
  }

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: 'prefix_sum',
      constants: {
        wg_size: workgroupSize,
      },
    },
  });
  return pipeline
}

async function initBuffers(device) {
  // Initialize buffers (no need to return them)
  ABuffer = device.createBuffer({
    mappedAtCreation: false,
    size: vec_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  BBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: numWorkgroups * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  CBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: vec_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  CReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: vec_size * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  DBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  debugBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: debug_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  debugReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: debug_size * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  TimestampResolveBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 2 * 8, // size of u_long
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  TimestampReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 2 * 8, // size of u_long
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
}

async function run(device, pipeline, bindGroup) {
  const queue = device.queue;
  //const { ABuffer, BBuffer, CBuffer, CReadBuffer, DBuffer, debugBuffer, debugReadBuffer, TimestampResolveBuffer, TimestampReadBuffer } = await initBuffers(device);

  const A_host = Array(vec_size).fill(alt);
  const B_host = Array(numWorkgroups).fill(0);
  const D_host = [0];
  const debug_host = [par_lookback, 0];

  queue.writeBuffer(ABuffer, 0, new Uint32Array(A_host));
  queue.writeBuffer(BBuffer, 0, new Uint32Array(B_host));
  queue.writeBuffer(DBuffer, 0, new Uint32Array(D_host));
  queue.writeBuffer(debugBuffer, 0, new Uint32Array(debug_host));

  const encoder = device.createCommandEncoder();

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const timestampWrites = {
    querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  };
  const start = Date.now();
  const computePass = encoder.beginComputePass({ timestampWrites });
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup);
  computePass.dispatchWorkgroups(numWorkgroups, 1, 1);
  computePass.end();

  encoder.copyBufferToBuffer(CBuffer, 0, CReadBuffer, 0, vec_size * 4);
  encoder.copyBufferToBuffer(debugBuffer, 0, debugReadBuffer, 0, debug_size * 4);
  encoder.resolveQuerySet(querySet, 0, 2, TimestampResolveBuffer, 0);
  encoder.copyBufferToBuffer(TimestampResolveBuffer, 0, TimestampReadBuffer, 0, 2 * 8);
  
  const computeCommands = encoder.finish();
  queue.submit([computeCommands]);

  // Wait for the results
  await CReadBuffer.mapAsync(GPUMapMode.READ, 0, vec_size * 4);
  const output = new Uint32Array(CReadBuffer.getMappedRange());

  await debugReadBuffer.mapAsync(GPUMapMode.READ, 0, debug_size * 4);
  const debugOut = new Uint32Array(debugReadBuffer.getMappedRange());
 
  await TimestampReadBuffer.mapAsync(GPUMapMode.READ, 0, 2 * 8);
  const timestampOutput = new BigUint64Array(TimestampReadBuffer.getMappedRange());
  duration = Date.now() - start;
  if (output[vec_size - 1] == vec_size * alt) {
    console.log("Succesful.")
  }else{
    console.log("There was an incorrect value.")
  }

  const time = timestampOutput[1] - timestampOutput[0];
  console.log('Execution Time: ', time, 'ticks (ns)');
  //console.log(typeof(time))
  //console.log("Throughput: ", (vec_size * 4 * 2)/(time), " GBPS\n")

  const timeInSeconds = Number(time) / 1e9; // Convert BigInt nanoseconds to seconds
  const bytesTransferred = vec_size * 4 * 2; // Assuming 4 bytes per element, and 2 passes
  const gigabytesTransferred = bytesTransferred / 1e9; // Convert bytes to gigabytes                                                 

  const throughput = gigabytesTransferred / timeInSeconds

  console.log("Date.now duration: ", duration, "ns?")
  //console.log("Throughput: ", throughput.toFixed(5), " GBPS");
  console.log("Throughput: ", throughput, " GBPS");
  
  document.getElementById("throughput-display").innerText = `Throughput: ${throughput} GBPS`;


  if (checkResults) {
    for (let i = 1; i < vec_size; i++) {
      console.log(`output[${i - 1}]: ${output[i - 1]}`);
    }
  }
}


async function main() {
  //await new Promise(r => setTimeout(r, 200000));
  vec_size = numWorkgroups * workgroupSize * BATCH_SIZE * 4;
  const requiredFeatures = ["timestamp-query", "subgroup"];

  // Check if WebGPU is available in the browser
  if (!navigator.gpu) {
    document.getElementById("webgpu-suppported").innerText = `WebGPU support: not supported`;
    console.error("WebGPU not supported in this browser.");
    return;
  }else{
    document.getElementById("webgpu-suppported").innerText = `WebGPU support: supported`;
  }

  

  // Request a high-performance adapter and enable the required features
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) {
    console.error("Failed to get a valid adapter.");
    return;
  }

//   // Log adapter details
//   console.log(`Adapter Name: ${adapter.name}`);
//   console.log(`Adapter Vendor: ${adapter.vendor}`);
//   console.log(`Adapter Description: ${adapter.description}`);


//   console.log("Adapter Name:", adapter.name || "Unknown");
// console.log("Adapter Vendor:", adapter.vendor || "Unknown");
// console.log("Adapter Description:", adapter.description || "Unknown");

// // Log other potentially useful information
// console.log("Adapter Limits:", adapter.limits);
// console.log("Is Fallback Adapter:", adapter.isFallbackAdapter);

  // Request the device from the adapter
  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query", "subgroups"],
});
  device.lost.then(info => {
    console.error("Device lost:", info.message);
  }); 

  // Check if 'timestamp-query' and 'subgroup' are supported
  if (adapter.features.has('timestamp-query')) {
    console.log('Timestamp queries are supported');
  } else {
    console.log('Timestamp queries are not supported on this device.');
  }

  if (adapter.features.has('subgroups')) {
    console.log('Subgroup operations are supported');
  } else {
    console.log('Subgroup operations are not supported on this device.');
  }

  // Dummy error callback for uncaptured WebGPU errors
  device.onuncapturederror = (event) => {
    console.error("Uncaptured error:", event.error);
  };

  // Initialize all WebGPU components
  const bindGroupLayout = await initBindGroupLayout(device);
  await initBuffers(device);
  const bindGroup = await initBindGroup(device, bindGroupLayout);
  const pipeline = await initComputePipeline(device, bindGroupLayout);
  // Run the compute pass (dispatching the work to the GPU)
  await run(device, pipeline, bindGroup);
}

main();