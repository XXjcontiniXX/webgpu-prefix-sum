// WebGPU bindings in JavaScript
const workgroupSize = 128;
const numWorkgroups = 2;
const BATCH_SIZE = 2;
let deviceID = 0;
let alt = 1;
let checkResults = false;
let vec_size;
let debug_size = 2;
let par_lookback = 1;
let bindGroupLayout;

let device, adapter;

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

async function initBindGroupLayout() {
  bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        resource: { buffer: { type: 'storage' } },
        visibility: GPUShaderStage.COMPUTE,
      },
      {
        binding: 1,
        resource: { buffer: { type: 'storage' } },
        visibility: GPUShaderStage.COMPUTE,
      },
      {
        binding: 2,
        resource: { buffer: { type: 'storage' } },
        visibility: GPUShaderStage.COMPUTE,
      },
      {
        binding: 3,
        resource: { buffer: { type: 'storage' } },
        visibility: GPUShaderStage.COMPUTE,
      },
      {
        binding: 4,
        resource: { buffer: { type: 'storage' } },
        visibility: GPUShaderStage.COMPUTE,
      },
    ],
  });
}

async function initBindGroup() {
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

  try {
    const bindGroup = device.createBindGroup(bindGroupDesc);
    return bindGroup;
  } catch (error) {
    console.error("Error creating bind group:", error);
    throw error;
  }
}

async function initComputePipeline() {
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

  return pipeline;
}

async function initBuffers() {
  // Initialize buffers (no need to return them)
  ABuffer = device.createBuffer({
    size: vec_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  BBuffer = device.createBuffer({
    size: numWorkgroups * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  CBuffer = device.createBuffer({
    size: vec_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  CReadBuffer = device.createBuffer({
    size: vec_size * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  DBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  debugBuffer = device.createBuffer({
    size: debug_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  debugReadBuffer = device.createBuffer({
    size: debug_size * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  TimestampResolveBuffer = device.createBuffer({
    size: 2 * 8, // size of u_long
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  TimestampReadBuffer = device.createBuffer({
    size: 2 * 8, // size of u_long
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
}

async function run() {
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

  const computePass = encoder.beginComputePass({ timestampWrites });
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup, 0, []);
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

  const time = timestampOutput[1] - timestampOutput[0];
  console.log('Execution Time: ', time);

  if (checkResults) {
    for (let i = 1; i < vec_size; i++) {
      console.log(`output[${i - 1}]: ${output[i - 1]}`);
    }
  }
}


async function main() {
  vec_size = numWorkgroups * workgroupSize * BATCH_SIZE * 4;

  // Check if WebGPU is available in the browser
  if (!navigator.gpu) {
    console.error("WebGPU not supported in this browser.");
    return;
  }

  // Request a high-performance adapter and enable the required features
  adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
    features: ['timestamp-query', 'subgroup'],  // Enable specific features
  });

  if (!adapter) {
    console.error("Failed to get a valid adapter.");
    return;
  }

  // Log adapter details
  console.log(`Adapter Name: ${adapter.name}`);
  console.log(`Adapter Vendor: ${adapter.vendor}`);
  console.log(`Adapter Description: ${adapter.description}`);


  console.log("Adapter Name:", adapter.name || "Unknown");
console.log("Adapter Vendor:", adapter.vendor || "Unknown");
console.log("Adapter Description:", adapter.description || "Unknown");

// Log other potentially useful information
console.log("Adapter Limits:", adapter.limits);
console.log("Is Fallback Adapter:", adapter.isFallbackAdapter);

  // Request the device from the adapter
  device = await adapter.requestDevice();
  device.lost.then(info => {
    console.error("Device lost:", info.message);
  });

  // Check if 'timestamp-query' and 'subgroup' are supported
  if (adapter.features.has('timestamp-query')) {
    console.log('Timestamp queries are supported');
  } else {
    console.log('Timestamp queries are not supported on this device.');
  }

  if (adapter.features.has('subgroup')) {
    console.log('Subgroup operations are supported');
  } else {
    console.log('Subgroup operations are not supported on this device.');
  }

  // Dummy error callback for uncaptured WebGPU errors
  device.onuncapturederror = (event) => {
    console.error("Uncaptured error:", event.error);
  };

  // Initialize all WebGPU components
  await initBindGroupLayout();
  await initBuffers();
  await initComputePipeline();
  await initBindGroup();

  // Run the compute pass (dispatching the work to the GPU)
  await run(device);
}

main();