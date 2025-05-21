// WebGPU bindings in JavaScript
let deviceID = 0;
let checkResults = false;



//const THREADS = [32, 64, 128, 256]

// const THREADS = [32, 64, 128, 256]

// const WORKGROUPS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

// const BATCH_SIZES = [1, 2, 4]

// const PAR_LOOKBACK = [1, 0]

const THREADS = [256]
const WORKGROUPS = [8192]
const BATCH_SIZES = [4]
const PAR_LOOKBACK = [1]

let VEC_SIZES = {
  [1 << 10]: [], [1 << 11]: [], [1 << 12]: [],
  [1 << 13]: [], [1 << 14]: [], [1 << 15]: [],
  [1 << 16]: [], [1 << 17]: [], [1 << 18]: [],
  [1 << 19]: [], [1 << 20]: [], [1 << 21]: [],
  [1 << 22]: [], [1 << 23]: [], [1 << 24]: [],
  [1 << 25]: []
};



const PER_THREAD_SIZE = 4;






async function deviceLostCallback(reason, message) {
  console.log(`Device lost: reason ${reason}`);
  if (message) console.log(` (message: ${message})`);
}

async function loadShader(device, path, TUNING_CONFIG) {
  const response = await fetch(path);
  let shaderSource = await response.text();
  let fullShaderSource =  `enable subgroups;\ndiagnostic(off, subgroup_uniformity);\nconst BATCH_SIZE = ${TUNING_CONFIG.batch_size};\n` + shaderSource;

  //fullShaderSource += shaderSource.replace("const BATCH_SIZE = 4;", `const BATCH_SIZE = ${BATCH_SIZE};`);

  const shaderModule = device.createShaderModule({ code: fullShaderSource });
  //console.log(fullShaderSource);
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

async function initBindGroup(device, bindGroupLayout, TUNING_CONFIG, buffers) {
  const entries = [];
  const vec_size = TUNING_CONFIG.numWorkgroups * TUNING_CONFIG.workgroupSize * TUNING_CONFIG.batch_size * PER_THREAD_SIZE;

  // AEntry
  if (buffers.ABuffer) {
    const AEntry = {
      binding: 0,
      resource: {
        buffer: buffers.ABuffer,
        offset: 0,
        size: vec_size * 4, // 4 bytes per int
      },
    };
    entries.push(AEntry);
  }

  // BEntry
  if (buffers.BBuffer) {
    const BEntry = {
      binding: 1,
      resource: {
        buffer: buffers.BBuffer,
        offset: 0,
        size: TUNING_CONFIG.numWorkgroups * 4, // 4 bytes per int
      },
    };
    entries.push(BEntry);
  }

  // CEntry
  if (buffers.CBuffer) {
    const CEntry = {
      binding: 2,
      resource: {
        buffer: buffers.CBuffer,
        offset: 0,
        size: vec_size * 4, // 4 bytes per int
      },
    };
    entries.push(CEntry);
  }

  // DEntry
  if (buffers.DBuffer) {
    const DEntry = {
      binding: 3,
      resource: {
        buffer: buffers.DBuffer,
        offset: 0,
        size: 4, // 4 bytes per int
      },
    };
    entries.push(DEntry);
  }

  // DebugEntry
  if (buffers.debugBuffer) {
    const debugEntry = {
      binding: 4,
      resource: {
        buffer: buffers.debugBuffer,
        offset: 0,
        size: 4 * TUNING_CONFIG.debug_size, // 4 bytes per int
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

async function initComputePipeline(device, bindGroupLayout, TUNING_CONFIG) {
  const shaderModule = await loadShader(device, 'prefix-sum.wgsl', TUNING_CONFIG);

  if (!Number.isFinite(TUNING_CONFIG.workgroupSize)) {
    throw new Error(`Invalid workgroupSize: ${TUNING_CONFIG.workgroupSize}`);
  }

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: 'prefix_sum',
      constants: {
        wg_size: TUNING_CONFIG.workgroupSize,
      },
    },
  });
  return pipeline
}

async function initBuffers(device, TUNING_CONFIG) {
  const vec_size = TUNING_CONFIG.numWorkgroups * TUNING_CONFIG.workgroupSize * TUNING_CONFIG.batch_size * PER_THREAD_SIZE;
  //console.log("size: ", vec_size);

  const ABuffer = device.createBuffer({
    mappedAtCreation: false,
    size: vec_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const BBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: TUNING_CONFIG.numWorkgroups * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const CBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: vec_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const CReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: vec_size * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const DBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const debugBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: TUNING_CONFIG.debug_size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const debugReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: TUNING_CONFIG.debug_size * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const TimestampResolveBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 2 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  const TimestampReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 2 * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    ABuffer,
    BBuffer,
    CBuffer,
    CReadBuffer,
    DBuffer,
    debugBuffer,
    debugReadBuffer,
    TimestampResolveBuffer,
    TimestampReadBuffer,
  };
}

async function run(device, pipeline, bindGroup, TUNING_CONFIG, buffers) {
  const vec_size = TUNING_CONFIG.numWorkgroups * TUNING_CONFIG.workgroupSize * TUNING_CONFIG.batch_size * PER_THREAD_SIZE;
  //console.log("workgroups: ", TUNING_CONFIG.numWorkgroups)
  //console.log("threads: ", TUNING_CONFIG.workgroupSize)
  //console.log("batch_size: ", TUNING_CONFIG.batch_size)
  const queue = device.queue;
  //const { buffers.ABuffer, buffers.BBuffer, buffers.CBuffer, buffers.CReadBuffer, buffers.DBuffer, buffers.debugBuffer, buffers.debugReadBuffer, buffers.TimestampResolveBuffer, buffers.TimestampReadBuffer } = await initBuffers(device);

  const A_host = Array(vec_size).fill(TUNING_CONFIG.alt);
  const B_host = Array(TUNING_CONFIG.numWorkgroups).fill(0);
  const D_host = [0];
  const debug_host = [TUNING_CONFIG.par_lookback, 0];

  await queue.writeBuffer(buffers.ABuffer, 0, new Uint32Array(A_host));
  await queue.writeBuffer(buffers.BBuffer, 0, new Uint32Array(B_host));
  await queue.writeBuffer(buffers.DBuffer, 0, new Uint32Array(D_host));
  await queue.writeBuffer(buffers.debugBuffer, 0, new Uint32Array(debug_host));

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
  computePass.dispatchWorkgroups(TUNING_CONFIG.numWorkgroups, 1, 1);
  computePass.end();

  encoder.copyBufferToBuffer(buffers.CBuffer, 0, buffers.CReadBuffer, 0, vec_size * 4);
  encoder.copyBufferToBuffer(buffers.debugBuffer, 0, buffers.debugReadBuffer, 0, TUNING_CONFIG.debug_size * 4);
  encoder.resolveQuerySet(querySet, 0, 2, buffers.TimestampResolveBuffer, 0);
  encoder.copyBufferToBuffer(buffers.TimestampResolveBuffer, 0, buffers.TimestampReadBuffer, 0, 2 * 8);
  
  
  const computeCommands = encoder.finish();
  queue.submit([computeCommands]);
  // Wait for the results
  await device.queue.onSubmittedWorkDone();  // Make sure GPU has finished all tasks

  await buffers.CReadBuffer.mapAsync(GPUMapMode.READ, 0, vec_size * 4);
  const output = new Uint32Array(buffers.CReadBuffer.getMappedRange());
  
  await buffers.debugReadBuffer.mapAsync(GPUMapMode.READ, 0, TUNING_CONFIG.debug_size * 4);
  const debugOut = new Uint32Array(buffers.debugReadBuffer.getMappedRange());
  
  await buffers.TimestampReadBuffer.mapAsync(GPUMapMode.READ, 0, 2 * 8);
  const timestampOutput = new BigUint64Array(buffers.TimestampReadBuffer.getMappedRange());
  
  duration = Date.now() - start;
  let incorrect = 0;
  if (output[vec_size - 1] == vec_size * TUNING_CONFIG.alt) {
    console.log("Succesful.")
  }else{
    incorrect = 1;
    console.log("real: ", output[vec_size - 1], "ideal: ", vec_size * TUNING_CONFIG.alt)
    console.log("There was an incorrect value(s).")
  }

  const time = timestampOutput[1] - timestampOutput[0];
  //console.log('Execution Time: ', time, 'ticks (ns)');
  //console.log(typeof(time))
  //console.log("Throughput: ", (vec_size * 4 * 2)/(time), " GBPS\n")

  const timeInSeconds = Number(time) / 1e9; // Convert BigInt nanoseconds to seconds
  const bytesTransferred = vec_size * 4 * 2; // Assuming 4 bytes per element, and 2 passes
  const gigabytesTransferred = bytesTransferred / 1e9; // Convert bytes to gigabytes                                                 

  const throughput = gigabytesTransferred / timeInSeconds

  //console.log("Date.now duration: ", duration, "ns?")
  //console.log("Throughput: ", throughput.toFixed(5), " GBPS");
  console.log("Throughput: ", throughput, " GBPS");
  
  document.getElementById("throughput-display").innerText = `Throughput: ${throughput} GBPS`;


  if (checkResults) {
    for (let i = 1; i < vec_size; i++) {
      console.log(`output[${i - 1}]: ${output[i - 1]}`);
    }
  }
  return [throughput, incorrect];
}



async function main() {

  const ITERS = 2;
  const WARM_UPS = 2;
  
  for (let i = 0; i < THREADS.length; i++) {
    for (let j = 0; j < WORKGROUPS.length; j++) {
      for (let k = 0; k < BATCH_SIZES.length; k++) {
        let size = THREADS[i] * WORKGROUPS[j] * BATCH_SIZES[k] * PER_THREAD_SIZE;
        if (size > 1 << 25) {
          continue;
        }
        for (let l = 0; l < PAR_LOOKBACK.length; l++) {
          let incorrect = 0;
          let throughput = 0;
          //console.log("vec_size: ", size)
          for (let p = 1; p <= ITERS + WARM_UPS; p++) { // iters PLUS warmups to get warmups
              if (p >= WARM_UPS) {
                // througput, threads, workgroups, batch_size, par_lookback
                //console.log("yomain")
                const [t, inc] = await main_helper(THREADS[i], WORKGROUPS[j], BATCH_SIZES[k], PAR_LOOKBACK[l], p);
                //console.log("gurt")
                throughput += t;
                incorrect += inc;
                
              }else{
                //console.log("yohelper")
                await main_helper(THREADS[i], WORKGROUPS[j], BATCH_SIZES[k], PAR_LOOKBACK[l], p)
                //console.log("gurt")
              }
          }
          VEC_SIZES[size].push([throughput / (ITERS + 1), THREADS[i], WORKGROUPS[j], BATCH_SIZES[k], PAR_LOOKBACK[l], incorrect])
        }
      }
    }
  }
  
  for (let i = 10; i < 26; i++) {
    VEC_SIZES[1 << i].sort((a, b) => b[0] - a[0])
    console.log("vec_Size: ", 1 << i)
    console.log(VEC_SIZES[1 << i][0])
  }
}


async function main_helper(thx, wkrgx, btchsx, plbkx, p) {

  const TUNING_CONFIG = {
    workgroupSize: thx,
    numWorkgroups: wkrgx,
    batch_size: btchsx,
    par_lookback: plbkx,
    alt: p,
    debug_size: 2
  };
  
  const requiredFeatures = ["timestamp-query", "subgroups"];

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
    requiredFeatures: requiredFeatures,
});
  device.lost.then(info => {
    console.error("Device lost:", info.message);
  }); 

  // // Check if 'timestamp-query' and 'subgroup' are supported
  // if (adapter.features.has('timestamp-query')) {
  //   console.log('Timestamp queries are supported');
  // } else {
  //   console.log('Timestamp queries are not supported on this device.');
  // }

  // if (adapter.features.has('subgroups')) {
  //   console.log('Subgroup operations are supported');
  // } else {
  //   console.log('Subgroup operations are not supported on this device.');
  // }

  // Dummy error callback for uncaptured WebGPU errors
  device.onuncapturederror = (event) => {
    console.error("Uncaptured error:", event.error);
  };

  // Initialize all WebGPU components
  
  const bindGroupLayout = await initBindGroupLayout(device, TUNING_CONFIG);
  const buffers = await initBuffers(device, TUNING_CONFIG);
  const bindGroup = await initBindGroup(device, bindGroupLayout, TUNING_CONFIG, buffers);
  const pipeline = await initComputePipeline(device, bindGroupLayout, TUNING_CONFIG);
  // Run the compute pass (dispatching the work to the GPU)
  let [throughput, incorrect] = await run(device, pipeline, bindGroup, TUNING_CONFIG, buffers);
  //console.log("tp: ", throughput, "inc: ", incorrect)
  return [throughput, incorrect];
}


main();