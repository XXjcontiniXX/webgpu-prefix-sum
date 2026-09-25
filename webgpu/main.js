// WebGPU bindings in JavaScript
let checkResults = false;

const params = new URLSearchParams(location.search);
// Each tuning axis can be narrowed from the URL, e.g. ?threads=128&workgroups=2048
// which is handy for a quick smoke test before committing to the full sweep.
const list = (key, fallback) => {
  const v = params.get(key);
  return v ? v.split(",").map(Number) : fallback;
};

const THREADS = list("threads", [32, 64, 128, 256]);
const WORKGROUPS = list("workgroups", [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]);
const BATCH_SIZES = list("batch", [1, 2, 4]);
const PAR_LOOKBACK = list("lookback", [1, 0]);

const ALL_SHADERS = [
  { key: "v2", name: "prefix-sum-v2", path: "prefix-sum-v2.wgsl" },
  { key: "v1", name: "prefix-sum-v1", path: "prefix-sum-v1.wgsl" },
];
// ?shaders=v2 picks a subset; default runs both.
const SHADERS = (() => {
  const want = params.get("shaders");
  if (!want) return ALL_SHADERS;
  const keys = want.split(",");
  return keys.map(k => ALL_SHADERS.find(s => s.key === k)).filter(Boolean);
})();

// const THREADS = [128]
// const WORKGROUPS = [2048]
// const BATCH_SIZES = [4]
// const PAR_LOOKBACK = [1]

// Input sizes swept by the benchmark, as powers of two (inclusive range).
// 2^12 is the smallest size the tuning grid can produce (32 threads * 32 workgroups * 1 batch * 4),
// 2^25 is the cap enforced in the sweep below.
const MIN_POW = Number(params.get("minpow")) || 12;
const MAX_POW = Number(params.get("maxpow")) || 25;

let VEC_SIZES = {};

// Peak memory bandwidth of the GPU under test, for the reference line on the plot.
// Override per-machine with ?peak=<GB/s> (default is the RTX 5070's 672 GB/s).
const THEORETICAL_THROUGHPUT_GBPS = Number(params.get("peak")) || 672;
// Per-run wall-clock budget; a config that blows past it is recorded as failed
// instead of wedging the whole sweep.
const RUN_TIMEOUT_MS = Number(params.get("timeout")) || 120000;
// ?verify=1 checks every output element instead of just the last one. Slow; use for
// correctness runs, not for timing sweeps.
const VERIFY_FULL = params.get("verify") === "1";
const SHADER_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"];
const PLOT_CANVAS_ID = "throughput-chart";
let GPU_LABEL = params.get("gpu") || "GPU";



const PER_THREAD_SIZE = 4;






async function deviceLostCallback(reason, message) {
  console.log(`Device lost: reason ${reason}`);
  if (message) console.log(` (message: ${message})`);
}

async function loadShader(device, path, TUNING_CONFIG) {
  const response = await fetch(path);
  let shaderSource = await response.text();

  let num_subgroups = TUNING_CONFIG.workgroupSize / 32


  let fullShaderSource =  `enable subgroups;\nrequires subgroup_id;\ndiagnostic(off, subgroup_uniformity);\nconst NUM_SUBGROUPS = ${num_subgroups};\nconst BATCH_SIZE = ${TUNING_CONFIG.batch_size};\n` + shaderSource;

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

async function initComputePipeline(device, bindGroupLayout, TUNING_CONFIG, shaderPath) {
  const shaderModule = await loadShader(device, shaderPath, TUNING_CONFIG);

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

  // Typed arrays directly: at 2^25 elements the Array->Uint32Array conversion this
  // replaced dominated wall-clock time (it never affected the timestamped GPU pass).
  const A_host = new Uint32Array(vec_size).fill(TUNING_CONFIG.alt);
  const B_host = new Uint32Array(TUNING_CONFIG.numWorkgroups);
  const D_host = new Uint32Array([0]);
  const debug_host = new Uint32Array([TUNING_CONFIG.par_lookback, 0]);

  queue.writeBuffer(buffers.ABuffer, 0, A_host);
  queue.writeBuffer(buffers.BBuffer, 0, B_host);
  queue.writeBuffer(buffers.DBuffer, 0, D_host);
  queue.writeBuffer(buffers.debugBuffer, 0, debug_host);

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
  if (VERIFY_FULL) {
    // Element-wise: the last element alone can hide interior errors that cancel out.
    for (let i = 0; i < vec_size; i++) {
      if (output[i] !== (i + 1) * TUNING_CONFIG.alt) {
        incorrect = 1;
        console.log(`MISMATCH at ${i}: got ${output[i]} want ${(i + 1) * TUNING_CONFIG.alt}`
          + ` (wg=${TUNING_CONFIG.workgroupSize} ngroups=${TUNING_CONFIG.numWorkgroups}`
          + ` batch=${TUNING_CONFIG.batch_size} plb=${TUNING_CONFIG.par_lookback})`);
        break;
      }
    }
  } else if (output[vec_size - 1] != vec_size * TUNING_CONFIG.alt) {
    incorrect = 1;
  }

  const time = timestampOutput[1] - timestampOutput[0];
  //console.log('Execution Time: ', time, 'ticks (ns)');
  //console.log(typeof(time))
  //console.log("Throughput: ", (vec_size * 4 * 2)/(time), " GBPS\n")

  const timeInSeconds = Number(time) / 1e9; // Convert BigInt nanoseconds to seconds
  const bytesTransferred = vec_size * 4 * 2; // Assuming 4 bytes per element, and 2 passes
  const gigabytesTransferred = bytesTransferred / 1e9; // Convert bytes to gigabytes                                                 

  const throughput = gigabytesTransferred / timeInSeconds

  document.getElementById("throughput-display").innerText = `Throughput: ${throughput} GBPS`;


  if (checkResults) {
    for (let i = 1; i < vec_size; i++) {
      console.log(`output[${i - 1}]: ${output[i - 1]}`);
    }
  }

  buffers.CReadBuffer.unmap();
  buffers.debugReadBuffer.unmap();
  buffers.TimestampReadBuffer.unmap();
  querySet.destroy();
  return [throughput, incorrect];
}

function ensurePlotCanvas() {
  let canvas = document.getElementById(PLOT_CANVAS_ID);
  if (!canvas) {
    const container = document.createElement("div");
    container.style.maxWidth = "1000px";
    container.style.marginTop = "16px";

    const title = document.createElement("h2");
    title.textContent = `Shader Throughput (${GPU_LABEL})`;

    canvas = document.createElement("canvas");
    canvas.id = PLOT_CANVAS_ID;
    canvas.style.width = "100%";
    canvas.style.height = "500px";
    canvas.width = 1000;
    canvas.height = 500;

    const legend = document.createElement("div");
    legend.id = "throughput-legend";
    legend.style.fontFamily = "sans-serif";
    legend.style.fontSize = "12px";
    legend.style.marginTop = "6px";

    container.appendChild(title);
    container.appendChild(canvas);
    container.appendChild(legend);

    document.body.appendChild(container);
  }
  return canvas;
}

function updateLegend(resultsByShader) {
  const legend = document.getElementById("throughput-legend");
  if (!legend) return;
  legend.innerHTML = "";

  resultsByShader.forEach((shader, idx) => {
    const item = document.createElement("span");
    item.style.display = "inline-block";
    item.style.marginRight = "12px";
    item.style.color = SHADER_COLORS[idx % SHADER_COLORS.length];
    item.textContent = shader.name;
    legend.appendChild(item);
  });

  const line = document.createElement("span");
  line.style.display = "inline-block";
  line.style.marginLeft = "12px";
  line.style.color = "#d62728";
  line.textContent = `Theoretical limit (${THEORETICAL_THROUGHPUT_GBPS} GBPS)`;
  legend.appendChild(line);
}

function drawPlot(resultsByShader) {
  const canvas = ensurePlotCanvas();
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const logicalWidth = canvas.clientWidth || canvas.width;
  const logicalHeight = canvas.clientHeight || canvas.height;
  canvas.width = Math.floor(logicalWidth * dpr);
  canvas.height = Math.floor(logicalHeight * dpr);
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, logicalWidth, logicalHeight);

  const margin = { top: 30, right: 20, bottom: 60, left: 70 };
  const plotWidth = logicalWidth - margin.left - margin.right;
  const plotHeight = logicalHeight - margin.top - margin.bottom;

  const sizeSet = new Set();
  let maxThroughput = 0;
  resultsByShader.forEach(shader => {
    shader.data.forEach(point => {
      sizeSet.add(point.size);
      if (Number.isFinite(point.throughput)) {
        if (point.throughput > maxThroughput) maxThroughput = point.throughput;
      }
    });
  });
  const sizes = Array.from(sizeSet).sort((a, b) => a - b);
  if (sizes.length === 0) return;

  const yMax = Math.max(THEORETICAL_THROUGHPUT_GBPS, maxThroughput) * 1.05;
  const yMin = 0;

  const xForIndex = (i) => {
    if (sizes.length === 1) return margin.left + plotWidth / 2;
    return margin.left + (i / (sizes.length - 1)) * plotWidth;
  };
  const yForValue = (v) => {
    return margin.top + plotHeight - ((v - yMin) / (yMax - yMin)) * plotHeight;
  };

  // Axes
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  // Y ticks
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#333";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = (yMax / yTicks) * i;
    const y = yForValue(v);
    ctx.strokeStyle = "#e0e0e0";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
    ctx.fillStyle = "#333";
    ctx.fillText(v.toFixed(0), margin.left - 8, y);
  }

  // X labels
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  sizes.forEach((size, i) => {
    const x = xForIndex(i);
    const power = Math.log2(size);
    const label = Number.isFinite(power) ? `2^${power}` : `${size}`;
    ctx.fillText(label, x, margin.top + plotHeight + 8);
  });

  // Theoretical line
  const yTheory = yForValue(THEORETICAL_THROUGHPUT_GBPS);
  ctx.strokeStyle = "#d62728";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(margin.left, yTheory);
  ctx.lineTo(margin.left + plotWidth, yTheory);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#d62728";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`Theoretical ${THEORETICAL_THROUGHPUT_GBPS} GBPS`, margin.left + 4, yTheory - 2);

  // Series
  resultsByShader.forEach((shader, idx) => {
    const color = SHADER_COLORS[idx % SHADER_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    sizes.forEach((size, i) => {
      const point = shader.data.find(p => p.size === size);
      if (!point || !Number.isFinite(point.throughput)) return;
      const x = xForIndex(i);
      const y = yForValue(point.throughput);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Points
    sizes.forEach((size, i) => {
      const point = shader.data.find(p => p.size === size);
      if (!point || !Number.isFinite(point.throughput)) return;
      const x = xForIndex(i);
      const y = yForValue(point.throughput);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  updateLegend(resultsByShader);
}



async function main() {
  const WARMUPS = Number(params.get("warmups") ?? NaN) || 2;
  const RUNS = Number(params.get("runs")) || 5;

  // Check if WebGPU is available in the browser
  if (!navigator.gpu) {
    document.getElementById("webgpu-suppported").innerText = `WebGPU support: not supported`;
    console.error("WebGPU not supported in this browser.");
    return;
  } else {
    document.getElementById("webgpu-suppported").innerText = `WebGPU support: supported`;
  }

  if (!navigator.gpu.wgslLanguageFeatures.has("subgroup_id")) {
    throw new Error(`WGSL subgroup_id and num_subgroups built-in values are not available`);
  }

  const requiredFeatures = ["timestamp-query", "subgroups"];

  // A config whose decoupled-lookback spin never makes forward progress hangs the
  // GPU and takes the device down with it. Re-acquire adapter and device so the
  // sweep can skip that config and carry on instead of losing every later size.
  let deviceLost = null;
  let deviceEpoch = 0;
  async function acquireDevice() {
    deviceEpoch++;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error("Failed to get a valid adapter.");
    // Without requiredLimits a device gets WebGPU's defaults -- a 128 MiB storage
    // binding, which caps the sweep at 2^25 u32 and keeps every size inside this
    // card's L2. Ask for what the adapter actually supports so larger, genuinely
    // DRAM-bound sizes are reachable.
    const requiredLimits = {};
    for (const k of ["maxBufferSize", "maxStorageBufferBindingSize",
                     "maxComputeWorkgroupsPerDimension", "maxComputeInvocationsPerWorkgroup"]) {
      if (adapter.limits[k] !== undefined) requiredLimits[k] = adapter.limits[k];
    }
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    device.lost.then(info => {
      deviceLost = info.message || "unknown";
      console.error("Device lost:", info.message);
    });
    device.onuncapturederror = (event) => {
      console.error("Uncaptured error:", event.error);
    };
    return { adapter, device };
  }

  let { adapter, device } = await acquireDevice();
  console.log(adapter);

  const adapterInfo = adapter.info || {};
  if (!params.get("gpu")) {
    GPU_LABEL = [adapterInfo.vendor, adapterInfo.architecture].filter(Boolean).join(" ") || "GPU";
  }
  window.__BENCH_META__ = {
    gpu: GPU_LABEL,
    vendor: adapterInfo.vendor,
    architecture: adapterInfo.architecture,
    subgroupMinSize: adapterInfo.subgroupMinSize,
    subgroupMaxSize: adapterInfo.subgroupMaxSize,
    peakGBps: THEORETICAL_THROUGHPUT_GBPS,
    warmups: WARMUPS,
    runs: RUNS,
    minPow: MIN_POW,
    maxPow: MAX_POW,
  };

  const resultsByShader = [];
  const allConfigs = [];
  const failures = [];

  // Count the configs up front so progress logging is meaningful.
  const configs = [];
  for (const workgroupSize of THREADS) {
    for (const numWorkgroups of WORKGROUPS) {
      for (const batch_size of BATCH_SIZES) {
        const size = workgroupSize * numWorkgroups * batch_size * PER_THREAD_SIZE;
        if (size < (1 << MIN_POW) || size > (1 << MAX_POW)) continue;
        for (const par_lookback of PAR_LOOKBACK) {
          configs.push({ size, workgroupSize, numWorkgroups, batch_size, par_lookback });
        }
      }
    }
  }
  const totalConfigs = configs.length * SHADERS.length;
  console.log(`PROGRESS total_configs=${totalConfigs} (${configs.length} per shader x ${SHADERS.length} shaders), ${WARMUPS} warmups + ${RUNS} timed runs each`);

  let done = 0;
  let recoveries = 0;
  const MAX_RECOVERIES = Number(params.get("maxrecoveries")) || 40;

  // After a device loss the replacement device sometimes serves large buffers out
  // of system memory instead of VRAM, which reads as ~26 GB/s (about PCIe x16) and
  // silently poisons every later measurement. Re-measure a fixed reference config
  // after each recovery and reject the device if it comes back degraded.
  const CANARY = { workgroupSize: 256, numWorkgroups: 1024, batch_size: 4, par_lookback: 1 };
  const CANARY_FLOOR = 0.6;
  let canaryBaseline = null;

  async function canary() {
    try {
      const [t] = await withTimeout(
        main_helper(device, CANARY.workgroupSize, CANARY.numWorkgroups, CANARY.batch_size,
                    CANARY.par_lookback, 1, SHADERS[0].path),
        RUN_TIMEOUT_MS);
      return t;
    } catch (e) { return null; }
  }

  // Bring the device back after a hang; returns false once we've given up.
  async function recoverDevice(why) {
    if (recoveries >= MAX_RECOVERIES) {
      console.log(`RECOVER giving up after ${recoveries} device recoveries`);
      return false;
    }
    recoveries++;
    console.log(`RECOVER #${recoveries} recreating device (${why})`);
    deviceLost = null;
    ({ adapter, device } = await acquireDevice());

    // Confirm the new device performs like the original before trusting its numbers.
    for (let attempt = 0; attempt < 3; attempt++) {
      const t = await canary();
      if (canaryBaseline === null) { canaryBaseline = t; break; }
      if (t !== null && t >= canaryBaseline * CANARY_FLOOR) {
        console.log(`CANARY ok epoch=${deviceEpoch} ${t.toFixed(1)} GB/s (baseline ${canaryBaseline.toFixed(1)})`);
        break;
      }
      console.log(`CANARY DEGRADED epoch=${deviceEpoch} ${t === null ? "failed" : t.toFixed(1) + " GB/s"} vs baseline ${canaryBaseline.toFixed(1)} — recreating device`);
      deviceLost = null;
      ({ adapter, device } = await acquireDevice());
    }
    return true;
  }

  // The first dispatch in a fresh page reliably loses the Dawn instance once in this
  // headless setup, which would otherwise cost us whichever config happens to be
  // first. Spend a throwaway dispatch absorbing it.
  for (let attempt = 0; attempt < 3; attempt++) {
    try { await canary(); } catch (e) { /* discarded */ }
    if (!deviceLost) break;
    console.log(`WARMUP absorbing startup device loss: ${deviceLost}`);
    deviceLost = null;
    ({ adapter, device } = await acquireDevice());
  }

  canaryBaseline = await canary();
  console.log(`CANARY baseline ${canaryBaseline === null ? "unavailable" : canaryBaseline.toFixed(1) + " GB/s"}`);

  for (const shader of SHADERS) {
    for (let i = MIN_POW; i <= MAX_POW; i++) {
      VEC_SIZES[1 << i] = [];
    }

    for (const cfg of configs) {
      if (deviceLost && !(await recoverDevice(deviceLost))) break;

      let best = -Infinity;
      let bestCorrect = -Infinity;
      let incorrect = 0;
      let error = null;

      for (let r = 0; r < WARMUPS + RUNS; r++) {
        try {
          const [t, inc] = await withTimeout(
            main_helper(device, cfg.workgroupSize, cfg.numWorkgroups, cfg.batch_size, cfg.par_lookback, 1, shader.path),
            RUN_TIMEOUT_MS
          );
          if (r >= WARMUPS) {
            if (t > best) best = t;
            if (inc === 0 && t > bestCorrect) bestCorrect = t;
            incorrect += inc;
          }
        } catch (e) {
          error = String(e && e.message ? e.message : e);
          break;
        }
        if (deviceLost) { error = "device lost: " + deviceLost; break; }
      }

      done++;
      if (error) {
        failures.push({ shader: shader.name, ...cfg, error });
        console.log(`PROGRESS ${done}/${totalConfigs} FAILED ${shader.name} size=2^${Math.log2(cfg.size)} wg=${cfg.workgroupSize} ngroups=${cfg.numWorkgroups} batch=${cfg.batch_size} plb=${cfg.par_lookback} :: ${error}`);
        // A timed-out run usually means the GPU is still spinning on a wedged
        // dispatch; drop the device so the next config starts clean.
        if (!(await recoverDevice(error))) break;
      } else {
        VEC_SIZES[cfg.size].push([best, cfg.workgroupSize, cfg.numWorkgroups, cfg.batch_size, cfg.par_lookback, incorrect, shader.name]);
        allConfigs.push({ shader: shader.name, ...cfg, best, bestCorrect: Number.isFinite(bestCorrect) ? bestCorrect : null, incorrect, epoch: deviceEpoch });
        console.log(`PROGRESS ${done}/${totalConfigs} ${shader.name} size=2^${Math.log2(cfg.size)} wg=${cfg.workgroupSize} ngroups=${cfg.numWorkgroups} batch=${cfg.batch_size} plb=${cfg.par_lookback} best=${best.toFixed(2)} GB/s incorrect=${incorrect}/${RUNS}`);
      }
    }

    console.log(`\nBest results for ${shader.name}`);
    const shaderResults = [];
    for (let i = MIN_POW; i <= MAX_POW; i++) {
      const size = 1 << i;
      VEC_SIZES[size].sort((a, b) => b[0] - a[0]);
      const bestEntry = VEC_SIZES[size][0];
      console.log("vec_size: ", size, bestEntry);
      // Best config that produced a correct scan on every timed run.
      const bestCorrectEntry = VEC_SIZES[size].find(e => e[5] === 0);
      if (bestEntry && Number.isFinite(bestEntry[0])) {
        shaderResults.push({
          size,
          throughput: bestEntry[0],
          config: { workgroupSize: bestEntry[1], numWorkgroups: bestEntry[2], batch_size: bestEntry[3], par_lookback: bestEntry[4] },
          incorrect: bestEntry[5],
          correctThroughput: bestCorrectEntry ? bestCorrectEntry[0] : null,
          correctConfig: bestCorrectEntry
            ? { workgroupSize: bestCorrectEntry[1], numWorkgroups: bestCorrectEntry[2], batch_size: bestCorrectEntry[3], par_lookback: bestCorrectEntry[4] }
            : null,
        });
      }
    }
    resultsByShader.push({ name: shader.name, data: shaderResults });
  }

  drawPlot(resultsByShader);

  window.__BENCH_RESULTS__ = { meta: window.__BENCH_META__, resultsByShader, allConfigs, failures, recoveries, deviceEpochs: deviceEpoch, canaryBaseline };
  window.__BENCH_DONE__ = true;
  console.log("BENCHMARK_COMPLETE");
}

function withTimeout(promise, ms) {
  let timer;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`run timed out after ${ms} ms`)), ms); }),
  ]);
}


async function main_helper(device, thx, wkrgx, btchsx, plbkx, alt, shaderPath) {

  const TUNING_CONFIG = {
    workgroupSize: thx,
    numWorkgroups: wkrgx,
    batch_size: btchsx,
    par_lookback: plbkx,
    alt: alt,
    debug_size: 2
  };
  
  // Initialize all WebGPU components
  const bindGroupLayout = await initBindGroupLayout(device, TUNING_CONFIG);
  const buffers = await initBuffers(device, TUNING_CONFIG);
  const bindGroup = await initBindGroup(device, bindGroupLayout, TUNING_CONFIG, buffers);
  const pipeline = await initComputePipeline(device, bindGroupLayout, TUNING_CONFIG, shaderPath);
  // Run the compute pass (dispatching the work to the GPU)
  let [throughput, incorrect] = await run(device, pipeline, bindGroup, TUNING_CONFIG, buffers);
  //console.log("tp: ", throughput, "inc: ", incorrect)

  buffers.ABuffer.destroy();
  buffers.BBuffer.destroy();
  buffers.CBuffer.destroy();
  buffers.CReadBuffer.destroy();
  buffers.DBuffer.destroy();
  buffers.debugBuffer.destroy();
  buffers.debugReadBuffer.destroy();
  buffers.TimestampResolveBuffer.destroy();
  buffers.TimestampReadBuffer.destroy();

  return [throughput, incorrect];
}


main().catch(err => {
  window.__BENCH_ERROR__ = String(err && err.stack ? err.stack : err);
  window.__BENCH_DONE__ = true;
  console.error("BENCHMARK_FAILED", err);
});
