// Kernel 1 - Prefix-sum with tuning


// inputArgs: fill: make a fill type that fills with a single value
// inputArgs: type: [u32, vec4...] need to be able iterable like sizes
// validArgs: need to be able to match buffer type with certain memload code


// useful to enum everything then types are less ambiguous then I can write vec4/vec2 and have it mean 4 or 2, instead of just writing numbers for the types.
// So that I can use the inputArgs.sizes field literally.


const type = {
    vec4: {size: 4, memload:
    `   
        var values: array<u32, BATCH_SIZE>;
        values[0] = in[my_id];
        for (var i: u32 = 1; i < BATCH_SIZE; i++) {
            values[i] = in[my_id + i] + values[i - 1];
        }

        scratch[local_id.x] = values[BATCH_SIZE - 1];
    `
    },
    vec2: {size: 2, memload:                 
    `
        var values: array<vec2<u32>, BATCH_SIZE>;
        for (var i: u32 = 0; i < BATCH_SIZE; i++) {
            values[i] = in[my_id + i];
        }

        values[0].y += values[0].x;

        for (var i: u32 = 1u; i < BATCH_SIZE; i = i++) {
            let prev = values[i - 1u].y;

            values[i].x += prev;
            values[i].y += values[i].x;
        }

        scratch[local_id.x] = values[BATCH_SIZE - 1u].y;
    `
    },
    u32:  {size: 1, memload:
    `
        var values: array<vec4<u32>, BATCH_SIZE>;
        for (var i: u32 = 0; i < BATCH_SIZE; i++) {
            values[i] = in[my_id + i];
        }

        values[0].y += values[0].x;
        values[0].z += values[0].y;
        values[0].w += values[0].z;

        for (var i: u32 = 1u; i < BATCH_SIZE; i++) {
            let prev = values[i - 1u].w;

            values[i].x += prev;
            values[i].y += values[i].x;
            values[i].z += values[i].y;
            values[i].w += values[i].z;
        }

        scratch[local_id.x] = values[BATCH_SIZE - 1].w;
    `
    },          
}


const kernel = {
    // Basic kernel information
    name: "Prefix-sum (Tuned)",
    description: "Prefix-sum with configurable workgroup sizes",

    // Input arguments for the kernel
    inputArgs: [
        {
            name: "in_buffer",
	        storage_type: "storage",
            types: [type.u32, type.vec2, type.vec4],
            fill: "fill_u32", //could make enum
            sizes: [1 << 10, 1 << 11, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25],
            io_type: "input", //could make enum
            order: 0
        },
        {
            name: "out_buffer",
	        storage_type: "storage",
            types: [type.u32, type.vec2, type.vec4],
            fill: "dont fill", //could make enum
            sizes: [1 << 10, 1 << 11, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25],
            io_type: "output", //could make enum
            order: 1
        },
        {   
            name: "prefix_states",
	        storage_type: "storage",
            types: "atomic_u32[]",
            fill: "fill_u32", //could make enum
            sizes: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            io_type: "input", //could make enum
            order: 2
        },
        {   
            name: "partition",
	        storage_type: "storage",
            types: "atomic_u32",
            values: [0]
        },
        {
            name: "vectorSize",
	        storage_type: "uniform",
            types: ["int"],
            values: [1 << 10, 1 << 11, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25]
        }
    ],

    // Tuning arguments
    tuningArgs: [
        {
            name: "workgroupSizeY",
            type: "interpolate",
            values: ["1"],
            description: "Workgroup size in Y dim",
            order: 0
        },
        {
            name: "workgroupSizeX",
            type: "interpolate",
            values: ["32", "64", "128", "256"],
            description: "Workgroup size in X dim",
            order: 1
        },
        {
            name: "memLoad",
            type: "interpolate",
            values: [type.u32.memload
                    , 
                    type.vec2.memload
                    , 
                    type.vec4.memload
                    ],
            description: "Type of element loaded from global memory",
            order: 2
        },
        {
            name: "workGroupReduction",
            type: "interpolate",
            values: [
                `   
                        if (sid == 0) {
                            let rake_batch_size = {{workgroupSizeX}} / subgroup_size;
                            let start = local_id.x * rake_batch_size;
                            for (var i = start + 1; i < start + rake_batch_size; i++) {
                                scratch[i] += scratch[i - 1];
                            }
                            let partial_sum = scratch[start + rake_batch_size - 1];
                            let prefix = subgroupExclusiveAdd(partial_sum);
                            for (var i = start; i < start + rake_batch_size; i++) {
                                scratch[i] += prefix;
                            }
                        }

                `
                    ,                    
                    
                `
                        let ai: u32 = 2u * local_id.x + 1u;
                        let bi: u32 = 2u * local_id.x + 2u;

                        if (BLOCK_SIZE >= 2u) {
                        if (local_id.x < (BLOCK_SIZE >> 1u)) {
                            scratch[1u * bi - 1u] += scratch[1u * ai - 1u];
                        }

                        if ((BLOCK_SIZE >> 0u) > sg_size) {
                            workgroupBarrier();
                        }
                        }
                        if (BLOCK_SIZE >= 4u)  { if (local_id.x < (BLOCK_SIZE >> 2u))  { scratch[2u * bi - 1u] += scratch[2u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 8u)  { if (local_id.x < (BLOCK_SIZE >> 3u))  { scratch[4u * bi - 1u] += scratch[4u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 16u) { if (local_id.x < (BLOCK_SIZE >> 4u))  { scratch[8u * bi - 1u] += scratch[8u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 32u) { if (local_id.x < (BLOCK_SIZE >> 5u))  { scratch[16u * bi - 1u] += scratch[16u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 64u) { if (local_id.x < (BLOCK_SIZE >> 6u))  { scratch[32u * bi - 1u] += scratch[32u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 128u){ if (local_id.x < (BLOCK_SIZE >> 7u))  { scratch[64u * bi - 1u] += scratch[64u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 256u){ if (local_id.x < (BLOCK_SIZE >> 8u))  { scratch[128u * bi - 1u] += scratch[128u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 512u){ if (local_id.x < (BLOCK_SIZE >> 9u))  { scratch[256u * bi - 1u] += scratch[256u * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 1024u){if (local_id.x < (BLOCK_SIZE >> 10u)) { scratch[512u * bi - 1u] += scratch[512u * ai - 1u]; } workgroupBarrier(); }

                        if (local_id.x == 0u) {
                        inclusive_scan = scratch[BLOCK_SIZE - 1u];
                        scratch[BLOCK_SIZE - 1u] = 0u;
                        }
                        workgroupBarrier();

                        // traverse down the tree building the scan in place
                        if (BLOCK_SIZE >= 2u) {
                        if (local_id.x < 1u) {
                            scratch[(BLOCK_SIZE >> 1u) * bi - 1u] += scratch[(BLOCK_SIZE >> 1u) * ai - 1u];
                            scratch[(BLOCK_SIZE >> 1u) * ai - 1u] = scratch[(BLOCK_SIZE >> 1u) * bi - 1u] - scratch[(BLOCK_SIZE >> 1u) * ai - 1u];
                        }
                        }
                        if (BLOCK_SIZE >= 4u)  { if (local_id.x < 2u)   { scratch[(BLOCK_SIZE >> 2u) * bi - 1u] += scratch[(BLOCK_SIZE >> 2u) * ai - 1u]; scratch[(BLOCK_SIZE >> 2u) * ai - 1u] = scratch[(BLOCK_SIZE >> 2u) * bi - 1u] - scratch[(BLOCK_SIZE >> 2u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 8u)  { if (local_id.x < 4u)   { scratch[(BLOCK_SIZE >> 3u) * bi - 1u] += scratch[(BLOCK_SIZE >> 3u) * ai - 1u]; scratch[(BLOCK_SIZE >> 3u) * ai - 1u] = scratch[(BLOCK_SIZE >> 3u) * bi - 1u] - scratch[(BLOCK_SIZE >> 3u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 16u) { if (local_id.x < 8u)   { scratch[(BLOCK_SIZE >> 4u) * bi - 1u] += scratch[(BLOCK_SIZE >> 4u) * ai - 1u]; scratch[(BLOCK_SIZE >> 4u) * ai - 1u] = scratch[(BLOCK_SIZE >> 4u) * bi - 1u] - scratch[(BLOCK_SIZE >> 4u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 32u) { if (local_id.x < 16u)  { scratch[(BLOCK_SIZE >> 5u) * bi - 1u] += scratch[(BLOCK_SIZE >> 5u) * ai - 1u]; scratch[(BLOCK_SIZE >> 5u) * ai - 1u] = scratch[(BLOCK_SIZE >> 5u) * bi - 1u] - scratch[(BLOCK_SIZE >> 5u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 64u) { if (local_id.x < 32u)  { scratch[(BLOCK_SIZE >> 6u) * bi - 1u] += scratch[(BLOCK_SIZE >> 6u) * ai - 1u]; scratch[(BLOCK_SIZE >> 6u) * ai - 1u] = scratch[(BLOCK_SIZE >> 6u) * bi - 1u] - scratch[(BLOCK_SIZE >> 6u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 128u){ if (local_id.x < 64u)  { scratch[(BLOCK_SIZE >> 7u) * bi - 1u] += scratch[(BLOCK_SIZE >> 7u) * ai - 1u]; scratch[(BLOCK_SIZE >> 7u) * ai - 1u] = scratch[(BLOCK_SIZE >> 7u) * bi - 1u] - scratch[(BLOCK_SIZE >> 7u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 256u){ if (local_id.x < 128u) { scratch[(BLOCK_SIZE >> 8u) * bi - 1u] += scratch[(BLOCK_SIZE >> 8u) * ai - 1u]; scratch[(BLOCK_SIZE >> 8u) * ai - 1u] = scratch[(BLOCK_SIZE >> 8u) * bi - 1u] - scratch[(BLOCK_SIZE >> 8u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 512u){ if (local_id.x < 256u) { scratch[(BLOCK_SIZE >> 9u) * bi - 1u] += scratch[(BLOCK_SIZE >> 9u) * ai - 1u]; scratch[(BLOCK_SIZE >> 9u) * ai - 1u] = scratch[(BLOCK_SIZE >> 9u) * bi - 1u] - scratch[(BLOCK_SIZE >> 9u) * ai - 1u]; } workgroupBarrier(); }
                        if (BLOCK_SIZE >= 1024u){if (local_id.x < 512u){ scratch[(BLOCK_SIZE >> 10u) * bi - 1u] += scratch[(BLOCK_SIZE >> 10u) * ai - 1u]; scratch[(BLOCK_SIZE >> 10u) * ai - 1u] = scratch[(BLOCK_SIZE >> 10u) * bi - 1u] - scratch[(BLOCK_SIZE >> 10u) * ai - 1u]; } workgroupBarrier(); }

                        let temp_tree: u32 = select(inclusive_scan, scratch[local_id.x + 1u], local_id.x != BLOCK_SIZE - 1u);

                        workgroupBarrier();

                        scratch[local_id.x] = temp_tree;
                `
            ],
            description: "Workgroup reduction strategy",
            order: 3
        },
        {
            name: "batchSize",
            type: "interpolate",
            values: [1, 2, 4, 8],
            description: "Number of elements from global memory per thread",
            order: 5
        },
        {
            name: "lookBackType",
            type: "interpolate",
            values: [
                `
                        if (part_id != 0 && sid == 0) {
                            var lookback_id : i32 = i32(part_id) - (i32(subgroup_size) - i32(subgroup_invocation_id));
                            var done = false;
                            var flag : u32 = FLG_P;
                            var agg : u32 = 0;
                            while(!done) {
                                if (lookback_id >= 0) {
                                let flagg = atomicLoad(&prefix_states[lookback_id]);  
                                agg = flagg & 0x3FFFFFFF;
                                flag = flagg >> ANTI_MASK;
                                }
                                if (subgroupAll(flag == FLG_A || flag == FLG_P)) {
                                var local_prefix = 0u;
                                if (subgroupAny(flag == FLG_P)) {
                                    // we will terminate after this iteration
                                    done = true;
                                    let inclusive = select(0, subgroup_invocation_id, flag == FLG_P);
                                    let max_inclusive = subgroupMax(inclusive);

                                    // load thread with highest FLG_P and higher prefixes
                                    if (max_inclusive <= subgroup_invocation_id) {
                                    local_prefix = agg;
                                    }

                                // if no thread has inclusive prefix, all threads load exclusive prefix
                                } else {
                                    // every thread looks back another partition
                                    local_prefix = agg;
                                    lookback_id = lookback_id - i32(subgroup_size);
                                }
                                let scanned_prefix = subgroupInclusiveAdd(local_prefix);

                                // last thread has the full prefix, update the workgroup level exclusive prefix
                                if (subgroup_invocation_id == subgroup_size - 1) {
                                    exclusive_prefix += scanned_prefix;
                                }
                                }
                            }
                            // finally last thread in subgroup updates this workgroup's prefix/flag
                            if (subgroup_invocation_id == subgroup_size - 1) {
                                //debug[0] = i32(sid);
                                atomicStore(&prefix_states[part_id], (FLG_P << ANTI_MASK) | ((exclusive_prefix + scratch[{{workgroupSizeX}} - 1]) & MASK_));
                            }
                            
                        }
                `
                    ,
                
                `
                        if (part_id != 0 && local_id.x == 0) {
                            var lookback_id = part_id - 1;
                            // spin and lookback until full prefix is set
                            while (lookback_id >= 0) {
                                let flagg = atomicLoad(&prefix_states[lookback_id]);     
                                let agg = flagg & 0x3FFFFFFF;
                                let flag = flagg >> ANTI_MASK;

                                if (flag == FLG_P) {
                                    exclusive_prefix += agg;
                                    break;
                                } else if (flag == FLG_A) {
                                    exclusive_prefix += agg;
                                    lookback_id -= 1;
                                }
                            }
                            atomicStore(&prefix_states[part_id], (FLG_P << ANTI_MASK) | ((exclusive_prefix + scratch[{{workgroupSizeX}} - 1]) & MASK_));
                        }
                `
            ],
            description: "Decoupled lookback strategy",
            order: 4
        },
    ],

    wgsl_shader:


`   
    const FLG_A = 1;
    const FLG_P = 2;
    const ANTI_MASK = 30u;
    const MASK_ = ~(3u << ANTI_MASK);

    var<workgroup> wg_broadcast: u32;
    var<workgroup> exclusive_prefix: u32;
    var<workgroup> inclusive_scan: u32;
    var<workgroup> scratch: array<u32, {{workgroupSizeX}}>;

    @compute @workgroup_size({{workgroupSizeX}}) fn prefix_sum(
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(subgroup_size) subgroup_size: u32, 
        @builtin(local_invocation_id) local_id: vec3<u32>) {

        if(local_id.x == 0u){
            wg_broadcast = atomicAdd(&part, 1);
        }

        let part_id = workgroupUniformLoad(&wg_broadcast);

        let sid = local_id.x / subgroup_size;
        let my_id = part_id * {{workgroupSizeX}} * {{batchSize}} + local_id.x * {{batchSize}};

        {{memLoad}} // type of memory load

        workgroupBarrier();

        {{workGroupReduction}} // reduction

        workgroupBarrier();

        if (local_id.x == 0) {

            atomicStore(&prefix_states[part_id], (FLG_A << ANTI_MASK) | (scratch[local_id.x - 1] & MASK_));

            if (part_id == 0) {
            atomicStore(&prefix_states[part_id], (FLG_P << ANTI_MASK) | (scratch[local_id.x - 1] & MASK_));
            }
            exclusive_prefix = 0;
        }
        workgroupBarrier();

        {{lookBackType}} // type of decoupled lookback  

        workgroupBarrier();

        var total_exclusive_prefix : u32 = exclusive_prefix;

        if (local_id.x != 0) {
            total_exclusive_prefix += scratch[local_id.x - 1];
        }

        for (var i : u32 = 0; i < {{batchSize}}; i++) {
            out[my_id + i] = values[i] + total_exclusive_prefix; 
        }`
}

jsReference: function PrefixSumCPU(in_buffer, out_buffer, vectorSize) {        
        out_buffer[0] = in_buffer[0];
        for (let i = 1; i < vectorSize; i++) {
            out_buffer[i] = out_buffer[i-1] + in_buffer[i];
        }
    }

validArgs: function F(in_buffer, out_buffer, vectorSize, workgroupSizeX, workgroupSizeY, batchSize, memLoad, numWorkgroups) {
    //check for valid prefix-sum input
    var first = in_buffer.size == out_buffer.size == vectorSize
    var second = in_buffer.type == out_buffer.type == memLoad 
    var third = vectorSize == workgroupSizeX * workgroupSizeY * numWorkgroups * in_buffer.type * batchSize
    var fourth = int(workgroupSizeX) * int(workgroupSizeY) <= 256
    return first && second && third && fourth
    }

numWorkgroups: function F(in_buffer, out_buffer, vectorSize, workgroupSizeX, workgroupSizeY, batchSize) {
	//calculate number of workgroups
	var numWorkgroupsX = vectorSize / workgroupSizeX * in_buffer.type * batchSize 
	var numWorkgroupsY = 1
	return (numWorkgroupsX , numWorkgroupsY)
    }