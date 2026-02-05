@group(0) @binding(0) var<storage, read_write> in: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> prefix_states: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> part: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> debug: array<u32>;

const FLG_A = 1;
const FLG_P = 2;
const ANTI_MASK = 30u;
const MASK_ = ~(3u << ANTI_MASK);

//const NUM_SUBGROUPS = 8;

override wg_size: u32;

var<workgroup> wg_broadcast: u32;
var<workgroup> exclusive_prefix: u32;
var<workgroup> inclusive_scan: u32;

// size of min_subgroup_size * batch_size, every subgroup has batch_size

// must change this to match with num_subgroups * BATCH_SIZE
var<workgroup> scratch: array<u32, NUM_SUBGROUPS*BATCH_SIZE>;


fn calc_lookback_id(
  subgroup_invocation_id: u32,  // Now passed as an argument
  subgroup_size: u32,           // Now passed as an argument
  part_id: i32, 
  lookback_amt: i32
) -> i32 {
  
  if (lookback_amt > part_id) {
    if (subgroup_invocation_id == subgroup_size - 1) {
      return 0;
    }
    return -1;
  } else {
    return part_id - lookback_amt;
  }
}

@compute @workgroup_size(wg_size) fn prefix_sum(
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_subgroups) num_subgroups: u32, 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id : u32) {

  if(local_id.x == 0u){
      wg_broadcast = atomicAdd(&part, 1);
  }


  let part_id = workgroupUniformLoad(&wg_broadcast);

  let thread_id = subgroup_id * subgroup_size + subgroup_invocation_id;

  let base = part_id * wg_size * BATCH_SIZE + subgroup_id * (subgroup_size * BATCH_SIZE);

  let batch_groups = BATCH_SIZE * num_subgroups;

  var values: array<vec4<u32>, BATCH_SIZE>;

  //{VEC4-REDUCTION}

var subgroup_carry: u32 = 0;

  for (var i: u32 = 0; i < BATCH_SIZE; i++) {
    let idx = base + i * subgroup_size + subgroup_invocation_id;
    values[i] = in[idx];

    values[i].y += values[i].x;
    values[i].z += values[i].y;
    values[i].w += values[i].z;

    let lane_sum = values[i].w;

    let ex = subgroupExclusiveAdd(lane_sum);
    values[i] += ex + subgroup_carry;

    let total_i = subgroupBroadcast(ex + lane_sum, 32 - 1u);
    subgroup_carry += total_i;
  }

  if (subgroup_invocation_id == subgroup_size - 1) {
    scratch[subgroup_id] = subgroup_carry;
  }

  workgroupBarrier();

  if (subgroup_id == 0) {
    let valid = select(0u, scratch[subgroup_invocation_id], subgroup_invocation_id < batch_groups);
    scratch[subgroup_invocation_id] = subgroupExclusiveAdd(valid);
  }
  
  workgroupBarrier();

  if (thread_id == 0) { // This has to be this rather than get_local_id == 0 bcz exprfx mst be synced by subbarrier in lookback

    atomicStore(&prefix_states[part_id], (FLG_A << ANTI_MASK) | (scratch[batch_groups - 1] & MASK_));

    if (part_id == 0) {
      atomicStore(&prefix_states[part_id], (FLG_P << ANTI_MASK) | (scratch[batch_groups - 1] & MASK_));
    }
    exclusive_prefix = 0;
  }
  workgroupBarrier();



  var p : u32 = debug[0];

  if (p == 1) {
    if (part_id != 0 && subgroup_id == 0) {
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
        //debug[0] = i32(subgroup_id);
        atomicStore(&prefix_states[part_id], (FLG_P << ANTI_MASK) | ((exclusive_prefix + scratch[batch_groups - 1]) & MASK_));
      }
      
    }
  }else{
    if (part_id != 0 && local_id.x == 0) {
      var lookback_id = part_id - 1;
      // spin and lookback until full prefix is set
      while (lookback_id >= 0) {
        let flagg = atomicLoad(&prefix_states[lookback_id]);     
        let agg = flagg & 0x3FFFFFFF;
        let flag = flagg >> ANTI_MASK;

        if (flag == FLG_P) {
                if (part_id == 6) {
                  debug[1] = agg;
                }
          exclusive_prefix += agg;
          break;
        } else if (flag == FLG_A) {
          exclusive_prefix += agg;
          lookback_id -= 1;
        }
      }
      atomicStore(&prefix_states[part_id], (FLG_P << ANTI_MASK) | ((exclusive_prefix + scratch[batch_groups - 1]) & MASK_));
    }
  }

  workgroupBarrier();  


  // each workgroup has prefix within device
  var total_exclusive_prefix : u32 = exclusive_prefix;

  // each subgroup has prefix within workgroup
  if (subgroup_id != 0) {
    total_exclusive_prefix += scratch[subgroup_id];
  }

  // each lane has prefix within subgroup
  for (var i : u32 = 0; i < BATCH_SIZE; i++) {
    let idx = base + i * subgroup_size + subgroup_invocation_id;
      out[idx] = values[i] + total_exclusive_prefix;
  }
}
