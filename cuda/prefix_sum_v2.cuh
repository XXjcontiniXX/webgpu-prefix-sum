// CUDA port of webgpu/prefix-sum-v2.wgsl
//
// Single-pass decoupled-lookback inclusive scan over u32, four elements per
// thread per batch step (uint4, matching the WGSL kernel's vec4<u32>).
//
// Most of the port is a direct substitution:
//   subgroup / subgroup_size      -> warp / 32
//   subgroupExclusiveAdd          -> shuffle-based warp scan
//   subgroupBroadcast(x, 31)      -> __shfl_sync(..., x, 31)
//   subgroupAll / subgroupAny     -> __all_sync / __any_sync
//   subgroupMax                   -> __reduce_max_sync            (sm_80+)
//   atomicStore / atomicLoad      -> cuda::atomic_ref, device scope
//   override wg_size / BATCH_SIZE -> template parameters
//
// Three things in the WGSL exist only to work around WGSL/WebGPU and are gone here:
//
//  1. The atomic partition ticket. The WGSL kernel draws its partition id from
//     atomicAdd(&part, 1) because WebGPU guarantees nothing about workgroup launch
//     order, so a block spinning on a lower-numbered predecessor could wait on one
//     that has not been dispatched. CUDA dispatches blocks in increasing blockIdx
//     order, so a lower blockIdx is already resident or retired and the lookback
//     spin cannot deadlock -- this is the same guarantee CUB's DeviceScan leans on.
//     partition id is just blockIdx.x, which also removes the ticket atomic, the
//     shared broadcast slot and one __syncthreads() per block.
//
//  2. The clamped scratch index. WGSL's select() evaluates both arms, so the
//     original had to clamp the index as well as mask the value to avoid reading
//     past scratch. C++'s ternary short-circuits, so a plain guard is enough.
//
//  3. The debug[0] switch between parallel and serial lookback. The serial path
//     never won a tuning configuration and caused nearly every GPU hang, so the
//     WGSL sweeps excluded it; only the parallel path is ported.
//
// Aggregates share a word with a 2-bit flag, so this carries the original's
// 30-bit limit on the running sum (N <= 2^30 for an all-ones input).
#pragma once

#include <cuda/atomic>

namespace psum {

constexpr unsigned FLG_X = 0u;   // not ready
constexpr unsigned FLG_A = 1u;   // block aggregate published
constexpr unsigned FLG_P = 2u;   // inclusive prefix published
constexpr unsigned ANTI_MASK = 30u;
constexpr unsigned MASK_ = ~(3u << ANTI_MASK);   // 0x3FFFFFFF
constexpr unsigned FULL = 0xFFFFFFFFu;

// The flag and the aggregate share one 32-bit word, so a single relaxed atomic
// access carries both and no extra fence is needed to pair them.
using state_ref = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;

__device__ __forceinline__ unsigned warp_inclusive_add(unsigned v, unsigned lane) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const unsigned n = __shfl_up_sync(FULL, v, off);
    if (lane >= (unsigned)off) v += n;
  }
  return v;
}

__device__ __forceinline__ uint4 scan4(uint4 v) {
  v.y += v.x;
  v.z += v.y;
  v.w += v.z;
  return v;
}

__device__ __forceinline__ void add4(uint4 &v, unsigned s) {
  v.x += s; v.y += s; v.z += s; v.w += s;
}

template <int BLOCK_SIZE, int BATCH_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void prefix_sum(
    const uint4 *__restrict__ in,
    uint4 *__restrict__ out,
    unsigned *__restrict__ prefix_states) {

  constexpr unsigned NUM_WARPS = BLOCK_SIZE / 32;

  __shared__ unsigned exclusive_prefix;
  __shared__ unsigned workgroup_aggregate;
  __shared__ unsigned scratch[NUM_WARPS];

  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned part_id = blockIdx.x;

  const unsigned base = part_id * (BLOCK_SIZE * BATCH_SIZE) + warp_id * (32u * BATCH_SIZE);

  // Issue every load before consuming any of them, so the batch's memory latency
  // overlaps instead of serialising behind each warp scan.
  uint4 values[BATCH_SIZE];
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) values[i] = in[base + i * 32u + lane];

  unsigned warp_carry = 0u;
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    uint4 v = scan4(values[i]);
    const unsigned inc = warp_inclusive_add(v.w, lane);
    add4(v, (inc - v.w) + warp_carry);
    values[i] = v;
    warp_carry += __shfl_sync(FULL, inc, 31);
  }

  if (lane == 31u) scratch[warp_id] = warp_carry;
  __syncthreads();

  // Scan the per-warp totals. Width is NUM_WARPS; the last lane also publishes
  // the inclusive total, because the exclusive scan alone drops the final warp.
  if (warp_id == 0) {
    const bool in_range = lane < NUM_WARPS;
    const unsigned valid = in_range ? scratch[lane] : 0u;
    const unsigned inc = warp_inclusive_add(valid, lane);
    if (in_range) scratch[lane] = inc - valid;
    if (lane == NUM_WARPS - 1u) workgroup_aggregate = inc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const unsigned flag = (part_id == 0u) ? FLG_P : FLG_A;
    state_ref(prefix_states[part_id])
        .store((flag << ANTI_MASK) | (workgroup_aggregate & MASK_), cuda::memory_order_relaxed);
    exclusive_prefix = 0u;
  }
  __syncthreads();

  // Parallel lookback: warp 0 inspects 32 predecessor partitions per iteration.
  if (part_id != 0u && warp_id == 0) {
    int lookback_id = (int)part_id - (32 - (int)lane);
    bool done = false;
    unsigned flag = FLG_P;   // lanes whose partition is negative stay "ready" with agg 0
    unsigned agg = 0u;

    while (!done) {
      if (lookback_id >= 0) {
        const unsigned f = state_ref(prefix_states[lookback_id]).load(cuda::memory_order_relaxed);
        agg = f & MASK_;
        flag = f >> ANTI_MASK;
      }
      if (__all_sync(FULL, flag == FLG_A || flag == FLG_P)) {
        unsigned local_prefix = 0u;
        if (__any_sync(FULL, flag == FLG_P)) {
          done = true;
          // Take the highest lane holding an inclusive prefix, plus every lane
          // above it. Stale lanes sit below that index and contribute nothing.
          const unsigned max_inclusive = __reduce_max_sync(FULL, (flag == FLG_P) ? lane : 0u);
          if (max_inclusive <= lane) local_prefix = agg;
        } else {
          local_prefix = agg;
          lookback_id -= 32;
        }
        const unsigned scanned = warp_inclusive_add(local_prefix, lane);
        if (lane == 31u) exclusive_prefix += scanned;
      }
    }
    if (lane == 31u) {
      state_ref(prefix_states[part_id])
          .store((FLG_P << ANTI_MASK) | ((exclusive_prefix + workgroup_aggregate) & MASK_),
                 cuda::memory_order_relaxed);
    }
  }
  __syncthreads();

  unsigned total = exclusive_prefix;              // prefix of this block within the device
  if (warp_id != 0) total += scratch[warp_id];    // prefix of this warp within the block

#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    uint4 v = values[i];
    add4(v, total);
    out[base + i * 32u + lane] = v;
  }
}

}  // namespace psum
