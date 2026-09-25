# CUDA port

`../webgpu/prefix-sum-v2.wgsl` ported to CUDA, measured against
`cub::DeviceScan::InclusiveSum` and a tuned streaming-copy ceiling.

```bash
make                                    # nvcc -O3 -arch=sm_120
for r in 1 2 3; do ./bench results/run$r.json; done
python3 analyze.py                      # per-size table
python3 tune.py                         # which single policy to ship
```

Build for another architecture with `make ARCH=sm_90`. Needs CUDA 13 and sm_80 or
newer (`__reduce_max_sync`).

| file | |
|---|---|
| `prefix_sum_v2.cuh` | Port of `prefix-sum-v2.wgsl`. Loads issued up front, then scanned. |
| `bench.cu` | Sweep, plus CUB and a copy ceiling tuned on the same grid. |
| `analyze.py` | Best correct throughput per size, across sweeps. |
| `tune.py` | Finds the single (block, batch) minimising L1 distance from the per-size best envelope. |

## What changed in the port

Two things in the WGSL exist only to work around WGSL/WebGPU.

**The atomic partition ticket is gone.** The WGSL kernel draws its partition id
from `atomicAdd(&part, 1)` because WebGPU guarantees nothing about workgroup
launch order, so a workgroup spinning on a lower-numbered predecessor could be
waiting on one that has not been dispatched. CUDA dispatches blocks in increasing
`blockIdx` order, so `part_id = blockIdx.x` suffices — the same guarantee CUB's
own scan relies on. That also removes the ticket atomic, a shared broadcast slot
and one `__syncthreads()` per block.

**The clamped scratch index is gone.** WGSL's `select()` evaluates both arms, so
the original clamped the index as well as masking the value. C++'s ternary
short-circuits.

Carried over unchanged: the packed flag/aggregate word, and with it the 30-bit
limit on the running sum.

## Results

RTX PRO 6000 Blackwell Max-Q, 188 SMs, 128 MiB L2, CUDA 13.0, CUB 3.0. Best
correct throughput across three sweeps, GB/s:

| size | v2 | CUB | copy ceiling | v2/CUB |
|---|---|---|---|---|
| 2^16 | 131.1 | 72.8 | 178.1 | 1.80 |
| 2^19 | 686.2 | 514.0 | 1032.1 | 1.34 |
| 2^20 | 1032.1 | 1032.1 | 2064.1 | 1.00 |
| 2^22 | 2745.0 | 2345.8 | 4128.3 | 1.17 |
| 2^24 | 4096.0 | 4096.0 | 5966.3 | 1.00 |
| 2^25 | 1724.6 | 1747.6 | 1772.0 | 0.99 |

`python3 analyze.py` prints the full table from whatever is in `results/`. The
numbers above are one set of three sweeps; re-running moves them by a few percent,
more than that below 2^19.

**Items-per-thread is the load-bearing tuning knob.** With the CTA size free,
batch 1 → 2 → 4 moves geomean-vs-ideal 0.754 → 0.889 → 0.994 At batch 4, 256
versus 128 threads is 0.968 versus 0.947 — the two swap places between sweeps.
CTA size is a threshold (don't pick 32), not a gradient. CUB's own policy history agrees: items went
12 → 15 → 22 across architecture generations.

**Most of this range is an L2 measurement.** A scan of N `u32` touches 8N bytes,
so the working set fits 128 MiB of L2 through N = 2^24, where a tuned copy
sustains ~5970 GB/s against 1792 GB/s of DRAM peak. At 2^25 it spills and copy
falls to 1771 GB/s — 98.8% of theoretical. That is the shape of the curve, not a
property of any scan.

## Timing

CUDA events around the kernel launch, validated against wall clock over 50
back-to-back launches. Throughput counts one read plus one write per element.
Every timed run is verified on the GPU: an inclusive scan of all ones must give
`out[i] == i + 1`. No configuration of either kernel failed across three sweeps.

Below about 2^19 a launch costs ~4 µs and the kernel costs less, so repeat sweeps
disagree by 12–19% there against ≤1.5% above. Read the small sizes as
"launch-bound", not as a ranking.
