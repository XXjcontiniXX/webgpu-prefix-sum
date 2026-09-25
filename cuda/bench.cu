// Throughput benchmark: the two ported kernels against cub::DeviceScan::InclusiveSum.
//
// Mirrors the WGSL harness in webgpu/main.js:
//   - same tuning grid: block size x block count x batch, size = block*grid*batch*4
//   - same size range, 2^12 .. 2^25 u32 elements
//   - 2 warm-up runs then 5 timed runs per configuration, best of the 5
//   - throughput = (N * 4 bytes * 2) / kernel time, i.e. one read plus one write
//   - timing covers the kernel only, as the WGSL timestamps covered the compute pass
// CUB has no tuning grid of its own to sweep, so it gets one entry per size.
#include <cub/device/device_scan.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "prefix_sum_v2.cuh"

#define CHECK(x)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),          \
                   __FILE__, __LINE__);                                                 \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

static const int BLOCK_SIZES[] = {32, 64, 128, 256};
static const int GRID_SIZES[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
static const int BATCH_SIZES[] = {1, 2, 4};
static const int MIN_POW = 12;
static const int MAX_POW = 25;
static const int WARMUPS = 2;
static const int RUNS = 5;
static const int PER_THREAD = 4;   // uint4

// Streaming copy, given the same block/grid/batch shape as the scans so it is an
// honest ceiling rather than a differently-tuned kernel. A scan reads N and writes
// N just as this does, so scan/copy is the efficiency figure worth reading.
template <int BLOCK_SIZE, int BATCH_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void copy_ceiling(
    const uint4 *__restrict__ a, uint4 *__restrict__ b) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned base = blockIdx.x * (BLOCK_SIZE * BATCH_SIZE) + warp_id * (32u * BATCH_SIZE);
  uint4 v[BATCH_SIZE];
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) v[i] = a[base + i * 32u + lane];
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) b[base + i * 32u + lane] = v[i];
}

__global__ void fill_ones(unsigned *p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1u;
}

// Exact check: an inclusive scan of all ones must give out[i] == i + 1.
__global__ void verify(const unsigned *out, size_t n, unsigned *bad) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && out[i] != (unsigned)(i + 1)) atomicAdd(bad, 1u);
}

struct Ctx {
  unsigned *d_in = nullptr;
  unsigned *d_out = nullptr;
  unsigned *d_states = nullptr;
  unsigned *d_bad = nullptr;
  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  cudaEvent_t beg, end;
};

struct Row {
  const char *kernel;
  size_t size;
  int block, grid, batch;
  double best_gbps;
  int incorrect;   // number of timed runs that produced a wrong scan
};

static double gbps_of(size_t n, float ms) {
  const double bytes = (double)n * 4.0 * 2.0;
  return bytes / (ms * 1.0e-3) / 1.0e9;
}

static int check_output(Ctx &c, size_t n) {
  CHECK(cudaMemset(c.d_bad, 0, sizeof(unsigned)));
  const int t = 256;
  verify<<<(unsigned)((n + t - 1) / t), t>>>(c.d_out, n, c.d_bad);
  CHECK(cudaGetLastError());
  unsigned bad = 0;
  CHECK(cudaMemcpy(&bad, c.d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
  return bad != 0;
}

enum Kind { K_V2 = 0, K_COPY = 1 };
static const char *kind_name(Kind k) {
  return k == K_V2 ? "cuda-v2" : "copy-ceiling";
}

template <Kind K, int BLOCK, int BATCH>
static void launch(Ctx &c, int grid) {
  if constexpr (K == K_V2) {
    psum::prefix_sum<BLOCK, BATCH><<<grid, BLOCK>>>(
        (const uint4 *)c.d_in, (uint4 *)c.d_out, c.d_states);
  } else {
    copy_ceiling<BLOCK, BATCH><<<grid, BLOCK>>>((const uint4 *)c.d_in, (uint4 *)c.d_out);
  }
}

template <Kind K, int BLOCK, int BATCH>
static Row measure(Ctx &c, int grid) {
  const size_t n = (size_t)BLOCK * grid * BATCH * PER_THREAD;
  Row r{kind_name(K), n, BLOCK, grid, BATCH, -1.0, 0};

  for (int i = 0; i < WARMUPS + RUNS; ++i) {
    CHECK(cudaMemset(c.d_states, 0, (size_t)grid * sizeof(unsigned)));
    CHECK(cudaMemset(c.d_out, 0, n * sizeof(unsigned)));
    CHECK(cudaEventRecord(c.beg));
    launch<K, BLOCK, BATCH>(c, grid);
    CHECK(cudaEventRecord(c.end));
    CHECK(cudaEventSynchronize(c.end));
    CHECK(cudaGetLastError());
    if (i >= WARMUPS) {
      float ms = 0.f;
      CHECK(cudaEventElapsedTime(&ms, c.beg, c.end));
      const double g = gbps_of(n, ms);
      if (g > r.best_gbps) r.best_gbps = g;
      if constexpr (K != K_COPY) r.incorrect += check_output(c, n);
    }
  }
  return r;
}

static Row measure_cub(Ctx &c, size_t n) {
  Row r{"cub-DeviceScan", n, 0, 0, 0, -1.0, 0};
  for (int i = 0; i < WARMUPS + RUNS; ++i) {
    CHECK(cudaMemset(c.d_out, 0, n * sizeof(unsigned)));
    CHECK(cudaEventRecord(c.beg));
    CHECK(cub::DeviceScan::InclusiveSum(c.d_temp, c.temp_bytes, c.d_in, c.d_out, (int)n));
    CHECK(cudaEventRecord(c.end));
    CHECK(cudaEventSynchronize(c.end));
    if (i >= WARMUPS) {
      float ms = 0.f;
      CHECK(cudaEventElapsedTime(&ms, c.beg, c.end));
      const double g = gbps_of(n, ms);
      if (g > r.best_gbps) r.best_gbps = g;
      r.incorrect += check_output(c, n);
    }
  }
  return r;
}

// Runtime (block, batch) -> compile-time dispatch.
template <Kind K>
static bool dispatch(Ctx &c, int block, int batch, int grid, Row &out) {
#define CASE(B, T)                                        \
  if (block == (B) && batch == (T)) {                     \
    out = measure<K, B, T>(c, grid);                     \
    return true;                                          \
  }
  CASE(32, 1) CASE(32, 2) CASE(32, 4)
  CASE(64, 1) CASE(64, 2) CASE(64, 4)
  CASE(128, 1) CASE(128, 2) CASE(128, 4)
  CASE(256, 1) CASE(256, 2) CASE(256, 4)
#undef CASE
  return false;
}

int main(int argc, char **argv) {
  const char *out_path = (argc > 1) ? argv[1] : "results.json";

  int dev = 0;
  CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, dev));

  const size_t max_n = (size_t)1 << MAX_POW;
  const size_t max_grid = (size_t)1 << MAX_POW;  // grid can reach N/(32*1*4)

  Ctx c;
  CHECK(cudaMalloc(&c.d_in, max_n * sizeof(unsigned)));
  CHECK(cudaMalloc(&c.d_out, max_n * sizeof(unsigned)));
  CHECK(cudaMalloc(&c.d_states, max_grid * sizeof(unsigned)));
  CHECK(cudaMalloc(&c.d_bad, sizeof(unsigned)));
  CHECK(cudaEventCreate(&c.beg));
  CHECK(cudaEventCreate(&c.end));

  fill_ones<<<(unsigned)((max_n + 255) / 256), 256>>>(c.d_in, max_n);
  CHECK(cudaDeviceSynchronize());

  CHECK(cub::DeviceScan::InclusiveSum(nullptr, c.temp_bytes, c.d_in, c.d_out, (int)max_n));
  CHECK(cudaMalloc(&c.d_temp, c.temp_bytes));

  std::vector<Row> rows;

  // Sweep sizes directly and derive the grid, the way CUB does: a policy fixes
  // (block, batch) and num_tiles follows from N. A fixed grid list would have let
  // each (block, batch) pair cover only part of the size range.
  for (int pass = 0; pass < 2; ++pass) {
    for (int block : BLOCK_SIZES)
      for (int pw = MIN_POW; pw <= MAX_POW; ++pw)
        for (int batch : BATCH_SIZES) {
          const size_t n = (size_t)1 << pw;
          const size_t tile = (size_t)block * batch * PER_THREAD;
          if (n % tile) continue;
          const size_t grid_sz = n / tile;
          if (grid_sz < 1 || grid_sz > 2147483647ull) continue;
          const int grid = (int)grid_sz;
          Row r;
          const bool ok = (pass == 0) ? dispatch<K_V2>(c, block, batch, grid, r)
                                      : dispatch<K_COPY>(c, block, batch, grid, r);
          if (!ok) continue;
          rows.push_back(r);
          std::fprintf(stderr, "%-18s 2^%-2d block=%-4d grid=%-6d batch=%d  %8.1f GB/s  wrong=%d/%d\n",
                       r.kernel, (int)log2((double)r.size), r.block, r.grid, r.batch,
                       r.best_gbps, r.incorrect, RUNS);
        }
  }

  for (int p = MIN_POW; p <= MAX_POW; ++p) {
    const size_t n = (size_t)1 << p;
    for (Row r : {measure_cub(c, n)}) {
      rows.push_back(r);
      std::fprintf(stderr, "%-18s 2^%-2d %26s  %8.1f GB/s  wrong=%d/%d\n",
                   r.kernel, p, "", r.best_gbps, r.incorrect, RUNS);
    }
  }

  FILE *f = std::fopen(out_path, "w");
  std::fprintf(f, "{\n  \"meta\": {\n");
  std::fprintf(f, "    \"gpu\": \"%s\",\n", prop.name);
  std::fprintf(f, "    \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
  std::fprintf(f, "    \"smCount\": %d,\n", prop.multiProcessorCount);
  std::fprintf(f, "    \"l2Bytes\": %d,\n", prop.l2CacheSize);
  std::fprintf(f, "    \"warmups\": %d,\n    \"runs\": %d,\n", WARMUPS, RUNS);
  std::fprintf(f, "    \"minPow\": %d,\n    \"maxPow\": %d,\n", MIN_POW, MAX_POW);
  std::fprintf(f, "    \"cubVersion\": %d\n", CUB_VERSION);
  std::fprintf(f, "  },\n  \"allConfigs\": [\n");
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row &r = rows[i];
    std::fprintf(f,
                 "    {\"kernel\":\"%s\",\"size\":%zu,\"block\":%d,\"grid\":%d,\"batch\":%d,"
                 "\"best\":%.4f,\"incorrect\":%d}%s\n",
                 r.kernel, r.size, r.block, r.grid, r.batch, r.best_gbps, r.incorrect,
                 (i + 1 == rows.size()) ? "" : ",");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::fprintf(stderr, "\nWrote %s (%zu rows)\n", out_path, rows.size());
  return 0;
}
