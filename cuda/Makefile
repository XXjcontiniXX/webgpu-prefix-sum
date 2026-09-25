NVCC    ?= /usr/local/cuda/bin/nvcc
ARCH    ?= sm_120
NVFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

bench: bench.cu prefix_sum_v2.cuh
	$(NVCC) $(NVFLAGS) bench.cu -o bench

clean:
	rm -f bench results*.json

.PHONY: clean
