# Convenience targets. Each subdirectory also stands alone -- see its README.

.PHONY: help cuda cuda-run cuda-report webgpu webgpu-headless clean

help:
	@echo "make cuda            build the CUDA benchmark"
	@echo "make cuda-run        build, then run three sweeps into cuda/results/"
	@echo "make cuda-report     print the per-size table and the policy search"
	@echo "make webgpu          serve the WGSL benchmark and open it in Chrome"
	@echo "make webgpu-headless run the WGSL sweep headless (no display needed)"
	@echo "make clean           remove build output"

cuda:
	$(MAKE) -C cuda

cuda-run: cuda
	cd cuda && for r in 1 2 3; do ./bench results/run$$r.json; done

cuda-report:
	cd cuda && python3 analyze.py && python3 tune.py

webgpu:
	cd webgpu && ./run.sh

webgpu-headless:
	cd webgpu && npm install --silent && node bench-headless.mjs run.json

clean:
	$(MAKE) -C cuda clean
