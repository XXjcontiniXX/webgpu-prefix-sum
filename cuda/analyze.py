#!/usr/bin/env python3
"""Aggregate the CUDA sweeps: best correct throughput per size, across repeats."""
import json, math, sys, glob, collections, statistics

V2, CUB, COPY = "cuda-v2", "cub-DeviceScan", "copy-ceiling"
files = sorted(glob.glob("results/run*.json"))
per_run = []
meta = None
for fn in files:
    d = json.load(open(fn)); meta = d["meta"]
    best = collections.defaultdict(dict)   # kernel -> size -> row
    for r in d["allConfigs"]:
        if r["incorrect"]: continue
        k, s = r["kernel"], r["size"]
        if s not in best[k] or r["best"] > best[k][s]["best"]:
            best[k][s] = r
    per_run.append(best)

sizes = sorted(per_run[0][CUB].keys())

def across(kernel, size, field="best"):
    vals = [pr[kernel][size][field] for pr in per_run if size in pr[kernel]]
    return vals or None

def cfg(kernel, size):
    # config of the best repeat
    rows = [pr[kernel][size] for pr in per_run if size in pr[kernel]]
    if not rows: return "-"
    r = max(rows, key=lambda x: x["best"])
    return f"{r['block']}t x {r['grid']} blk x b{r['batch']}" if r["block"] else "-"

print(f"GPU {meta['gpu']}  sm{meta['sm']}  {meta['smCount']} SMs  "
      f"L2 {meta['l2Bytes']>>20} MiB  CUB {meta['cubVersion']}")
print(f"{meta['warmups']} warm-ups + {meta['runs']} timed runs per config, "
      f"best across {len(files)} sweeps\n")

hdr = f"{'size':>5} | {'v2':>9} {'CUB':>9} {'copy':>9} | {'v2/CUB':>8} {'v2/copy':>8} | spread"
print(hdr); print("-"*len(hdr))
for s in sizes:
    v = {k: max(across(k, s) or [float('nan')]) for k in (V2, CUB, COPY)}
    sp = max((max(a)-min(a))/max(a) for a in (across(k,s) for k in (V2,CUB)) if a and len(a)>1)
    print(f"2^{int(math.log2(s)):<3d} | {v[V2]:9.1f} {v[CUB]:9.1f} {v[COPY]:9.1f} | "
          f"{v[V2]/v[CUB]:8.3f} {v[V2]/v[COPY]:8.3f} | {sp:5.1%}")

print(f"\n{'size':>5} | best config, revised")
for s in sizes:
    print(f"2^{int(math.log2(s)):<3d} | {cfg(V2,s)}")
