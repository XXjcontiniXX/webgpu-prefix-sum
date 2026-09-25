#!/usr/bin/env python3
"""Find the single kernel config minimising L1 distance from the per-size best envelope."""
import json, math, glob, collections

V2, CUB = "cuda-v2", "cub-DeviceScan"
best = collections.defaultdict(list); cub = collections.defaultdict(list)
for fn in sorted(glob.glob("results/run*.json")):
    for r in json.load(open(fn))["allConfigs"]:
        if r["incorrect"]: continue
        if r["kernel"] == V2: best[(r["size"], r["block"], r["batch"])].append(r["best"])
        elif r["kernel"] == CUB: cub[r["size"]].append(r["best"])
M  = {k: max(v) for k, v in best.items()}
CU = {k: max(v) for k, v in cub.items()}
sizes = sorted(CU)

# The ideal line: best config at each size (what the throughput plot showed).
ideal = {s: max(v for k, v in M.items() if k[0] == s) for s in sizes}

def score(get):
    """L1 under three metrics; `get` maps size -> achieved GB/s."""
    rel = sum(abs(1 - get(s) / ideal[s]) for s in sizes)                 # relative shortfall
    lg  = sum(abs(math.log10(get(s)) - math.log10(ideal[s])) for s in sizes)
    raw = sum(abs(get(s) - ideal[s]) for s in sizes)
    return rel, lg, raw

blocks = sorted({k[1] for k in M}); batches = sorted({k[2] for k in M})

print("=== A. CUB parity: one (block, batch); grid follows N, as CUB's policy does ===")
cands = []
for b in blocks:
    for t in batches:
        if any((s, b, t) not in M for s in sizes): continue
        rel, lg, raw = score(lambda s, b=b, t=t: M[(s, b, t)])
        cands.append((rel, lg, raw, b, t))
cands.sort()
print(f"{'block':>5} {'batch':>5} | {'L1 rel':>7} {'L1 log':>7} {'L1 raw GB/s':>12} | {'geomean vs ideal':>16}")
for rel, lg, raw, b, t in cands:
    geo = math.exp(sum(math.log(M[(s,b,t)]/ideal[s]) for s in sizes)/len(sizes))
    print(f"{b:5d} {t:5d} | {rel:7.3f} {lg:7.3f} {raw:12.0f} | {geo:16.3f}")
relw, lgw, raww = cands[0], min(cands, key=lambda c: c[1]), min(cands, key=lambda c: c[2])
print(f"\nwinner by L1-rel: block={relw[3]} batch={relw[4]}   "
      f"by L1-log: block={lgw[3]} batch={lgw[4]}   by L1-raw: block={raww[3]} batch={raww[4]}")

print("\n=== B. Your definition: fix batch only, let CTA size vary per size ===")
for t in batches:
    get = lambda s, t=t: max(M[(s, b, t)] for b in blocks if (s, b, t) in M)
    rel, lg, raw = score(get)
    geo = math.exp(sum(math.log(get(s)/ideal[s]) for s in sizes)/len(sizes))
    print(f"batch={t} | L1 rel {rel:6.3f}  L1 log {lg:6.3f}  geomean vs ideal {geo:.3f}")

# Head to head vs CUB
b, t = relw[3], relw[4]
getB = lambda s: max(M[(s, bb, t)] for bb in blocks if (s, bb, t) in M)
print(f"\n=== Head to head ===")
print(f"{'size':>5} | {'ideal':>8} {'fixed':>8} {'batch-only':>10} {'CUB':>8} | {'fixed/CUB':>9} {'bo/CUB':>7} {'ideal/CUB':>9}")
print("-" * 82)
g = lambda a: math.exp(sum(math.log(x) for x in a) / len(a))
f_r, b_r, i_r = [], [], []
for s in sizes:
    fx, bo = M[(s, b, t)], getB(s)
    f_r.append(fx/CU[s]); b_r.append(bo/CU[s]); i_r.append(ideal[s]/CU[s])
    print(f"2^{int(math.log2(s)):<3d} | {ideal[s]:8.1f} {fx:8.1f} {bo:10.1f} {CU[s]:8.1f} | "
          f"{fx/CU[s]:9.3f} {bo/CU[s]:7.3f} {ideal[s]/CU[s]:9.3f}")
print(f"\ngeomean vs CUB   fixed({b}t,b{t}) {g(f_r):.3f}   batch-only {g(b_r):.3f}   per-size ideal {g(i_r):.3f}")
big = [i for i, s in enumerate(sizes) if s >= 2**20]
print(f"geomean 2^20+    fixed {g([f_r[i] for i in big]):.3f}   "
      f"batch-only {g([b_r[i] for i in big]):.3f}   ideal {g([i_r[i] for i in big]):.3f}")
