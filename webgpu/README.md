# WGSL kernels

Two WGSL scans swept across a tuning grid in Chrome. Both are single-pass
decoupled-lookback inclusive scans over `u32`, four elements per thread per batch
step (`vec4<u32>`).

| file | |
|---|---|
| `prefix-sum-v1.wgsl` | The [ScanBox](https://jamescontini.com/Thesis.pdf) kernel. Thread-contiguous runs, raking block scan through shared memory. |
| `prefix-sum-v2.wgsl` | Warp-carried scan over coalesced loads, every load issued before the first scan. |
| `main.js` | Harness: builds the tuning grid, times each configuration, checks every result. |
| `index.html` | Page that loads `main.js`. |

## Run it

```bash
./run.sh                    # serves this directory, opens Chrome with WebGPU flags
PORT=9000 ./run.sh          # different port
CHROME=/path/to/chrome ./run.sh
```

Needs Chrome 144+. The script picks a Chrome automatically on Linux, macOS and
Windows (Git Bash), serves over `127.0.0.1`, and shuts the server down when Chrome
closes. Progress and results print to the browser console; the page also draws a
throughput chart.

### Headless, for reproducible numbers

```bash
npm install
node bench-headless.mjs run.json
```

Needs no display, so it works over SSH. Any extra argv entries pass through to
Chrome as flags, e.g. `--enable-dawn-features=disable_robustness`.

## v1 vs v2

![WebGPU prefix-sum throughput, v1 vs v2, Nvidia RTX 5070](prefix-sum-v1-vs-v2.png)

Measured on an Nvidia RTX 5070. v2's coalesced layout leads from 2<sup>19</sup>
through 2<sup>24</sup>, widest around 2<sup>20</sup>–2<sup>22</sup>; the two
converge at the small sizes, where a dispatch costs more than the scan.

## Narrowing a sweep

`main.js` reads its grid from the URL, which is useful for a quick check before
committing to the full run:

| param | default |
|---|---|
| `threads` | `32,64,128,256` |
| `workgroups` | `32,…,32768` |
| `batch` | `1,2,4` |
| `lookback` | `1,0` — `1` is parallel lookback, `0` the serial spin |
| `shaders` | `v2,v1` |
| `minpow` / `maxpow` | `12` / `25` |
| `warmups` / `runs` | `2` / `5` |
| `peak` | `672` — GB/s reference line for the chart |
| `verify` | unset; `1` checks every output element instead of the last |

```
index.html?threads=128&workgroups=2048&batch=4&runs=2
```

Pass `HTTP_PORT`, `CDP_PORT` or `QUERY` as environment variables to
`bench-headless.mjs` to do the same headless.

## Notes on the sweeps

The reported sweeps use `lookback=1`. The harness recovers from a lost device and
re-measures a reference configuration afterwards to confirm the replacement is not
degraded; that check exists because a device recreated after a hang sometimes
serves large buffers from system memory at about PCIe speed.
