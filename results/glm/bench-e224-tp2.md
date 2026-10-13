## e224-tp2 (2026-10-13T06:08:52Z)

GLM-5.3-Flash e224, NVFP4 MTP K=2, fp8 KV 20 GiB/node, max-num-seqs 8, reasoning_effort=low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 26.63 | 28.4 | 0.404 |
| C2 | 43.21 | 24.05 | 0.499 |
| C3 | 59.46 | 21.94 | 0.516 |
| C4 | 70.6 | 19.61 | 0.537 |
| C5 | 70.66 | 15.8 | 0.605 |
| C6 | 86.67 | 16.08 | 0.642 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 32.73 | 25.92 | 24.4 | 23.64 | 18.62 | 18.01 |
| json | 30.8 | 24.23 | 22.55 | 21.85 | 16.8 | 17.92 |
| narrative | 21.43 | 17.78 | 15.82 | 14.13 | 11.58 | 11.87 |
| prose | 25.47 | 22.37 | 19.52 | 17.36 | 13.96 | 14.4 |
| math | 31.54 | 27.99 | 24.54 | 22.97 | 17.52 | 20.02 |
| reasoning | 29.04 | 22.41 | 18.87 | 17.42 | 14.16 | 13.32 |
| summary | 27.85 | 22.52 | 19.86 | 17.7 | 14.92 | 14.58 |
| structured | 29.6 | 26.62 | 23.41 | 20.92 | 16.37 | 16.35 |
| format | 27.1 | 26.65 | 28.46 | 20.49 | 18.25 | 18.27 |
| ceiling_count | 33.99 | 31.48 | 26.97 | 25.07 | 20.31 | 22.49 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.86 | 1.84 | 1.88 | 1.9 | 1.87 | 1.89 |
| json | 1.61 | 1.61 | 1.65 | 1.64 | 1.6 | 1.57 |
| narrative | 0.81 | 0.84 | 0.87 | 0.9 | 0.87 | 0.89 |
| prose | 1.17 | 1.24 | 1.23 | 1.26 | 1.2 | 1.2 |
| math | 1.7 | 1.79 | 1.8 | 1.77 | 1.78 | 1.78 |
| reasoning | 1.5 | 1.39 | 1.35 | 1.38 | 1.35 | 1.23 |
| summary | 1.37 | 1.4 | 1.4 | 1.33 | 1.37 | 1.32 |
| structured | 1.61 | 1.86 | 1.77 | 1.72 | 1.72 | 1.74 |
| format | 1.34 | 1.86 | 1.89 | 1.82 | 1.82 | 1.89 |
| ceiling_count | 2.0 | 2.0 | 2.0 | 2.0 | 1.99 | 1.99 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 1.01 | 1478.6 |
| 8000 | 5938 | 3.193 | 1859.8 |
| 32000 | 23844 | 12.226 | 1950.2 |
| 64000 | 47771 | 24.043 | 1986.9 |
