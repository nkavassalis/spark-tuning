## e192-tp2 (2026-10-13T00:56:33Z)

GLM-5.3-Flash e192, NVFP4 MTP K=2, fp8 KV 20 GiB/node, max-num-seqs 8, reasoning_effort=low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 21.02 | 22.4 | 0.499 |
| C2 | 40.93 | 22.87 | 0.488 |
| C3 | 55.28 | 21.23 | 0.523 |
| C4 | 73.09 | 20.39 | 0.529 |
| C5 | 69.82 | 15.8 | 0.623 |
| C6 | 88.04 | 16.7 | 0.6 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 25.53 | 22.88 | 27.82 | 25.47 | 19.12 | 19.96 |
| json | 22.39 | 21.27 | 23.48 | 22.1 | 16.64 | 18.7 |
| narrative | 15.85 | 17.84 | 15.57 | 14.34 | 11.14 | 11.73 |
| prose | 18.43 | 18.54 | 16.37 | 15.9 | 12.72 | 12.76 |
| math | 24.75 | 28.09 | 24.62 | 23.13 | 18.72 | 19.91 |
| reasoning | 22.25 | 23.1 | 18.79 | 19.53 | 14.15 | 15.74 |
| summary | 21.55 | 23.23 | 19.5 | 18.38 | 14.62 | 15.31 |
| structured | 24.55 | 24.3 | 22.99 | 22.43 | 16.3 | 17.37 |
| format | 26.29 | 26.62 | 21.93 | 22.26 | 18.77 | 18.82 |
| ceiling_count | 26.97 | 29.48 | 29.27 | 25.18 | 20.6 | 22.0 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.87 | 1.7 | 1.9 | 1.88 | 1.86 | 1.86 |
| json | 1.31 | 1.48 | 1.59 | 1.63 | 1.6 | 1.62 |
| narrative | 1.25 | 0.83 | 0.85 | 0.92 | 0.8 | 0.87 |
| prose | 1.48 | 0.92 | 1.0 | 1.18 | 1.11 | 1.06 |
| math | 1.86 | 1.78 | 1.76 | 1.79 | 1.79 | 1.79 |
| reasoning | 1.57 | 1.45 | 1.43 | 1.52 | 1.35 | 1.47 |
| summary | 1.59 | 1.4 | 1.33 | 1.41 | 1.36 | 1.39 |
| structured | 1.76 | 1.46 | 1.68 | 1.89 | 1.59 | 1.81 |
| format | 1.89 | 1.85 | 1.69 | 1.81 | 1.88 | 1.65 |
| ceiling_count | 1.85 | 1.98 | 2.0 | 1.97 | 1.99 | 2.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 0.953 | 1568.2 |
| 8000 | 5938 | 2.993 | 1984.0 |
| 32000 | 23844 | 12.203 | 1954.0 |
| 64000 | 47771 | 23.782 | 2008.7 |
