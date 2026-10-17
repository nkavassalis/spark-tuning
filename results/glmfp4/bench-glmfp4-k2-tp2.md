## glmfp4-k2-tp2 (2026-10-17T02:38:43Z)

nvidia GLM-5.3-Flash FP4-Local, TP=2, MTP K=2,  nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, reasoning_effort low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 34.33 | 37.61 | 0.396 |
| C2 | 49.08 | 27.86 | 0.626 |
| C3 | 61.38 | 23.69 | 0.743 |
| C4 | 75.12 | 21.3 | 0.808 |
| C5 | 73.02 | 16.58 | 0.886 |
| C6 | 97.12 | 18.54 | 0.924 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 42.88 | 32.57 | 25.0 | 24.28 | 17.77 | 20.24 |
| json | 44.98 | 31.09 | 28.87 | 26.36 | 20.22 | 23.55 |
| narrative | 30.08 | 21.95 | 19.3 | 16.25 | 12.38 | 13.99 |
| prose | 32.61 | 25.63 | 21.73 | 18.57 | 14.89 | 15.78 |
| math | 39.18 | 29.32 | 25.72 | 23.76 | 18.0 | 22.04 |
| reasoning | 30.06 | 23.59 | 18.4 | 16.9 | 12.84 | 14.15 |
| summary | 34.97 | 25.77 | 23.69 | 19.6 | 15.99 | 15.53 |
| structured | 42.46 | 30.05 | 26.36 | 21.49 | 17.63 | 19.24 |
| format | 41.23 | 30.75 | 24.13 | 24.48 | 19.49 | 22.35 |
| ceiling_count | 43.18 | 33.41 | 30.97 | 28.25 | 22.66 | 24.04 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.76 | 1.8 | 1.74 | 1.75 | 1.79 | 1.77 |
| json | 1.74 | 1.6 | 1.64 | 1.61 | 1.62 | 1.66 |
| narrative | 0.83 | 0.81 | 0.92 | 0.85 | 0.85 | 0.9 |
| prose | 1.0 | 1.08 | 1.05 | 1.1 | 1.11 | 1.04 |
| math | 1.44 | 1.47 | 1.46 | 1.51 | 1.48 | 1.5 |
| reasoning | 0.9 | 1.02 | 0.91 | 1.07 | 0.97 | 1.01 |
| summary | 1.2 | 1.21 | 1.23 | 1.22 | 1.22 | 1.22 |
| structured | 1.73 | 1.72 | 1.67 | 1.6 | 1.65 | 1.68 |
| format | 1.64 | 1.69 | 1.47 | 1.65 | 1.63 | 1.69 |
| ceiling_count | 1.83 | 1.85 | 1.85 | 1.85 | 1.85 | 1.84 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 0.884 | 1690.3 |
| 8000 | 5938 | 2.887 | 2056.9 |
| 32000 | 23844 | 10.871 | 2193.3 |
| 64000 | 47771 | 21.873 | 2184.0 |
