## glmfp4-k4-tp2 (2026-10-17T03:05:33Z)

nvidia GLM-5.3-Flash FP4-Local, TP=2, MTP K=4,  nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, reasoning_effort low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 34.07 | 37.76 | 0.419 |
| C2 | 46.85 | 27.22 | 0.679 |
| C3 | 49.76 | 18.8 | 0.841 |
| C4 | 71.63 | 21.37 | 0.866 |
| C5 | 84.73 | 19.87 | 0.936 |
| C6 | 80.39 | 15.51 | 1.017 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 46.16 | 31.8 | 22.69 | 24.57 | 23.2 | 18.22 |
| json | 47.42 | 34.23 | 25.13 | 28.66 | 25.8 | 21.43 |
| narrative | 25.0 | 18.17 | 12.07 | 13.47 | 12.04 | 9.43 |
| prose | 26.19 | 23.0 | 13.5 | 15.89 | 14.74 | 10.9 |
| math | 41.91 | 31.83 | 19.29 | 24.99 | 21.86 | 17.62 |
| reasoning | 25.17 | 21.2 | 14.64 | 15.46 | 13.96 | 11.15 |
| summary | 36.18 | 24.29 | 16.88 | 18.65 | 19.99 | 13.72 |
| structured | 44.86 | 32.27 | 22.55 | 26.46 | 23.26 | 18.93 |
| format | 46.95 | 28.19 | 22.41 | 24.17 | 23.96 | 18.2 |
| ceiling_count | 58.84 | 42.27 | 29.94 | 33.64 | 34.53 | 24.78 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 2.75 | 2.65 | 2.66 | 2.72 | 2.69 | 2.7 |
| json | 2.58 | 2.51 | 2.71 | 2.7 | 2.71 | 2.74 |
| narrative | 0.9 | 0.96 | 0.99 | 0.96 | 0.96 | 0.95 |
| prose | 0.99 | 1.36 | 1.2 | 1.31 | 1.3 | 1.23 |
| math | 2.26 | 2.29 | 2.15 | 2.22 | 2.26 | 2.26 |
| reasoning | 1.01 | 1.29 | 1.42 | 1.41 | 1.35 | 1.3 |
| summary | 1.86 | 1.74 | 1.75 | 1.79 | 1.81 | 1.72 |
| structured | 2.64 | 2.48 | 2.75 | 3.07 | 2.71 | 2.81 |
| format | 2.72 | 1.95 | 2.64 | 2.2 | 2.65 | 2.45 |
| ceiling_count | 3.64 | 3.49 | 3.59 | 3.5 | 3.53 | 3.46 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 0.902 | 1657.2 |
| 8000 | 5938 | 2.907 | 2042.5 |
| 32000 | 23844 | 10.986 | 2170.4 |
| 64000 | 47771 | 22.07 | 2164.5 |
