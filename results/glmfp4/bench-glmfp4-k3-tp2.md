## glmfp4-k3-tp2 (2026-10-17T02:11:49Z)

nvidia GLM-5.3-Flash FP4-Local, TP=2, MTP K=3,  nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, reasoning_effort low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 35.33 | 39.04 | 0.41 |
| C2 | 48.41 | 27.57 | 0.667 |
| C3 | 51.47 | 19.29 | 0.785 |
| C4 | 75.25 | 21.72 | 0.842 |
| C5 | 73.0 | 16.74 | 0.932 |
| C6 | 93.69 | 18.42 | 0.961 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 46.91 | 31.07 | 21.23 | 24.27 | 18.47 | 20.5 |
| json | 46.88 | 33.32 | 24.52 | 27.68 | 19.69 | 24.46 |
| narrative | 27.43 | 20.22 | 13.74 | 15.26 | 11.67 | 12.37 |
| prose | 30.19 | 23.2 | 15.52 | 17.71 | 12.76 | 14.2 |
| math | 41.56 | 31.08 | 21.12 | 24.5 | 18.58 | 21.13 |
| reasoning | 32.58 | 21.48 | 14.61 | 15.89 | 12.93 | 13.71 |
| summary | 36.48 | 24.34 | 17.87 | 19.35 | 16.06 | 17.64 |
| structured | 43.25 | 30.85 | 21.62 | 24.14 | 18.37 | 21.62 |
| format | 46.09 | 32.55 | 23.38 | 26.7 | 22.17 | 20.17 |
| ceiling_count | 49.56 | 37.27 | 26.66 | 29.56 | 23.78 | 26.52 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 2.43 | 2.34 | 2.24 | 2.4 | 2.38 | 2.37 |
| json | 2.19 | 2.06 | 2.22 | 2.21 | 2.07 | 2.25 |
| narrative | 0.89 | 0.93 | 0.99 | 1.02 | 0.97 | 0.95 |
| prose | 1.07 | 1.21 | 1.27 | 1.25 | 1.17 | 1.23 |
| math | 1.93 | 1.97 | 1.97 | 1.96 | 1.93 | 1.95 |
| reasoning | 1.31 | 1.2 | 1.18 | 1.25 | 1.27 | 1.27 |
| summary | 1.6 | 1.53 | 1.54 | 1.48 | 1.6 | 1.58 |
| structured | 2.16 | 2.21 | 2.19 | 2.1 | 2.26 | 2.22 |
| format | 2.3 | 2.33 | 2.33 | 2.34 | 2.33 | 2.25 |
| ceiling_count | 2.56 | 2.58 | 2.67 | 2.56 | 2.6 | 2.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 0.901 | 1657.4 |
| 8000 | 5938 | 2.899 | 2048.6 |
| 32000 | 23844 | 10.988 | 2169.9 |
| 64000 | 47771 | 22.075 | 2164.0 |
