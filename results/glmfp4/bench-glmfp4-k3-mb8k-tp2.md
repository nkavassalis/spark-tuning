## glmfp4-k3-mb8k-tp2 (2026-10-17T11:01:46Z)

nvidia GLM-5.3-Flash FP4-Local, TP=2, MTP K=3, mb8k nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, reasoning_effort low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 34.79 | 38.42 | 0.416 |
| C2 | 46.64 | 27.08 | 0.65 |
| C3 | 52.02 | 19.47 | 0.805 |
| C4 | 76.52 | 22.06 | 0.827 |
| C5 | 74.32 | 16.84 | 0.918 |
| C6 | 95.94 | 18.57 | 0.954 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 45.47 | 31.35 | 21.64 | 23.78 | 19.66 | 22.56 |
| json | 45.45 | 31.57 | 25.6 | 26.59 | 21.59 | 23.34 |
| narrative | 26.93 | 20.01 | 13.79 | 14.63 | 11.23 | 12.14 |
| prose | 30.3 | 22.22 | 15.77 | 19.52 | 13.5 | 15.02 |
| math | 41.27 | 30.5 | 21.0 | 23.97 | 19.33 | 22.12 |
| reasoning | 32.58 | 20.89 | 14.73 | 17.36 | 12.29 | 12.88 |
| summary | 36.52 | 26.44 | 17.16 | 21.99 | 14.58 | 15.89 |
| structured | 40.28 | 30.87 | 22.47 | 23.83 | 18.0 | 21.82 |
| format | 46.95 | 29.88 | 23.07 | 26.9 | 21.42 | 21.36 |
| ceiling_count | 49.97 | 37.25 | 26.29 | 30.68 | 23.09 | 27.33 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 2.32 | 2.24 | 2.35 | 2.37 | 2.38 | 2.33 |
| json | 2.13 | 2.03 | 2.33 | 2.16 | 2.21 | 2.17 |
| narrative | 0.85 | 0.92 | 0.99 | 0.93 | 0.96 | 0.9 |
| prose | 1.09 | 1.13 | 1.23 | 1.28 | 1.21 | 1.27 |
| math | 1.89 | 2.01 | 1.9 | 1.85 | 1.97 | 1.96 |
| reasoning | 1.34 | 1.13 | 1.17 | 1.39 | 1.19 | 1.19 |
| summary | 1.6 | 1.59 | 1.47 | 1.65 | 1.54 | 1.52 |
| structured | 1.93 | 2.26 | 2.29 | 2.28 | 2.21 | 2.33 |
| format | 2.37 | 1.71 | 2.31 | 2.31 | 2.35 | 2.28 |
| ceiling_count | 2.62 | 2.6 | 2.64 | 2.62 | 2.59 | 2.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 0.895 | 1668.8 |
| 8000 | 5938 | 2.714 | 2188.1 |
| 32000 | 23844 | 10.591 | 2251.3 |
| 64000 | 47771 | 21.295 | 2243.3 |
