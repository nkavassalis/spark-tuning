## glmfp4-k5-tp2 (2026-10-17T01:43:51Z)

nvidia GLM-5.3-Flash FP4-Local, TP=2, MTP K=5,  nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, reasoning_effort low per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 33.38 | 37.11 | 0.448 |
| C2 | 44.7 | 25.87 | 0.72 |
| C3 | 58.49 | 22.71 | 0.819 |
| C4 | 71.1 | 20.75 | 0.904 |
| C5 | 67.38 | 15.79 | 1.002 |
| C6 | 91.9 | 18.06 | 1.027 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 47.67 | 31.78 | 24.62 | 24.1 | 19.01 | 23.5 |
| json | 47.49 | 32.61 | 29.67 | 27.15 | 20.86 | 25.7 |
| narrative | 23.95 | 16.59 | 14.11 | 12.16 | 9.21 | 10.55 |
| prose | 25.68 | 20.19 | 17.6 | 14.53 | 11.15 | 12.13 |
| math | 40.6 | 28.9 | 25.65 | 22.79 | 17.08 | 19.45 |
| reasoning | 27.04 | 17.73 | 17.7 | 16.12 | 10.77 | 11.46 |
| summary | 30.94 | 22.14 | 19.06 | 17.4 | 14.5 | 15.22 |
| structured | 44.53 | 31.44 | 27.19 | 26.47 | 21.35 | 21.57 |
| format | 46.13 | 31.49 | 28.79 | 26.03 | 18.22 | 23.0 |
| ceiling_count | 62.04 | 53.14 | 41.34 | 36.29 | 27.7 | 34.13 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 3.33 | 3.23 | 3.03 | 3.15 | 3.09 | 3.17 |
| json | 2.93 | 2.79 | 2.83 | 2.93 | 2.95 | 2.97 |
| narrative | 1.03 | 0.94 | 1.02 | 0.95 | 0.95 | 1.01 |
| prose | 1.17 | 1.38 | 1.32 | 1.34 | 1.26 | 1.19 |
| math | 2.49 | 2.44 | 2.39 | 2.35 | 2.3 | 2.45 |
| reasoning | 1.4 | 1.17 | 1.62 | 1.74 | 1.34 | 1.33 |
| summary | 1.72 | 1.7 | 1.82 | 1.73 | 1.94 | 1.87 |
| structured | 3.0 | 2.98 | 3.02 | 3.38 | 3.35 | 3.21 |
| format | 3.03 | 2.92 | 3.07 | 2.87 | 2.49 | 2.93 |
| ceiling_count | 4.35 | 4.44 | 4.3 | 4.25 | 4.31 | 4.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 0.925 | 1615.6 |
| 8000 | 5938 | 2.928 | 2027.8 |
| 32000 | 23844 | 11.177 | 2133.3 |
| 64000 | 47771 | 22.368 | 2135.7 |
