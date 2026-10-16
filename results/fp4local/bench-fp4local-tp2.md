## fp4local-tp2 (2026-10-16T10:51:30Z)

nvidia FP4-Local, TP=2, MTP K=4, nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.78 | 65.91 | 0.27 |
| C2 | 90.12 | 53.18 | 0.415 |
| C3 | 107.74 | 42.86 | 0.513 |
| C4 | 135.7 | 40.73 | 0.553 |
| C5 | 162.36 | 38.82 | 0.581 |
| C6 | 161.79 | 32.06 | 0.634 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 77.3 | 62.36 | 48.37 | 44.77 | 41.55 | 36.12 |
| json | 87.89 | 70.5 | 56.31 | 52.61 | 55.95 | 44.13 |
| narrative | 40.07 | 33.64 | 25.75 | 24.92 | 23.41 | 19.28 |
| prose | 41.38 | 34.56 | 30.25 | 26.39 | 25.01 | 20.86 |
| math | 83.76 | 67.0 | 54.24 | 53.86 | 48.23 | 41.82 |
| reasoning | 65.8 | 56.09 | 40.63 | 39.37 | 35.16 | 29.74 |
| summary | 46.72 | 33.01 | 29.6 | 27.78 | 26.29 | 21.8 |
| structured | 67.7 | 53.59 | 44.61 | 42.67 | 43.46 | 33.52 |
| format | 82.56 | 67.85 | 55.95 | 54.21 | 50.28 | 41.25 |
| ceiling_count | 96.37 | 83.05 | 72.33 | 70.91 | 63.94 | 55.41 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 2.83 | 2.86 | 2.75 | 2.62 | 2.64 | 2.8 |
| json | 3.39 | 3.36 | 3.25 | 3.17 | 3.39 | 3.24 |
| narrative | 0.93 | 0.99 | 0.87 | 0.86 | 0.97 | 0.89 |
| prose | 1.04 | 1.08 | 1.22 | 1.05 | 1.14 | 1.05 |
| math | 3.15 | 3.15 | 3.03 | 3.19 | 3.04 | 3.28 |
| reasoning | 2.26 | 2.34 | 2.08 | 2.21 | 2.2 | 2.22 |
| summary | 1.32 | 1.06 | 1.29 | 1.24 | 1.27 | 1.34 |
| structured | 2.52 | 2.58 | 2.61 | 2.61 | 2.67 | 2.57 |
| format | 3.17 | 3.1 | 3.17 | 3.14 | 3.23 | 3.12 |
| ceiling_count | 3.81 | 3.9 | 3.97 | 3.9 | 3.9 | 3.89 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 0.766 | 2601.6 |
| 8000 | 7874 | 2.89 | 2725.0 |
| 32000 | 31551 | 11.727 | 2690.5 |
| 64000 | 63221 | 23.581 | 2681.0 |
