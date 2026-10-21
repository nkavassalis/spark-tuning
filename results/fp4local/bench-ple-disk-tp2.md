## ple-disk-tp2 (2026-10-21T11:43:52Z)

FP4-Local TP=2 K=2, ple-disk, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 55.94 | 62.13 | 0.276 |
| C2 | 86.07 | 49.95 | 0.42 |
| C3 | 113.01 | 44.89 | 0.525 |
| C4 | 139.68 | 41.27 | 0.555 |
| C5 | 157.64 | 37.28 | 0.593 |
| C6 | 182.13 | 35.67 | 0.597 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 72.64 | 56.3 | 49.43 | 46.75 | 41.47 | 39.8 |
| json | 69.98 | 56.08 | 51.18 | 45.8 | 43.72 | 41.18 |
| narrative | 48.44 | 36.94 | 33.56 | 32.27 | 27.7 | 26.04 |
| prose | 45.55 | 42.27 | 36.46 | 34.59 | 29.33 | 27.47 |
| math | 69.88 | 57.92 | 50.76 | 46.04 | 41.04 | 41.47 |
| reasoning | 67.17 | 52.54 | 44.77 | 40.66 | 36.86 | 36.16 |
| summary | 47.0 | 39.06 | 35.79 | 31.01 | 29.23 | 27.26 |
| structured | 68.56 | 53.44 | 49.34 | 46.94 | 43.99 | 38.83 |
| format | 69.96 | 54.99 | 52.75 | 47.33 | 42.18 | 42.84 |
| ceiling_count | 71.66 | 61.41 | 57.17 | 53.87 | 47.55 | 48.09 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.88 | 1.8 | 1.68 | 1.79 | 1.77 | 1.76 |
| json | 1.82 | 1.8 | 1.77 | 1.74 | 1.82 | 1.8 |
| narrative | 0.94 | 0.76 | 0.83 | 0.9 | 0.9 | 0.82 |
| prose | 0.84 | 1.03 | 0.98 | 0.98 | 0.97 | 0.91 |
| math | 1.84 | 1.88 | 1.86 | 1.85 | 1.85 | 1.86 |
| reasoning | 1.69 | 1.64 | 1.58 | 1.57 | 1.64 | 1.57 |
| summary | 0.94 | 0.98 | 1.05 | 0.93 | 1.05 | 1.0 |
| structured | 1.88 | 1.84 | 1.78 | 1.7 | 1.8 | 1.77 |
| format | 1.88 | 1.85 | 1.88 | 1.85 | 1.84 | 1.87 |
| ceiling_count | 1.97 | 1.97 | 1.99 | 1.97 | 1.97 | 1.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 0.77 | 2586.7 |
| 8000 | 7874 | 2.881 | 2733.3 |
| 32000 | 31551 | 11.548 | 2732.3 |
| 64000 | 63221 | 23.528 | 2687.0 |
