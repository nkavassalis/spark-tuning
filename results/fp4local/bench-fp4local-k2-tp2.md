## fp4local-k2-tp2 (2026-10-16T14:51:23Z)

nvidia FP4-Local, TP=2, MTP K=2, nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.51 | 62.78 | 0.273 |
| C2 | 86.88 | 50.58 | 0.437 |
| C3 | 115.46 | 45.28 | 0.495 |
| C4 | 137.87 | 40.44 | 0.537 |
| C5 | 155.66 | 36.69 | 0.574 |
| C6 | 186.93 | 36.62 | 0.604 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 71.93 | 56.01 | 48.87 | 43.45 | 37.92 | 38.26 |
| json | 70.22 | 55.56 | 52.86 | 46.11 | 44.63 | 43.31 |
| narrative | 48.5 | 37.13 | 33.16 | 29.8 | 27.35 | 26.23 |
| prose | 46.57 | 40.48 | 36.52 | 32.6 | 29.56 | 29.04 |
| math | 70.61 | 58.18 | 53.56 | 44.88 | 40.53 | 43.52 |
| reasoning | 68.31 | 52.93 | 45.68 | 41.6 | 35.29 | 35.56 |
| summary | 47.49 | 39.87 | 37.95 | 31.7 | 30.4 | 28.2 |
| structured | 69.78 | 57.76 | 49.57 | 45.09 | 41.74 | 39.43 |
| format | 71.65 | 57.33 | 49.32 | 48.7 | 42.76 | 45.99 |
| ceiling_count | 72.05 | 61.62 | 57.06 | 54.61 | 46.95 | 47.71 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.88 | 1.8 | 1.76 | 1.72 | 1.72 | 1.72 |
| json | 1.82 | 1.8 | 1.83 | 1.8 | 1.84 | 1.81 |
| narrative | 0.94 | 0.79 | 0.81 | 0.77 | 0.87 | 0.82 |
| prose | 0.87 | 0.95 | 0.96 | 0.94 | 0.98 | 0.98 |
| math | 1.84 | 1.88 | 1.82 | 1.84 | 1.84 | 1.86 |
| reasoning | 1.69 | 1.64 | 1.62 | 1.61 | 1.55 | 1.56 |
| summary | 0.94 | 1.01 | 1.03 | 1.01 | 1.1 | 1.06 |
| structured | 1.88 | 1.88 | 1.82 | 1.79 | 1.81 | 1.8 |
| format | 1.88 | 1.85 | 1.84 | 1.81 | 1.85 | 1.88 |
| ceiling_count | 1.97 | 1.97 | 1.99 | 1.97 | 1.97 | 1.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 0.792 | 2517.2 |
| 8000 | 7874 | 2.891 | 2723.2 |
| 32000 | 31551 | 11.567 | 2727.8 |
| 64000 | 63221 | 23.657 | 2672.4 |
