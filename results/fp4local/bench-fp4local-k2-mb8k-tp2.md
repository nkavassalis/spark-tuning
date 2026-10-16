## fp4local-k2-mb8k-tp2 (2026-10-16T16:18:47Z)

nvidia FP4-Local, TP=2, MTP K=2, mb8k nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.83 | 63.35 | 0.278 |
| C2 | 86.91 | 50.84 | 0.46 |
| C3 | 117.79 | 46.15 | 0.513 |
| C4 | 138.51 | 40.92 | 0.535 |
| C5 | 156.23 | 36.78 | 0.588 |
| C6 | 182.93 | 36.02 | 0.618 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 72.88 | 56.27 | 50.5 | 42.77 | 37.34 | 37.9 |
| json | 70.35 | 57.32 | 50.34 | 46.25 | 44.3 | 41.7 |
| narrative | 48.5 | 39.59 | 34.61 | 31.81 | 27.43 | 27.15 |
| prose | 49.9 | 41.66 | 35.37 | 33.2 | 29.2 | 27.76 |
| math | 71.18 | 59.02 | 55.33 | 45.7 | 40.63 | 38.89 |
| reasoning | 66.63 | 52.13 | 46.56 | 43.3 | 38.17 | 34.66 |
| summary | 48.1 | 41.98 | 38.89 | 32.06 | 29.87 | 27.56 |
| structured | 71.3 | 54.57 | 52.47 | 45.23 | 41.84 | 42.48 |
| format | 71.3 | 55.0 | 51.27 | 48.0 | 42.25 | 46.04 |
| ceiling_count | 72.58 | 61.3 | 56.93 | 52.84 | 48.44 | 48.96 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.88 | 1.8 | 1.71 | 1.7 | 1.7 | 1.74 |
| json | 1.82 | 1.8 | 1.72 | 1.77 | 1.83 | 1.81 |
| narrative | 0.94 | 0.86 | 0.87 | 0.86 | 0.84 | 0.9 |
| prose | 0.99 | 0.95 | 0.93 | 0.98 | 0.94 | 0.92 |
| math | 1.84 | 1.93 | 1.83 | 1.85 | 1.85 | 1.86 |
| reasoning | 1.62 | 1.62 | 1.63 | 1.55 | 1.63 | 1.54 |
| summary | 0.92 | 1.01 | 1.05 | 0.99 | 1.05 | 1.04 |
| structured | 1.88 | 1.88 | 1.88 | 1.79 | 1.8 | 1.86 |
| format | 1.88 | 1.85 | 1.86 | 1.85 | 1.85 | 1.88 |
| ceiling_count | 1.97 | 1.97 | 1.98 | 1.98 | 1.97 | 1.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 0.768 | 2594.3 |
| 8000 | 7874 | 2.8 | 2812.5 |
| 32000 | 31551 | 11.321 | 2787.0 |
| 64000 | 63221 | 23.165 | 2729.1 |
