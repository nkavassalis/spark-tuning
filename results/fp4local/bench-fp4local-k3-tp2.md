## fp4local-k3-tp2 (2026-10-16T15:11:18Z)

nvidia FP4-Local, TP=2, MTP K=3, nvidia Spark recipe (b12x, compile on), max-num-seqs 8, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.96 | 66.05 | 0.271 |
| C2 | 89.45 | 52.6 | 0.42 |
| C3 | 106.15 | 42.3 | 0.496 |
| C4 | 138.38 | 41.01 | 0.545 |
| C5 | 154.53 | 36.83 | 0.61 |
| C6 | 175.16 | 35.09 | 0.605 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 75.57 | 58.53 | 47.43 | 50.01 | 42.6 | 38.34 |
| json | 82.94 | 62.82 | 51.94 | 51.1 | 47.56 | 44.09 |
| narrative | 44.98 | 36.57 | 28.34 | 28.65 | 23.68 | 24.08 |
| prose | 44.11 | 38.81 | 31.28 | 29.74 | 25.66 | 25.43 |
| math | 77.95 | 62.99 | 51.68 | 49.7 | 45.57 | 42.51 |
| reasoning | 73.19 | 58.23 | 43.74 | 38.36 | 35.94 | 33.58 |
| summary | 44.1 | 36.17 | 29.29 | 28.1 | 26.26 | 24.79 |
| structured | 73.04 | 53.38 | 45.37 | 44.09 | 39.7 | 39.34 |
| format | 78.58 | 65.91 | 51.62 | 49.38 | 44.52 | 43.68 |
| ceiling_count | 85.67 | 73.22 | 61.41 | 60.96 | 54.91 | 54.85 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 2.43 | 2.32 | 2.32 | 2.39 | 2.4 | 2.34 |
| json | 2.76 | 2.63 | 2.55 | 2.59 | 2.67 | 2.54 |
| narrative | 1.0 | 0.98 | 0.87 | 0.99 | 0.88 | 0.97 |
| prose | 0.99 | 1.11 | 1.08 | 1.07 | 1.09 | 1.07 |
| math | 2.55 | 2.62 | 2.49 | 2.63 | 2.58 | 2.61 |
| reasoning | 2.32 | 2.18 | 2.06 | 1.83 | 2.05 | 1.95 |
| summary | 1.03 | 1.11 | 1.06 | 1.03 | 1.2 | 1.16 |
| structured | 2.39 | 2.25 | 2.2 | 2.25 | 2.26 | 2.3 |
| format | 2.56 | 2.56 | 2.48 | 2.51 | 2.58 | 2.56 |
| ceiling_count | 2.92 | 2.92 | 2.96 | 2.94 | 2.92 | 2.92 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 0.775 | 2572.1 |
| 8000 | 7874 | 2.856 | 2756.9 |
| 32000 | 31551 | 11.499 | 2743.7 |
| 64000 | 63221 | 23.428 | 2698.5 |
