## qwen-C-clean (2026-10-15T11:05:14Z)

clean re-check of results/mimo/bench-qwen-C-tp2.json, 2026-10-15

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 62.44 | 67.78 | 0.196 |
| C6 | 218.42 | 42.44 | 0.413 |

### Per-stream tok/s by category

| category | C1 | C6 |
|---|---|---|
| coding | 74.3 | 44.84 |
| json | 77.16 | 50.74 |
| narrative | 53.43 | 31.39 |
| prose | 51.91 | 34.29 |
| math | 76.75 | 46.52 |
| reasoning | 68.39 | 41.15 |
| summary | 53.75 | 34.11 |
| structured | 77.81 | 48.63 |
| format | 76.56 | 50.32 |
| ceiling_count | 77.79 | 58.88 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C6 |
|---|---|---|
| coding | 1.83 | 1.79 |
| json | 1.94 | 1.86 |
| narrative | 1.0 | 0.83 |
| prose | 0.94 | 1.02 |
| math | 1.88 | 1.83 |
| reasoning | 1.68 | 1.48 |
| summary | 1.06 | 1.07 |
| structured | 2.0 | 1.9 |
| format | 1.9 | 1.85 |
| ceiling_count | 1.99 | 1.99 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 32000 | 31551 | 0.802 | 39348.6 |
