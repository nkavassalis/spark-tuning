## qwen-C-tp2 (2026-10-07T21:56:41Z)

Qwen config C: RadixArk NVFP4+FP8-dense, MTP K=2, GMU 0.835, max-num-seqs 8, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.33 | 66.05 | 0.342 |
| C2 | 96.07 | 55.32 | 0.394 |
| C3 | 137.61 | 53.59 | 0.378 |
| C4 | 174.52 | 49.07 | 0.299 |
| C5 | 197.73 | 45.69 | 0.404 |
| C6 | 228.21 | 43.96 | 0.37 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 74.1 | 55.77 | 57.38 | 57.47 | 48.6 | 48.3 |
| json | 74.28 | 61.64 | 62.72 | 53.6 | 53.09 | 50.12 |
| narrative | 48.31 | 44.52 | 39.13 | 36.47 | 34.96 | 33.09 |
| prose | 52.79 | 47.63 | 42.64 | 38.23 | 34.72 | 33.08 |
| math | 74.65 | 63.93 | 61.8 | 53.28 | 49.99 | 49.02 |
| reasoning | 63.63 | 56.39 | 48.94 | 48.08 | 43.69 | 42.96 |
| summary | 55.35 | 40.45 | 40.58 | 38.51 | 35.85 | 34.11 |
| structured | 74.99 | 62.87 | 62.85 | 59.88 | 56.78 | 53.48 |
| format | 76.34 | 64.66 | 66.23 | 56.09 | 53.53 | 51.51 |
| ceiling_count | 76.93 | 70.47 | 66.88 | 65.68 | 61.09 | 58.48 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.82 | 1.76 | 1.79 | 1.82 | 1.78 | 1.81 |
| json | 1.84 | 1.78 | 1.82 | 1.88 | 1.84 | 1.78 |
| narrative | 0.82 | 0.91 | 0.87 | 0.92 | 0.92 | 0.96 |
| prose | 1.01 | 1.06 | 1.01 | 0.99 | 0.93 | 0.94 |
| math | 1.87 | 1.81 | 1.83 | 1.87 | 1.8 | 1.81 |
| reasoning | 1.44 | 1.45 | 1.49 | 1.59 | 1.49 | 1.55 |
| summary | 1.1 | 1.01 | 1.08 | 1.07 | 1.15 | 1.07 |
| structured | 1.91 | 1.83 | 1.88 | 1.87 | 1.84 | 1.85 |
| format | 1.9 | 1.88 | 1.9 | 1.88 | 1.89 | 1.89 |
| ceiling_count | 1.99 | 1.99 | 1.97 | 1.98 | 1.99 | 1.99 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 1.798 | 1108.5 |
| 8000 | 7874 | 3.36 | 2343.6 |
| 32000 | 31551 | 10.41 | 3030.9 |
| 64000 | 63221 | 21.153 | 2988.7 |
