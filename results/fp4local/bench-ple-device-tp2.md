## ple-device-tp2 (2026-10-21T12:13:11Z)

FP4-Local TP=2 K=2, ple-device, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 57.57 | 63.88 | 0.264 |
| C2 | 87.85 | 51.2 | 0.422 |
| C3 | 115.6 | 45.41 | 0.468 |
| C4 | 141.24 | 41.42 | 0.508 |
| C5 | 157.99 | 36.77 | 0.541 |
| C6 | 187.54 | 36.6 | 0.585 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 73.63 | 56.69 | 47.71 | 45.37 | 38.19 | 38.49 |
| json | 70.91 | 56.59 | 51.19 | 47.36 | 43.9 | 42.43 |
| narrative | 49.79 | 37.5 | 35.41 | 31.65 | 27.75 | 27.76 |
| prose | 49.4 | 40.83 | 36.54 | 32.12 | 28.66 | 29.55 |
| math | 71.59 | 59.55 | 53.21 | 47.2 | 41.83 | 39.37 |
| reasoning | 68.94 | 53.0 | 46.24 | 43.13 | 36.24 | 36.18 |
| summary | 47.95 | 39.77 | 37.0 | 33.1 | 30.76 | 28.03 |
| structured | 70.96 | 58.48 | 51.5 | 45.72 | 40.37 | 41.15 |
| format | 71.73 | 58.36 | 49.85 | 47.12 | 43.23 | 46.43 |
| ceiling_count | 73.11 | 61.97 | 57.44 | 53.5 | 49.28 | 49.09 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 1.88 | 1.8 | 1.7 | 1.74 | 1.7 | 1.72 |
| json | 1.82 | 1.8 | 1.8 | 1.78 | 1.8 | 1.76 |
| narrative | 0.94 | 0.76 | 0.92 | 0.82 | 0.84 | 0.86 |
| prose | 0.94 | 0.97 | 0.96 | 0.89 | 0.91 | 1.0 |
| math | 1.84 | 1.93 | 1.83 | 1.88 | 1.82 | 1.84 |
| reasoning | 1.69 | 1.64 | 1.58 | 1.6 | 1.56 | 1.57 |
| summary | 0.94 | 0.98 | 1.08 | 1.04 | 1.03 | 1.02 |
| structured | 1.88 | 1.88 | 1.85 | 1.79 | 1.82 | 1.82 |
| format | 1.88 | 1.85 | 1.84 | 1.83 | 1.85 | 1.88 |
| ceiling_count | 1.97 | 1.97 | 1.98 | 1.97 | 1.97 | 1.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1993 | 0.743 | 2681.4 |
| 8000 | 7874 | 2.746 | 2867.7 |
| 32000 | 31551 | 11.036 | 2858.8 |
| 64000 | 63221 | 22.473 | 2813.2 |
