## spark-tuning-tp2 (2026-10-07T15:29:00Z)

recipe defaults, fp8 KV, GMU 0.90, DFlash 7, thinking off

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 42.82 | 48.48 | 0.302 |
| C2 | 63.72 | 37.78 | 0.459 |
| C3 | 85.02 | 33.84 | 0.516 |
| C4 | 101.0 | 30.42 | 0.544 |
| C5 | 122.11 | 29.49 | 0.583 |
| C6 | 139.63 | 28.57 | 0.613 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 67.57 | 56.53 | 48.32 | 44.35 | 38.4 | 39.07 |
| json | 51.09 | 41.08 | 36.34 | 32.94 | 32.33 | 29.86 |
| narrative | 20.62 | 15.25 | 13.2 | 12.34 | 11.5 | 10.74 |
| prose | 20.41 | 18.2 | 16.81 | 14.28 | 13.16 | 12.37 |
| math | 68.23 | 51.2 | 45.3 | 38.8 | 40.34 | 42.95 |
| reasoning | 34.2 | 28.42 | 23.16 | 20.67 | 19.08 | 18.8 |
| summary | 22.69 | 18.69 | 16.39 | 14.14 | 13.3 | 12.44 |
| structured | 77.43 | 55.55 | 50.72 | 46.13 | 49.6 | 47.45 |
| format | 74.12 | 55.14 | 54.36 | 50.12 | 47.66 | 43.42 |
| ceiling_count | 91.23 | 68.94 | 63.42 | 57.35 | 54.3 | 52.6 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 5.42 | 5.48 | 5.52 | 5.24 | 5.0 | 5.42 |
| json | 3.57 | 3.51 | 3.68 | 3.51 | 3.65 | 3.47 |
| narrative | 0.76 | 0.68 | 0.69 | 0.78 | 0.79 | 0.78 |
| prose | 0.84 | 1.08 | 1.27 | 1.1 | 1.12 | 1.1 |
| math | 5.07 | 5.05 | 4.84 | 4.66 | 4.92 | 5.15 |
| reasoning | 2.12 | 2.54 | 2.3 | 2.27 | 2.21 | 2.31 |
| summary | 1.08 | 1.26 | 1.22 | 1.25 | 1.29 | 1.23 |
| structured | 6.29 | 6.25 | 6.41 | 6.25 | 6.33 | 6.33 |
| format | 5.5 | 5.5 | 5.84 | 5.85 | 5.8 | 5.64 |
| ceiling_count | 6.97 | 6.89 | 6.91 | 6.92 | 6.9 | 6.94 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2004 | 0.874 | 2291.8 |
| 8000 | 7919 | 4.495 | 1761.8 |
| 32000 | 31836 | 21.43 | 1485.6 |
| 64000 | 63764 | 50.166 | 1271.1 |
