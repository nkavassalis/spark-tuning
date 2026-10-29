## deepseek-tp2 (2026-10-28T16:39:19Z)

DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw TensorFold TP=2, drafts on, API key locked, thinking off per request

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 71.53 | 84.01 | 0.261 |
| C2 | 101.56 | 61.95 | 0.392 |
| C3 | 123.16 | 50.33 | 0.511 |
| C4 | 133.07 | 40.98 | 0.656 |
| C5 | 117.86 | 47.45 | 1.372 |
| C6 | 121.57 | 43.74 | 1.945 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 103.91 | 79.21 | 61.17 | 46.17 | 61.71 | 56.32 |
| json | 80.99 | 45.93 | 50.34 | 41.56 | 48.19 | 47.09 |
| narrative | 56.31 | 40.31 | 35.3 | 30.61 | 33.68 | 32.91 |
| prose | 57.57 | 47.69 | 37.85 | 41.0 | 38.74 | 36.23 |
| math | 88.36 | 73.72 | 60.27 | 51.19 | 55.78 | 50.23 |
| reasoning | 85.02 | 57.71 | 51.64 | 40.86 | 47.35 | 42.55 |
| summary | 56.5 | 46.26 | 34.84 | 33.22 | 33.0 | 29.7 |
| structured | 117.09 | 89.58 | 59.68 | 43.35 | 51.41 | 46.17 |
| format | 110.32 | 77.1 | 61.86 | 40.84 | 57.16 | 52.48 |
| ceiling_count | 129.34 | 107.76 | 77.08 | 59.83 | 65.1 | 65.75 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding |  |  |  |  |  |  |
| json |  |  |  |  |  |  |
| narrative |  |  |  |  |  |  |
| prose |  |  |  |  |  |  |
| math |  |  |  |  |  |  |
| reasoning |  |  |  |  |  |  |
| summary |  |  |  |  |  |  |
| structured |  |  |  |  |  |  |
| format |  |  |  |  |  |  |
| ceiling_count |  |  |  |  |  |  |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1162 | 0.816 | 1423.4 |
| 8000 | 4504 | 2.452 | 1837.0 |
| 32000 | 18304 | 8.884 | 2060.2 |
| 64000 | 36536 | 17.739 | 2059.6 |
