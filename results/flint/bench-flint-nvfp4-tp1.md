## flint-nvfp4-tp1 (2026-10-29T05:23:47Z)

Recursive Flint nvfp4, vLLM nightly edad8b03, TP=1, compile off, API key locked; Flint always reasons, so thinking-off requests still reason

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 81.44 | 84.06 | 0.084 |
| C2 | 135.18 | 70.5 | 0.125 |
| C3 | 197.25 | 69.09 | 0.147 |
| C4 | 246.73 | 64.59 | 0.148 |
| C5 | 299.07 | 62.97 | 0.167 |
| C6 | 339.76 | 59.76 | 0.184 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 84.41 | 71.2 | 69.79 | 65.96 | 67.3 | 59.6 |
| json | 84.21 | 70.91 | 67.87 | 60.66 | 60.14 | 64.14 |
| narrative | 84.3 | 69.86 | 65.61 | 59.67 | 56.68 | 50.99 |
| prose | 84.45 | 70.92 | 69.21 | 61.07 | 59.11 | 56.72 |
| math | 84.31 | 71.54 | 71.8 | 68.25 | 65.56 | 67.18 |
| reasoning | 84.11 | 70.26 | 70.75 | 66.12 | 63.5 | 55.74 |
| summary | 83.14 | 68.81 | 67.42 | 60.53 | 58.3 | 52.2 |
| structured | 83.97 | 70.01 | 69.98 | 67.16 | 67.62 | 63.46 |
| format | 83.68 | 71.02 | 69.4 | 71.89 | 68.55 | 67.77 |
| ceiling_count | 84.74 | 71.25 | 69.06 | 63.5 | 61.43 | 59.58 |

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
| 2000 | 1055 | 0.177 | 5971.4 |
| 8000 | 4129 | 0.573 | 7210.6 |
| 32000 | 16677 | 2.429 | 6865.7 |
| 64000 | 33381 | 5.487 | 6083.3 |
