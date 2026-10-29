## flint-bf16-tp1 (2026-10-29T05:29:27Z)

Recursive Flint bf16, vLLM nightly edad8b03, TP=1, compile off, API key locked; Flint always reasons, so thinking-off requests still reason

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 33.37 | 34.63 | 0.234 |
| C2 | 59.34 | 31.18 | 0.33 |
| C3 | 81.3 | 28.52 | 0.373 |
| C4 | 98.95 | 26.03 | 0.405 |
| C5 | 117.7 | 24.81 | 0.435 |
| C6 | 134.45 | 23.56 | 0.439 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 34.7 | 31.31 | 27.5 | 25.99 | 23.23 | 22.69 |
| json | 34.68 | 31.14 | 28.11 | 25.21 | 26.11 | 23.42 |
| narrative | 34.67 | 31.04 | 28.0 | 23.67 | 21.58 | 19.38 |
| prose | 34.73 | 31.3 | 27.03 | 23.56 | 21.01 | 19.68 |
| math | 34.52 | 31.54 | 29.39 | 31.06 | 30.97 | 29.53 |
| reasoning | 34.66 | 31.23 | 28.64 | 23.51 | 21.86 | 20.94 |
| summary | 34.6 | 30.48 | 27.2 | 24.09 | 20.62 | 18.75 |
| structured | 34.52 | 31.07 | 29.98 | 26.68 | 27.45 | 27.73 |
| format | 34.55 | 31.47 | 30.84 | 30.54 | 30.44 | 29.93 |
| ceiling_count | 34.36 | 31.3 | 27.39 | 25.39 | 23.42 | 23.23 |

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
| 2000 | 1055 | 0.383 | 2753.8 |
| 8000 | 4129 | 1.083 | 3814.1 |
| 32000 | 16677 | 4.233 | 3939.6 |
| 64000 | 33381 | 8.935 | 3735.9 |
