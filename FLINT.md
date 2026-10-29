# Recursive Flint (BF16 and NVFP4) on the Sparks

**Status 2026-10-29:** measured 2026-10-28/29 (speed, quality, the post-cutoff AtCoder set; NVFP4 on the head, BF16 on the worker,
one variant per node), then stopped and **removed from both nodes** on 2026-10-29 (weights, `~/flint-cache`, the vLLM nightly
image). `flint/setup_nodes.sh && flint/up.sh` rebuilds it (~103 GB download), `flint/down.sh --fp4local` goes back. The user's
plan for Flint is the int4 build on a Mac mini next to the cluster
([mac-tuning](https://github.com/nkavassalis/mac-tuning)), as an escalation model for hard problems. Current cluster state:
[AGENTS.md Purpose](AGENTS.md#purpose).

Recursive Flint is a 39.8B mixture-of-experts coding model (~2.5B active, 49 layers: 35 sliding-window (513) + 14 full
attention, 262K context, knowledge cutoff June 2026; [model card](https://huggingface.co/RecursiveSI/flint)). Its int4 MLX build
was measured on a Mac mini in the sister repo [mac-tuning](https://github.com/nkavassalis/mac-tuning) (FLINT.md there). There it had
the best LiveCodeBench of any model on either machine, but was slow to prefill and served one request at a time. This page
asks two questions. Do the full BF16 weights or the NVFP4 build do better than that int4 quant? And is Flint on a Spark a
viable model for this cluster's work (coding agents, several at once, long histories)?

**Verdict:**
- **Neither Spark build beats the Mac's int4.** On every test the three tie (all p ≥ 0.11).
- **Flint NVFP4 is the better hard-problem coder than Qwen FP4-Local:**
  - LiveCodeBench 63.8 vs 51.2% (p=0.013).
  - Post-cutoff AtCoder 68.7 vs 53.7% (p<0.001), a tie with DeepSeek.
  - It ties Qwen on HumanEval+, MBPP+ and long context.
- **NVFP4 is fast for agents:**
  - 82 tok/s decode at 1K.
  - 340 tok/s aggregate at 6 streams, 1.8× Qwen.
  - ~7K tok/s cold prefill, 2.4× Qwen.
  - It loses to Qwen at 128K (39 vs 49 tok/s).
- **BF16 is too slow for this hardware** at 34 tok/s, and no better.
- **Flint uses only one Spark.** The user ruled out a split cluster (Flint on one node, Qwen TP=1 on the other) because Qwen at
  TP=1 would be too slow (an estimate, not measured).
- **Qwen stayed the live model** (user's call, 2026-10-29).

## Setup (2026-10-29)

| | NVFP4 | BF16 |
|---|---|---|
| Checkpoint | `RecursiveSI/flint-nvfp4-gguf` @ `6ec43e19`, one 22.9 GB GGUF (W4A16, NVFP4 weights + per-tensor scales) | `RecursiveSI/flint` @ `2a247be0`, 15 safetensors shards, 79.6 GB |
| Node | head | worker |
| Weights in memory | 21.9 GiB | 74.2 GiB |
| `--gpu-memory-utilization` | 0.50 | 0.75 (the worker also runs the quality harness) |
| KV pool (bf16 KV) | 1,350,396 tokens (5.2 × 262K) | 596,559 tokens (2.3 × 262K) |
| Free memory after load | 52 GiB | 22 GiB |
| Kernels (GB10, sm_121) | `MarlinNvFp4LinearKernel` + Marlin NvFp4 MoE (weight-only FP4), FlashInfer attention | FlashInfer CUTLASS unquantized MoE, FlashInfer attention |
| Load time (cold, incl. FlashInfer JIT) | 175 s | 459 s |

- **Engine:** vLLM nightly `edad8b03` (arm64, CUDA 13.0, pinned by digest in `flint/lib.sh`). Flint support was merged into
  vLLM main on 2026-10-22, after v0.32.0. Flint's own fused kernels are gated to SM100 (B200/B300), so on GB10 vLLM runs its
  stock layers. The `cu134` nightlies need a newer driver than the nodes have (580, CUDA 13.0).
- **One variant per Spark, TP=1.** Flint's kernels are single-GPU and the model fits on one node, so TP=2 over the CX7 link would
  only add communication. Running the two variants side by side kept the one-model-per-node rule
  (`stop_other_models` in `scripts/models.sh` now accepts several names to keep).
- **Flags:** vLLM's Flint recipe (`--tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice
  --exclude-tools-when-tool-choice-none`; the NVFP4 server is given the GGUF *folder* plus `--hf-config-path` /
  `--tokenizer` from the BF16 repo), `--max-model-len 262144 --max-num-seqs 8`, bf16 KV cache.
  Our standard additions: **torch.compile off** (`mode 0`, AGENTS.md), FULL_DECODE_ONLY CUDA graphs, and the API key from `cluster.env`.
  The NVFP4 GGUF is converted at load by Flint's own loader (`vllm/models/flint/gguf_nvfp4.py`), a lossless repack into
  ModelOpt NVFP4. It is not vLLM's generic GGUF path.
- **Smoke test:** both split reasoning from content, and both return a parsed `read_file` tool call (`tool_calls`, finish
  `tool_calls`).
- Downloads: `flint/setup_nodes.sh`, plain HTTPS in a memory-capped container, every LFS file checked against its sha256.

## Speed (2026-10-29, nothing else on either server, API key locked)

**Decode**, `scripts/bench_decode.py` (500 tokens, `ignore_eos`, T=0.6, median of 3, the prompts every model ran). Flint has
no draft model, so there is no acceptance column. Files: `results/flint/flint-{nvfp4,bf16}-bench_decode-2026-10-29.json`.

| ctx | **Flint NVFP4** code / code_ts / edit | Flint BF16 | Qwen FP4-Local live (thinking default = `xhigh`) | DeepSeek (`high`) | Flint int4, Mac |
|---|---|---|---|---|---|
| 1K | **82.4 / 82.2 / 81.9** | 34.1 / 34.1 / 34.1 | 50.9 / 47.3 / 58.4 | 61.0 / 69.1 / 81.5 | 114–120 |
| 32K | 64.3 / 64.2 / 61.4 | 30.7 / 30.7 / 30.1 | — | 72.1 / 69.4 / 74.9 | 84–91 (16K) |
| 64K | 52.9 / 52.9 / 49.3 | 27.8 / 27.8 / 26.8 | — | 63.5 / 63.0 / 70.2 | 45–51 |
| 128K | 39.0 / 39.0 / 35.2 | 23.4 / 23.4 / 22.0 | 49.3 / 44.1 / 52.5 | 60.9 / 70.5 / 69.1 | 26–31 |
| mean of the six 1K/128K cells | **59.9** | 28.5 | 50.4 | 68.7 | — |

- Flint always reasons, so its column is the like-for-like match to the others' thinking-on decode.
- **Flint NVFP4 decodes 1.6× as fast as Qwen at 1K, but loses speed with context.** At 128K it is 0.8× Qwen. The 14
  full-attention layers read the whole KV cache on every token, and no MTP draft hides that cost.
- **BF16 is 2.4× slower than NVFP4** at 1K. Decode is memory-bound and BF16 moves ~4× the expert bytes per token.
- **The Mac's int4 is faster than either Spark build at 1K**, though both machines have the same 273 GB/s memory bandwidth.
  The cause was not isolated. The Spark runs vLLM's stock layers (Flint's fused kernels are SM100-only) with torch.compile
  off, while `rec` on the Mac has its own kernels. From 64K up, the Spark NVFP4 is faster.

**Concurrency and cold prefill**, `bench/mimobench.py` (T=0, keyed copy, run on the head; "thinking off" is ignored, Flint
reasons anyway): `results/flint/bench-flint-{nvfp4,bf16}-tp1.{md,json}`.

| C | **Flint NVFP4** aggregate / per-stream | Flint BF16 | Qwen FP4-Local live | DeepSeek |
|---|---|---|---|---|
| 1 | 81.4 / 84.1 | 33.4 / 34.6 | 57.6 / 63.9 | 71.5 / 84.0 |
| 2 | 135.2 / 70.5 | 59.3 / 31.2 | 87.9 / 51.2 | 101.6 / 62.0 |
| 3 | 197.3 / 69.1 | 81.3 / 28.5 | 115.6 / 45.4 | 123.2 / 50.3 |
| 4 | 246.7 / 64.6 | 99.0 / 26.0 | 141.2 / 41.4 | 133.1 / 41.0 |
| 5 | 299.1 / 63.0 | 117.7 / 24.8 | 158.0 / 36.8 | 117.9 / 47.5 |
| 6 | **339.8 / 59.8** | 134.5 / 23.6 | 187.5 / 36.6 | 121.6 / 43.7 |

| cold prefill target | **Flint NVFP4** tokens / TTFT / tok/s | Flint BF16 | Qwen FP4-Local | Flint int4, Mac |
|---|---|---|---|---|
| 8K | 4,129 / 0.57 s / 7,211 | 4,129 / 1.08 s / 3,814 | — | ~600 tok/s |
| 32K | 16,677 / **2.43 s** / 6,866 | 16,677 / 4.23 s / 3,940 | 31,551 / 11.0 s / 2,859 | — |
| 64K | 33,381 / **5.49 s** / 6,083 | 33,381 / 8.94 s / 3,736 | 63,221 / 22.5 s / 2,813 | ~270 at 100K+ |

- **Several agents at once suit NVFP4 well.** Per-stream speed falls only from 84 to 60 tok/s at 6 streams, so the aggregate
  (340) is 1.8× Qwen FP4-Local's and 2.8× DeepSeek's.
- **Prefill is the other big win.** Per token it is 2.4× Qwen FP4-Local. Flint's tokenizer also needs about half the tokens
  for this prose, so the same text is ready ~4× sooner. The Mac's weak point (a cold 100K prompt took 6.5–9 min there) is gone.

**Prefix cache**, `scripts/prefix_probe.py`: `results/flint/flint-{nvfp4,bf16}-prefix-probe-{24000,96000}-2026-10-29.json`.

| step | NVFP4, 21,213 tokens | BF16, 21,213 tokens |
|---|---|---|
| cold | 21,213 computed, 3.23 s | 21,213, 5.62 s |
| warm, immediately again | **13 recomputed**, 0.19 s | 13, 0.36 s |
| 1 token changed at the start | 21,214 recomputed, 3.24 s | 21,214, 5.63 s |
| 1 token changed at the end | 15 recomputed, 0.19 s | 15, 0.38 s |
| after 3 competing prompts | 13, 0.19 s | 13, 0.37 s |
| after 600 s idle | 13, 0.18 s | 13, 0.36 s |
| 85,413 tokens: warm / 1 token changed at the end | 5 / 7 recomputed, 0.34 s | 5 / 7, 0.49 s |

- Warm hits are free (block granularity, as on Qwen FP4-Local), and survive competing prompts and 10 minutes idle.
- The 85K "cold" row shared its first 21,200 tokens with the 24K probe (same nonce). It recomputed 64,213 tokens in 16.6 s
  (NVFP4) / 23.1 s (BF16), i.e. 3,870 / 2,780 tok/s at 21-85K.
- One run each.

## Quality (2026-10-29, T=0.6, 32K budget, API key locked)

The settings of the Mac int4 runs in mac-tuning (`FLINT-t06`, `FLINT-atc-t06`): T=0.6, top_p 0.95, top_k 20, 32,768 tokens for
every task (Flint always reasons; the card asks for at least 32K), no reasoning switch. Concurrency 8 instead of the Mac's 1.
`flint/run_suite.sh` runs it from the worker. Result folders `results/quality/{FL-nvfp4-t06,FL-bf16-t06,FL-nvfp4-atc-t06}`
(time to solve was stopped at the 10:00 EDT handback, 29/41 HE+ answers, not scored:
`results/quality/TTS32-fl-nvfp4-t06`). McNemar exact p on the same problems; one run per cell.

| Test | **Flint NVFP4** | Flint BF16 | Flint int4 (Mac) | Qwen FP4-Local `medium` / `xhigh` (32K) | DeepSeek (T=0.6) |
|---|---|---|---|---|---|
| LiveCodeBench (80, 2025-01..04) | 63.8 | 68.8 | 71.2 | 51.2 / 48.8 | 76.2 |
| HumanEval+ | 92.1 | 93.3 | 91.5 | 95.7 / — (16K) | 94.5 |
| post-cutoff AtCoder (147, sample tests) | **68.7** | not run (see below) | 68.0 | 53.7 / 53.1 | 70.1 |
| MBPP+ (378) | 80.7 | — | 79.1 | 79.9 (thinking off) | 78.0 (thinking off) |
| long context (144, 64K/128K) | 92.4 | — | — (no logprobs on `rec`) | 97.2 | 94.4 (T=1.0) |

- **The three Flint builds tie.** LCB80: BF16 vs NVFP4 9 / 5 (p=0.42), BF16 vs Mac int4 3 / 5 (p=0.73), NVFP4 vs Mac int4 2 / 8
  (p=0.11). HumanEval+: every p ≥ 0.45. AtCoder: NVFP4 vs Mac int4 7 / 6 (p=1.0). Neither the full weights nor NVFP4 measurably
  beats the Mac's int4. Most of the problems only the Mac solved sat at the 32K budget edge (6 of 8 for NVFP4), where one draw
  finishes and another doesn't. vLLM at concurrency 8 doesn't reproduce a seeded sample (COMPARISON.md), so each Spark run is an
  independent draw, while `rec` on the Mac is deterministic.
- **Flint beats Qwen on hard problems.** LCB80, NVFP4 against Qwen FP4-Local: 12 / 2 (p=0.013, `medium`) and 14 / 2 (p=0.004,
  `xhigh`). Against GLM FP4-Local (T=1.0) 7 / 3 (p=0.34) and MiMo (T=1.0) 6 / 2 (p=0.29): ties. DeepSeek is better: 4 / 14 (p=0.031).
- **On problems newer than every model, Flint NVFP4 ties the best.** AtCoder 101/147: against Qwen FP4-Local 27 / 5 and 29 / 6
  (p<0.001 both), against DeepSeek 4 / 6 (p=0.75). Rescored on sample tests only (`results/quality/samples-only/FL-*-t06`),
  NVFP4's lcb80 reads 72.5% (BF16 76.2%), so the drop to the new problems is −4 points, against −8 for the Mac int4, −14 for
  DeepSeek and ~0 for Qwen.
- **MBPP+ ties everything** except MiMo, which loses (NVFP4 over MiMo 34 / 10, p<0.001). Against Mac int4 13 / 7 (p=0.26), Qwen
  FP4-Local thinking off 18 / 15, GLM FP4-Local `low` 16 / 20, DeepSeek thinking off 24 / 14 (p=0.14).
- **Long context is a weak point:** 133/144, below GLM FP4-Local (1 / 11, p=0.006) and GLM E224 (0 / 11, p=0.001). It ties Qwen
  FP4-Local (4 / 11, p=0.12), DeepSeek (6 / 9) and MiMo (11 / 8). Most misses are two-hop retrieval at 128K (20/24).
- **BF16 on AtCoder was stopped after 55 of 147 problems** (the handback; it ran ~8 problems an hour). It scored 44/55 on those
  55, which are easier than the set as a whole. On the same 55 it ties NVFP4 (2 / 2), the Mac int4 (6 / 2, p=0.29) and DeepSeek
  (2 / 2), and beats Qwen `medium` (10 / 1, p=0.012). `results/quality/FL-bf16-atc-t06` holds the 55.
- GSM8K was not run (every model here scores ~96%).

## How to run it

```bash
flint/setup_nodes.sh            # image on both nodes; GGUF (22.9 GB) to the head, BF16 shards (79.6 GB) to the worker; sha256-checked
flint/up.sh both                # or nvfp4 / bf16; stops every other model; API key from cluster.env
flint/down.sh --fp4local        # stop both, relaunch the live Qwen FP4-Local (--qwen: config C)
```
Quality suite: copy `flint/run_suite.sh` to the worker's harness folder and run `./flint_suite.sh <nvfp4|bf16> <api> <key file>`.

## Caveats

- One run per cell. The Spark runs sample at concurrency 8 and don't reproduce seeds, while the Mac runs do.
- Tool calling was smoke-tested only (one parsed `read_file` call). No real pi session was run.
- Flint's own fused kernels are SM100-only, so these are vLLM's stock layers with torch.compile off. A GB10 kernel path or
  compile could be faster (not tried).
