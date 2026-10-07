# Qwen3.8-Flash-Next (config C) vs MiMo-V2.6-Flash on the same dual DGX Spark cluster

Measured 2026-10-07 on the same two Sparks, same vLLM TP=2 over CX7, one model at a time, nothing else on
the server. Each model runs its tested deployment:

| | Qwen, config C ([QWEN.md](QWEN.md), [FP8DENSE.md](FP8DENSE.md)) | MiMo ([MIMO.md](MIMO.md)) |
|---|---|---|
| Weights | RadixArk NVFP4 experts + FP8-dense linears | XiaomiMiMo MXFP4 experts + fp8 attention |
| Spec decode | MTP, K=2 | DFlash, 7 draft tokens |
| KV / context | fp8, GMU 0.835, 262K, max-num-seqs 8 | fp8, GMU 0.90, 300K, max-num-seqs 8 |
| KV pool (boot log 2026-10-07) | 4.25M tokens (16.2× full 262K) | 2.05M tokens (6.8× full 300K) |
| Server sampling defaults | vLLM defaults; thinking on by template default | temperature 1.0 / top_p 0.95 / rep-pen 1.05; thinking off |

## Summary

| | Qwen C | MiMo | Winner |
|---|---|---|---|
| Single-user code decode, 1K ctx, thinking off (tok/s, median of 3) | 66–70 | 37–52 | Qwen +35–78% |
| Single-user code decode, 128K ctx, thinking off | 65–68 | 25–30 | Qwen ×2.2–2.6 |
| Recipe bench aggregate, C1 / C6 (tok/s) | 58.3 / 228.2 | 42.8 / 139.6 | Qwen +36% / +63% |
| Cold prefill at 32K / 64K (tok/s) | 3,031 / 2,989 | 1,486 / 1,271 | Qwen ×2.0–2.4 |
| Cold prefill at 2K (tok/s) | 1,109 | 2,292 | MiMo (likely a single-run outlier for Qwen, see below) |
| HumanEval+ (greedy, thinking off) | **93.3** | 88.4 | Qwen (p=0.057) |
| MBPP+ (greedy, thinking off) | **79.9** | 74.3 | Qwen (p=0.001, significant) |
| GSM8K | 96.4 | 96.4 | tie |
| HumanEval+ (thinking) | **92.7** | 89.0 | Qwen (p=0.21, not significant) |
| LiveCodeBench (thinking, 80 problems, 16K-token budget) | 41.2 | **46.2** | MiMo (p=0.34, not significant) |
| Long context 64K/128K retrieval + two-hop + reverse | **95.1%** (137/144) | 90.3% (130/144) | Qwen (two-hop: 41/48 vs 34/48) |
| Multimodal input (image / video / audio) | no | yes (recipe-verified; not tested here) | MiMo |

**Bottom line for single-user coding with pi on this cluster:** Qwen config C is faster on every axis except
2K prefill, and at least as accurate on every test except LiveCodeBench in thinking mode, where MiMo is
ahead but the difference isn't significant. The gaps are largest at long context, which is where agent sessions
spend their time. MiMo's case is multimodal input and a 300K window; on throughput it loses even at C6.

p-values are exact McNemar tests on paired per-problem pass/fail (`quality/quality_eval.py compare-pass`).
For scale, in the FP8-dense evaluation two runs of the *same* Qwen config differed by 1–2 points, so treat
anything without a small p-value as a tie.

---

## Speed

### Single-user decode (`scripts/bench_decode.py`, 500 tokens, `ignore_eos`)

Same prompts, T=0.6, median of 3. Thinking was **off for both** (MiMo's server default; Qwen sent
`enable_thinking: false`). Files: `results/mimo/qwen-bench_decode-vsmimo-thinkoff-2026-10-07.json`,
`results/mimo/bench_decode-vsqwenC-2026-10-07.json`.

| ctx | task | Qwen C tok/s (acc.len) | MiMo tok/s (acc.len) | Qwen / MiMo |
|---|---|---|---|---|
| 1K | code | 66.3 (2.55) | 37.2 (3.50) | 1.78× |
| 1K | code_ts | 68.0 (2.62) | 40.7 (3.93) | 1.67× |
| 1K | edit | 70.1 (2.73) | 52.2 (5.00) | 1.34× |
| 128K | code | 65.2 (2.56) | 24.6 (3.45) | 2.65× |
| 128K | code_ts | 65.1 (2.55) | 29.7 (4.17) | 2.19× |
| 128K | edit | 67.6 (2.72) | 30.4 (4.30) | 2.22× |

The reason is step cost, not drafting. In engine steps per second (tok/s ÷ acceptance length), Qwen runs
~25–26 steps/s at both 1K and 128K. MiMo runs ~10.5 at 1K and ~7 at 128K. DFlash accepts 1.4–1.8× more tokens per
step than MTP K=2, but each MiMo step costs ~2.5–3.6× more and gets slower with context.

The wider single-run grid (T=0 and 0.6, 1K and 64K, adds prose) shows the same picture
(`results/mimo/qwen-bench_decode-wide-thinkoff-2026-10-07.json` vs `results/mimo/bench_decode-2026-10-07.json`):

| ctx | T | code | code_ts | edit | prose |
|---|---|---|---|---|---|
| 1K | 0.0 | 71.1 vs 40.0 | 68.7 vs 47.6 | 72.5 vs 58.0 | 56.9 vs 23.1 |
| 1K | 0.6 | 67.8 vs 38.1 | 67.3 vs 44.6 | 70.8 vs 44.5 | 53.6 vs 23.3 |
| 64K | 0.0 | 66.3 vs 31.5 | 69.3 vs 35.9 | 70.4 vs 46.7 | 52.6 vs 17.5 |
| 64K | 0.6 | 67.0 vs 28.4 | 61.3 vs 30.1 | 69.8 vs 45.3 | 52.1 vs 16.9 |

(Qwen vs MiMo, tok/s.) Prose is MiMo's worst case: DFlash accepts ~1 token per step on it.

**Thinking on vs off matters for Qwen's speed.** With the server/template default (thinking on, the way pi
talks to the `qwen` provider today) the same Qwen cells measure 48–56 tok/s, the same as the 2026-09-27 numbers
(`results/mimo/qwen-bench_decode-vsmimo-default-2026-10-07.json`). MTP acceptance is lower on reasoning text
(acc.len 1.9–2.2 vs 2.55–2.73). Even so, Qwen with thinking on is faster than MiMo with thinking off in every cell.

### Recipe bench (`bench/mimobench.py` from the MiMo recipe, temperature 0, thinking off, unique prefixes)

Files: `results/mimo/bench-qwen-C-tp2.{md,json}` and `results/mimo/bench-spark-tuning-tp2.{md,json}`.

| C | Qwen C aggregate | MiMo aggregate | Qwen C per-stream | MiMo per-stream |
|---|---|---|---|---|
| 1 | 58.3 | 42.8 | 66.1 | 48.5 |
| 2 | 96.1 | 63.7 | 55.3 | 37.8 |
| 3 | 137.6 | 85.0 | 53.6 | 33.8 |
| 4 | 174.5 | 101.0 | 49.1 | 30.4 |
| 5 | 197.7 | 122.1 | 45.7 | 29.5 |
| 6 | 228.2 | 139.6 | 44.0 | 28.6 |

Per-stream at C1 by category (Qwen / MiMo): coding 74.1 / 67.6, json 74.3 / 51.1, math 74.7 / 68.2,
structured 75.0 / 77.4, format 76.3 / 74.1, reasoning 63.6 / 34.2, summary 55.4 / 22.7, prose 52.8 / 20.4,
narrative 48.3 / 20.6. MiMo nearly catches up on short, highly predictable output (structured, format,
code), where DFlash accepts 5–6 of 7 drafts. Open-ended text runs at less than half Qwen's speed. The
bench's "DFlash accepted tokens per draft step" column for Qwen is MTP K=2 acceptance (max 2), not DFlash.

Cold prefill (one request, unique prefix):

| prompt tokens | Qwen C tok/s (TTFT) | MiMo tok/s (TTFT) |
|---|---|---|
| ~2K | 1,109 (1.80 s) | 2,292 (0.87 s) |
| ~8K | 2,344 (3.36 s) | 1,762 (4.50 s) |
| ~32K | 3,031 (10.4 s) | 1,486 (21.4 s) |
| ~64K | 2,989 (21.2 s) | 1,271 (50.2 s) |

The Qwen 2K cell is out of line with its own 8K–64K trend. It's a single run and probably includes a one-off
warm-up cost, so don't read it as a real MiMo advantage without a rerun. At agent-sized prompts (≥32K) Qwen
prefills 2–2.4× faster.

---

## Quality

Same harness, data and sandbox for both (`quality/run_quality.sh`, `quality/run_deep.sh`, run on the worker).
Qwen numbers are the existing config-C runs from 2026-09-27/28 (`results/quality/C-radix-fp8dense`,
`results/quality/C-deep`), not rerun. MiMo: `results/quality/M-mimo-tp2`, `results/quality/M-deep`.

| Test | n | Qwen C | MiMo | Qwen-only / MiMo-only passes | McNemar p |
|---|---|---|---|---|---|
| HumanEval+ (base) | 164 | 95.1 | 90.9 | | |
| **HumanEval+ (plus)**, greedy, thinking off | 164 | **93.3** | 88.4 | 11 / 3 | 0.057 |
| MBPP+ (base) | 378 | 93.1 | 88.9 | | |
| **MBPP+ (plus)**, greedy, thinking off | 378 | **79.9** | 74.3 | 31 / 10 | **0.001** |
| GSM8K, greedy, thinking off | 1319 | 96.4 | 96.4 | 21 / 22 | 1.0 |
| HumanEval+ (plus), thinking, 16K budget | 164 | **92.7** | 89.0 | 11 / 5 | 0.21 |
| LiveCodeBench (80 recent problems), thinking, 16K budget | 80 | 41.2 | **46.2** | 3 / 7 | 0.34 |
| Long context: retrieve / two-hop / reverse at 64K + 128K | 144 | **95.1%** | 90.3% | two-hop 41/48 vs 34/48 | — |

Conditions and caveats:
- Thinking-off tests are greedy (temperature 0) for both. MiMo requests also get its server-default
  repetition penalty 1.05, because the comparison is "as deployed".
- Thinking tests use each model's recommended sampling: Qwen 0.6 / top_p 0.95 / top_k 20, MiMo the
  checkpoint's 1.0 / 0.95 (`THINK_TEMP`/`THINK_TOP_P`/`THINK_TOP_K` in `run_deep.sh`). Both are single seeded runs.
- LiveCodeBench is budget-bound for both: 45/80 (Qwen) and 39/80 (MiMo) responses hit the 16K-token cap,
  so this measures "solves it within 16K tokens of thinking". MiMo thinks more tersely (HumanEval+ mean
  completion 833 tokens vs Qwen's 1,877).
- No extraction failures: every HumanEval+/MBPP+ answer from both models contained a code block. Truncations:
  MiMo 0/164, 0/378, 1/1319; Qwen 1, 1, 19.
- Long-context prompts are the same text, so token counts differ by tokenizer (mean 95K MiMo vs 102K Qwen).

### Corpus NLL (bits per byte)

Per-token NLL can't be compared across tokenizers, so `quality_eval.py bpb` normalises by bytes. On the fixed
corpus (Python stdlib source + license text, 769 KB): **Qwen C 0.048 bits/byte, MiMo 0.154 bits/byte**. That's a
3× gap, but this corpus is mostly well-known code that is surely in both models' training data, so it measures
memorisation of that text as much as modelling quality. It was built to catch quantisation damage within
one model, not to rank models. Use it as a side note only.

---

## Not measured

- Real pi agent sessions end to end (task success, wall-clock per task, tool-call cap trip rate).
- MiMo multimodal input on this cluster.
- Second runs of the MiMo quality suite (no noise floor for MiMo itself), and a second Qwen prefill run at 2K.
- Concurrency above 6 (both deployments cap at max-num-seqs 8).
