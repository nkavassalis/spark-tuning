# FP8-dense hybrid checkpoint: measured speed and quality

**Question:** the deployment repo's `FP8_DENSE=true` lane converts every BF16 dense projection
(GDN / attention / HyperConnection / shared expert / lm_head, 591 linears) to FP8 E4M3 with
per-output-channel weight scales and dynamic per-token activation scales. The NVFP4 routed experts
are untouched. It was estimated to give ~1.5× batch-1 decode, and had **never been run on a GPU or
evaluated for quality**.

**Answer (2026-09-27, 2× DGX Spark, vLLM TP=2+EP, MTP K=2):**

- **Speed: +35% decode** vs the same checkpoint in BF16 (39.0 → 52.5 tok/s mean, 19.0 → 25.8 engine
  steps/s), MTP acceptance unchanged. **+18% vs the previous production config** (44.4 tok/s).
- **Quality: no detectable change on any task benchmark.** HumanEval+ −0.6, MBPP+ +0.5,
  GSM8K ±0.0 points, all well inside the measured run-to-run noise.
- **A small but real shift in token probabilities:** +0.0030 nats/token NLL (+2.3% relative,
  perplexity 1.1399 → 1.1434) on a fixed 190K-token text. That is ~19× the rerun noise, and 6× the
  difference between two independently built NVFP4 checkpoints.

## Configs

| | Checkpoint | Dense weights | MTP draft vocab | MTP experts |
|---|---|---|---|---|
| **A** (previous production) | `nvidia/Qwen3.8-Flash-Next-NVFP4` | BF16 | 47k code subset | FP8 block |
| **B** (control) | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b71922` | BF16 | full 248k | BF16 |
| **C** (FP8-dense) | B converted with `files/fp8dense/build.sh` | **FP8 per-channel** | full 248k | BF16 |

**B vs C is the clean comparison:** same source weights and the same everything else. C differs
only in FP8 dense projections. C must use the RadixArk build: the converter hard-codes that
checkpoint's layout. It also can't use the reduced draft vocabulary, because both overlay
`nvidia/mtp.py`. A is included as the reference for "what you had before".

Everything else is identical across A/B/C: fp8 KV cache, 262K context, MTP K=2, GMU 0.835,
`--performance-mode interactivity`, NCCL settings, control plane on the CX7 link
(see [README](README.md)).

## Speed

`scripts/bench_decode.py --tasks code,code_ts,edit --contexts 1000,128000 --temps 0.6 --repeats 3`.
The median of 3 runs is shown, 500 decoded tokens, single stream. "st/s" = engine steps/s
(tok/s ÷ mean acceptance length), which removes MTP-acceptance noise.

| ctx | task | A tok/s | A st/s | B tok/s | B st/s | **C tok/s** | **C st/s** |
|---|---|---|---|---|---|---|---|
| 1K | code | 45.2 | 21.37 | 39.0 | 19.37 | **52.8** | **25.78** |
| 1K | code_ts | 43.5 | 21.89 | 37.6 | 19.12 | **52.4** | **26.26** |
| 1K | edit | 45.4 | 21.91 | 41.4 | 19.21 | **55.4** | **26.10** |
| 128K | code | 44.5 | 21.12 | 40.5 | 18.84 | **49.1** | **25.44** |
| 128K | code_ts | 41.1 | 21.01 | 35.9 | 18.78 | **50.0** | **25.72** |
| 128K | edit | 46.6 | 21.19 | 39.3 | 18.76 | **55.2** | **25.35** |
| **mean** | | **44.4** | **21.42** | **39.0** | **19.01** | **52.5** | **25.78** |

Mean acceptance length: A 2.07, B 2.05, C 2.04. So **FP8 did not hurt MTP drafting**, even though
the drafter's lm_head is FP8 too. The +35.6% step rate is below the 1.5× estimate from the byte
model, probably because non-GEMM work (MoE routing, NCCL, GDN, QSA) doesn't shrink.

Why B is slower than A: B drafts over the full 248k vocabulary and has BF16 MTP experts (nvidia's are
FP8). That's the ~11% step-rate gap, and the reason A vs C understates the FP8 gain.

Raw data: `results/idxshare-off-2026-09-27.json` (A), `results/fp8dense-{B-radix,C-fp8dense}-speed.json`.

## Quality

Harness: `quality/` (`run_quality.sh` + `quality_eval.py`), run from the worker against the API.
Generation is **greedy, thinking disabled**, with 8 concurrent requests. Model-written code runs
under evalplus 0.3.1 in a container with `--network none`, 6 GB memory, 512 pids.

### 1. Token log-probabilities (most sensitive)

190,191 tokens of fixed text (`quality/corpus.jsonl`: 88 chunks of Python 3.12.7 stdlib source +
LICENSE prose). Log-probs come from `prompt_logprobs` (prefill only), sent one request at a time.
Compared token by token against B.

| run | mean NLL (nats/tok) | ΔNLL vs B | mean \|Δlogprob\| vs B | top-1 agreement vs B |
|---|---|---|---|---|
| B radix BF16 | 0.13098 | — | — | — |
| **C radix FP8-dense** | 0.13397 | **+0.00298** | 0.0462 | 98.56% |
| A nvidia BF16 | 0.13146 | +0.00048 | 0.0470 | 98.53% |
| A rerun (noise floor, A vs A) | 0.13162 | +0.00016 vs A | 0.0360 vs A | 98.84% vs A |

How to read it:
- **Rerunning the same config is not bit-identical.** |Δlogprob| is 0.036 and top-1 agreement 98.8%
  (batch/chunking differences), so a ΔNLL around ±0.0002 is noise.
- **FP8 adds a real, systematic +0.0030 nats/token** (+2.3% relative NLL; perplexity +0.31%). It's
  biased in one direction. The per-token wobble between two *different BF16 checkpoints* (A vs B)
  is the same size, but it nets to ~zero.
- Caveat: stdlib code is probably memorized, so its NLL is low. The relative increase could differ
  on novel text.

### 2. Task accuracy (paired, per problem)

| benchmark | n | A | B | **C** | C−B | C vs B flips (lost / gained) | A vs A-rerun flips (noise) |
|---|---|---|---|---|---|---|---|
| HumanEval (base) | 164 | 96.95 | 95.73 | 95.12 | −0.6 | | |
| **HumanEval+** | 164 | 95.12 | 93.90 | **93.29** | −0.6 | 2 / 1 (p=1.00) | 2 / 2 |
| MBPP (base) | 378 | 93.92 | 92.33 | 93.12 | +0.8 | | |
| **MBPP+** | 378 | 79.89 | 79.37 | **79.89** | +0.5 | 3 / 5 (p=0.73) | 3 / 2 |
| **GSM8K** (exact match) | 1319 | 96.51 | 96.36 | **96.36** | ±0.0 | 11 / 11 (p=1.00) | — |

p = exact two-sided McNemar test on discordant pairs. Rough 95% interval on the C−B difference,
from the discordant counts: **HumanEval+ ±2.1, MBPP+ ±1.5, GSM8K ±0.7 points**. So these
benchmarks rule out an FP8 regression larger than about 1–2 points, and can't see anything smaller.
The NLL result above says the true effect is small and nonzero.

Raw data: `results/quality/<run>/` (every generation, evalplus per-task results, GSM8K scores,
xz-compressed per-token log-probs).

### Not measured

- **Thinking mode.** All evals ran with `enable_thinking: false`. Long reasoning chains may compound
  small per-token errors more than short answers do.
- **Long-context quality** (64K–256K agentic contexts). Speed was measured at 128K; quality wasn't.
- Sampled decoding (T=0.6) quality. It should track greedy, but wasn't verified.

## Bugs hit on the way (upstream `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` @ `d2f54b7`)

1. **`files/overlay/apply_patches.py` is missing.** `FP8_DENSE=true` dies at start.sh Step 4c.
   Fix: `configs/fp8dense-overlay/build_overlay.sh` rebuilds the overlay from the shipped `.diff` files.
2. **Duplicate `modelopt.py` mount.** With `FP8_DENSE=true`, start.sh mounts both
   `overlay/modelopt.py` and the MXFP8 `modelopt_patched.py` on the same container path. Fix: build
   the overlay's modelopt.py *on top of* `modelopt_patched.py`, and skip the second mount
   (`configs/start.sh.patch`).
3. `EXTRA_DOCKER_ARGS` was dropped on both nodes' `docker run` (also in `configs/start.sh.patch`).
4. Not a repo bug, but worth knowing: `snapshot_download(revision=<commit sha>)` writes no
   `refs/main`, so offline vLLM fails with `LocalEntryNotFoundError`. Write it with
   `printf %s <sha> > refs/main` (no newline) on both nodes. Also, `huggingface_hub`'s xet transfer
   was OOM-killed under a 3 GB container cap; `HF_HUB_DISABLE_XET=1 max_workers=2` downloaded at
   ~10 GB/min using 173 MB.

## Reproduce

```bash
# head, deployment repo root
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4        # 135 GB, verify with verify-weights.py
./files/fp8dense/build.sh                             # ~2 min, CPU only
bash /path/to/spark-tuning/configs/fp8dense-overlay/build_overlay.sh
git apply /path/to/spark-tuning/configs/start.sh.patch
# .env: MODEL_ID="RadixArk/Qwen3.8-Flash-Next-NVFP4"  MTP_DRAFT_VOCAB=""  FP8_DENSE=true
./stop.sh && ./start.sh --no-download                 # rsync -aH both checkpoints first to keep hard links
# anywhere
scripts/bench_decode.py --tasks code,code_ts,edit --contexts 1000,128000 --temps 0.6 --repeats 3
quality/run_quality.sh C-fp8dense                     # on a node with docker; ~70 min
```
