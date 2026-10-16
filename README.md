# spark-tuning

Tuning notes, scripts and measured results for a **dual DGX Spark (GB10)** cluster: two Sparks joined by
their ConnectX-7 200G QSFP link and serving one model with vLLM tensor parallel 2. Everything here is measured
on real hardware; numbers that were not measured are labelled as such.

The cluster runs one model at a time. Each model has its own write-up:

| Model | Deployment recipe | Write-up | Status |
|---|---|---|---|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` (+ RadixArk FP8-dense), MTP K=2 | [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | [QWEN.md](QWEN.md), [FP8DENSE.md](FP8DENSE.md) | **serving** (config C) |
| `XiaomiMiMo/MiMo-V2.6-Flash-RL`, DFlash 7 | [tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe) | [MIMO.md](MIMO.md) | benchmarked, **removed from the nodes**; `mimo/setup_nodes.sh && mimo/up.sh` to bring it back |
| `autotrust/GLM5.3-Flash-E192-DGX-Spark` (+ E224), NVFP4 MTP K=2 | stock vLLM 0.31.0 + `glm/` (after autotrust's [E224 two-Spark runbook](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark/blob/main/DEPLOY-2X-DGX-SPARK.md)) | [GLM.md](GLM.md) | benchmarked 2026-10-13, **removed from the nodes**; `glm/setup_nodes.sh && glm/up.sh` to bring it back |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` (QAD NVFP4 + MXFP8), MTP K=2–4 | community b12x vLLM (`eugr/spark-vllm-b12x`) + `fp4local/` | [FP4LOCAL.md](FP4LOCAL.md) | benchmarked 2026-10-16 at TP=2: as smart as config C, slower under concurrency; checkpoint still on the nodes, `fp4local/up.sh` |

Head-to-head speed and quality on this cluster: **[QWEN_VS_MIMO.md](QWEN_VS_MIMO.md)**. Short version: for
single-user coding Qwen config C is 1.3–2.7× faster and at least as accurate. MiMo is ahead only on multimodal input
and the 300K window.

## Layout

| Path | What |
|---|---|
| `QWEN.md`, `FP8DENSE.md` | Qwen3.8-Flash-Next tuning: MTP, NCCL/control plane, FP8-dense evaluation, failures |
| `MIMO.md` | MiMo-V2.6-Flash setup on this cluster, tool-call loop mitigations, measurements |
| `QWEN_VS_MIMO.md` | same-cluster head-to-head: decode, concurrency, prefill, HumanEval+/MBPP+/GSM8K/LCB/long context |
| `GLM.md` | GLM-5.3-Flash E192/E224 setup, speed and quality on this cluster, compared with Qwen C and MiMo |
| `FP4LOCAL.md` | nvidia's QAD FP4-Local quant of the same Qwen vs our config C: speed (K sweep), quality, KV, how to switch |
| `scripts/` | Qwen benchmark / health scripts (`bench_decode.py`, `spec_acceptance.py`, `check_cluster.sh`, `mtp_ab.sh`, `prefix_probe.py` for prefix-cache behaviour, `watch_stalls.sh` stall sampler, `stage_pi_compaction.py` for the pi client staging) |
| `PI.md` | the client side: how pi's request shape produces stalls, and the client-side tuning |
| `quality/` | quality-eval harness (NLL, HumanEval+/MBPP+, GSM8K, thinking-mode, long context) |
| `configs/` | diffs/patches applied to the Qwen deployment repo |
| `mimo/` | MiMo setup / switch scripts, tool-call-cap proxy, functional tests |
| `glm/` | GLM setup / switch scripts (`setup_nodes.sh`, `up.sh`, `down.sh`) and the in-container vLLM patch (`patch_vllm.py`) |
| `fp4local/` | FP4-Local setup / switch scripts (`setup_nodes.sh`, `up.sh [LIVE=1]`, `down.sh [--qwen]`) |
| `results/` | raw benchmark and eval outputs |

## Setup

Scripts never hard-code hosts. Copy `cluster.env.example` to `cluster.env` (git-ignored) and fill in the
SSH targets of the head and worker and the API base URL. Python scripts read `API` from the environment or
take `--url`.

Requirements on the machine you run from: SSH key access to both nodes, Python 3 (stdlib only), and
Node 18+ only if you want to run the proxy unit tests locally. The nodes need Docker (user in the
`docker` group) and the stock DGX Spark CX7 link (`enp1s0f0np0` / `rocep1s0f0`).

## Switching models

```bash
mimo/setup_nodes.sh      # (re)provision: recipe checkout, ~178 GB weights on both nodes, patches. Does not stop Qwen.
mimo/up.sh               # stop Qwen, start MiMo + tool-call-cap proxy (~11-15 min)
mimo/down.sh --qwen      # stop MiMo, relaunch Qwen (~11 min)
GLM_VARIANT=E224 glm/setup_nodes.sh && GLM_VARIANT=E224 glm/up.sh   # GLM-5.3-Flash (not installed now); glm/down.sh --qwen to go back
fp4local/up.sh               # nvidia FP4-Local (installed); LIVE=1 serves it as qwen3.8-flash-next with no key; fp4local/down.sh --qwen

MiMo is **not installed** on the nodes right now (removed 2026-10-07); `setup_nodes.sh` rebuilds it from scratch.
```

## Tests

```bash
node --test mimo/toolcap/*.test.cjs               # proxy unit tests (no network, no GPU)
API=http://<head>:8000 mimo/test_mimo.py  # MiMo functional checks; needs the MiMo setup running
scripts/check_cluster.sh                  # Qwen health/config check (while Qwen is serving)
scripts/prefix_probe.py                   # prefix-cache behaviour: cold / warm / competing / idle
scripts/stage_pi_compaction.py --check    # pi client staging (see PI.md); --revert to undo
```
- 2026-10-15 — single-stream decode reported as slower (~20 vs ~27 tok/s): **not a regression**. Same-tool re-check gives
  C1 62.4 (recorded 58.3) and C6 218.4 (recorded 228.2); `.env` is byte-identical to `.env.C-fp8dense`, same image, no throttling.
  The ~20 tok/s is 4 concurrent agents with thinking on (measured 23-29 tok/s per stream at C4, 98-112 tok/s aggregate). See `QWEN.md`.
