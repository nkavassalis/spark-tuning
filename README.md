# spark-tuning

Tuning notes, scripts and measured results for a **dual DGX Spark (GB10)** cluster: two Sparks joined by
their ConnectX-7 200G QSFP link and serving one model with vLLM tensor parallel 2. Everything here is measured
on real hardware; numbers that were not measured are labelled as such.

The cluster runs one model at a time. Each model has its own write-up:

| Model | Deployment recipe | Write-up | Status |
|---|---|---|---|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` (+ RadixArk FP8-dense), MTP K=2 | [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | [QWEN.md](QWEN.md), [FP8DENSE.md](FP8DENSE.md) | **serving** (config C) |
| `XiaomiMiMo/MiMo-V2.6-Flash-RL`, DFlash 7 | [tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe) | [MIMO.md](MIMO.md) | benchmarked, **removed from the nodes**; `mimo/setup_nodes.sh && mimo/up.sh` to bring it back |

Head-to-head speed and quality on this cluster: **[QWEN_VS_MIMO.md](QWEN_VS_MIMO.md)**. Short version: for
single-user coding Qwen config C is 1.3–2.7× faster and at least as accurate. MiMo is ahead only on multimodal input
and the 300K window.

## Layout

| Path | What |
|---|---|
| `QWEN.md`, `FP8DENSE.md` | Qwen3.8-Flash-Next tuning: MTP, NCCL/control plane, FP8-dense evaluation, failures |
| `MIMO.md` | MiMo-V2.6-Flash setup on this cluster, tool-call loop mitigations, measurements |
| `QWEN_VS_MIMO.md` | same-cluster head-to-head: decode, concurrency, prefill, HumanEval+/MBPP+/GSM8K/LCB/long context |
| `scripts/` | Qwen benchmark / health scripts (`bench_decode.py`, `spec_acceptance.py`, `check_cluster.sh`, `mtp_ab.sh`) |
| `quality/` | quality-eval harness (NLL, HumanEval+/MBPP+, GSM8K, thinking-mode, long context) |
| `configs/` | diffs/patches applied to the Qwen deployment repo |
| `mimo/` | MiMo setup / switch scripts, tool-call-cap proxy, functional tests |
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

MiMo is **not installed** on the nodes right now (removed 2026-10-07); `setup_nodes.sh` rebuilds it from scratch.
```

## Tests

```bash
node --test mimo/toolcap/*.test.cjs               # proxy unit tests (no network, no GPU)
API=http://<head>:8000 mimo/test_mimo.py  # MiMo functional checks; needs the MiMo setup running
scripts/check_cluster.sh                  # Qwen health/config check (while Qwen is serving)
```
