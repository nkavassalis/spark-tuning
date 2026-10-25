# spark-tuning

Tuning notes, scripts and measured results for a **dual DGX Spark (GB10)** cluster: two Sparks joined by
their ConnectX-7 200G QSFP link and serving one model with vLLM across both nodes (TP=2, with or without expert parallelism; see
[Parallel layout](#parallel-layout)). Everything here is measured
on real hardware; numbers that were not measured are labelled as such.

The cluster runs one model at a time. Each model has its own write-up:

| Model | Deployment recipe | Write-up | Status |
|---|---|---|---|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` (+ RadixArk FP8-dense), MTP K=2 (config C) | [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | [QWEN.md](QWEN.md), [FP8DENSE.md](FP8DENSE.md) | installed, **replaced** by FP4-Local on 2026-10-16 (`fp4local/down.sh --qwen` brings it back) |
| `XiaomiMiMo/MiMo-V2.6-Flash-RL`, DFlash 7 | [tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe) | [MIMO.md](MIMO.md) | benchmarked, **removed from the nodes**; `mimo/setup_nodes.sh && mimo/up.sh` to bring it back |
| `autotrust/GLM5.3-Flash-E192-DGX-Spark` (+ E224), NVFP4 MTP K=2 | stock vLLM 0.31.0 + `glm/` (after autotrust's [E224 two-Spark runbook](https://huggingface.co/autotrust/GLM5.3-Flash-E224-DGX-Spark/blob/main/DEPLOY-2X-DGX-SPARK.md)) | [GLM.md](GLM.md) | benchmarked 2026-10-13, **removed from the nodes**; `glm/setup_nodes.sh && glm/up.sh` to bring it back |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` (QAD NVFP4 + MXFP8), MTP K=2–4 | community b12x vLLM (`eugr/spark-vllm-b12x`) + `fp4local/` | [FP4LOCAL.md](FP4LOCAL.md) | **serving** since 2026-10-16 (TP=2, K=2): as smart as config C, a bit slower, bigger KV, free warm cache hits; `fp4local/down.sh --qwen` to go back |
| `nvidia/GLM-5.3-Flash-FP4-Local` (QAD NVFP4 + MXFP8, all 288 experts), MTP K=3 | community b12x vLLM + `glmfp4/` (nvidia's 2-Spark recipe, text only) | [GLMFP4.md](GLMFP4.md) | benchmarked 2026-10-17, **removed from the nodes**: best LiveCodeBench here (55%), ties elsewhere, ~35–40% slower decode than Qwen FP4-Local; `glmfp4/setup_nodes.sh && DEFAULT_EFFORT=high SPEC_TOKENS=3 glmfp4/up.sh` to bring it back |

Head-to-head speed and quality on this cluster: **[QWEN_VS_MIMO.md](QWEN_VS_MIMO.md)**. Short version: for
single-user coding Qwen config C is 1.3–2.7× faster and at least as accurate. MiMo is ahead only on multimodal input
and the 300K window.

> [!NOTE]
> **The server's default temperature is 1.0; every Qwen thinking score here is at 0.6, and pi sends 0.6 (since 2026-10-19).**
> A request that sends no `temperature` gets the checkpoint's `generation_config.json`: **1.0 / top_p 0.95 / top_k 20**. Every Qwen
> thinking-on score in this repo was measured at **0.6** / 0.95 / 20 (thinking off: greedy). Until 2026-10-19 pi sent no temperature,
> so it ran at 1.0. Measured at the server default on 2026-10-19 (FP4-Local, `medium`, 1.0):
> HumanEval+ is unchanged (95.1 vs 95.7), but **LiveCodeBench drops to 37.5% from 43.8%**. All 6 lost problems are hard ones
> (hard: 4/46 vs 9/46) and 49 vs 43 of 80 hit the 16K cap. Paired: vs `medium` at 0.6, 6 lost / 1 gained, p=0.125; vs `xhigh` at 0.6
> p=0.016, vs `low` at 0.6 p=0.021. One run each. **The repo's 0.6 scores are what pi gets now; 1.0 is what any client gets that
> sends nothing.** Client-side fix (2026-10-19): pi sends 0.6 via `"samplingParams": {"temperature": 0.6}` on the model
> ([pi/models.json.example](pi/models.json.example)). A top-level `"temperature"` key on the model is **silently ignored** by pi.
> The server default is still 1.0 for any other client (server fix, not applied: `--override-generation-config '{"temperature":0.6}'`).
> Details: [FP4LOCAL.md](FP4LOCAL.md) "Sampling temperature".

## Layout

| Path | What |
|---|---|
| `QWEN.md`, `FP8DENSE.md` | Qwen3.8-Flash-Next tuning: MTP, NCCL/control plane, FP8-dense evaluation, failures |
| `MIMO.md` | MiMo-V2.6-Flash setup on this cluster, tool-call loop mitigations, measurements |
| `QWEN_VS_MIMO.md` | same-cluster head-to-head: decode, concurrency, prefill, HumanEval+/MBPP+/GSM8K/LCB/long context |
| `GLM.md` | GLM-5.3-Flash E192/E224 setup, speed and quality on this cluster, compared with Qwen C and MiMo |
| `FP4LOCAL.md` | nvidia's QAD FP4-Local quant of the same Qwen vs our config C: speed (K sweep), quality, KV, how to switch |
| `GLMFP4.md` | nvidia's QAD FP4-Local GLM-5.3-Flash: K sweep, speed and quality vs Qwen and the pruned GLMs, time-to-solve |
| `COMPARISON.md` | every model × every reasoning setting we ran (quality, time to solve, speed) on one page, plus how to use them |
| `SGLANG.md` | can SGLang serve FP4-Local instead of the b12x vLLM build? Not yet (CSF scales, NVFP4 PLE table); researched, not measured |
| `scripts/` | Qwen benchmark / health scripts (`bench_decode.py`, `spec_acceptance.py`, `check_cluster.sh`, `mtp_ab.sh`, `prefix_probe.py` for prefix-cache behaviour, `watch_stalls.sh` stall sampler, `stage_pi_compaction.py` for the pi client staging, `models.sh` one-model-at-a-time guard, `check_repo.py` drift check) |
| `PI.md` | the client side: how pi's request shape produces stalls, and the client-side tuning |
| `quality/` | quality-eval harness (NLL, HumanEval+/MBPP+, GSM8K, thinking-mode, long context, time to solve) |
| `configs/` | diffs/patches applied to the Qwen deployment repo |
| `mimo/` | MiMo setup / switch scripts, tool-call-cap proxy, functional tests |
| `glm/` | GLM setup / switch scripts (`setup_nodes.sh`, `up.sh`, `down.sh`) and the in-container vLLM patch (`patch_vllm.py`) |
| `fp4local/` | FP4-Local setup / switch scripts (`setup_nodes.sh`, `up.sh [LIVE=1]`, `down.sh [--qwen]`) |
| `glmfp4/` | GLM FP4-Local setup / switch scripts (`setup_nodes.sh`, `up.sh`, `down.sh [--fp4local\|--qwen]`) |
| `results/` | raw benchmark and eval outputs |

## Parallel layout

Every deployment here uses **TP=2**: tensor parallelism across the two Sparks. Attention and the other dense layers are split
in half, one half per GPU, and recombined with an all-reduce over the CX7 link. **Expert parallelism (EP) is not a separate count**:
in vLLM it is a flag on top of TP=2 (`--enable-expert-parallel`) that changes only the MoE expert layers.

- **TP only:** every expert's matrices are split in half, so every token's expert work runs on both GPUs, followed by an all-reduce.
- **TP + EP:** each GPU holds about half the experts whole, and tokens are exchanged between the GPUs to reach their routed experts
  (config C: `--all2all-backend allgather_reducescatter`). vLLM names the workers `Worker_TP0_EP0` / `Worker_TP1_EP1`.

| Deployment | Layout | Notes |
|---|---|---|
| Qwen config C (and configs A/B before it) | **TP=2 + EP** | `configs/running-vllm-args.txt` |
| MiMo-V2.6-Flash | TP=2; EP not enabled by anything in this repo | the recipe's own launch script isn't vendored here, so not verified |
| GLM-5.3-Flash E192/E224 | TP=2 only | never tried with EP |
| nvidia FP4-Local (**serving**) | TP=2 only | EP refused by the b12x build ([FP4LOCAL.md](FP4LOCAL.md#why-no-expert-parallelism-its-the-software-not-the-format)) |
| nvidia GLM FP4-Local | TP=2 only | EP fails at startup on the same b12x build ([GLMFP4.md](GLMFP4.md#tuning-attempts-k3-lossless)) |

So cross-model speed comparisons against config C also compare layouts. This matters most under concurrency: FP4-Local's
~20% lower aggregate at 3–6 streams is the main gap, and EP is the main config difference (cause not proven). GLM's 2–3× decode gap
is far too large for layout alone to explain.

## Setup

Scripts never hard-code hosts. Copy `cluster.env.example` to `cluster.env` (git-ignored) and fill in the
SSH targets of the head and worker and the API base URL. Python scripts read `API` from the environment or
take `--url`.

pi client: copy [`pi/models.json.example`](pi/models.json.example) to `~/.pi/agent/models.json` and replace `<head>`.
It sends temperature 0.6 (`samplingParams`; a top-level `temperature` is ignored by pi) and turns on image input ([PI.md](PI.md)).

Requirements on the machine you run from: SSH key access to both nodes, Python 3 (stdlib only), and
Node 18+ only if you want to run the proxy unit tests locally. The nodes need Docker (user in the
`docker` group) and the stock DGX Spark CX7 link (`enp1s0f0np0` / `rocep1s0f0`).

## Switching models

```bash
mimo/setup_nodes.sh      # (re)provision: recipe checkout, ~178 GB weights on both nodes, patches. Does not stop Qwen.
mimo/up.sh               # stop Qwen, start MiMo + tool-call-cap proxy (~11-15 min)
mimo/down.sh --qwen      # stop MiMo, relaunch config C (~11 min)
mimo/down.sh --fp4local  # stop MiMo, relaunch the live FP4-Local
GLM_VARIANT=E224 glm/setup_nodes.sh && GLM_VARIANT=E224 glm/up.sh   # GLM-5.3-Flash (not installed, 2026-10-25); glm/down.sh --fp4local to go back
LIVE=1 SPEC_TOKENS=2 fp4local/up.sh   # the live model: nvidia FP4-Local as qwen3.8-flash-next, no key; plain fp4local/up.sh = benchmark mode (K=4, API key); fp4local/down.sh --qwen
glmfp4/setup_nodes.sh && DEFAULT_EFFORT=high SPEC_TOKENS=3 glmfp4/up.sh   # nvidia GLM FP4-Local (not on the head, 2026-10-25; worker keeps a copy); glmfp4/down.sh --fp4local

MiMo is **not on the head** (2026-10-25: removed 2026-10-07, reinstalled 2026-10-23 and 2026-10-24, removed from the head 2026-10-25; the worker
keeps its weights and image). `setup_nodes.sh` re-downloads it on the head and rsyncs whatever the worker lacks.
```

## Tests

```bash
scripts/check_repo.py                     # doc/script drift check (offline; AGENTS.md "Keeping the docs current")
scripts/test_models.sh                    # one-model-at-a-time guard, offline
node --test mimo/toolcap/*.test.cjs               # proxy unit tests (no network, no GPU)
API=http://<head>:8000 mimo/test_mimo.py  # MiMo functional checks; needs the MiMo setup running
scripts/check_cluster.sh                  # config C health/config check (inspects vllm-fn only, not the live FP4-Local)
scripts/prefix_probe.py                   # prefix-cache behaviour: cold / warm / competing / idle
scripts/stage_pi_compaction.py --check    # pi client staging (see PI.md); --revert to undo
```
- 2026-10-15 — single-stream decode reported as slower (~20 vs ~27 tok/s): **not a regression**. Same-tool re-check gives
  C1 62.4 (recorded 58.3) and C6 218.4 (recorded 228.2); `.env` is byte-identical to `.env.C-fp8dense`, same image, no throttling.
  The ~20 tok/s is 4 concurrent agents with thinking on (measured 23-29 tok/s per stream at C4, 98-112 tok/s aggregate). See `QWEN.md`.

## License

[MIT](LICENSE). Third-party files keep their own licenses: `mimo/toolcap/` is a copy of the MiMo recipe's proxy
(MIT, `mimo/toolcap/LICENSE.upstream`), and the patches in `configs/` and `glm/patch_vllm.py` modify vLLM (Apache-2.0).
