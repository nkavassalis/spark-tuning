# spark-tuning

Tuning notes, scripts and results for a dual **DGX Spark** (GB10) cluster serving
`nvidia/Qwen3.8-Flash-Next-NVFP4` with vLLM (TP=2 + EP, MTP speculative decoding), deployed with
[MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks).

**Goal:** high-quality, high-speed single-user code generation. 1M context not needed.
**Constraint:** no change may alter model output quality. Only lossless or neutral changes were applied
(speculative decoding is verified by the target model, so the output distribution is unchanged).

| Node | Hostname | LAN (2.5GbE, `enP7s7`) | CX7 200G (`enp1s0f0np0`) |
|---|---|---|---|
| head / rank 0 | sparky | 10.1.13.99 | 192.168.100.10 |
| worker / rank 1 | raijin | 10.1.13.98 | 192.168.100.11 |

API: `http://10.1.13.99:8000/v1` — model `qwen3.8-flash-next`, `max_model_len` 262144.

---

## TL;DR

| Change | Why | Output change? |
|---|---|---|
| `HEAD_IP`/`WORKER_IP` → CX7 IPs (`192.168.100.x`) | vLLM's per-step scheduler→worker broadcast (`--master-addr`, `VLLM_HOST_IP`) was on the 2.5GbE LAN | none |
| MTP draft length **3 → 2** | 3rd draft position accepted only ~20–40% on real traffic | none (spec decode is lossless) |
| NCCL pinning (`NCCL_NET=IB`, `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_IGNORE_CPU_AFFINITY=1`) | from the repo's own tuning report §3.5 | none |
| `--performance-mode interactivity` | capture every CUDA-graph size 1…32 (less padding at odd MTP batch sizes) | none |
| `start.sh` bugfix: pass `EXTRA_DOCKER_ARGS` to both nodes' `docker run` | it was silently dropped on both head and worker | n/a |
| Context kept at **262144** (native, no YaRN) | already not 1M; lowering it does not speed up decode | none |

Single-stream code decode, before → after (500 tokens, 1 run each — see caveats):

| | 1K ctx, T=0 | 1K ctx, T=0.6 | 64K ctx, T=0 | 64K ctx, T=0.6 |
|---|---|---|---|---|
| before (K=3, LAN control plane) | 45.1 | 42.5 | 40.5 | 37.6 |
| after, launch 1 | 46.1 | 43.4 | 44.6 | 48.0 |
| after, launch 2 (current) | 45.3 | 47.1 | 49.5 | 45.6 |

**Read this honestly:** at short context the gain is within noise (~0–3%). At 64K it is consistently
+10–25%. Copy-from-context tasks got ~10% *slower* (they benefit from long drafts). The changes were
applied as a bundle, so the gains are **not individually attributed** — `scripts/mtp_ab.sh` exists to
isolate the K=2 vs K=3 effect (see [Open items](#open-items)).

### Update: FP8-dense weights — +35% decode, no detectable accuracy loss

We then evaluated the recipe's never-benchmarked `FP8_DENSE` lane. It uses per-channel FP8 for the
591 BF16 dense linears; the NVFP4 experts are untouched. Full write-up: **[FP8DENSE.md](FP8DENSE.md)**.

| | decode tok/s (mean of 6 cells) | HumanEval+ | MBPP+ | GSM8K | NLL (nats/tok) |
|---|---|---|---|---|---|
| A: nvidia BF16-dense (previous prod) | 44.4 | 95.1 | 79.9 | 96.5 | 0.1315 |
| B: RadixArk BF16-dense (control) | 39.0 | 93.9 | 79.4 | 96.4 | 0.1310 |
| **C: RadixArk FP8-dense** | **52.5** | 93.3 | 79.9 | 96.4 | 0.1340 |

C vs B, the clean comparison: **+35% decode**, and task scores change by −0.6 / +0.5 / 0.0 points,
within run-to-run noise. Follow-ups found no detectable change in thinking mode (HumanEval+,
LiveCodeBench) or at 64K/128K context: pooled over 2,249 paired items, 24 vs 21 flips. Per-token NLL rises by a small but real +0.003 nats (+2.3%). Two
upstream bugs had to be fixed to run it at all (see FP8DENSE.md).

---

## Speculative decoding (MTP) — the main lever

The model ships a Multi-Token-Prediction head. vLLM uses it to draft K tokens per step; the target model
verifies all K in one forward pass and keeps the accepted prefix (+1 bonus token). Rejected drafts are
discarded, so **output is identical in distribution to non-speculative decoding** — K is a pure
speed knob.

### Cost / benefit model

Per engine step: 1 target forward + K draft forwards. Each draft forward re-reads the MTP layer and a
(reduced, 47k-id) draft lm_head. From the repo's byte model (`docs/CLAUDE/fable5-1-report.md`), one
draft pass ≈ 0.8 GiB vs ≈ 6.8 GiB for the target step, so **c ≈ 0.12** relative cost per draft.

```
tokens/step  E(K) = 1 + p0 + p1 + ... + p(K-1)      (p_i = unconditional per-position acceptance)
step cost    T(K) = 1 + c*K
throughput  ~ E(K) / T(K)
```

The K-th position pays for itself only if `p(K-1) > c·E(K-1) / (1 + c·(K-1))` — about **0.25–0.30**
here. There is a second, unmodelled cost: the QSA sparse-attention backend can't fuse multi-step
draft decode and **rebuilds attention metadata per draft step** (repo README "Gotchas"), and that
cost grows with context. That is the most plausible reason K=2 helped most at 64K (hypothesis, not
profiled).

### What was observed

Per-position acceptance on your **real traffic** with K=3 (server log, 10 s windows):

```
0.74 / 0.50 / 0.40     0.66 / 0.46 / 0.36     0.56 / 0.33 / 0.19     0.68 / 0.50 / 0.32
```

and during the synthetic benchmark: `0.54/0.29/0.12`, `0.57/0.30/0.22`, `0.68/0.45/0.30`.
The repo's published numbers (89% / 74.5% / 60%) are for greedy prose and do not describe this workload.

With K=2 (current, `results/current-K2-2026-09-27.json`) acceptance is `p0 ≈ 0.51–0.79`,
`p1 ≈ 0.25–0.62`, mean acceptance length 1.76–2.41 tokens/step. Temperature matters: `code_ts` at
T=0.6 drops to 0.51/0.25 — rejection sampling accepts less at higher temperature.

**Why real code is slower than the published ~54 tok/s:** those are greedy, short-prompt prose runs
with high MTP acceptance. Your agent traffic had ~135K-token prompts (3.7% of a 3.65M-token KV pool
for one request) and 36–55% average draft acceptance.

### How to re-check K on your own traffic (no restart)

```bash
scripts/spec_acceptance.py --watch 60          # rolling acceptance while you use the agent normally
```

It prints per-position acceptance and an estimated throughput for every K ≤ current. If `p1`
(the last position at K=2) stays well above ~0.45 on your real use, K=3 may be worth re-testing;
if it falls below ~0.25, try K=1.

---

## Scripts

All run from any machine that can reach `10.1.13.99`. Python 3 stdlib only. On another cluster,
override the defaults: `--url` for the Python scripts, `HEAD`/`WORKER`/`API`/`REPO` env vars for the shell scripts.

| Script | Restarts server? | Purpose |
|---|---|---|
| `scripts/check_cluster.sh` | no | health, served args, NCCL env on both nodes, confirms control plane on CX7 and that a request moves bytes over RoCE |
| `scripts/bench_decode.py` | no | decode tok/s + TTFT + **per-run MTP acceptance** for `code`, `code_ts`, `edit` (refactor real code in context), `prose`, `copy` × contexts × temperatures; `--json` for comparisons |
| `scripts/spec_acceptance.py` | no | snapshot / delta / `--watch` of acceptance counters + draft-length estimate |
| `scripts/mtp_ab.sh --yes [K...]` | **yes** (~11 min per K) | relaunches with each K, benchmarks, restores the original K and re-checks health; writes `results/mtp_ab-<ts>/` |
| `scripts/compare_results.py a.json b.json` | no | side-by-side tok/s and % delta |
| `quality/run_quality.sh <label>` | no | NLL on a fixed corpus + HumanEval+/MBPP+ (sandboxed execution) + GSM8K; ~70 min; run on a node with docker. `quality_eval.py compare-nll` / `compare-pass` for paired comparisons |

Examples:

```bash
scripts/check_cluster.sh
scripts/bench_decode.py --contexts 1000,64000,128000 --temps 0.6 --repeats 3 --json results/x.json
BENCH_ARGS="--tasks code,edit --contexts 1000,128000 --temps 0.6 --repeats 3" scripts/mtp_ab.sh --yes 2 3
```

Benchmarks are only valid when nothing else is using the server (the metrics are server-global).

---

## Configuration

Deployment repo on the head: `~/Qwen3.8-Flash-Next-Dual-DGX-Sparks`. Launch with `./stop.sh && ./start.sh --launch`
on the head (cold start ≈ 11 min; the first request after launch is slow while FlashInfer autotunes).

- `configs/env.diff` — `.env` changes (original backed up on the head as `.env.bak-20260926-2124`; HF token redacted here)
- `configs/start.sh.patch` — the `EXTRA_DOCKER_ARGS` fix
- `configs/running-vllm-args.txt` — the exact `vllm serve` command line now running

To revert everything: on the head, `cp .env.bak-20260926-2124 .env && git checkout start.sh && ./stop.sh && ./start.sh --launch`.

---

## Things tried that failed / were rejected

### torch.compile — **do not enable** (hung the head)

`--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'` (report §3.6 item 5, "0–8%,
keep mode 0 as fallback"). During startup the head exhausted its unified memory (weights +
KV reservation already leave ~4–7 GiB free) and userspace locked up: ping answered, SSH did not, for
20+ minutes. It needed a hard power cycle.

Side effect: during the ~9.5 h hang its DHCP lease on the LAN expired and the router handed out
`10.1.13.100` on reboot. The MAC did **not** change (checked: permanent burned-in address, same NetworkManager profile). The head now has a
static `10.1.13.99/24` (NetworkManager "Wired connection 3", gw/DNS `10.1.13.1`). **Add a router
DHCP reservation or exclude `.99` from the pool** so it's never handed to another host.

### `index_share_for_mtp_iteration` — tested, no gain, reverted (2026-09-27)

What it does: the MTP drafter's attention layer uses QSA sparse attention, whose indexer scores
the whole context and picks top-k positions for each token. With the flag on, draft step 0 computes
the top-k indices and draft steps 1+ reuse them (skipping scoring + top-k + expand), while still
writing their tokens into the indexer cache. Output is unchanged (the target model re-indexes in all
12 QSA layers and verifies every draft); only draft acceptance can drop (step 1's indices are one
position stale). It is wired up for this model in this image (`speculator.py` +
`set_skip_topk`/`compact_topk_indices` in the patched `qwen3_8_flash_next/nvidia/mtp.py`).

A/B at K=2, T=0.6, 3 repeats, 1K and 128K context (`results/idxshare-{off,on}-2026-09-27.json`).
Raw tok/s moved −5.6% … +8.2%, but that tracked acceptance-length noise. Normalised to engine
steps/s (tok/s ÷ acceptance length), which is what the flag can affect:

| ctx | task | steps/s off | steps/s on | Δ |
|---|---|---|---|---|
| 1K | code | 21.37 | 21.47 | +0.5% |
| 1K | code_ts | 21.89 | 21.51 | −1.8% |
| 1K | edit | 21.91 | 21.37 | −2.5% |
| 128K | code | 21.12 | 20.95 | −0.8% |
| 128K | code_ts | 21.01 | 20.99 | −0.1% |
| 128K | edit | 21.19 | 21.18 | −0.0% |

Within noise (run-to-run spread is ±8%), slightly negative on average, and no context-length trend.
With K=2 it skips one indexer pass on one layer per step, which is too small to see. Reverted to
keep the better-tested path. It might be worth re-testing only if K is ever raised to 3+.

Side observation: engine step rate is ~21 steps/s at both 1K and 128K, so at this config decode
speed is set almost entirely by **MTP acceptance**, not context length.

### Rejected for quality reasons

- **YaRN / 1M context:** not needed; native 262K keeps rope unscaled.

(`FP8_DENSE=true` was originally here pending a quality evaluation. It has now been evaluated:
see [FP8DENSE.md](FP8DENSE.md).)

### Not worth it for single-stream decode

- Lowering `MAX_MODEL_LEN` below 262144: frees KV-cache pool, no decode speed-up.
- Second CX7 HCA (`roceP2p1s0f0`): no IP on either node; doubles bandwidth (prefill) but
  decode collectives are latency-bound.
- KV cache dtype: already fp8.

---

## Open items

1. ~~Isolate K=2 vs K=3~~: done on the FP8-dense config. K=2 is 2.7% faster (FP8DENSE.md).
2. ~~`index_share_for_mtp_iteration`~~ — tested, no gain, reverted (see above).
3. ~~`FP8_DENSE`~~ (**production since 2026-09-27**): evaluated, +35% decode, no detectable task-accuracy change. See [FP8DENSE.md](FP8DENSE.md).
4. ~~FP8-dense + reduced draft vocabulary~~: built and measured, no step-rate gain because the draft ids sit
   almost entirely on rank 0 (see FP8DENSE.md "Follow-up").
5. ~~Quality in thinking mode and at long context~~: done, no detectable change (FP8DENSE.md).
6. `QSA_PROFILE`: deprioritised. The step rate is flat from 1K to 128K, so attention isn't the bottleneck.
7. Router DHCP reservation for both Sparks.

## Timeline

- 2026-09-26 21:18 — baseline measured (`results/baseline-K3-2026-09-26.txt`)
- 21:25 — relaunch with tuned `.env`; re-measured (`results/after-tune1-K2-2026-09-27.txt`)
- ~21:50 — torch.compile relaunch → head hung; worker container exited
- 2026-09-27 07:31 — head power-cycled; came back on `10.1.13.100` (DHCP); tuned config relaunched via the CX7 link
- ~07:50 — head pinned back to static `10.1.13.99`; reference benchmark `results/current-K2-2026-09-27.json`
- ~12:00 — `index_share_for_mtp_iteration` A/B: no gain, reverted
- 13:00–17:30 — FP8-dense evaluation: RadixArk download + verify, build, configs A/B/C speed + quality
  (`FP8DENSE.md`, `results/quality/`). Server left running config C (FP8-dense).
- ~19:30 — FP8-dense + draft vocab (config D): no gain; back on C
- 2026-09-27 20:00 → 09-28 01:46 — thinking-mode + long-context quality (B vs C), then the K=2 vs K=3 A/B on C.
  Production: **C (RadixArk NVFP4 + FP8-dense, MTP K=2)**
