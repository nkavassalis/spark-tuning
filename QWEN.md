# Qwen3.8-Flash-Next on the dual DGX Spark cluster

> **Status 2026-10-07:** serving config C (GMU 0.835, MAX_NUM_SEQS 8) again, after a day on MiMo-V2.6-Flash
> ([MIMO.md](MIMO.md)). Head-to-head: [QWEN_VS_MIMO.md](QWEN_VS_MIMO.md); Qwen won on speed and on quality.
> The staged 2026-10-04 change below was dropped, never used. Switch models with `mimo/up.sh` / `mimo/down.sh --qwen`.
> **2026-10-16:** nvidia's QAD quant `Qwen3.8-Flash-Next-FP4-Local` was measured against config C ([FP4LOCAL.md](FP4LOCAL.md)): same quality,
> slower under concurrency and with thinking on, so config C stays in production.

Tuning notes, scripts and results for a dual **DGX Spark** (GB10) cluster serving
`nvidia/Qwen3.8-Flash-Next-NVFP4` with vLLM (TP=2 + EP, MTP speculative decoding), deployed with
[MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks).

**Goal:** high-quality, high-speed single-user code generation. 1M context not needed.
**Constraint:** no change may alter model output quality. Only lossless or neutral changes were applied
(speculative decoding is verified by the target model, so the output distribution is unchanged).

Two nodes: head (rank 0) and worker (rank 1), each with a 2.5GbE LAN port and the ConnectX-7 200G
QSFP link (`enp1s0f0np0` / HCA `rocep1s0f0`, the DGX Spark defaults) cabled directly between them.

API: `http://<head>:8000/v1` — model `qwen3.8-flash-next`, `max_model_len` 262144.

---

## TL;DR

| Change | Why | Output change? |
|---|---|---|
| `HEAD_IP`/`WORKER_IP` → CX7 link IPs | vLLM's per-step scheduler→worker broadcast (`--master-addr`, `VLLM_HOST_IP`) was on the 2.5GbE LAN | none |
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

All run from any machine that can reach the head. Python 3 stdlib only. Point them at your cluster with
the `API` env var (or `--url`) for the Python scripts and `HEAD`/`WORKER`/`API`/`REPO` for the shell scripts;
the shell scripts also source an untracked `cluster.env` (copy `cluster.env.example`).

| Script | Restarts server? | Purpose |
|---|---|---|
| `scripts/check_cluster.sh` | no | health, served args, NCCL env on both nodes, confirms control plane on CX7 and that a request moves bytes over RoCE |
| `scripts/bench_decode.py` | no | decode tok/s + TTFT + **per-run MTP acceptance** for `code`, `code_ts`, `edit` (refactor real code in context), `prose`, `copy` × contexts × temperatures; `--json` for comparisons |
| `scripts/spec_acceptance.py` | no | snapshot / delta / `--watch` of acceptance counters + draft-length estimate |
| `scripts/mtp_ab.sh --yes [K...]` | **yes** (~11 min per K) | relaunches with each K, benchmarks, restores the original K and re-checks health; writes `results/mtp_ab-<ts>/` |
| `scripts/watch_stalls.sh` | no | 2 s sampler for the stall symptom: scheduler state + wait reasons, engine progress, GPU util, host memory/swap/major-faults/PSI, RoCE traffic, worker memory. `--once` for a snapshot |
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

Deployment repo: a checkout of `Qwen3.8-Flash-Next-Dual-DGX-Sparks` on the head. Launch with `./stop.sh && ./start.sh --launch`
on the head (cold start ≈ 11 min; the first request after launch is slow while FlashInfer autotunes).

- `configs/env.diff` — `.env` changes (original backed up on the head as `.env.bak-20260926-2124`; HF token redacted here)
- **Dropped 2026-10-07 (never went live):** the head `.env` is back to the tested config C
  (copy kept as `.env.staged-20261004-unused`). What had been staged:
  `MAX_NUM_SEQS` 8→16 (KV pool admits 17.06× full-context sessions; the scheduler cap was the limit),
  `GPU_MEMORY_UTILIZATION` 0.835→0.86 (+~3 GiB KV ≈ +350K tokens; pool was 4,471,258 tok / 17.06× at FP8-dense).
  If it's ever tried: check the boot log (`Free memory on device`) and `free --giga` on **both** nodes, and revert
  GMU to 0.835 if OS-available dips below ~4 GiB on the head (unified memory; see torch.compile incident).
- `configs/start.sh.patch` — the `EXTRA_DOCKER_ARGS` fix
- `configs/running-vllm-args.txt` — the exact `vllm serve` command line now running

To revert everything: on the head, `cp .env.bak-20260926-2124 .env && git checkout start.sh && ./stop.sh && ./start.sh --launch`.

---

## Multi-agent stalls (2026-10-15): measurement, not a fix yet

Symptom reported: with 3-4 pi agents on it, the server shows high load but no agent gets prefilled or generated
tokens for a while. What the server itself recorded, from `docker logs vllm-fn` and `/metrics`:

- **It happens**: 14 ten-second windows today (~2.3 min), in three episodes (09:29:45-09:30:35, 09:34:25-09:34:45,
  09:35:55-09:36:45): `Running: 2-3`, prompt throughput **0.0**, generation **0.7-2.6 tok/s**, KV usage *creeping up*.
- **Server's own latency record** (790 requests since the 10-13 launch): 4 requests in the TTFT 40-80 s bucket,
  91 of 188,969 inter-token gaps in the 2.5-5 s bucket, none over 5 s. Mean TTFT 1.7 s. So a few percent of turns
  get a multi-second freeze, and a handful get tens of seconds.
- **Not the queue and not the KV pool**: `num_preemptions_total` 0, `waiting_by_reason{capacity|deferred}` 0, KV usage
  8-16% of a 4.25M-token pool while only 3-4 of the 8 slots are used.
- **The engine stops stepping, it doesn't draft badly**: during the windows only 16 draft tokens were logged in 10 s
  (against ~800 when healthy) while per-position acceptance stayed at 0.6-0.8. Acceptance is fine; steps stop.
- **Ruled out**: Triton JIT (6 JIT-during-inference warnings total, none in a stall window), disk I/O (container
  `read_bytes` 0, `vmstat` bi/bo ~0), NCCL/RoCE errors (none logged; link counters clean), GPU memory (96/120 GiB),
  and — from the episodes captured on 2026-10-15 with `scripts/watch_stalls.sh` — **memory reclaim**: every frozen
  window sampled `psi_mem 0.00`, `swapout 0 KB/s`, `majfault_delta 0`, worker `avail` steady at ~8.5 GiB. Memory
  pressure only appeared at 10:19:06-15 when `avail` dipped to 4.0 GiB (PSI "some" peaked at 0.72), i.e. it followed
  the stall rather than causing it.

Two live suspects, both consistent with "busy but no tokens":

1. **Async scheduling is enabled** on this deployment. It isn't passed explicitly: `SchedulerConfig.async_scheduling`
   defaults to `None` and is resolved to True, and it is *not* auto-disabled here because our method `mtp` is inside
   `EagleModelTypes` (checked in the image: `vllm/config/vllm.py:1184`, `config/speculative.py:37`). Async scheduling
   plus speculative decoding is precisely the combination the MiMo recipe turns off (`--no-async-scheduling`) because
   of vllm#46669 (corrupted output under concurrency). A desync there looks exactly like this.
2. **Host memory pressure on unified memory.** Head `MemAvailable` is ~5.2 GiB (the guardrail from the torch.compile
   incident is "keep above ~4 GiB"), **2.4 GiB of the vLLM process is swapped out right now** (Worker_TP 1.44 GiB,
   EngineCore 0.26 GiB, API server 0.6 GiB), with 8.15M major faults since boot. When the kernel reclaims the engine's
   own host pages, the next step that touches them page-faults from NVMe: load without progress.

Contributing: pi resends the whole history, and prompt sizes here are p50 ~54K / max ~172K tokens (`request_prompt_tokens`
histogram), so every new turn admits a large prompt in one go — visible as the KV creep during the stall windows.

**Next step**: `scripts/watch_stalls.sh` samples scheduler state (incl. wait reasons), engine progress, GPU util,
host memory/swap/major-faults/PSI and RoCE traffic every 2 s into `results/stall-watch/`, so the next stall is
captured with the cause attached rather than inferred from 10-s log lines. Two candidate fixes, each needing a
restart (~11 min), to be A/B'd against the stall counter:

| Candidate | Change | Expected cost |
|---|---|---|
| Disable async scheduling | add `--no-async-scheduling` to the launch | a few % decode at small batch (it exists to close GPU gaps) |
| Give the OS room | `GPU_MEMORY_UTILIZATION` **down** to 0.80 (~+7 GiB OS headroom) | smaller KV pool (still >10x peak usage) |

Note this is the *opposite* direction to the `MAX_NUM_SEQS=16 / GMU=0.86` change that was staged on 2026-10-04 and
dropped on 2026-10-07: raising GMU takes memory away from the OS, and 16 in-flight 100K+-token prompts admit more KV
at once. If stalls are memory-driven, that staged change would have made them worse.

## Speed re-check (2026-10-15): nothing regressed — here is what the ~20 tok/s actually is

Reported: single-stream coding decode feels like ~20 tok/s against ~27 remembered from the config-C days. Measured
against the recorded baselines (`results/mimo/bench-qwen-C-tp2.json`, `results/mimo/qwen-bench_decode-vsmimo-*.json`),
the server has **not** regressed; what changed is the measurement context.

Drift checks, all clean:

| What | Now | Config-C era | Verdict |
|---|---|---|---|
| `.env` on the head | byte-identical to `.env.C-fp8dense` (diff empty) | same | no drift |
| Launch flags | `--gpu-memory-utilization 0.835 --max-num-seqs 8 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --speculative-config {mtp,K=2} --compilation-config {mode:0,FULL_DECODE_ONLY} --performance-mode interactivity` | same | no drift |
| Image | `vllm/vllm-openai:qwen38-flash-next`, built 2026-08-26 | same image | no drift |
| KV pool | 4,481,625 tokens / 17.10x (boot log 10-13) | 4.25M / 16.2x (10-07) | +5%, bigger now |
| Clocks / throttling | 2501 MHz under load, 61 C, 32 W, no `HW Slowdown` / `SW Power Cap` / thermal bits set | - | not throttled |
| OS updates touching driver/CUDA/firmware since 10-12 | none in `/var/log/apt/history.log`; no reboot since 09-27 | - | no drift |

Same-tool comparison (recipe bench, temperature 0, thinking off, 9 categories, `results/qwen-c-rcheck-mimobench-2026-10-15.json`):

| Cell | Recorded 2026-10-07 | Re-measured 2026-10-15 | Delta |
|---|---|---|---|
| C1 aggregate | 58.3 tok/s | **62.4** | +7% |
| C6 aggregate | 228.2 tok/s | **218.4** | -4% (single run; run-to-run spread on this bench was already measured at 6-15%) |
| C1 coding, per stream | 74.1 | 74.3 | 0% |
| C1 math, per stream | 74.7 | 76.8 | +3% |

So the ~20 tok/s is the *session* condition, not the machine. Decomposition, all T=0.6, `scripts/bench_decode.py`,
`results/qwen-c-rcheck-thinkon-2026-10-15.json` and the concurrency runs in `results/concurrency-2026-10-15/`
(one JSON per stream plus a SUMMARY per condition; thinking ON = what pi sends):

| Condition | Per-stream tok/s | Engine steps/s per stream |
|---|---|---|
| 1 stream, box idle, thinking **off**, 1K | 63.9 | 24.6 (acc 2.60) |
| 1 stream, box idle, thinking **on**, 1K | 52.8 | 25.8 (acc 2.05) |
| 3 streams, thinking on, 1K | 32.0 | 14.6 |
| 4 streams, thinking on, 1K | 26.9-29.3 | 12.7-13.9 |
| 4 streams, thinking on, 32K | 22.7-26.1 | 11.2-12.9 |
| 1 stream, thinking on, 64K | 43.5 | 24.6 |

Read it this way: at 4 concurrent agents a single stream gets 23-29 tok/s while the cluster as a whole does 98-112
tok/s (vs 53 for one stream). Concurrency costs per-stream speed and buys aggregate throughput; context size costs
much less for Qwen than it does for MiMo (Qwen stays within ~20% from 1K to 128K). Thinking on costs ~20% at C1 and
shows up as lower MTP acceptance (2.05 vs 2.60 tokens/step), which is why the pi-facing number is near 20 while the
idle-box number is 52-64.

**Benchmarking rule learned the hard way today**: an agent session running on the server while you benchmark makes
the same tool report C1 aggregate 37 instead of 62 (-40%) and 1K decode 52.8 instead of 63.9. (Those contaminated
numbers are stdout only: `mimobench.py` writes its JSON at the end and that run died on a missing output directory, so
there is no raw file for them — the *clean* run is banked as `results/qwen-c-rcheck-mimobench-2026-10-15.json`.)
Two of my three early
readings today were contaminated this way. Check `run<=1` from `scripts/watch_stalls.sh` (or `/metrics`) before
believing a speed number, and re-run anything with TTFT above ~0.3 s at C1.

## The stall, caught end to end (2026-10-15 14:11) — it is re-crunch, and it is now quantified

`scripts/watch_stalls.sh` with the fixed columns caught a 50-second episode and the counters identify the cause.
Raw rows in `results/stall-watch/` (gitignored live logs); this is the shape:

| time (UTC) | run | wait | wait_cap | kv% | gen_delta | prefill_recomputed | cache-hit alloc | gpu | psi | RoCE MB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 14:11:09 | 1 | 0 | 0 | 2.4 | 3 | 0 | 0 | 96 | 0.00 | 1261 |
| 14:11:12 | 1 | 1 | **1** | 2.7 | **0** | 0 | 0 | 96 | 0.00 | 1504 |
| 14:11:30 | 1 | 1 | 1 | 4.5 | 0 | 0 | 0 | 96 | 0.00 | 1391 |
| 14:11:56 | 1 | 1 | 1 | 6.5 | 0 | 0 | 0 | 96 | 0.00 | 1303 |
| 14:11:59 | 2 | 0 | 0 | 10.2 | 0 | 0 | 0 | 96 | 0.00 | 1231 |
| 14:12:02 | 2 | 0 | 0 | 10.0 | 50 | **182,012** | **140,080** | 94 | 0.00 | 970 |
| 14:12:05 | 1 | 0 | 0 | 3.3 | 282 | 0 | 0 | 94 | 0.00 | 176 |

What it says:

- 50 s with one request running and one waiting **on capacity** (`wait_cap=1`), zero output tokens, GPU "96% busy",
  RoCE sustained at 1.3-1.5 GB/s, KV creeping 2.7%→6.5%, and `psi_mem 0.00` throughout (no memory involvement).
- When it finished, one counter flush landed: **182,012 prompt tokens recomputed** plus 140,080 served from cache
  (the by-source counters are recorded at request completion, so they read 0 during the prefill and jump after it —
  which is also why the 10-second `loggers.py` line showed "prompt throughput 0.0" while KV was climbing; trust the
  counters, not that line). 182,012 tokens over ~50 s = **3,640 tok/s**, matching independently measured cold-prefill
  rate (3,031-3,570 tok/s). KV creep of ~4,500 tokens/s matches the same thing.
- So the frozen window was a **~180K-token prompt being re-prefilled from scratch** — a prefix-cache miss, not a
  hang, not NCCL, not reclaim. One such turn is 50 s of engine time in which every other session sees no tokens and
  the new one is blocked on capacity.
- Scale of it: the `request_prompt_tokens` histogram shows 386 requests in (100K, 200K] and 97 above 200K since the
  10-13 launch, against a cumulative prefix hit rate of 95.3%. Most turns hit; the misses are enormous.

**Why a miss happens on a prompt the server has already seen.** Corrected the same day: it is *not* editing
`AGENTS.md` mid-session. pi reads context files inside `resourceLoader.reload()` (`resource-loader.js:468`, one call
site: the explicit settings/extension reload path), and the injected copy in a session that has been running since
10-07 is still the 10-07 text even after several same-day edits to the file on disk. So live sessions keep a stable
prefix and my doc commits re-crunched nothing. What *does* diverge the prefix (hash from token 0): **session
start/resume** (the file is re-read then, so an edited `AGENTS.md` costs exactly one full re-crunch at that boundary),
model change, thinking-level change, tool/skill set changes, and compaction. All of those are session-boundary or
structural events, which is also why the misses cluster at odd times rather than continuously.

**The per-turn cost of a warm hit is roughly constant, not proportional** (`scripts/prefix_probe.py --steps cold,warm`,
3 sizes, `results/prefix-probe-tail-{8000,24000,96000}-2026-10-15.json`): in steady state the recomputed tail is
2,535 of 23,959 at 24K (89.4% hit) and 2,623 of 96,559 at 96K (97.3% hit) — i.e. ~2.5-3K tokens, about 0.8 s, at any
size. Caveat found while banking the raw data: the **first** warm pass right after a cold prefill recomputes more
(9,215 on that same 96K prompt, 2,995 on a fresh 8K one), because the final chunk isn't cached yet; the turn after
that is flat. An earlier draft of this section used the first-warm figure as steady state — corrected 2026-10-15.
Big contexts therefore do not slow the *hit* path at all; they only make each **miss** catastrophic. That is the whole
argument for smaller sessions: it bounds the tail risk, not the steady state.

**`--performance-mode interactivity` is not a scheduler policy** — checked in the image (`config/vllm.py:411`): the
modes change CUDA graph granularity and kernel selection only. There is no prefill-priority setting implied by it.

### Knobs for this, in the order worth trying

| Change | Where | Effect | Cost |
|---|---|---|---|
| Keep pi sessions smaller; compact before 100K | client | A full miss at 3.6K tok/s is 25 s at 90K, 50 s at 180K; nothing server-side can change that | losing old context |
| Don't edit injected instruction files while agents are running | workflow | Removes the self-inflicted misses | batch doc edits instead |
| `--max-num-batched-tokens 8192` → 4096 | restart | A 180K prefill becomes ~44 chunks instead of ~22, so decode of the other sessions gets a step every ~1 s instead of ~2 s | prefill throughput somewhat lower; the stalled request's own TTFT slightly worse |
| `--long-prefill-token-threshold 16384` | restart | Caps how much of a *long* prompt one iteration chews, the purpose-built knob ("a request is considered long if the prompt is longer than this") | same direction as above |
| `GPU_MEMORY_UTILIZATION` 0.835 → 0.80 | restart | ~+7 GiB OS headroom; `avail` is at 4.1 GiB, the guardrail floor | smaller KV pool (19% peak usage → no practical loss) |
| `--no-async-scheduling` | restart | **Demoted**: the captured episodes are fully explained by long prefill, so this is a hypothesis again, not a suspect | unknown throughput cost |


## When does KV actually get flushed? (2026-10-15, measured with `scripts/prefix_probe.py`)

Symptom that prompted it: an agent that has been idle for a while seems to get its whole prompt re-crunched, even
though nothing restarted since it was last seen. Measured, not inferred: the server counts prompt tokens by source
(`vllm:prompt_tokens_by_source_total{source="local_compute"|"local_cache_hit"}`), so `scripts/prefix_probe.py` reads
those counters around each request and reports recomputed vs cached for that request alone (this fork returns
`usage.prompt_tokens_details = null`, so the response body cannot be used).

Probe: a 23,959-token prompt with a random nonce, sent cold, warm, after competing prompts, and after sitting idle.
Files `results/prefix-probe-2026-10-15.json`, `results/prefix-probe-idle-1007.json`.

| Step | Recomputed | Cached | Verdict |
|---|---|---|---|
| first sight | 23,959 | 0 | baseline: a cold 24K prompt is ~8 s of prefill |
| same bytes again, immediately | 7,479 | 16,480 (68.8%) | mechanism works, but not all of it comes back |
| same bytes again, after 3 competing prompts of the same size | 2,535 | 21,424 (89.4%) | competing traffic did **not** evict it |
| same bytes again after 900 s with nothing sent | 2,535 | 21,424 (89.4%) | **idle does not flush anything** |

So, on this deployment:

- **There is no timer.** Nothing expires a cached prefix; an idle agent keeps its KV as long as the blocks are there.
  `vllm:engine_sleep_state` reads `awake` (not `discard_all`), and nothing on the nodes calls `/sleep` — no timer or
  cron unit touches the server.
- **A warm prompt still re-crunches a fixed tail.** The same prompt recomputed exactly 2,535 tokens (10.6%) on every
  repetition, including after the idle gap. Expect ~10% of a prompt to be recomputed even with a perfect cache hit;
  on a 130K agent context that is ~13K tokens, ~4 s of prefill, every turn.
- **Eviction is pressure-driven (LRU), and there was no pressure**: KV usage during all of this was 5-19% of a
  4.48M-token pool. Idle prefixes are not evicted while the pool is this empty.
- **Therefore a full re-crunch of a prompt the server has already seen means the prefix diverged, not that KV was
  flushed.** The prefix cache matches from token 0 (`prefix_caching_hash_algo=sha256`, block_size 8), so anything
  that changes the *front* of the request invalidates everything after it: the system prompt, the tool list, and (for
  pi) the project instructions file. Editing `AGENTS.md` mid-day invalidates every session's prompt, which is exactly
  the kind of day this was.
- **What this means for the stall symptom.** A pi turn re-sends the whole history (p50 ~54K, max ~172K tokens here).
  A full miss at 3.5K tok/s is 15-50 s during which the engine logs prompt throughput while decode of everyone else
  crawls. Captured episodes (10:08, 10:18-10:19 in `results/stall-watch/`) show `gen_delta ~0` with KV creeping and
  RoCE at 1.3-1.6 GB/s, and the engine logging `prompt throughput 0.0` at the same time — consistent with *admission
  of huge mostly-cached prompts* rather than recompute, but the sampler columns that would prove it
  (`prefix_cache_hit_alloc`) were broken by a parsing bug that day, so this is the leading explanation, not a
  confirmed one. The same windows show PSI memory 0.00 and no swap-out, so memory pressure was **not** the cause.

### Why is there 4.3 GiB in swap on a box that reports free RAM? (2026-10-15)

Because on a GB10 there is no second pool of memory: `free` on the head reads total 121 GiB, **used 117 GiB**, free
2.1 GiB, buff/cache 5.7 GiB, available 4.1 GiB (worker: available 7.5 GiB). The engine's ~101 GiB of GPU/UVM
reservation is the same silicon the OS is counting, so the whole host has ~4 GiB of slack, and `vm.swappiness=60`
(the distro default) means the kernel prefers paging anonymous/shared memory out to discarding page cache. Swap is a
16 GB file, 4.3 GiB used on the head and 2.5 GiB on the worker.

The important detail: **the swapped bytes are not the engine's hot pages.** Every vLLM process reports
`VmSwap: 0 kB` (API server RSS 2.4 GiB, EngineCore 1.1 GiB, Worker_TP0 2.5 GiB), while the container's
`memory.swap.current` was 4.4 GiB — that combination means shared/tmpfs-backed buffers (NCCL and `/dev/shm` segments,
host `Shmem` is 2.4 GiB) are what got swapped, not weights, KV metadata or Python heap. Consistent with this, the
captured stall windows show `psi=0.00`, `swapout=0`, `majfault_delta=0`, so reclaim was not on the critical path at
the moment of a freeze. Memory is therefore a **risk** (4 GiB available is at the guardrail line set after the
torch.compile hang) rather than a demonstrated cause. Without root we cannot lower `vm.swappiness`; the lever we do
have is `GPU_MEMORY_UTILIZATION`, which is also the lever for the stall fix — see the candidates below.

## Things tried that failed / were rejected

### torch.compile — **do not enable** (hung the head)

`--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'` (report §3.6 item 5, "0–8%,
keep mode 0 as fallback"). During startup the head exhausted its unified memory (weights +
KV reservation already leave ~4–7 GiB free) and userspace locked up: ping answered, SSH did not, for
20+ minutes. It needed a hard power cycle.

Side effect: during the ~9.5 h hang its DHCP lease on the LAN expired and it came back on a different
address. The MAC did **not** change. Fix: give the head a static LAN address (and/or a router DHCP
reservation) so it's never handed to another host.

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
- 2026-09-27 07:31 — head power-cycled; came back on a new DHCP address; tuned config relaunched via the CX7 link
- ~07:50 — head pinned back to a static LAN address; reference benchmark `results/current-K2-2026-09-27.json`
- ~12:00 — `index_share_for_mtp_iteration` A/B: no gain, reverted
- 13:00–17:30 — FP8-dense evaluation: RadixArk download + verify, build, configs A/B/C speed + quality
  (`FP8DENSE.md`, `results/quality/`). Server left running config C (FP8-dense).
- ~19:30 — FP8-dense + draft vocab (config D): no gain; back on C
- 2026-09-27 20:00 → 09-28 01:46 — thinking-mode + long-context quality (B vs C), then the K=2 vs K=3 A/B on C.
  Production: **C (RadixArk NVFP4 + FP8-dense, MTP K=2)**
- 2026-10-04 — `.env` staged: MAX_NUM_SEQS 16, GMU 0.86 (dropped 2026-10-07 before ever going live). Confirmed client restarts do **not**
  evict the server KV pool (container up since 09-28, prefix-cache hit rate 95%, utilization ~6%); the slow
  first question after a client restart is a prefill miss on a rebuilt prompt prefix, not eviction.
- 2026-10-15 — multi-agent stall investigated (3-4 agents): ~2.3 min of frozen engine steps in three episodes, 4 requests with TTFT 40-80 s.
  Queue, KV pool, preemptions, JIT, disk and RoCE all clean. Suspects: async scheduling on with MTP, and host
  memory pressure (5.2 GiB available, 2.4 GiB of the engine swapped out). `scripts/watch_stalls.sh` added to catch the next one.
- 2026-10-15 — prefix-cache flush question answered by measurement (`scripts/prefix_probe.py`): no timer, idle does not flush, competing traffic did not evict;
  a warm prompt still re-crunches a fixed ~10% tail (2,535 of 23,959 tokens). A full re-crunch means the prompt diverged from its cached prefix.
- 2026-10-15 — captured two stall episodes live; memory reclaim ruled out (PSI 0.00 during every frozen window). Speed re-check: no regression, ~20 tok/s = 4 agents + thinking.
