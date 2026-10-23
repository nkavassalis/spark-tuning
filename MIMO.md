# MiMo-V2.6-Flash-RL on the dual DGX Spark cluster

Xiaomi's [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (~310B total / ~12B active,
fp8 attention + MXFP4 experts) served with vLLM TP=2 across both Sparks with the bundled DFlash drafter
(7 draft tokens), following
[tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe)
pinned at commit `9c699a2`. The recipe's image, four patches and serving defaults are used **unchanged**.
This page covers what we added around it, what we measured, and the agent caveats.

**Status (2026-10-07):** set up, tested, benchmarked, then **removed from both nodes**. Qwen config C serves the
cluster; MiMo is kept only as this documentation plus the scripts and results here. To run it again for future
model investigation: `mimo/setup_nodes.sh` (re-downloads 178 GB weights, ~25 min, then rsyncs to the worker in
~6 min) and `mimo/up.sh` (~14 min to serving). Agents then use `http://<head>:8000/mimo/v1` (the tool-call-cap
proxy); the raw endpoint is `http://<head>:8888/v1`; model id `mimo-v2.6-flash`.

---

## TL;DR

| | Result |
|---|---|
| Boot | ~13.5 min cold (worker first, then head), `mimo/up.sh` |
| KV pool | 13.34 GiB = **2,047,527 tokens**, 6.83× full 300K requests (recipe reported 1.84–1.87M) |
| Recipe bench C1 / C6 aggregate | **42.8 / 139.6 tok/s** (recipe: 45.6 / 155.8). C1 code 67.6, math 68.2, prose 20.4 |
| Cold prefill 2K / 64K | 2,292 / 1,271 tok/s (recipe: 1,947 / 1,209) |
| Functional checks | chat, tool-call parsing, storm guard: all pass (`mimo/test_mimo.py`) |
| Quality (thinking off, greedy) | HumanEval+ 88.4, MBPP+ 74.3, GSM8K 96.4 (Qwen C: 93.3 / 79.9 / 96.4) |
| Quality (thinking) | HumanEval+ 89.0, LiveCodeBench 46.2 (Qwen C: 92.7 / 41.2); long context 90.3% (Qwen 95.1%) |
| Temperature 0.6 vs server 1.0 (2026-10-23) | thinking off: **MBPP+ 75.4 vs 70.6 (p=0.011)**, HumanEval+ tie; thinking on: tie. 1.0 is 1.3–1.7× slower to solve. Storm trade-off: see "Sampling temperature" |
| vs Qwen config C, single-user code, both thinking off | Qwen is 1.3–1.8× faster at 1K and 2.2–2.7× faster at 128K |

On this cluster MiMo is slower than the tuned Qwen deployment on everything we measured except 2K prefill. It is no
better on quality: worse on MBPP+ (significant), tied or not significantly different elsewhere. Full
head-to-head: **[QWEN_VS_MIMO.md](QWEN_VS_MIMO.md)**. MiMo's advantages are multimodal input (image/video/audio) and
the 300K window.

---

## Tool-call loops ("storms"): what to watch

Upstream observed, and we confirmed with a reproduction (`mimo/test_mimo.py`, raw endpoint returned all 12
requested parallel calls), that the model will happily put many tool calls in one response. In long agent
sessions this becomes a *storm*: dozens to hundreds of calls in one turn, usually right after a large file write,
because the model plans an imagined trajectory instead of waiting for results. Per the recipe's measurements
it is the model's most likely continuation, not a serving bug:

- **temperature 0.6 makes it worse** than 1.0 (4/4 vs 2/4 reproductions), so do *not* lower temperature for "precision";
- repetition penalty does **not** fix storms (it does fix the separate near-greedy "same `grep` 400 times" loop);
- thinking on avoids it, at ~24K reasoning tokens per step;
- streaming vs non-streaming and async scheduling make no difference.

Mitigations in effect here:

| Mitigation | Where | What it does / doesn't do |
|---|---|---|
| `--generation-config auto` + `repetition_penalty 1.05` | server default (recipe `REP_PENALTY`) | requests that send **no** sampling params get temperature 1.0 / top_p 0.95 / rep-pen 1.05. Stops the near-greedy repeated-identical-call loop. **A client that sends its own `temperature` overrides this** — check your agent's settings. |
| Tool-call cap proxy, `toolCap: 6` | `mimo/toolcap/` on the head, `:8000/mimo/v1` | on a **streamed** response, when a 7th tool call opens it aborts the generation upstream and ends the stream with `finish_reason: tool_calls`. The agent runs the first 6 and asks again with real results. Verified: 12-call request → 6 complete calls, `finish_reason=tool_calls`, proxy log `tool-call storm guard tripped at 7 calls`. |
| `--no-async-scheduling` | server (recipe default) | not a storm fix: prevents token corruption under concurrency with spec decode (vllm#46669) |
| Thinking off by default | server (recipe default) | keeps reasoning out of `content`. Turning thinking on per request avoids storms at a large token cost. |

**Caveats to be wary of:**

1. **The proxy only caps streamed (SSE) responses.** Non-streaming requests pass through untouched and can
   still return hundreds of calls. Use a streaming client, or set a modest `max_tokens`.
2. **Bypassing the proxy disables the guard.** Port 8888 is the raw server, for benchmarks only. Point agents at
   `:8000/mimo/v1`.
3. **The cap limits damage; it doesn't stop the behaviour.** Upstream reports one harness (OMP, whose system prompt
   pushes parallel calls) still storms on almost every step and only crawls forward under the cap. If an agent
   keeps hitting the cap every turn (`grep tripped ~/mimo-toolcap/proxy.log` on the head), check its system
   prompt for "call tools in parallel" wording before anything else.
4. A legitimate turn that needs >6 parallel calls gets split across turns (costs a round-trip, not correctness).
5. vLLM also supports a per-request `repetition_detection` stop, if your client can send extra body fields.
   Not used or tested here.

---

## What we did (and why it differs from the recipe quick start)

All scripts read hosts from the untracked `cluster.env`; nothing site-specific is in the repo.

| Step | Script | Notes |
|---|---|---|
| Recipe checkout on both nodes, pinned commit | `mimo/setup_nodes.sh` | `RECIPE_COMMIT` in `mimo/lib.sh`; bump on purpose |
| Render `launch/mimo.env` on each node | `mimo/setup_nodes.sh` from `mimo/mimo.env.example` | the head's link IP, subnet and `$HOME` are discovered on the nodes at run time |
| Weights + cache under `$HOME`, not `/var/tmp` | `mimo/mimo.env.example` | systemd-tmpfiles ages out `/var/tmp` (30 days on Ubuntu) and would eat 178 GB of weights |
| Download on the head only, rsync to the worker over the CX7 link | `mimo/setup_nodes.sh` | 178 GB: ~25 min from HF, then 5.5 min head→worker at ~530 MB/s. No NFS (would need root). |
| Drop the page cache before launch without sudo | `drop_caches` in `mimo/lib.sh` | recipe's `serve.sh` tries `sudo -n`, which fails silently here; at GMU 0.90 vLLM's probe doesn't count page cache. We run `echo 3 > /proc/sys/vm/drop_caches` in a privileged container (SSH user is in the docker group). |
| Stop Qwen → worker → head → proxy → wait for ready | `mimo/up.sh` | the recipe's `serve.sh` exits 1 on the worker even on success (last line is `[ "$R" = 0 ] && echo`), so `up.sh` checks the container instead |
| Tool-call-cap proxy as a container on the head | `mimo/up.sh`, `mimo/toolcap/` | `node:22-alpine`, `--restart unless-stopped`, host network, `:8000`. One-line change to upstream (`upstream.diff`): bind address from `TOOLCAP_HOST` (upstream hardcodes loopback). |
| Switch back | `mimo/down.sh [--qwen]` | `--qwen` relaunches the Qwen repo's `./start.sh --launch` |

Serving config (recipe defaults): TP=2 (expert parallelism not enabled by anything in this repo; Qwen config C runs TP=2 + EP), fp8 KV, GMU 0.90, `max-model-len` 300000, `max-num-seqs` 8, DFlash 7, marlin
MXFP4 MoE, DeepGEMM off, thinking off, `--no-async-scheduling`, `--generation-config auto`, rep-pen 1.05.
Boot log: free memory at start 111.41/121.63 GiB; weights + non-torch 88.94 GiB; peak activation 7.19 GiB.
After boot the OS reports ~6 GiB available on the head and ~9 GiB on the worker, about the same headroom as the
Qwen deployment. **The torch.compile hang in [QWEN.md](QWEN.md) applies here too: do not raise GMU or enable
compilation modes without watching `free` on both nodes.**

---

## Measurements (2026-10-07, single run each, nothing else on the server)

### Recipe bench (`bench/mimobench.py`, prompt set v1, temperature 0, unique prefixes)

Full output: [results/mimo/bench-spark-tuning-tp2.md](results/mimo/bench-spark-tuning-tp2.md) / `.json`.

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | recipe aggregate |
|---|---|---|---|---|
| C1 | 42.82 | 48.48 | 0.302 | 45.58 |
| C2 | 63.72 | 37.78 | 0.459 | 71.03 |
| C3 | 85.02 | 33.84 | 0.516 | 99.69 |
| C4 | 101.00 | 30.42 | 0.544 | 120.37 |
| C5 | 122.11 | 29.49 | 0.583 | 136.47 |
| C6 | 139.63 | 28.57 | 0.613 | 155.77 |

We're 6–15% under the recipe at C1–C6. Possible causes we haven't tested: single run vs their two-pair average,
fp8 KV pool size, and the recipe's note that single-stream cells move ~10% between passes. Prefill is 2–18% *faster*
than the recipe's numbers (2,292 / 1,762 / 1,486 / 1,271 tok/s at 2K / 8K / 32K / 64K).

DFlash accepted tokens per draft step (of 7) at C1: counting 6.97, structured 6.29, format 5.50, coding 5.42,
math 5.07, json 3.57, reasoning 2.12, summary 1.08, prose 0.84, narrative 0.76. Prose is where DFlash doesn't help.

### Our single-user code bench vs the tuned Qwen (`scripts/bench_decode.py`, 500 decode tokens)

First comparison, against Qwen's 2026-09-27 numbers. Those Qwen runs used the template default (thinking on),
which makes Qwen slower. The same-day thinking-off comparison is in [QWEN_VS_MIMO.md](QWEN_VS_MIMO.md) and shows a
larger gap. T=0.6, median of 3 (`results/mimo/bench_decode-vsqwenC-2026-10-07.json` vs
`results/fp8dense-C-fp8dense-speed.json`):

| ctx | task | Qwen C tok/s (acc.len) | MiMo tok/s (acc.len) | Δ |
|---|---|---|---|---|
| 1K | code | 52.8 (2.05) | 37.2 (3.50) | −30% |
| 1K | code_ts | 52.4 (2.00) | 40.7 (3.93) | −22% |
| 1K | edit | 55.4 (2.12) | 52.2 (5.00) | −6% |
| 128K | code | 49.1 (1.93) | 24.6 (3.45) | −50% |
| 128K | code_ts | 50.0 (1.95) | 29.7 (4.17) | −41% |
| 128K | edit | 55.2 (2.18) | 30.4 (4.30) | −45% |

Engine step rate (tok/s ÷ acceptance length) explains it. MiMo runs ~10.6 steps/s at 1K and ~7.1 at 128K;
Qwen config C runs ~25 steps/s, flat with context. DFlash accepts 2× more tokens per step than Qwen's MTP, but
each MiMo step costs more than 2× as much and gets slower as context grows. Our guess is the 9 full-attention
layers, which we haven't profiled.

A wider single-run grid (T=0 and 0.6, 1K and 64K, including prose) is in
`results/mimo/bench_decode-2026-10-07.json`: 1K T=0 code 40.0 / edit 58.0 / prose 23.1;
64K T=0 code 31.5 / edit 46.7 / prose 17.5.

---

## Sampling temperature: 0.6 vs the server's 1.0 (2026-10-23)

MiMo was reinstalled for one day to answer a single question: does temperature 0.6 (Qwen's setting) change MiMo's quality, compared
with the checkpoint's 1.0 that the server applies to any request that sends nothing? The two thinking-off runs before this were
greedy, so the temperature MiMo actually runs at in pi (server default 1.0, thinking off) had never been measured. Every run used the
same harness, data and sandbox as 2026-10-07, top_p 0.95, no top_k, and the server's rep-pen 1.05 ("as deployed"). Each problem
carries a fixed seed (`crc32` of the prompt or task id), so both temperatures draw the same random stream per problem. One run each,
paired McNemar. The thinking-off sampled mode is new: `NOTHINK_TEMP` / `NOTHINK_TOP_P` in `quality/run_quality.sh` (unset = the old
greedy request, byte for byte).

**Quality** (full sets, concurrency 8, 16K budget with thinking on):

| Test | 2026-10-07 (off: greedy; on: 1.0) | **0.6** | **1.0** | 0.6 vs 1.0 (lost / gained at 1.0) |
|---|---|---|---|---|
| HumanEval+, thinking off | 88.4 | 84.8 | 86.0 | 8 / 10, p=0.82 |
| **MBPP+, thinking off** | 74.3 | **75.4** | 70.6 | 32 / 14, **p=0.011** |
| HumanEval+, thinking on | 89.0 | 92.1 | 92.1 | 4 / 4, p=1.0 |
| LiveCodeBench, thinking on | 46.2 | 48.8 | 45.0 | 5 / 2, p=0.45 |

- **Thinking off, 1.0 is measurably worse on MBPP+**: −4.8 points against 0.6 (p=0.011) and −3.7 against greedy (p=0.076).
  HumanEval+ shows nothing at any temperature.
- **Thinking on, temperature doesn't matter that we can see.** The 1.0 rerun reproduced the 0.6 HumanEval+ score exactly. Against
  2026-10-07 at identical settings and seeds it moved 89.0 → 92.1 (3 / 8 flips, p=0.27). That is the run-to-run spread, about ±3
  points: the first, apparently better, 0.6 result was this noise, not temperature.
- Lengths barely move: thinking-off HumanEval+ / MBPP+ 323 / 259 tokens at 0.6, 352 / 288 at 1.0 (greedy 327 / 253). Thinking on:
  742 / 9,331 at 0.6, 792 / 9,855 at 1.0. About half of LiveCodeBench hits the 16K cap at every setting (38 and 40 of 80).

**Time to solve** (`quality/run_tts.sh`: one problem at a time, wall clock per request, 16K budget, nothing else on the server; the
COMPARISON.md subsets: every 4th HumanEval+ problem = 41, every 8th LiveCodeBench = 10):

| Setting | HE+ solved / 41 | s per problem | s per solved | mean tokens | tok/s | LCB solved / 10 | s per problem | mean tokens |
|---|---|---|---|---|---|---|---|---|
| thinking off, **0.6** | 34 | **7.2** | **8.7** | 356 | 49.2 | 1 | **110** | 4,069 |
| thinking off, 1.0 (server default) | 35 | 9.5 | 11.1 | 411 | 43.3 | 0 | 160 | 4,327 |
| thinking on, **0.6** | 37 | **21.8** | **24.1** | 904 | 41.5 | 1 | **414** | 14,199 |
| thinking on, 1.0 | 37 | 37.4 | 41.4 | 1,205 | 32.2 | 2 | 467 | 13,134 |

**At 1.0 MiMo is slower to the same answer**: 1.3× (thinking off) and 1.7× (thinking on) longer per HumanEval+ problem, at the same
solve count. Two effects stack: it writes more tokens (+15–33%), and it writes them more slowly (43 vs 49 and 32 vs 42 tok/s,
prefill included). The slower rate is consistent with DFlash drafts being accepted less often at a higher temperature, as with
Qwen's MTP. That is inferred: draft acceptance wasn't recorded. n=10 can't separate LiveCodeBench solve counts.

**What this means for pi.** pi with no `samplingParams` runs MiMo thinking off at 1.0, the slowest and, on MBPP+, the worst setting
measured here. 0.6 would be better on both. **But the recipe measured 0.6 making tool-call storms more frequent** (4/4 vs 2/4
reproductions, see "Tool-call loops" above), and nothing here tests storms: these are single-turn prompts with no tools. So 0.6 is a
trade (better code, faster, more storm risk), not a free win. The server default stays at 1.0, and the storm rule above still
stands. Untested: anything between 0.6 and 1.0, a storm reproduction at 0.6 with the tool-call cap in place, and GSM8K / long
context at either temperature.

Run on the worker with `API=http://<head-link-ip>:8888 MODEL=mimo-v2.6-flash` and, per row:
`TASKS="humaneval mbpp" NOTHINK_TEMP=0.6 NOTHINK_TOP_P=0.95 quality/run_quality.sh <label>`,
`TASKS="humaneval lcb" THINK_TEMP=0.6 THINK_TOP_P=0.95 THINK_TOP_K= quality/run_deep.sh <label>`,
`THINK_KWARGS='{"chat_template_kwargs":{"enable_thinking":false}}' THINK_TEMP=0.6 THINK_TOP_P=0.95 THINK_TOP_K= quality/run_tts.sh <label>`
(drop `THINK_KWARGS` for thinking on). Files: `results/quality/M-t{06,10}-{off,deep}/`, `results/quality/TTS-mimo-{off,on}-t{06,10}/`.
Lowest free memory over the whole day: 3.5 GiB (memory guard at 1.5 GiB never fired).

## What is on the nodes / how it was removed

**Reinstalled 2026-10-23 for the temperature study above, then removed again the same day.** Removed: each node's
`models/MiMo-V2.6-Flash-RL`, `mimo-cache`, `mimo-recipe`, `mimo-toolcap`, `mimo-setup.log`, plus `docker rmi` of the recipe image
and `node:22-alpine`. Disk back to 508 GB free on the head and 3.2 TB on the worker. `~/.venv-hf` stays: `fp4local/` and `glmfp4/`
setup use it too. Every result file was copied into this repo before removal. (First removal: 2026-10-07, same list plus the venv.)

Gotcha, hit both times: root-owned files under `mimo-cache` and the recipe checkout make a plain `rm -rf` fail. On 10-07 the
audio-lib step (`docker run` as root) caused it. The end-of-setup chown added then made it worse: `$(id -u):$(id -g)` sat inside
single quotes, so it expanded in the container, as root, and chowned **everything** to `root:root` (seen 2026-10-23). Fixed: the ids
now go in as `-e OWNER=...` from the node's shell (tested on the worker: a root-created file came back user-owned, plain `rm` worked).
Serving also writes root-owned files (Triton cache etc.) into `mimo-cache` *after* setup, though. So clean up with a container:

```bash
docker run --rm -v $HOME:/h --entrypoint sh <any local image> -c 'rm -rf /h/mimo-cache /h/mimo-recipe'
```

## Not done / open items

- ~~Quality~~: done 2026-10-07, see [QWEN_VS_MIMO.md](QWEN_VS_MIMO.md). Run with
  `API=http://<head-link-ip>:8888 MODEL=mimo-v2.6-flash quality/run_quality.sh <label>` and, for thinking mode,
  `THINK_TEMP=1.0 THINK_TOP_P=0.95 THINK_TOP_K= quality/run_deep.sh <label>` (on the worker).
- **Real agent sessions**: the storm guard has only been checked with a synthetic 12-call request. We haven't
  measured how often it trips in a real long session with our agent.
- Multimodal paths (image/video/audio), needle-in-a-haystack, and 300K-context requests: the recipe verified them
  on its hardware. We haven't run its `tests/mimo_vision.py`, `mimo_media.py` or `mimo_needle.py` here.
- Untested tuning knobs from the recipe: `SPEC_K` (7 is the recipe's measured best on TP4), GMU 0.88 at TP4 only.
