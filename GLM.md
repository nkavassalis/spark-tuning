# GLM-5.3-Flash (autotrust E192 / E224 NAS-pruned NVFP4) on the dual DGX Spark cluster

[autotrust/GLM5.3-Flash-E192-DGX-Spark](https://huggingface.co/autotrust/GLM5.3-Flash-E192-DGX-Spark) is an
unofficial build of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash): 192 of 288 routed experts
kept per layer (pruned by neural architecture search), NVFP4 experts, 18B active parameters per token, 124 GiB of weights,
full 154,880-token vocabulary, and a bundled NVFP4 MTP draft (`mtp-nvfp4/`). E224 keeps 224 experts (141 GiB). We served both
with stock vLLM 0.31.0 at TP=2 over the CX7 link, with MTP K=2. Revisions tested: E192 `ef3f9cf`, E224 `45d2363`.

**Status (2026-10-13):** evaluated overnight, then the cluster went back to Qwen config C and GLM was **removed from both
nodes** (both checkpoints, the vLLM 0.31.0 image, `~/glm-cache`). To run it again: `GLM_VARIANT=E224 glm/setup_nodes.sh`
(158 GB download + rsync to the worker, ~45 min) then `GLM_VARIANT=E224 glm/up.sh` (~10 min). Use `E192` for the smaller build.
Removal note: the container writes root-owned files into `~/glm-cache`, so delete it with a privileged container
(`docker run --rm --privileged -v $HOME:/h --entrypoint sh <image> -c 'rm -rf /h/glm-cache'`), then `~/models/GLM5.3-Flash-*`.

## Summary (E192)

| | Qwen C | MiMo | **GLM E192** | Verdict |
|---|---|---|---|---|
| Single-user code decode, 1K, thinking off (tok/s, median of 3) | 66–70 | 37–52 | **20–32** | Qwen ×2.2–3.6 |
| Single-user code decode, 128K, thinking off | 65–68 | 25–30 | **20–31** | Qwen ×2.1–3.2; GLM ≈ MiMo |
| Decode at template default (Qwen thinking on / GLM effort max), 1K–128K | 48–56 | — | **27–29** | Qwen ×1.8–2.1 |
| Engine steps/s (flat 1K→128K?) | ~25.5 (yes) | 10.6 → 7.1 (no) | **~11.5 (yes)**, some runs 7.5 | |
| Recipe bench aggregate C1 / C6 (tok/s) | 58.3 / 228.2 | 42.8 / 139.6 | **21.0 / 88.0** | Qwen ×2.6–2.8 |
| Cold prefill 32K / 64K (tok/s) | 3,031 / 2,989 | 1,486 / 1,271 | **1,954 / 2,009** | Qwen ×1.5 |
| HumanEval+ / MBPP+ / GSM8K (thinking off, greedy) | 93.3 / 79.9 / 96.4 | 88.4 / 74.3 / 96.4 | **92.1 / 79.9 / 96.3** | GLM = Qwen; both > MiMo on MBPP+ |
| HumanEval+ (thinking) | 92.7 | 89.0 | **92.7** | tie |
| LiveCodeBench (thinking, 16K budget) | 41.2 | 46.2 | **47.5** | GLM best, not significant (p=0.30 vs Qwen) |
| Long context 64K/128K (retrieve, two-hop, reverse) | 95.1% | 90.3% | **97.9%** | GLM best; significant vs MiMo (p=0.007) |
| Free memory per node after boot | ~4–7 GiB | ~6–9 GiB | **19 GiB** | GLM |

**Bottom line:** GLM E192 is at least as accurate as anything we've run here. It ties Qwen config C on code (HumanEval+, MBPP+) and math,
and leads on LiveCodeBench and long-context two-hop retrieval, though only the long-context lead over MiMo is statistically
significant. It is also **2–3× slower than Qwen** at single-user decode (11.5 vs 25.5 engine steps/s with similar MTP acceptance)
and ~1.5× slower at prefill. For pi as a coding agent, Qwen config C stays the better default: same pass rates at more than twice the
speed. GLM is the one to try when correctness at long context matters more than latency. If you do, **set `reasoning_effort`**:
the default is `max`.

**E224** (224 experts, 9 GiB more weights per node) runs at the same speed, still leaves 11–13 GiB free per node, and scored
**100% on long context** (significantly ahead of Qwen, p=0.016) and 82.0 on MBPP+ (+2.1 over E192, not significant). It also has
better corpus likelihood (0.141 vs 0.196 bits/byte). Its GSM8K and thinking-mode tests weren't run. If you want GLM on this cluster,
**E224 is the better pick**: it fits fine at 262K context and loses nothing in speed. See [E224](#e224-224-experts-kept-158-gb).

---

## Setup (what we did, and why)

All scripts read hosts from the untracked `cluster.env`. Nothing site-specific is in the repo.

| Step | Script | Notes |
|---|---|---|
| Download on the head, rsync to the worker over CX7 | `glm/setup_nodes.sh` | 139 GB (E192) / 158 GB (E224). rsync ran at 280–370 MB/s. Same path on both nodes (vLLM's multi-node loader needs that). |
| Enable the MTP code path | `glm/setup_nodes.sh` | `text_config.num_nextn_predict_layers` 0 → 1 in the target `config.json` (backup `config.json.bak-nomtp`), per the model card. Inert without `--speculative-config`. |
| Stock image, patched at start | `glm/patch_vllm.py`, run by `glm/up.sh` inside the container | `vllm/vllm-openai:v0.31.0` (arm64, CUDA 13, torch 2.13). Two small patches from autotrust's E224 two-Spark runbook so vLLM can load a **standalone** draft directory: build the draft with its own config/quant config, and don't defer the draft's lm_head. Idempotent, only touch the spec-decode path. |
| Launch | `glm/up.sh` / `glm/down.sh [--qwen]` | Stops Qwen, drops the page cache (privileged container, no sudo), starts the head, waits for its rendezvous port, starts the headless worker, waits for `/v1/models`. Container `vllm_glm`, caches in `~/glm-cache`. |

Serving config (same for both variants):

| Setting | Value | Why |
|---|---|---|
| `--speculative-config` | MTP, `/model/mtp-nvfp4`, `num_speculative_tokens` 2 | the card's recommendation. The NVFP4 draft is the fixed 2026-10-10 export (shared expert NVFP4, proper exclude list). |
| `--kv-cache-memory-bytes` | 20 GiB per node | a fixed KV pool instead of a GMU fraction. E192: 2,111,321 tokens, 8.05× a full 262K request. |
| `--gpu-memory-utilization` | 0.80 | only the startup free-memory gate when the KV size is fixed |
| `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` | 262144 / 8 / 8192 | same context and concurrency as Qwen config C; the model supports 1M |
| Parallel layout | TP=2 only (no `--enable-expert-parallel`) | Qwen config C runs TP=2 **+ EP**, so the speed comparisons are also TP-only vs TP+EP. GLM with EP was never tried; the 2–3× decode gap is too large for layout alone. |
| `--compilation-config` | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | **torch.compile off**, CUDA graphs for decode. A compile-time memory spike hung the head once (QWEN.md). |
| `--no-enable-flashinfer-autotune` | | the runbook: autotune desyncs TP ranks across nodes |
| `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice` | | from the card. Without them, reasoning lands in `content` and tool requests 400. |
| NCCL / host env | same as Qwen's (`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` = CX7 iface, `NCCL_IB_HCA==rocep1s0f0`, `NCCL_NET=IB`, …), `VLLM_USE_DEEP_GEMM=0` | the runbook's "bootstrap over the management NIC" split wasn't needed. Qwen's settings worked first time. |

E192 memory after boot: 65.18 GiB of weights + draft per node; the OS reports **19 GiB available on each node**
(Qwen leaves ~4–7 GiB, MiMo ~6–9). Weight load took 2.5 min on the worker and 7 min on the head (page cache had just been
dropped, and an E224 download was writing to the same disk). Boot to ready: 569–588 s.

### Thinking is always on: what that means for clients

GLM-5.3's chat template has no thinking switch. It takes `reasoning_effort` (`low` / `high` / anything else = **`max`**),
passed as `chat_template_kwargs.reasoning_effort` (or the top-level OpenAI field). **A client that sends nothing gets
`max`.** pi doesn't send it, so pi would get max-effort reasoning on every turn.
At `low` the model emitted **zero reasoning tokens** on every request we inspected, so `low` is the counterpart
of Qwen/MiMo's "thinking off" in every table here. The harness gained env knobs for this (`NOTHINK_KWARGS`, `THINK_KWARGS`,
`NOTHINK_MAX_TOKENS` in `quality/`, `--chat-kwargs` in `scripts/bench_decode.py`). Their defaults reproduce the old
Qwen/MiMo requests exactly.

Functional checks (E192): chat at low and default effort (reasoning in the `reasoning` field, code in `content`);
a two-city weather request returned two parsed `get_weather` tool calls with `finish_reason=tool_calls`.

---

## Speed (E192, nothing else on the server)

### Single-user decode (`scripts/bench_decode.py`, 500 tokens, `ignore_eos`, T=0.6, median of 3)

Qwen C and MiMo columns are the 2026-10-07 runs from [QWEN_VS_MIMO.md](QWEN_VS_MIMO.md). "st/s" = engine steps per second
(tok/s ÷ mean acceptance length). Files: `results/glm/e192-bench_decode-low-2026-10-13.json`,
`results/glm/e192-bench_decode-default-2026-10-13.json`.

Thinking off (GLM: `reasoning_effort=low`):

| ctx | task | Qwen C tok/s (acc.len) | MiMo tok/s (acc.len) | **GLM E192 tok/s (acc.len)** | GLM st/s | GLM run range |
|---|---|---|---|---|---|---|
| 1K | code | 66.3 (2.55) | 37.2 (3.50) | **29.3** (2.54) | 11.5 | 29.3–29.4 |
| 1K | code_ts | 68.0 (2.62) | 40.7 (3.93) | **31.5** (2.74) | 11.5 | 31.0–31.6 |
| 1K | edit | 70.1 (2.73) | 52.2 (5.00) | **19.7** (2.66) | 7.4 | 19.1–31.6 |
| 128K | code | 65.2 (2.56) | 24.6 (3.45) | **20.4** (2.70) | 7.6 | 18.2–29.4 |
| 128K | code_ts | 65.1 (2.55) | 29.7 (4.17) | **31.1** (2.79) | 11.2 | 30.7–31.9 |
| 128K | edit | 67.6 (2.72) | 30.4 (4.30) | **28.6** (2.48) | 11.5 | 26.0–32.0 |

Template default (Qwen: thinking on; GLM: effort `max`, which is what pi would get):

| ctx | task | Qwen C tok/s (acc.len) | **GLM E192 tok/s (acc.len)** | GLM st/s |
|---|---|---|---|---|
| 1K | code | 56.4 (2.19) | **26.9** (2.32) | 11.6 |
| 1K | code_ts | 49.2 (1.91) | **26.7** (2.31) | 11.6 |
| 1K | edit | 54.8 (2.13) | **29.4** (2.53) | 11.6 |
| 128K | code | 51.3 (2.04) | **27.8** (2.39) | 11.6 |
| 128K | code_ts | 48.2 (1.89) | **26.6** (2.32) | 11.5 |
| 128K | edit | 52.5 (2.13) | **28.9** (2.50) | 11.6 |

What this says:

- **GLM runs ~11.5 engine steps/s, flat from 1K to 128K.** Qwen C runs ~25.5, also flat; MiMo falls from 10.6 to 7.1. The
  NVFP4 MTP draft accepts about as much as Qwen's MTP at K=2 (acc.len 2.3–2.8 vs 1.9–2.7), so the ~2.2× gap to Qwen is
  step cost (each GLM step reads 18B active parameters' worth of weights; we did not profile where the time goes).
  Decode step rate does not change with context up to 128K.
- **Some runs drop to ~7.5 steps/s** (1K edit, 128K code at effort low: 2 of 3 and 2 of 3 runs). Acceptance in those runs is normal, so
  it's step cost, not drafting. GPU clocks stayed at ~2.5 GHz with no throttle reasons on either node, and nothing else was using CPU.
  It never happened at effort `max` (all 18 runs between 26.2 and 30.4 tok/s, medians at 11.5–11.6 st/s). It isn't post-EOS output either: at effort low the natural answers to
  those prompts run past 500 tokens. **Cause not identified.** Treat 7.5–11.5 st/s as the real range.
- vs MiMo: similar step rate at 1K, where DFlash's longer accepted runs put MiMo 1.3–2.6× ahead. MiMo slows with context and GLM
  doesn't, so at 128K they're level (GLM 20–31 vs MiMo 25–30 at effort low; GLM 27–29 at effort max).

Wider single-run grid (effort low, T=0 and 0.6, 1K and 64K, includes prose; `results/glm/e192-bench_decode-wide-low-2026-10-13.json`).
Qwen / MiMo / GLM tok/s:

| ctx | T | code | code_ts | edit | prose |
|---|---|---|---|---|---|
| 1K | 0.0 | 71.1 / 40.0 / **30.3** | 68.7 / 47.6 / **32.0** | 72.5 / 58.0 / **32.4** | 56.9 / 23.1 / **24.1** |
| 1K | 0.6 | 67.8 / 38.1 / **30.0** | 67.3 / 44.6 / **31.2** | 70.8 / 44.5 / **31.2** | 53.6 / 23.3 / **16.6** |
| 64K | 0.0 | 66.3 / 31.5 / **18.5** | 69.3 / 35.9 / **22.8** | 70.4 / 46.7 / **21.6** | 52.6 / 17.5 / **19.8** |
| 64K | 0.6 | 67.0 / 28.4 / **29.3** | 61.3 / 30.1 / **26.4** | 69.8 / 45.3 / **24.1** | 52.1 / 16.9 / **18.3** |

The 64K T=0 row was mostly in the slow mode above (single runs).

### Recipe bench (MiMo recipe `bench/mimobench.py` @ `9c699a2`, temperature 0, unique prefixes)

Run on the head, as for Qwen and MiMo. The only change was the request's `chat_template_kwargs`
(`{"reasoning_effort":"low"}` instead of `{"enable_thinking":false}`, via a local copy that reads `BENCH_CHAT_KWARGS`).
The bench's own header still says "thinking off". Files: `results/glm/bench-e192-tp2.{md,json}`.

| C | Qwen C aggregate | MiMo aggregate | **GLM E192 aggregate** | Qwen / MiMo / **GLM** per-stream |
|---|---|---|---|---|
| 1 | 58.3 | 42.8 | **21.0** | 66.1 / 48.5 / **22.4** |
| 2 | 96.1 | 63.7 | **40.9** | 55.3 / 37.8 / **22.9** |
| 3 | 137.6 | 85.0 | **55.3** | 53.6 / 33.8 / **21.2** |
| 4 | 174.5 | 101.0 | **73.1** | 49.1 / 30.4 / **20.4** |
| 5 | 197.7 | 122.1 | **69.8** | 45.7 / 29.5 / **15.8** |
| 6 | 228.2 | 139.6 | **88.0** | 44.0 / 28.6 / **16.7** |

C1 per-stream by category (GLM): coding 25.5, format 26.3, math 24.8, structured 24.6, json 22.4, reasoning 22.3, summary 21.6,
prose 18.4, narrative 15.9. The table's "DFlash accepted tokens per draft step" column is MTP K=2 acceptance here (max 2): 1.25–1.89.
Per-stream speed holds up to C4 and then drops; aggregate scales ~4× from C1 to C6.

Cold prefill (one request, unique prefix):

| prompt (target) | Qwen C tok/s | MiMo tok/s | **GLM E192 tok/s (TTFT)** |
|---|---|---|---|
| ~2K | 1,109 | 2,292 | **1,568** (0.95 s) |
| ~8K | 2,344 | 1,762 | **1,984** (2.99 s) |
| ~32K | 3,031 | 1,486 | **1,954** (12.2 s) |
| ~64K | 2,989 | 1,271 | **2,009** (23.8 s) |

Prompt token counts differ by tokenizer: the same ~64K-target text is 47,771 GLM tokens. GLM prefill is flat at ~2,000 tok/s
from 8K up (the runbook measured 2,060 at 192K), about 2/3 of Qwen's.

---

## Quality (E192)

Same harness, data and sandbox as Qwen and MiMo (`quality/run_quality.sh`, `quality/run_deep.sh`, run on the worker).
Qwen C numbers are the 2026-09-27/28 runs (`results/quality/C-radix-fp8dense`, `C-deep`), MiMo's are from 2026-10-07
(`M-mimo-tp2`, `M-deep`). GLM: `results/quality/G192-low`, `results/quality/G192-deep`. One run each.

| Test | n | Qwen C | MiMo | **GLM E192** | GLM vs Qwen: Qwen-only / GLM-only (p) | GLM vs MiMo: MiMo-only / GLM-only (p) |
|---|---|---|---|---|---|---|
| HumanEval+ (base), greedy, thinking off | 164 | 95.1 | 90.9 | **95.1** | | |
| **HumanEval+ (plus)**, greedy, thinking off | 164 | 93.3 | 88.4 | **92.1** | 6 / 4 (p=0.75) | 6 / 12 (p=0.24) |
| MBPP+ (base) | 378 | 93.1 | 88.9 | **94.7** | | |
| **MBPP+ (plus)**, greedy, thinking off | 378 | 79.9 | 74.3 | **79.9** | 16 / 16 (p=1.0) | 16 / 37 (**p=0.005**) |
| GSM8K, greedy, thinking off | 1319 | 96.4 | 96.4 | **96.3** | 26 / 25 (p=1.0) | 22 / 21 (p=1.0) |
| HumanEval+ (base), thinking | 164 | 95.1 | 95.1 | **97.0** | | |
| **HumanEval+ (plus)**, thinking, 16K budget | 164 | 92.7 | 89.0 | **92.7** | 5 / 5 (p=1.0) | 7 / 13 (p=0.26) |
| **LiveCodeBench** (80 recent problems), thinking, 16K budget | 80 | 41.2 | 46.2 | **47.5** | 5 / 10 (p=0.30) | 6 / 7 (p=1.0) |
| **Long context**: retrieve / two-hop / reverse at 64K + 128K | 144 | 95.1% | 90.3% | **97.9%** | 3 / 7 (p=0.34) | 2 / 13 (**p=0.007**) |
| ↳ two-hop only | 48 | 41/48 | 34/48 | **45/48** | | |

p-values are exact McNemar tests on paired per-problem pass/fail (`quality/quality_eval.py compare-pass`). Two runs of the
same Qwen config differed by 1–2 points, so anything without a small p-value is a tie.

Conditions:
- "Thinking off" for GLM is `reasoning_effort=low` (it can't turn thinking off; at low it emitted no reasoning).
  Greedy, `max_tokens` 4096 (Qwen/MiMo: 1536, 1024 for GSM8K; GLM's mean completion was 204 / 178 / 110 tokens, so the
  bigger cap only matters for runaway loops). Truncated at the cap: 1/164, 2/378, 0/1319. Every HumanEval+/MBPP+ answer
  but one per task contained code.
- Thinking tests for GLM ran at **`reasoning_effort=high`** with the checkpoint's sampling (temperature 1.0, top_p 0.95, no top_k),
  seeded, same 16K-token budget as Qwen (0.6 / 0.95 / 20) and MiMo (1.0 / 0.95). Not at `max`, the default pi would get: there wasn't
  time for both (LiveCodeBench at `high` already took 2 h 40 min).
- `high` thinks only as much as a problem seems to need. On HumanEval+ the mean completion was **485 tokens** (564 characters of
  reasoning), against Qwen's 1,877 and MiMo's 833. On LiveCodeBench it reasoned at length (mean 8,380 tokens, 25K characters of
  reasoning), and **33/80 responses hit the 16K cap** (Qwen 45/80, MiMo 39/80). All 33 capped responses had no code and count as fails,
  so LiveCodeBench here means "solves it within 16K tokens".
- Long context is greedy at effort low with `max_tokens` 512 (Qwen/MiMo: thinking off, 96). Same prompts; mean 94,196 GLM tokens
  (MiMo 95K, Qwen 102K, by tokenizer). 3 misses, all two-hop.

Corpus NLL (`quality_eval.py bpb`, same fixed corpus of Python stdlib source + license text): **GLM E192 0.196 bits/byte**,
MiMo 0.154, Qwen C 0.048. As in QWEN_VS_MIMO.md, this corpus is mostly famous code and measures memorisation as much as
modelling. Treat it as a side note. It's also the only test here where GLM is clearly last. Whether that comes from the expert pruning is untested
(E224's number, below, is the closest thing to a check).

---

## E224 (224 experts kept, 158 GB)

Same scripts and serving config (`GLM_VARIANT=E224`), same NVFP4 draft file. Run after E192, with less time, so it got the speed suite
without the wide grid, plus HumanEval+, MBPP+, corpus NLL and the long-context test.

Memory: **74.05 GiB of weights + draft per node** (E192: 65.18), same 20 GiB KV pool (2,111,321 tokens). OS-available after boot:
**11 GiB on the head, 13 GiB on the worker**, more than Qwen leaves. So E224 fits comfortably at 262K context with this KV budget.
Boot to ready: 600 s.

Speed (`results/glm/e224-*.json`, `results/glm/bench-e224-tp2.*`): **the same as E192**, as expected with the same 18B active
parameters per token.

| ctx | task | effort low: E192 / **E224** tok/s | effort max: E192 / **E224** tok/s |
|---|---|---|---|
| 1K | code | 29.3 / **29.1** | 26.9 / **28.1** |
| 1K | code_ts | 31.5 / **31.0** | 26.7 / **26.8** |
| 1K | edit | 19.7 / **31.6** | 29.4 / **29.7** |
| 128K | code | 20.4 / **29.5** | 27.8 / **27.1** |
| 128K | code_ts | 31.1 / **31.4** | 26.6 / **26.6** |
| 128K | edit | 28.6 / **27.1** | 28.9 / **29.0** |

E224 ran at 11.4–11.7 steps/s in every cell, and **never showed the ~7.5 steps/s slow mode** (all 36 runs between 25.8 and 31.8 tok/s).
The two E192 cells that differ are exactly the ones where E192 dropped into it. Recipe bench C1–C6 aggregate: 26.6 / 43.2 / 59.5 / 70.6 /
70.7 / 86.7 tok/s (E192: 21.0 / 40.9 / 55.3 / 73.1 / 69.8 / 88.0). The C1 gap may be the same slow mode in E192's C1 pass (not checked).
Cold prefill 1,479 / 1,860 / 1,950 / 1,987 tok/s at 2K / 8K / 32K / 64K, the same as E192.

Quality (`results/quality/G224-low`, `G224-deep`; same conditions as E192's thinking-off rows):

| Test | n | Qwen C | E192 | **E224** | E192-only / E224-only (p) | Qwen-only / E224-only (p) |
|---|---|---|---|---|---|---|
| HumanEval+ (base / plus), greedy, effort low | 164 | 95.1 / 93.3 | 95.1 / 92.1 | **95.1 / 92.1** | 7 / 7 (p=1.0) | 6 / 4 (p=0.75) |
| MBPP+ (base / plus), greedy, effort low | 378 | 93.1 / 79.9 | 94.7 / 79.9 | **94.4 / 82.0** | 11 / 19 (p=0.20) | 14 / 22 (p=0.24) |
| Long context 64K + 128K (retrieve / two-hop / reverse), effort low | 144 | 95.1% | 97.9% | **100%** (two-hop 48/48) | 0 / 3 (p=0.25) | 0 / 7 (**p=0.016**) |
| Corpus NLL, bits/byte (side note) | | 0.048 | 0.196 | **0.141** | top-1 agreement 87.6% | |

E224 vs MiMo: MBPP+ 39 vs 10 flips, long context 14 vs 0 (both p<0.001). Truncated at the 4096 cap: 0/164 and 5/378.

**The extra 32 experts show up in the corpus likelihood, not in the task scores.** E224 is 0.055 bits/byte better than E192 on the
fixed corpus (−0.165 nats/token on the same tokens), and 12% of top-1 next-token predictions change between them. That's ~50× the
FP8-dense vs BF16 difference in FP8DENSE.md (+0.003 nats/token). But HumanEval+ is identical and MBPP+ is +2.1 points, not significant. So pruning to 192 experts costs measurable
"knowledge" (memorised text) and nothing we could detect on code tasks. Not run for E224 (time): GSM8K, thinking-mode HumanEval+ and
LiveCodeBench.

---

## How to run it again

```bash
GLM_VARIANT=E192 glm/setup_nodes.sh   # idempotent; downloads (139 GB) and rsyncs to the worker if missing
GLM_VARIANT=E192 glm/up.sh            # stops Qwen, ~10 min to ready; API http://<head>:8000/v1, model glm53-flash-e192
glm/down.sh --qwen                    # back to Qwen config C (~11 min)
```

Knobs on `glm/up.sh`: `SPEC_TOKENS` (0 = no MTP), `MAX_LEN`, `MAX_SEQS`, `KV_BYTES`, `GMU`, `CUDAGRAPH`.
For pi: set `reasoning_effort` explicitly (`low` for speed) or every turn runs at max effort.

## Caveats and not measured

- One run per test for GLM (no noise floor for GLM itself), same as MiMo. Two runs of the same Qwen config differed by 1–2 points.
- Thinking-mode tests at `reasoning_effort=high`, not `max` (pi's default). `max` would think longer: likely slower per answer,
  and on LiveCodeBench more answers would hit the 16K cap, so the score could go either way. Unmeasured.
- Different sampling across models in thinking mode (each model's recommended settings), as in QWEN_VS_MIMO.md.
- The intermittent ~7.5 steps/s decode mode is unexplained (see Speed).
- Not tuned: no K=3 test (the E224 runbook measured K=3 ≈ K=2 at temperature 1.0), no `--performance-mode interactivity`, no
  `max-num-seqs` above 8, torch.compile off throughout. Each of these might move speed. None was tried.
- Not exercised: vision input, the 1M-token window (we served 262K), real pi agent sessions, tool calling beyond one smoke test.
- The checkpoint is an unofficial derivative (autotrust's NAS pruning of zai-org's model). The only other two-Spark E192 numbers we know
  of are in the repo's `MTP-NVFP4-ISSUES.md` (a user report: 66.11 GiB per node, 23.4 tok/s single-stream with E224's draft at spec=3,
  HumanEval 95.7% with a different harness). Our 27–31 tok/s at K=2 is in the same range.
