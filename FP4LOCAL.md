# nvidia Qwen3.8-Flash-Next-FP4-Local on the dual DGX Spark: speed and quality vs our quants

**Question (2026-10-16):** nvidia published a smaller quant of the same model,
[`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (revision `b9ce18a`,
uploaded 2026-10-15). Is it as smart as our config C (RadixArk NVFP4 + FP8-dense, [FP8DENSE.md](FP8DENSE.md)), and is it faster?

What it is: the same Qwen3.8-Flash-Next, but **quantization-aware distilled** (QAD) instead of post-training rounded. NVFP4
routed experts, MXFP8 attention and shared experts, NVFP4 MTP experts, about 96 GiB on disk (our C: 123 GiB). The NVFP4 scale
factors use a new compressed format that upstream vLLM can't read yet, so it needs the community **b12x** vLLM build
(`eugr/spark-vllm-b12x`, pinned here by digest: `nightly-20261015`, vLLM `0.1.dev21830+g8244d1c44`, torch 2.13). nvidia
validated it on **one** Spark (TP=1). We ran it TP=2 across both Sparks so every number compares with the recorded config C runs.

## Summary

> [!WARNING]
> **pi does not run Qwen at the sampling settings this repo benchmarks.** pi sends no `temperature`, so vLLM applies the
> checkpoint's `generation_config.json`: **temperature 1.0 / top_p 0.95 / top_k 20**. Every Qwen thinking-on score in this repo was
> measured at **0.6** / 0.95 / 20 (thinking off: greedy). Re-measured at pi's real settings on 2026-10-19 (FP4-Local, `medium`, 1.0):
> HumanEval+ is unchanged (95.1 vs 95.7), but **LiveCodeBench drops to 37.5% from 43.8%**. All 6 lost problems are hard ones
> (hard: 4/46 vs 9/46) and 49 vs 43 of 80 hit the 16K cap. Paired: vs `medium` at 0.6, 6 lost / 1 gained, p=0.125; vs `xhigh` at 0.6
> p=0.016, vs `low` at 0.6 p=0.021. One run each. **Read the repo's Qwen thinking scores as "at 0.6", not "what pi gets".**
> Fix, *not applied* (it changes live output): serve with `--override-generation-config '{"temperature":0.6}'`, or have the client
> send 0.6. Details: [FP4LOCAL.md](FP4LOCAL.md) "Sampling temperature".


| | config C (prod) | **FP4-Local** (best K per row) | Verdict |
|---|---|---|---|
| HumanEval+ / MBPP+ / GSM8K, thinking off | 93.3 / 79.9 / 96.4 | **94.5 / 79.9 / 96.4** | tie (p ≥ 0.73) |
| HumanEval+ / LiveCodeBench, thinking on | 92.7 / 41.2 | **92.7 / 46.2** | tie / FP4-Local ahead, not significant (p=0.125) |
| Same at pi's `reasoning_effort` `medium` / `low` (2026-10-18; above at `xhigh`, the template default) | — | **95.7 / 43.8**, **94.5 / 47.5** | tie with `xhigh` (every p ≥ 0.125); ~2.7–3× faster on easy problems |
| Long context 64K/128K | 95.1% | **97.2%** | tie (p=0.25) |
| Corpus NLL vs BF16-dense (nats/token) | +0.0030 | **+0.0023** | FP4-Local slightly closer to BF16 |
| Decode, thinking off, 6-cell mean (tok/s) | 67.0 | **68.7** (K=4) | tie (+2%) |
| Decode, **thinking on** (pi's default), 6-cell mean | 52.1 | 49.5 (K=2) / 45.8 (K=4) | config C +5% / +14% |
| Recipe bench aggregate C1 / C4 / C6 | 58.3 / 174.5 / 228.2 | 59.0 / 138.4 / 186.9 | tie at C1, **config C +19–26% at C3–C6** |
| Cold prefill 32K / 64K (tok/s) | 3,031 / 2,989 | 2,691 / 2,681 | config C +11% |
| Warm-hit TTFT on a cached 128K prompt | 1.5–3.5 s | **0.22–0.26 s** | FP4-Local, 6–15× |
| KV pool (TP=2, fp8, 262K) | 4.25M tokens | **5.11–5.49M** tokens | FP4-Local +20–29% |
| Memory free per node after boot | ~4–7 GiB | **14–18 GiB** | FP4-Local |

**Bottom line:** it is **as smart as config C**, possibly a little smarter: it tied or beat config C on every test, and none of
the differences is significant. It is **not faster for how this cluster is used.** Single-stream thinking-off decode is even. With
thinking on it's 5% slower at its best draft length (K=2) and 14% slower with nvidia's K=4. With 3–6 concurrent streams it delivers
~20% less total throughput at every K. Its real wins are elsewhere: a ~25% bigger KV pool, 10+ GiB more free memory per node,
and **zero-recompute warm prefix-cache hits** (config C rebuilds ~2.5K tokens on every hit). **Decision (2026-10-16): FP4-Local is
the live model**, at MTP K=2 (fastest K with thinking on and under concurrency), otherwise nvidia's recipe. The user judged the
bigger KV pool, the memory headroom and the cheap warm hits worth the slower raw speed. It's served as `qwen3.8-flash-next` with no
API key, so pi needed no change. Expert parallelism can't be used with this checkpoint, and an 8192 prefill batch changed nothing
(see "Tuning attempts"). `fp4local/down.sh --qwen` goes back to config C in ~11 min.

---

## Setup

Scripts: `fp4local/` (`lib.sh`, `setup_nodes.sh`, `up.sh`, `down.sh [--qwen]`), same pattern as `glm/` and `mimo/`.

| Step | Notes |
|---|---|
| Download | `hf download` into the head's HF cache at the pinned revision: 102 GB, ~8 min. **`huggingface_hub` 2.x stores file contents in a hub-wide store** (`~/.cache/huggingface/hub/blobs/xx/<sha256>`) behind two levels of relative symlinks; the per-model `blobs/` holds only links. `setup_nodes.sh` rsyncs both trees to the worker (480 MB/s over CX7, 3.5 min). |
| Image | `eugr/spark-vllm-b12x@sha256:03a95654…` on both nodes (9.7 GB). Flags such as `--nnodes/--headless` exist in this build; it needs a GPU attached even to print `--help`. |
| Launch | `fp4local/up.sh`: stops Qwen, refuses if any other container is running, drops the page cache, head then headless worker. Container `vllm_fp4local`, caches in `~/fp4local-cache`. |
| Serving config | **nvidia's DGX Spark recipe from the model card, verbatim** (env and flags), except TP 1 → 2 plus the multi-node and NCCL/CX7 settings config C uses. That means: b12x GEMM/MoE/GDN backends, `--load-format b12x`, `--quantization modelopt_mixed`, MTP **K=4**, PLE table on **disk** (`VLLM_PLE_TABLE_MEMORY=disk`), block size 16, `max-num-batched-tokens` 4096, `max-num-seqs` 8, GMU 0.8, 262K context, fp8 KV, **torch.compile on** (`VLLM_USE_AOT_COMPILE=1` + `fuse_act_quant` pass). |
| Parallel layout | **TP=2 only, no expert parallelism.** Config C runs TP=2 **+ EP** (each GPU holds half the experts whole). Every speed comparison below is therefore also TP-only vs TP+EP; EP isn't available on this build (see "Why no expert parallelism"). |
| torch.compile | AGENTS.md forbids it without asking because a compile-time memory spike once hung the head. **Approved by the user for this test**, on the condition that the Qwen server was stopped first. A watchdog sampled `MemAvailable` on both nodes every 5 s and would have killed the container below 2 GiB. The lowest reading during boot and compile was **15.5 GiB**: weights are 36.3 GiB per node at TP=2, so the old overcommit risk wasn't close. |
| Lock-out | The head served with `--api-key` (random key, only in the untracked `cluster.env` and a mode-600 file on the node). Unauthenticated requests got 401, **including `/metrics`** in this build. All harness scripts send `Authorization: Bearer $API_KEY` when `API_KEY` is set (`scripts/bench_decode.py`, `quality/*.py`, both runners). `num_requests_running` was 0 before and after every speed run. |

Boot: 697 s cold (weights 107–162 s, then b12x autotuning of its kernels for ~6 min; cached in `~/fp4local-cache/b12x` afterwards).
KV pool **5,108,909 tokens** (19.5× a full 262K request; config C: 4.25M). OS-available after boot: **15 GiB head, 18 GiB worker**
(config C: ~4–7 GiB). Functional checks: chat with thinking on and off (reasoning in `reasoning`, code in `content`), and a two-city
request returned two parsed `get_weather` tool calls (`qwen3_xml` parser).

Same tokenizer and chat template as config C: every benchmark prompt has the identical token count (1,033 / 128,033 / …), and output
lengths match (HumanEval+ mean 267 vs 266 tokens), so per-token NLL is directly comparable here.

---

## Speed (TP=2, MTP K=4, nothing else on the server)

`scripts/bench_decode.py`, 500 tokens, `ignore_eos`, T=0.6, median of 3 (same prompts and settings as the config C files).
Config C: `results/mimo/qwen-bench_decode-vsmimo-{thinkoff,default}-2026-10-07.json`. FP4-Local:
`results/fp4local/fp4local-bench_decode-{thinkoff,default}-2026-10-16.json`. st/s = engine steps/s (tok/s ÷ acceptance length).

Thinking off:

| ctx | task | config C tok/s (acc.len, st/s) | **FP4-Local K=4 tok/s (acc.len, st/s)** | ratio |
|---|---|---|---|---|
| 1K | code | 66.3 (2.55, 26.0) | **69.0** (3.39, 20.3) | 1.04× |
| 1K | code_ts | 68.0 (2.62, 26.0) | **72.0** (3.51, 20.5) | 1.06× |
| 1K | edit | 70.1 (2.73, 25.7) | **77.4** (3.96, 19.6) | 1.10× |
| 128K | code | 65.2 (2.56, 25.5) | **61.4** (3.14, 19.6) | 0.94× |
| 128K | code_ts | 65.1 (2.55, 25.5) | **63.5** (3.22, 19.7) | 0.98× |
| 128K | edit | 67.6 (2.72, 24.9) | **68.5** (3.51, 19.5) | 1.01× |

Template default (**thinking on**, which is how pi talks to Qwen):

| ctx | task | config C tok/s (acc.len, st/s) | **FP4-Local K=4 tok/s (acc.len, st/s)** | ratio |
|---|---|---|---|---|
| 1K | code | 56.4 (2.19, 25.8) | **47.7** (2.39, 20.0) | 0.85× |
| 1K | code_ts | 49.2 (1.91, 25.7) | **44.3** (2.23, 19.9) | 0.90× |
| 1K | edit | 54.8 (2.13, 25.8) | **49.7** (2.51, 19.8) | 0.91× |
| 128K | code | 51.3 (2.04, 25.1) | **42.9** (2.19, 19.6) | 0.84× |
| 128K | code_ts | 48.2 (1.89, 25.4) | **41.5** (2.13, 19.5) | 0.86× |
| 128K | edit | 52.5 (2.13, 24.7) | **48.8** (2.51, 19.4) | 0.93× |

**Why:** FP4-Local runs ~20 engine steps/s against config C's ~25.5 (both flat from 1K to 128K), because each step drafts 4
tokens instead of 2. On code with thinking off, the draft accepts 3.1–4.0 tokens per step and more than makes up for it. On
reasoning text and prose, acceptance falls to 2.1–2.5 and the longer draft is mostly wasted. K is a pure speed knob
(speculative decoding is verified by the target model), so the K=2 re-run below tests this directly.

Wider single-run grid (thinking off, T=0 and 0.6, 1K and 64K; `results/fp4local/fp4local-bench_decode-wide-thinkoff-2026-10-16.json`
vs `results/mimo/qwen-bench_decode-wide-thinkoff-2026-10-07.json`). Config C / FP4-Local tok/s:

| ctx | T | code | code_ts | edit | prose |
|---|---|---|---|---|---|
| 1K | 0.0 | 71.1 / **77.3** | 68.7 / **68.9** | 72.5 / **77.5** | 56.9 / **45.7** |
| 1K | 0.6 | 67.8 / **75.2** | 67.3 / **66.9** | 70.8 / **76.2** | 53.6 / **45.8** |
| 64K | 0.0 | 66.3 / **69.1** | 69.3 / **67.3** | 70.4 / **76.6** | 52.6 / **43.7** |
| 64K | 0.6 | 67.0 / **62.7** | 61.3 / **71.2** | 69.8 / **77.5** | 52.1 / **46.7** |

Code and edits: roughly even to +16%. Prose: −10 to −20%.

**Warm prefix-cache hits are much cheaper.** In the same runs, time-to-first-token on a repeated (cached) 128K prompt was
**0.22–0.26 s** against config C's **1.5–3.5 s** (TTFT columns of the same files). PI.md traced pi's per-turn pause to exactly
this warm-hit recompute tail (~2.5K tokens on config C). Here first-token latency on a warm hit is 6–15× lower; the next section
measures why.

### Why warm hits are free here: prefix-match granularity, not block size

*(Corrected 2026-10-16. An earlier version said FP4-Local uses 16-token cache blocks; it doesn't.)*

The model mixes attention layers (KV per token) with Gated DeltaNet layers (one fixed-size recurrent state per sequence). vLLM
pads the attention page up to the recurrent-state page, so both deployments run **big physical blocks**:

| | config C | FP4-Local |
|---|---|---|
| Physical block (boot log: "Setting attention block size to … tokens to ensure that attention page size is >= mamba page size") | **1,648** tokens | **2,832** tokens (the recipe's `--block-size 16` is overridden for the pages) |
| Prefix-cache hit granularity (`prefix_match_unit` / `hash_block_size`) | 1,648 (no block size or match unit set, so it defaults to the block) | fine (the recipe's `--block-size 16` keeps it small) |
| Where a warm hit can resume | last full block, minus one more block for MTP lookahead | the exact prompt end |

Read from the source of both images (`vllm/v1/core/kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`,
`config/cache.py`). Both builds have the same fine-grained mode: `resolve_kv_cache_block_sizes` sets the hash granularity to
`--prefix-match-unit` if given, else the GCD of the cache groups' block sizes, and `enable_partial_hash_hits` turns on only when that
is smaller than the physical block (Mamba `align` mode, no context parallelism). Under it, vLLM publishes partial-block cache entries
and a saved recurrent state at each prompt's end (`_cache_partial_tail_block`), so a repeat resumes at the end of the prompt.
Without it, hits floor to whole blocks, and under MTP/EAGLE the lookup backs off one more alignment unit
(`reachable_hit_positions`).

That reproduces config C's measured tails exactly (remainder + one 1,648-token block): 7,939 = 4 blocks + 1,347 → **2,995**;
23,959 = 14 + 887 → **2,535**; 96,559 = 58 + 975 → **2,623**. FP4-Local recomputed **0** at all three sizes. FP4-Local's resolved
hash granularity isn't printed in its log, so "fine" above is inferred from the flag, the code and the measured zero.

**One build-level asymmetry behind that configuration** (checked line by line, 2026-10-16; config C image
`vllm 0.1.dev20073+g8e685d198`, this image `0.1.dev21830+g8244d1c44`). Both builds have the fine-grained mode, but they
*enable* it differently in `v1/core/kv_cache_coordinator.py`:

- old: `enable_partial_hash_hits = dcp_world_size == 1 and has_partial_mamba_group`, and that predicate requires a
  **Mamba** group with `block_size > hash_block_size`.
- this build: `allow_partial_hash_hits and (has_partial_mamba_group or has_partial_attention_group)` — the new route
  accepts a `FullAttentionManager`/`SlidingWindowManager` with `block_size > hash_block_size and
  supports_fine_grained_hash_lookup`. It also reads `num_prefill_checkpoint_blocks > 1` and splits
  `lookup_alignment_tokens` from `state_materialization_alignment_tokens`, keeping both the fine and the coarse
  recurrent states so a missing fine state falls back instead of missing.

That matters for the implication below: the recipe passes no `--prefix-match-unit`, so the fine hash granularity comes
from the small requested block size while the attention pages are grown to 2,832 — a shape the **attention** route
accepts and the mamba-only route may not (if `--block-size 16` applies to the mamba group, then `mamba.block_size ==
hash_block_size` and the old predicate stays false). So config C should get free hits with **`--prefix-match-unit` set
explicitly**; `--block-size 16` alone is the weaker bet on the old image. Neither has been tried.

**Implication (untested):** this is configuration, not something only the b12x build can do. Config C's image has the same
`--prefix-match-unit` flag and partial-hit code, so config C would likely get zero-recompute warm hits with `--prefix-match-unit 16`
(or a small `--block-size`). That has not been tried.

Measured with the same probe and prompts as config C (`scripts/prefix_probe.py --steps cold,warm`, server's own
`prompt_tokens_by_source` counters; `results/fp4local/fp4local-k2-prefix-probe-tail-{8000,24000,96000}-2026-10-16.json` vs
`results/prefix-probe-tail-*-2026-10-15.json`):

| prompt | config C recomputed on a warm hit | **FP4-Local recomputed** | FP4-Local warm request, end to end |
|---|---|---|---|
| 7,939 | 2,995 | **0** (100% cached) | 0.21 s |
| 23,959 | 2,535 | **0** | 0.24 s |
| 96,559 | 2,623 (9,215 on the first repeat) | **0** | 0.35 s |

"0" is the server counter; vLLM may still recompute the final token internally to sample, but nothing measurable beyond that.
**Scope of that zero** (measured later the same day, see `PI.md`): it applies to a *byte-identical* repeat, which is what the
published partial tail entry serves. A prompt that shares a long prefix but is not identical floors to a 4,096-token boundary -
recompute = `(prompt mod 4096) + 4096`, exact at 30K (5,386) and 120K (6,274). So "resume at the exact prompt end" is true for
repeats, not for appends.
For pi, which re-sends the whole history every turn, this removes a ~0.8 s fixed cost per turn on config C. Cold misses are
unaffected (and ~10% slower here, see cold prefill above).

### Recipe bench (MiMo recipe `bench/mimobench.py`, temperature 0, thinking off, unique prefixes; run on the head)

Same script as config C's `results/mimo/bench-qwen-C-tp2.*`, with only an `Authorization` header added (local copy).
Files: `results/fp4local/bench-fp4local-tp2.{md,json}`.

| C | config C aggregate / per-stream | **FP4-Local K=4 aggregate / per-stream** |
|---|---|---|
| 1 | 58.3 / 66.1 | **58.8 / 65.9** |
| 2 | 96.1 / 55.3 | **90.1 / 53.2** |
| 3 | 137.6 / 53.6 | **107.7 / 42.9** |
| 4 | 174.5 / 49.1 | **135.7 / 40.7** |
| 5 | 197.7 / 45.7 | **162.4 / 38.8** |
| 6 | 228.2 / 44.0 | **161.8 / 32.1** |

Single-stream is a tie. **Under concurrency FP4-Local falls behind (−29% aggregate at C6):** with 4 drafts per stream the batch
spends more compute on rejected tokens, and this config has no expert parallelism (config C runs TP=2 + EP). C1 per-stream by
category: coding 77.3, json 87.9, math 83.8, format 82.6, structured 67.7, reasoning 65.8, summary 46.7, prose 41.4, narrative 40.1.

Cold prefill (one request, unique prefix):

| prompt tokens | config C tok/s | **FP4-Local tok/s (TTFT)** |
|---|---|---|
| ~2K | 1,109 | **2,602** (0.77 s) |
| ~8K | 2,344 | **2,725** (2.89 s) |
| ~32K | 3,031 | **2,691** (11.7 s) |
| ~64K | 2,989 | **2,681** (23.6 s) |

Flat at ~2,700 tok/s: faster than config C below 8K, ~10% slower at agent-sized prompts. The cold-miss case from PI.md (182K
tokens) would take ~68 s instead of ~50 s. That is an extrapolation, not measured.

### Draft length (K) sweep: K=4 is nvidia's choice, K=2 is ours

Speculative decoding is lossless (the target verifies every token), so K changes speed only and every quality number above
holds for all three. Same benchmarks, warm boots (281 s; the b12x kernel tuning is cached). Files:
`results/fp4local/fp4local{,-k3,-k2}-bench_decode-{thinkoff,default}-2026-10-16.json`, `bench-fp4local{,-k3,-k2}-tp2.*`.
Cells: tok/s (acceptance length, engine steps/s).

| ctx | task | config C | FP4-Local K=4 | K=3 | K=2 |
|---|---|---|---|---|---|
| | *thinking off* | | | | |
| 1K | code | 66.3 | 69.0 (3.39, 20.3) | 66.2 (3.01, 22.0) | 64.0 (2.60, 24.6) |
| 1K | code_ts | 68.0 | 72.0 (3.51, 20.5) | 69.9 (3.18, 22.0) | 64.0 (2.59, 24.8) |
| 1K | edit | 70.1 | 77.4 (3.96, 19.6) | 73.1 (3.42, 21.4) | 66.8 (2.73, 24.5) |
| 128K | code | 65.2 | 61.4 (3.14, 19.6) | 66.0 (3.08, 21.4) | 61.7 (2.56, 24.1) |
| 128K | code_ts | 65.1 | 63.5 (3.22, 19.7) | 64.8 (2.99, 21.7) | 62.9 (2.61, 24.1) |
| 128K | edit | 67.6 | 68.5 (3.51, 19.5) | 62.0 (2.92, 21.3) | 60.4 (2.56, 23.6) |
| | **mean** | **67.0** | **68.7** | **67.0** | **63.3** |
| | *thinking on (template default)* | | | | |
| 1K | code | 56.4 | 47.7 (2.39, 20.0) | 43.8 (2.00, 21.9) | 50.0 (2.05, 24.3) |
| 1K | code_ts | 49.2 | 44.3 (2.23, 19.9) | 47.0 (2.16, 21.7) | 46.5 (1.90, 24.4) |
| 1K | edit | 54.8 | 49.7 (2.51, 19.8) | 54.3 (2.47, 22.0) | 57.4 (2.33, 24.6) |
| 128K | code | 51.3 | 42.9 (2.19, 19.6) | 48.3 (2.27, 21.3) | 48.5 (2.03, 23.9) |
| 128K | code_ts | 48.2 | 41.5 (2.13, 19.5) | 43.4 (2.05, 21.2) | 43.6 (1.82, 23.9) |
| 128K | edit | 52.5 | 48.8 (2.51, 19.4) | 52.0 (2.45, 21.3) | 51.4 (2.18, 23.6) |
| | **mean** | **52.1** | **45.8** | **48.1** | **49.5** |

Recipe bench aggregate (thinking off), tok/s:

| C | config C | K=4 | K=3 | K=2 |
|---|---|---|---|---|
| 1 | 58.3 | 58.8 | 59.0 | 56.5 |
| 2 | 96.1 | 90.1 | 89.5 | 86.9 |
| 3 | 137.6 | 107.7 | 106.2 | 115.5 |
| 4 | 174.5 | 135.7 | 138.4 | 137.9 |
| 5 | 197.7 | 162.4 | 154.5 | 155.7 |
| 6 | 228.2 | 161.8 | 175.2 | 186.9 |

Reading it: each extra draft token costs FP4-Local ~2 steps/s (24.2 → 21.6 → 19.8 st/s for K=2/3/4). At K=2 its step rate is within
5% of config C's (25.5), so per-step the two models cost about the same. With thinking off, code accepts enough drafts that K=4 wins;
on reasoning text it doesn't, and K=2 is best. **Under concurrency no K closes the gap**: C3–C6 are 16–29% behind config C at
every K. So it isn't the draft length (see "Tuning attempts" for EP and the prefill batch). Cold prefill at K=2: 2,517 / 2,723 / 2,728 / 2,672 tok/s, the same as K=4.
KV pool: 5.11M (K=4), 5.31M (K=3), 5.49M (K=2) tokens.

### Tuning attempts (K=2, both lossless)

| Change | Result |
|---|---|
| `--enable-expert-parallel` (config C uses EP) | **Not possible on this build.** Both ranks fail at startup with `NotImplementedError: NVFP4-CSF supports TP without EP/DP`. It's a software limitation in the b12x vLLM build, not the scale format; see "Why no expert parallelism" below. |
| `max-num-batched-tokens` 4096 → 8192 (config C's value) | No change worth having: recipe bench C1/C3/C6 56.8 / 117.8 / 182.9 (4096: 56.5 / 115.5 / 186.9), thinking-on grid within ±5%, cold prefill +2–3% (2,594 / 2,813 / 2,787 / 2,729 tok/s). It costs 6% of the KV pool (5.18M vs 5.49M tokens). Kept 4096. Files: `results/fp4local/*k2-mb8k*`. |

### Why no expert parallelism (it's the software, not the format)

Read from the source in the pinned image (`eugr/spark-vllm-b12x@sha256:03a95654…`, vLLM `0.1.dev21830+g8244d1c44`), 2026-10-16.
Two separate restrictions, neither inherent to the checkpoint:

1. **The b12x MoE backend has no EP path at all**, compressed scales or not. In
   `vllm/model_executor/layers/fused_moe/b12x.py`, `_supports_parallel_config()` is false whenever `use_ep`, `ep_size != 1`,
   all2all kernels or EPLB are on, and `supports_expert_map()` returns false. The hybrid A4 prefill path
   (`B12X_W4A16_A4_PREFILL`, set by nvidia's recipe) separately refuses DP/PCP/EP/sequence parallelism. So a plain
   ModelOpt NVFP4 checkpoint run through these kernels couldn't use EP either: the all-to-all token dispatch EP needs isn't
   wired into this backend yet.
2. **The compressed-scale MoE method assumes every rank holds every expert.**
   `vllm/model_executor/layers/quantization/nvfp4_csf.py` (`Nvfp4CsfMoEMethod`) raises the error above. It is built on
   the no-DP/EP prepare/finalize path (`MoEPrepareAndFinalizeNoDPEPModular`), decodes a layer's scale planes for **all experts**
   into one model-owned E4M3 scratch shared across layers, and prefetches the next layer's expansion on a side CUDA stream
   while attention runs (`_prefetches_all_experts`: "prefetch matches the call's full-expert expansion workload"). It also
   requires pipeline parallel 1 and no ubatching ("CSF shared scratch requires PP1 without ubatching"). Under EP each rank
   would own a subset of experts, so this scratch/prefetch design would need rework.

What the format itself allows: the scales are **lossless** and decode to ordinary E4M3 NVFP4 block scales at run time
(`b12x.moe.fused_moe.expand_scales`). They are stored per expert matrix, and the loader already slices them per TP rank on
32-channel boundaries (`nvfp4_csf_utils.prepare_nvfp4_csf_weights`, `tp_extent(..., 32)`). Splitting by expert instead is no
harder. Nothing in the encoding ties it to TP-only serving.

Routes to EP, if it's ever worth it:
1. **A future b12x / vLLM build adds EP.** Retest with `EP=1 SPEC_TOKENS=2 fp4local/up.sh`; it fails within ~2 minutes if still unsupported.
2. **Expand the scales offline** into a standard ModelOpt NVFP4 checkpoint and serve it with upstream kernels that support EP
   (config C runs FlashInfer/CUTLASS NVFP4 with EP on this cluster). No conversion tool ships in this image (not found). It would
   also need upstream support for the rest of the checkpoint (MXFP8 attention, the `qwen4_exp` model), which the model card says
   hasn't landed. It would give back part of the size saving (expanded scales are 1 byte per 16 weights) and possibly the b12x
   kernels' own speedups.

Unmeasured: whether EP would actually close the ~20% gap at 3–6 streams. It's the main config difference from config C, but
the gap's cause isn't proven.

---

## Quality (same harness, data and sandbox as configs A/B/C; run on the worker)

`quality/run_quality.sh` and `quality/run_deep.sh` with their defaults (thinking off greedy; thinking on at 0.6 / 0.95 / 20,
seeded, 16K budget), exactly as the A/B/C runs. FP4-Local: `results/quality/F4-fp4local`, `results/quality/F4-deep`. One run.

| Test | n | A nvidia NVFP4 | B RadixArk BF16-dense | **C RadixArk FP8-dense (prod)** | **FP4-Local** | C-only / F4-only (p) |
|---|---|---|---|---|---|---|
| HumanEval+ base / **plus**, thinking off | 164 | 97.0 / 95.1 | 95.7 / 93.9 | 95.1 / **93.3** | 97.6 / **94.5** | 3 / 5 (p=0.73) |
| MBPP+ base / **plus**, thinking off | 378 | 93.9 / 79.9 | 92.3 / 79.4 | 93.1 / **79.9** | 93.4 / **79.9** | 5 / 5 (p=1.0) |
| GSM8K, thinking off | 1319 | 96.5 | 96.4 | **96.4** | **96.4** | 14 / 15 (p=1.0) |
| HumanEval+ base / **plus**, thinking on, 16K budget | 164 | — | 95.1 / 92.7 | 95.1 / **92.7** | 95.7 / **92.7** | 4 / 4 (p=1.0) |
| **LiveCodeBench** (80 problems), thinking on, 16K budget | 80 | — | 43.8 | **41.2** | **46.2** | 0 / 4 (p=0.125) |
| Long context 64K + 128K (retrieve / two-hop / reverse) | 144 | — | 96.5% (two-hop 43/48) | **95.1%** (41/48) | **97.2%** (44/48) | 0 / 3 (p=0.25) |
| Corpus NLL, nats/token (190,191 identical tokens) | | 0.1315 | **0.1310** | 0.1340 | **0.1333** | |

FP4-Local vs A, A2 (A's rerun) and B on HumanEval+/MBPP+/GSM8K: 1–8 flips each way, every p ≥ 0.75. For scale, the two runs of A
(same config, run twice) differed by 2–3 flips each way on HumanEval+ and MBPP+.

NLL detail (`quality_eval.py compare-nll`, reference B): A +0.0005, A2 +0.0006, C +0.0030, **FP4-Local +0.0023** nats/token; top-1
agreement with B 98.5% / 98.5% / 98.6% / **98.2%**. So on average likelihood FP4-Local sits between nvidia's old quant and our
config C. It disagrees with B on slightly more individual tokens than any rounding quant does, which is what distillation (weights
moved, not just rounded) would look like. None of it shows up in task scores.

Truncated at the cap (thinking off): HumanEval+ 2, MBPP+ 1, GSM8K 22 (config C: 1, 1, 19).
Thinking-mode runs: mean completion 1,825 tokens on HumanEval+ (C 1,877, B 1,942), capped 6/164 (C 7, B 8). LiveCodeBench: mean
10,443 tokens, **40/80 at the 16K cap** (C 45, B 43), so as before this measures "solves it within 16K tokens of thinking".
Long context: mean prompt 102,393 tokens (same prompts and tokenizer as B/C).

---

## Reasoning effort: `low` / `medium` / `xhigh`, and thinking off (2026-10-18)

**Question:** every thinking-on number above was taken without sending `reasoning_effort`. What does the template do with it,
what does pi send, and which level should agents use?

The chat template (thinking on only) accepts `chat_template_kwargs.reasoning_effort` = `low`, `medium` or `xhigh`; anything else is
an HTTP 400 (`high` included). **Omitted = `xhigh`** (`reasoning_effort|default('xhigh')`). The level only changes one sentence at the
top of the system prompt: `xhigh` adds "Reasoning effort is set to xhigh. Please think carefully ... prioritize correctness",
`low` adds "Reasoning effort is set to low. Keep your thinking brief and focused", and **`medium` adds nothing**. So every Qwen
"thinking on" number in this repo (quality, time to solve, `bench_decode.py --thinking on/default`) is at **`xhigh`**, while
**pi sends `medium`** by default (seen in its request bodies, [PI.md](PI.md)). Changing level mid-session makes the next request
re-prefill the whole history (PI.md).

Same harness as the quality table above (thinking on at 0.6 / 0.95 / 20, seeded per task, 16K budget), K=2, the server locked with
`API_KEY`, nothing else on it. One run per level. Full sets at concurrency 8 (`results/quality/F4-deep-{medium,low}`; `xhigh` is
the existing `F4-deep`); time to solve one problem at a time on every 4th HumanEval+ and every 8th LiveCodeBench problem
(`results/quality/TTS-fp4local-{medium,low,off}`; `xhigh` is the existing `TTS-fp4local`, same subsets, seeds and K).
p is paired McNemar against `xhigh` on the same problems.

| Qwen FP4-Local (K=2) | HumanEval+ base / **plus** (164) | **LiveCodeBench** (80) | mean tokens HE+ / LCB | HE+ / LCB at 16K cap |
|---|---|---|---|---|
| `xhigh` (omitted; all earlier runs) | 95.7 / **92.7** | **46.2** | 1,826 / 10,443 | 6 / 40 |
| `medium` (pi's default) | 98.8 / **95.7** (1 / 6, p=0.125) | **43.8** (4 / 2, p=0.69) | 1,004 / 10,593 | 0 / 43 |
| `low` | 98.8 / **94.5** (2 / 5, p=0.45) | **47.5** (2 / 3, p=1.0) | 840 / 10,196 | 0 / 40 |

`medium` vs `low`: HE+ 4 / 2 (p=0.69), LCB 1 / 4 (p=0.38). **No level is significantly better than another on quality.** The noise
floor (A vs A2) is 2–3 flips each way.

| Time to solve, one at a time | HE+ subset (41): solved | s / problem | s / solved | mean tokens | LCB subset (10): solved | s / problem | mean tokens | capped |
|---|---|---|---|---|---|---|---|---|
| `xhigh` (2026-10-17) | 37 | 40.4 | 44.8 | 2,168 | 1 | 292 | 15,313 | 9 |
| `medium` | 36 | 14.7 | 16.7 | 945 | 1 | 301.5 | 15,726 | 9 |
| `low` | **39** | **13.4** | **14.1** | 854 | 1 | 291 | 15,340 | 9 |
| thinking off (greedy) | 38 | **4.8** | **5.2** | 304 | **3** | **173** | 10,508 | 1 |

Reading it:
- **`xhigh` buys nothing here and costs ~3× the time on easy problems.** `low` and `medium` write about half the tokens on
  HumanEval+, never hit the cap (`xhigh`: 6 of 164, most of its misses), and score the same or a little higher.
- **On hard problems the level doesn't change how long Qwen thinks.** All three levels run ~40 of 80 LiveCodeBench problems to the
  16K cap and average 10.2–10.6K tokens. The sentence shortens easy answers, not hard ones.
- **Thinking off is the fastest way to a right answer on easy problems** (4.8 s, 38/41; the full 164 thinking-off greedy run scores
  94.5 too, table above).
- **Thinking off on hard problems doesn't mean a short answer.** With thinking off Qwen reasons *in the visible answer*: 10.5K tokens
  per LCB problem on average, but it stopped by itself on 9 of 10 and solved 3 of 10 to 1 of 10 at every thinking level. n=10 and
  greedy vs sampled: a hint, not a result. The full 80-problem LiveCodeBench was never run with thinking off.
- End-to-end streaming rate (prefill included) at K=2: ~64 tok/s on short answers at every level, ~52 tok/s on 16K-token runs.

**How to use it:** keep pi at `medium` (its default) or use `low` (both measured at temperature 0.6; pi gets 1.0, see the next section); don't select `xhigh`. Use thinking off for mechanical edits and
small functions, where it is ~3× faster again at the same pass rate. Pick one level per session, because switching costs a full
re-prefill.

---

## Sampling temperature: what pi actually gets (2026-10-19)

**Question:** pi sends no `temperature`. What does the server use, and does it matter?

vLLM's default `--generation-config auto` loads the checkpoint's `generation_config.json`, and the server log says so on boot:
"Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 1.0, 'top_k': 20,
'top_p': 0.95}`". So **pi samples at 1.0 / 0.95 / 20**, thinking on or off. The harness has always used 0.6 / 0.95 / 20 for thinking
(the older Qwen recommendation) and greedy for thinking off. nvidia's model card gives no sampling recommendation (revision `b9ce18a`).

Re-run of the `medium` row above with only the temperature changed: 1.0 / 0.95 / 20, same prompts, per-task seeds, K=2, lock-out,
budget (`results/quality/F4-deep-medium-t1`, `results/quality/TTS-fp4local-medium-t1`). One run.

| Qwen FP4-Local, `medium` | HumanEval+ base / **plus** | **LiveCodeBench** (80) | LCB easy / medium / hard | mean tokens HE+ / LCB | LCB at 16K cap |
|---|---|---|---|---|---|
| T=0.6 (the harness) | 98.8 / **95.7** | **43.8** | 16/16, 10/18, **9/46** | 1,004 / 10,593 | 43 |
| **T=1.0 (what pi gets)** | 98.2 / **95.1** (1 / 0, p=1.0) | **37.5** (6 / 1, p=0.125) | 16/16, 10/18, **4/46** | 1,003 / 11,353 | 49 |

T=1.0 on LiveCodeBench against the other 0.6 runs: `xhigh` 7 / 0 (**p=0.016**), `low` 9 / 1 (**p=0.021**), GLM FP4-Local high 15 / 1
(p=0.001); against config C (`xhigh`, 0.6) 4 / 1 (p=0.38). Time to solve at 1.0: HE+ subset 38/41, 15.8 s per problem (0.6: 36/41,
14.7 s), 1,010 tokens; LCB subset 1/10, 332.5 s (0.6: 301.5 s).

Reading it:
- **Easy problems don't care.** HumanEval+ and the easy/medium LiveCodeBench problems are identical at both temperatures.
- **Hard problems do.** Every problem lost at 1.0 is a hard one: 4 of 46 solved against 9. At 1.0 the model wanders more, writes
  ~7% more tokens and runs 6 more problems out of budget. Same level only, the drop is not significant on its own (p=0.125). It is
  significant against both other 0.6 levels, and every 0.6 level beats it. Taken together: likely a real cost, not proven by one run.
- **So the repo's Qwen thinking scores overstate what pi gets on hard problems**, and the "`medium`/`low` tie `xhigh`" result above
  holds at 0.6 only.

**Fix (not applied; it changes live output, so it's the user's call):** add
`--override-generation-config '{"temperature":0.6}'` to the serve command (it is merged into the model's default config;
that a request's own `temperature` still wins is vLLM's usual default handling, not tested here), or have the client send 0.6. Untested here: whether thinking-off quality changes at 1.0
(all thinking-off numbers are greedy), and whether 0.6 is the best value rather than just better than 1.0.

---

## How to run it again

```bash
fp4local/setup_nodes.sh     # idempotent: 102 GB download + rsync to the worker, image pull
LIVE=1 SPEC_TOKENS=2 fp4local/up.sh   # the live config (no key, served as qwen3.8-flash-next too)
fp4local/up.sh              # benchmark mode: API_KEY lock-out; stops Qwen, ~12 min cold (~5 min warm); http://<head>:8000/v1, model qwen3.8-flash-next-fp4local
fp4local/down.sh --qwen     # back to config C (~11 min)
```

Knobs: `SPEC_TOKENS` (default 4, nvidia's), `MAX_SEQS`, `MAX_BATCHED`, `GMU`, `PLE_MEMORY` (`disk`). With `API_KEY` set in
`cluster.env`, `up.sh` locks the server and every client must send it.

## Caveats / not measured

- One run per quality test. The A-vs-A2 rerun is the noise floor: 2–3 flips each way on HumanEval+/MBPP+.
- Effort study (2026-10-18): one run per level; time-to-solve subsets are 41 and 10 problems. The thinking-off time-to-solve run is
  greedy while the thinking runs sample (0.6 / 0.95 / 20), so its LCB 3/10 is not a like-for-like comparison.
- TP=2 only, as asked. nvidia validated TP=1 on a single Spark; that wasn't run, so whether one Spark is enough for this
  workload is unmeasured.
- nvidia's recipe was used as written for everything except TP and K (EP impossible, 8192 batch no gain, above). Not tried: PLE in
  memory (`PLE_MEMORY` knob), `--performance-mode interactivity`.
- Live since 2026-10-16 ~16:30 UTC with no Docker restart policy (same as config C): after a node reboot, run `LIVE=1 SPEC_TOKENS=2
  fp4local/up.sh`. Multi-agent stall behaviour (QWEN.md) under the new server is not yet observed.
- The b12x build is a community nightly, not upstream vLLM. Pinned by digest here; behaviour can change between nightlies.
- Not exercised: vision input, 1M context, real pi agent sessions, multi-agent stall behaviour (QWEN.md).
