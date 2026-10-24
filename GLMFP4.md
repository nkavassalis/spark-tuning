# nvidia GLM-5.3-Flash-FP4-Local on the dual DGX Spark: speed, quality and time-to-solve

**Question (2026-10-17):** nvidia published a QAD quant of the full GLM-5.3-Flash,
[`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) (revision `76348dd`). How fast and how
smart is it on this cluster, against the earlier GLM builds (autotrust's pruned E192/E224, [GLM.md](GLM.md)) and the Qwen models
(config C and the live Qwen FP4-Local, [FP4LOCAL.md](FP4LOCAL.md))? And despite slower tokens, does it **solve problems faster**?

What it is: all 288 routed experts (E192/E224 are NAS-pruned to 192/224), **quantization-aware distilled**: NVFP4 routed experts,
MXFP8 attention, NVFP4 MTP experts, NVFP4/MXFP8 vision tower. 166 GiB on disk; 83.4 GiB of weights per node at TP=2. Same NVFP4-CSF
compressed scale format as Qwen FP4-Local, so the same community **b12x** vLLM build (`eugr/spark-vllm-b12x`, pinned by digest:
`nightly-20261015`, which was also `:latest` on 2026-10-17). nvidia validated it on two Sparks at TP=2, as we run it.

## Summary

| | Qwen C | Qwen FP4-Local (live, K=2) | GLM E224 | **GLM FP4-Local (K=3)** | Verdict |
|---|---|---|---|---|---|
| HumanEval+ / MBPP+ / GSM8K, thinking off (GLM: effort low) | 93.3 / 79.9 / 96.4 | 94.5 / 79.9 / 96.4 | 92.1 / 82.0 / — | **91.5 / 81.7 / 96.5** | tie (every p ≥ 0.21) |
| HumanEval+, thinking (Qwen: effort xhigh, the default; GLM: effort high), 16K budget | 92.7 | 92.7 (`medium` 95.7, `low` 94.5) | — (E192 92.7) | **92.1** | tie |
| **LiveCodeBench**, thinking, 16K budget | 41.2 | 46.2 (`medium` 43.8, `low` 47.5) | — (E192 47.5) | **55.0** | **GLM FP4: best; vs Qwen C p=0.003**, vs FP4-Local `xhigh` p=0.065, `medium` p=0.012, `low` p=0.15 |
| Temperature 0.6 vs 1.0 (2026-10-24; GLM only) | | | | `low` MBPP+ 83.6 / 83.1, `high` LCB **57.5** / 53.8 | **no significant effect** (every p ≥ 0.45; noise ±2–3) |
| Long context 64K/128K | 95.1% | 97.2% | 100% | **99.3%** | GLM ≥ Qwen (vs C p=0.07) |
| Corpus bits/byte (side note) | 0.048 | 0.048 | 0.141 | **0.087** | Qwen; GLM FP4 far better than the pruned builds |
| Decode, 6-cell mean, thinking off / on (GLM: effort low / max), tok/s | 67.0 / 52.1 | 63.3 / 49.5 | 30.0 / 27.9 | **41.0 / 36.8** | Qwen ×1.35–1.6; GLM FP4 +37% / +32% over E224 |
| Recipe bench aggregate C1 / C4 / C6 | 58.3 / 174.5 / 228.2 | 56.5 / 137.9 / 186.9 | 26.6 / 70.6 / 86.7 | **35.3 / 75.3 / 93.7** | Qwen ×1.6–2.4 |
| Cold prefill 32K / 64K (tok/s) | 3,031 / 2,989 | 2,728 / 2,672 | 1,950 / 1,987 | **2,170 / 2,164** | Qwen +25–40% |
| Warm prefix hit (recomputed tokens) | ~2.5K | 0 | — | **0** | |
| KV pool (fp8) | 4.25M | 5.49M | 2.11M | **2.02M** (500K max context) | |
| Memory free per node after boot | ~4–7 GiB | 14–18 GiB | 11–13 GiB | **7–8 GiB** | tight but stable |
| **Time to solve**, HumanEval+ subset (41), one at a time | — | 40.4 s/problem at `xhigh` (37/41); **14.7 at `medium`** (36/41), **13.4 at `low`** (39/41), 4.8 thinking off (38/41) | — | 18.5 s/problem at high (36/41), 99 s at max; **4.2 s at `low`** (35/41, 0.6; 2026-10-24) | **Qwen at `low`/`medium` 1.3–1.4× faster than GLM high**; GLM high beat only Qwen `xhigh` (2.2×) |
| **Time to solve**, LiveCodeBench subset (10), one at a time | — | 292 / 302 / 291 s/problem at `xhigh` / `medium` / `low` (1/10 each); 173 s thinking off (3/10, greedy) | — | 323 s/problem at high (2/10), 438 s at max (1/10) | mostly time to the 16K cap; n=10 can't separate solve counts |

**Bottom line:** GLM FP4-Local is the strongest model we've run here on hard competitive code. LiveCodeBench 55% is the only
significant quality gain over Qwen config C we've measured, and it ties everywhere else. But its tokens are 35–40% slower than the
live Qwen FP4-Local's, it has under half the KV pool and it has less memory headroom.

**On easy problems GLM at `reasoning_effort=high` beat Qwen only at Qwen's own slowest setting.** The 2026-10-17 comparison sent
Qwen no effort, which its template resolves to `xhigh` (found 2026-10-18): GLM high took 18.5 s per problem against Qwen `xhigh`'s 40.4 s.
At **`medium` (what pi actually sends) Qwen takes 14.7 s and at `low` 13.4 s**, so Qwen is the faster one, with the same pass rate
(36–39 of 41; full 164: Qwen `medium` 95.7, `low` 94.5, GLM high 92.1, every p ≥ 0.11). Qwen with thinking off takes 4.8 s
(38/41). **At GLM's template default (effort max) GLM is 99 s per problem.** On hard problems both mostly run out of the 16K budget;
GLM at high solved 2 of 10 to Qwen's 1 at every level ([FP4LOCAL.md](FP4LOCAL.md) "Reasoning effort").

**Removed from both nodes on 2026-10-17** after the tests (`glmfp4/setup_nodes.sh` re-downloads it, 166 GiB); the live model is unchanged (Qwen FP4-Local, K=2).
Reinstalled overnight 2026-10-23/24 for the temperature study ("Sampling temperature" below: no quality effect; `low` timed at
4.2 s per HumanEval+ problem) and removed again.

---

## Which one to run: GLM FP4-Local or Qwen FP4-Local

Based on the measurements above. Real agent sessions (multi-turn tool use, task success in pi) weren't measured for either model.

| Workload | Pick | Why (measured) |
|---|---|---|
| Hard algorithmic / competitive problems, where getting it right matters more than latency | **GLM** (effort high) | LiveCodeBench 55.0% (57.5% at 0.6, 2026-10-24; temperature not significant) vs 46.2% (p=0.065; vs config C 41.2%, p=0.003); hits the 16K cap less often (31 vs 40 of 80) |
| Everyday coding-agent turns: small functions, edits, fixes | **Qwen at `medium` or `low`** | equal pass rates; Qwen 14.7 / 13.4 s per problem vs GLM high 18.5 s (and GLM max 99 s). The earlier "GLM 2.2× faster" was against Qwen `xhigh` (40.4 s) |
| Thinking off: fast completions, boilerplate, mechanical edits | **Qwen** (but close) | ties on HumanEval+/MBPP+/GSM8K. Decode is 63 vs 41 tok/s, yet GLM `low` writes shorter answers (166 vs 304 tokens), so measured time to solve is about even: GLM `low` 4.2 s per problem (35/41, 0.6) vs Qwen 4.8 s (38/41, greedy). Qwen keeps it as the live model (no switch) and on the solve count; n=41 |
| Several agents at once | **Qwen** | aggregate 1.5–2× higher at C2–C6 (C6 187 vs 94 tok/s) |
| Big repos, many long sessions in parallel, giant cold prompts | **Qwen** | KV pool 5.49M vs 2.03M tokens; cold prefill 2,700 vs 2,170 tok/s (a 180K-token cold miss: ~67 s vs ~83 s, estimated from those rates) |
| Very long single context (128K–500K) | either; GLM serves up to 500K | long-context tests 99.3% vs 97.2% at 64K/128K (p=0.38); Qwen's window is 262K; GLM beyond 128K is untested |
| Images / video in the prompt | **not GLM as deployed**; Qwen untested | GLM runs `--language-model-only` here (memory); image input to Qwen FP4-Local was never tested on this cluster |
| Long unattended uptime, stability | **Qwen** | 14–18 GiB free per node vs 7–8 GiB; days of live traffic behind it; GLM ran only for benchmarks |

Short version: **Qwen FP4-Local stays the default** for speed, concurrency, KV room and headroom. **GLM FP4-Local is the one to
switch to for a run of hard problems** (LiveCodeBench 55.0 vs Qwen 43.8–47.5 at any effort), and it is only fast when told
`reasoning_effort=high`. The two can't both be served at
once (one model per cluster), so it is a switch (~5 min warm boot), not a per-request choice.

---

## Setup

Scripts: `glmfp4/` (`lib.sh`, `setup_nodes.sh`, `up.sh`, `down.sh [--fp4local|--qwen]`), same pattern as `fp4local/`.

| Step | Notes |
|---|---|
| Download | `hf download` into the head's HF cache at the pinned revision: 166 GiB in 11.5 min. rsync of the HF blob store to the worker over CX7: 178 GB at ~550 MB/s, ~5 min. Head disk afterwards: 343 GB free. |
| Image | the same pinned b12x image as `fp4local/`; it has the `Glm5NextForConditionalGeneration` model, the `humming` MoE kernels for the draft, and the RoCE all-reduce. |
| Launch | `glmfp4/up.sh`: stops config C or FP4-Local, refuses if any other container is running, drops the page cache, head then headless worker. Container `vllm_glmfp4`, caches in `~/glmfp4-cache`. |
| Serving config | **nvidia's 2 x DGX Spark recipe from the model card, verbatim** (env incl. `VLLM_ENABLE_ROCE_ALLREDUCE=1`, `NCCL_BUFFSIZE`/`NCCL_MAX_CTAS`, AOT compile, V2 model runner; flags incl. `--attention-backend B12X`, `--block-size 256`, b12x MoE/linear backends, `--load-format b12x`, fp8 KV, 500K context, GMU 0.87, MTP with a `humming` MoE backend for the draft), plus the CX7 NCCL settings every deployment here uses. |
| Deviations | **`--language-model-only`** (text only; see the boot section), **`max-num-seqs` 8** (card: 4) so the C1–C6 bench and the harness's concurrency 8 run as for every other model, **MTP K=3** (card: 5; see the K sweep), and one RoCE HCA (the card lists both port "twins"; only one has an IP here). |
| Parallel layout | **TP=2 only.** EP fails at startup on this build (see Tuning). |
| torch.compile | On, as in nvidia's recipe (AOT compile). Same terms as FP4-Local: recipe as written, everything else stopped first. A guard sampled `MemAvailable` on both nodes every 2–5 s and killed the container below 2 GiB. |
| Lock-out | `--api-key` from the untracked `cluster.env`; unauthenticated requests got 401 (`/metrics` included). `num_requests_running` was 0 before and after every speed run. |
| Reasoning control | Same template as the other GLM builds: `chat_template_kwargs.reasoning_effort` = `low` (little or no reasoning on easy prompts; long reasoning on hard ones, see "Sampling temperature"), `high`, anything else / omitted = **`max`**. Parsers `glm45` / `glm47`. Tool call (one `get_weather` request) parsed correctly. |

### Boot and memory: why `--language-model-only`

The recipe as written didn't survive our memory guard. **Twice the head dropped from ~25 GiB to ~1.4 GiB available within
seconds** right after "Loading weights took 111 s" (log: `b12x weights loaded: 91.99 GB selected on rank 0`), and the guard killed
it. The first attempt was at the recipe's GMU 0.87, the second at 0.84, so it isn't the KV budget: GMU applies after this point. We
didn't let it run on to see whether it would recover, because an overcommit has hung this head before (QWEN.md). The next steps in
vLLM after loading are weight post-processing and the profiling pass, which for this multimodal checkpoint runs dummy
images/videos through the vision encoder.

`--language-model-only` sets every multimodal limit to 0, so vLLM builds no vision tower and skips encoder profiling. It doesn't
touch the language model. With it the boot went through every time (8 boots): lowest `MemAvailable` during boot 5.8–7.6 GiB on the
head and 2.9–8.7 GiB on the worker; after boot **7 GiB (head) / 8–9 GiB (worker)** free. Which of the two (vision tower or encoder
profiling) caused the dip isn't isolated. **Cost:** no image/video input on this deployment.

Boot: 640 s cold (191 s weight load, then compile and b12x autotune), ~290 s warm. **KV pool 2,029,850 tokens** at K=5,
2.02–2.03M at K=2–4 (the card says ~2M), 4.06× a 500K-token request.

---

## Speed (TP=2, nothing else on the server)

`scripts/bench_decode.py`, 500 tokens, `ignore_eos`, T=0.6, median of 3, the same prompts as every other model here.
"Thinking off" for GLM is `reasoning_effort=low`; "default" sends nothing (= effort max). Cells: tok/s (acceptance length, engine
steps/s). Files: `results/glmfp4/glmfp4-k{5,4,3,2}-bench_decode-{low,default}-2026-10-17.json`.

### Draft length (K) sweep: nvidia's K=5 is the slowest; K=3 is ours

| ctx | task | K=5 (recipe) | K=4 | **K=3** | K=2 |
|---|---|---|---|---|---|
| | *effort low* | | | | |
| 1K | code | 35.5 (3.14, 11.3) | 38.7 (3.10, 12.5) | **38.9** (2.79, 14.0) | 38.7 (2.42, 16.0) |
| 1K | code_ts | 41.6 (3.81, 10.9) | 42.2 (3.42, 12.4) | **41.9** (3.02, 13.9) | 40.1 (2.57, 15.6) |
| 1K | edit | 44.4 (4.09, 10.8) | 45.6 (3.81, 12.0) | **45.1** (3.37, 13.4) | 41.5 (2.71, 15.3) |
| 128K | code | 34.0 (3.04, 11.2) | 40.2 (3.26, 12.3) | **40.6** (2.94, 13.8) | 40.2 (2.55, 15.8) |
| 128K | code_ts | 41.3 (3.81, 10.8) | 41.1 (3.39, 12.1) | **42.6** (3.12, 13.7) | 38.9 (2.50, 15.6) |
| 128K | edit | 31.0 (2.84, 10.9) | 35.2 (2.90, 12.1) | **37.0** (2.70, 13.7) | 37.1 (2.41, 15.4) |
| | **mean** | 38.0 | 40.5 | **41.0** | 39.4 |
| | *template default (effort max)* | | | | |
| 1K | code | 31.3 (2.85, 11.0) | 35.3 (2.84, 12.4) | **38.5** (2.77, 13.9) | 35.4 (2.25, 15.7) |
| 1K | code_ts | 28.1 (2.53, 11.1) | 30.3 (2.45, 12.4) | **33.3** (2.41, 13.8) | 34.5 (2.20, 15.7) |
| 1K | edit | 30.8 (2.80, 11.0) | 36.1 (2.94, 12.3) | **37.9** (2.74, 13.8) | 36.7 (2.36, 15.5) |
| 128K | code | 34.7 (3.12, 11.1) | 34.3 (2.77, 12.4) | **35.1** (2.53, 13.8) | 38.7 (2.46, 15.7) |
| 128K | code_ts | 29.4 (2.67, 11.0) | 31.1 (2.53, 12.3) | **33.3** (2.40, 13.9) | 32.8 (2.11, 15.6) |
| 128K | edit | 34.9 (3.18, 11.0) | 42.0 (3.49, 12.0) | **42.4** (3.10, 13.7) | 37.9 (2.47, 15.3) |
| | **mean** | 31.5 | 34.9 | **36.8** | 36.0 |

Reading it: each extra draft token costs ~1.5 engine steps/s (15.6 → 13.8 → 12.3 → 11.0 for K=2..5), all flat from 1K to 128K.
The 4th and 5th draft positions are accepted 43–54% of the time on code edits, 15–31% on fresh code, ~10–20% on reasoning text and
1–7% on prose (per-position columns in the JSON). So K=5 only ties K=3 on the most predictable text and loses everywhere else,
worst at effort max (−14%), which is what GLM writes most of the time. nvidia's RTX PRO 6000 recipe on the same card uses K=3; why
the Spark recipe says 5 isn't stated. K is lossless, so every quality number below holds for every K.

Against the other models (6-cell means): GLM FP4-Local K=3 **41.0 / 36.8** tok/s (low / max); E224 at K=2 30.0 / 27.9; Qwen FP4-Local
K=2 63.3 / 49.5 (thinking off / on); Qwen C 67.0 / 52.1. Per engine step GLM FP4-Local at K=2 (15.6 st/s) is **36% faster than E224 on
stock vLLM 0.31.0** (~11.5 st/s), the same model shape on different kernels (b12x, compile on). A GLM token carries 4.35 bytes of our
corpus against Qwen's 4.05, so per character the gap to Qwen is ~7% smaller than tok/s says.

Wider single-run grid at K=5 (effort low, T=0 and 0.6, 1K and 64K, with prose; `glmfp4-k5-bench_decode-wide-low-2026-10-17.json`):
code 36–37, code_ts 38–42, edit 32–45, **prose 25–26** tok/s (prose accepts 2.1–2.2 tokens per step).

### Recipe bench (MiMo recipe `bench/mimobench.py`, temperature 0, effort low, unique prefixes; run on the head)

Aggregate tok/s (`results/glmfp4/bench-glmfp4-k{5,4,3,2}-tp2.{md,json}`):

| C | Qwen C | Qwen FP4-Local K=2 | GLM E224 | GLM FP4 K=5 | K=4 | **K=3** | K=2 |
|---|---|---|---|---|---|---|---|
| 1 | 58.3 | 56.5 | 26.6 | 33.4 | 34.1 | **35.3** | 34.3 |
| 2 | 96.1 | 86.9 | 43.2 | 44.7 | 46.9 | **48.4** | 49.1 |
| 3 | 137.6 | 115.5 | 59.5 | 58.5 | 49.8 | **51.5** | 61.4 |
| 4 | 174.5 | 137.9 | 70.6 | 71.1 | 71.6 | **75.3** | 75.1 |
| 5 | 197.7 | 155.7 | 70.7 | 67.4 | 84.7 | **73.0** | 73.0 |
| 6 | 228.2 | 186.9 | 86.7 | 91.9 | 80.4 | **93.7** | 97.1 |

Single-stream GLM FP4-Local is +33% over E224; under load the two are within ~10% of each other (single runs; this bench varied
6–15% between repeats before). Per-stream speed at C6 is ~18 tok/s. K=2 and K=3 tie under load.

Cold prefill (one request, unique prefix), tok/s at the 2K / 8K / 32K / 64K targets (1,494 / 5,938 / 23,844 / 47,771 GLM tokens):
**K=3: 1,657 / 2,049 / 2,170 / 2,164**; K=2–5 all within 3%. E224: 1,479 / 1,860 / 1,950 / 1,987. Qwen FP4-Local: 2,517 / 2,723 /
2,728 / 2,672.

### Prefix cache

`scripts/prefix_probe.py --steps cold,warm` at K=5 (`results/glmfp4/glmfp4-k5-prefix-probe-tail-{8000,24000,96000}-2026-10-17.json`):

| prompt | cold e2e (recomputed) | warm e2e (recomputed) |
|---|---|---|
| 7,075 | 3.66 s (7,075) | 0.31 s (**0**) |
| 21,376 | 10.45 s (21,376) | 0.41 s (**0**) |
| 86,154 | 33.09 s (69,770; 16,384 already cached from an earlier bench prompt) | 0.48 s (**0**) |

Warm hits recompute nothing, as with Qwen FP4-Local: the recipe's `--block-size 256` keeps the prefix-match granularity fine
(see FP4LOCAL.md "Why warm hits are free here"). Cold prefill here is ~2,100 tok/s.

### Tuning attempts (K=3, lossless)

| Change | Result |
|---|---|
| `--enable-expert-parallel` | **Fails at startup** (`Engine core initialization failed`). The root-cause line fell outside the 60 log lines `up.sh` keeps; the checkpoint uses the same `nvfp4_csf.py` experts method that raises `NVFP4-CSF supports TP without EP/DP` for any EP config (FP4LOCAL.md "Why no expert parallelism"), so it's almost certainly the same limit. |
| `--max-num-batched-tokens 8192` (recipe: 4096) | Cold prefill +1–7% (64K: 2,243 vs 2,164 tok/s), decode unchanged (effort-max mean 36.3 vs 36.8), aggregate unchanged (C1 / C4 / C6: 34.8 / 76.5 / 95.9), KV pool −6.5% (1.89M vs 2.02M). Single run; **not adopted**. Files `*-k3-mb8k-*`. |

---

## Quality (same harness, data and sandbox as Qwen and the other GLMs; run on the worker)

Same conditions as GLM E192 (GLM.md): effort low, greedy, `max_tokens` 4096 for HumanEval+/MBPP+/GSM8K; thinking tests at **effort
high** with the checkpoint's sampling (1.0 / 0.95, no top_k), seeded, 16K budget; long context at effort low, greedy, 512 cap. Served
at K=3. `results/quality/GF4-glmfp4`, `results/quality/GF4-deep`. One run.

| Test | n | Qwen C | Qwen FP4-Local | GLM E192 | GLM E224 | **GLM FP4-Local** | vs Qwen C: C-only / GLM-only (p) | vs FP4-Local (p) | vs E192 (p) |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval+ base / **plus**, thinking off | 164 | 95.1 / 93.3 | 97.6 / 94.5 | 95.1 / 92.1 | 95.1 / 92.1 | **94.5 / 91.5** | 9 / 6 (0.61) | 10 / 5 (0.30) | 7 / 6 (1.0) |
| MBPP+ base / **plus**, thinking off | 378 | 93.1 / 79.9 | 93.4 / 79.9 | 94.7 / 79.9 | 94.4 / 82.0 | **94.7 / 81.7** | 10 / 17 (0.25) | 9 / 16 (0.23) | 8 / 15 (0.21) |
| GSM8K, thinking off | 1319 | 96.4 | 96.4 | 96.3 | — | **96.5** | 22 / 24 (0.88) | 21 / 22 (1.0) | 17 / 20 (0.74) |
| HumanEval+ base / **plus**, thinking, 16K | 164 | 95.1 / 92.7 | 95.7 / 92.7 | 97.0 / 92.7 | — | **97.0 / 92.1** | 7 / 6 (1.0) | 5 / 4 (1.0) | 6 / 5 (1.0) |
| **LiveCodeBench** (80), thinking, 16K | 80 | 41.2 | 46.2 | 47.5 | — | **55.0** | 1 / 12 (**0.003**) | 2 / 9 (0.065) | 4 / 10 (0.18) |
| Long context 64K + 128K | 144 | 95.1% | 97.2% | 97.9% | 100% | **99.3%** (two-hop 47/48) | 1 / 7 (0.07) | 1 / 4 (0.38) | 0 / 2 (0.50) |
| Corpus NLL, nats/token (176,964 GLM tokens) | | | | 0.591 | 0.426 | **0.261** | | | |
| Corpus bits/byte (tokenizer-independent) | | 0.048 | 0.048 | 0.196 | 0.141 | **0.087** | | | |

vs E224: HumanEval+ 7 / 6 (p=1.0), MBPP+ 12 / 11 (p=1.0), long context 1 / 0 (p=1.0). p-values are exact McNemar on paired
per-problem pass/fail (`quality_eval.py compare-pass`); two runs of the same Qwen config differed by 1–3 flips each way.

Reading it:
- **Code and math without thinking: a tie with everything.** No difference anywhere has p < 0.2.
- **LiveCodeBench is the one real gain: 55.0%**, the best score on this cluster, **significantly ahead of Qwen config C** (12 vs 1
  problems only one of them solved) and ahead of the live Qwen FP4-Local at p=0.065 (borderline). Like every model here it ran into
  the 16K cap: **31/80 capped** (all without code, so fails); Qwen C 45, FP4-Local 40, E192 33. Mean 7,918 completion tokens (E192
  8,381, FP4-Local 10,443).
- **Long context 99.3%**: one two-hop miss at 64K, effectively tied with E224's 100%.
- **Likelihood**: the full 288-expert QAD model recovers most of what pruning lost: −0.33 nats/token vs E192 and −0.17 vs E224 on the
  same tokens (top-1 agreement 86% / 90%). Still well behind Qwen in bits/byte, which as before mostly measures memorised famous code.
- At effort high it thinks only as much as a problem needs: HumanEval+ mean **429 completion tokens** (476 characters of reasoning;
  Qwen ~1,850), 1/164 capped. Thinking-off answers: mean 144 / 128 / 93 tokens, 0 capped.

---

## Time to solve (does it solve problems faster despite slower tokens?)

Tokens/s and pass rate don't answer this on their own: GLM thinks much less on easy problems. So we timed it directly. **One problem
at a time** (concurrency 1, nothing else on the server), thinking on, 16K budget, each model at its quality-run settings (GLM effort
high, 1.0/0.95; Qwen thinking on, 0.6/0.95/20) plus GLM at **effort max**, pi's default when nothing is sent. Every 4th HumanEval+
problem (41) and every 8th LiveCodeBench problem (10). Wall clock per request, from `quality/deep_eval.py gen-think --stride N`
at `CONCURRENCY=1` (new `elapsed_s` field). Files: `results/quality/TTS-{glmfp4-high,glmfp4-max,fp4local}/`.

| | n | GLM FP4-Local, effort high | GLM FP4-Local, effort max | Qwen FP4-Local (K=2), effort `xhigh` (none sent = template default) | Qwen, `medium` (pi's default) | Qwen, `low` | Qwen, thinking off (greedy) |
|---|---|---|---|---|---|---|---|
| HumanEval+ subset: solved | 41 | 36 | 35 | 37 | 36 | **39** | 38 |
| ↳ seconds per problem | | 18.5 | 99.3 | 40.4 | 14.7 | 13.4 | **4.8** |
| ↳ seconds per solved problem | | 21.1 | 116.4 | 44.8 | 16.7 | 14.1 | **5.2** |
| ↳ mean completion tokens | | 728 | 3,875 | 2,168 | 945 | 854 | 304 |
| LiveCodeBench subset: solved | 10 | 2 | 1 | 1 | 1 | 1 | **3** |
| ↳ seconds per problem | | 323 | 438 | 292 | 301.5 | 291 | **173** |
| ↳ mean completion tokens | | 10,680 | 15,043 | 15,313 | 15,726 | 15,340 | 10,508 |

The Qwen `medium` / `low` / thinking-off columns were added on 2026-10-18 (`results/quality/TTS-fp4local-{medium,low,off}`; same
subsets, seeds, K and lock-out). The 2026-10-17 run sent Qwen no `reasoning_effort`, i.e. `xhigh`. Thinking off is greedy (as the
thinking-off quality runs), the others sample, so its LCB 3/10 is a hint, not a like-for-like win; with thinking off Qwen reasons
in the visible answer (10.5K tokens on LCB) and stopped by itself on 9 of 10.

The LiveCodeBench subset is hard: in the full concurrency-8 runs the same 10 problems went 3/10 (GLM FP4-Local), 2/10 (E192), 1/10
(both Qwens). With n=10 the solve counts can't separate the models; the seconds per problem can.

Reading it:
- **Easy problems: GLM at effort high finishes 2.2× sooner than Qwen at `xhigh`, but not than Qwen at `medium` or `low`.** It
  takes 18.5 s per problem against Qwen `xhigh`'s 40.4 s and 21.1 s against 44.8 s per *solved* problem; Qwen `medium` / `low`
  take 14.7 / 13.4 s (16.7 / 14.1 per solved). The solve rate is the same: 36 vs 37 of 41, and the full 164 tie (5 / 4, p=1.0). Qwen streams faster
  end to end, prefill included (~54 vs ~39 tok/s), but writes 3× the tokens.
- **At effort max GLM loses that.** It writes 3,875 tokens per problem and takes 99 s: 2.5× slower than Qwen `xhigh` (~7× slower than Qwen `medium`), 5.4× slower than
  itself at high, and with one fewer solve. So the effort pi sends decides which way this goes: anything but `low`/`high` (including pi's default `medium`) gets max.
  The fix for that is in the next section.
- **Hard problems: the wall clock is mostly time to the cap.** Qwen hit the 16K budget on 9 of 10, GLM high on 6 of 10 and GLM max on
  9 of 10. "Seconds per problem" here is roughly 16K ÷ token speed, so Qwen's faster tokens win it (292 vs 323 s). Per *solved* hard
  problem GLM at high is ahead (1,613 vs 2,918 s), but n=10 with 2 vs 1 solves can't support that comparison.
- **The Qwen numbers aren't a regression.** On these exact subsets its concurrency-8 run on 2026-10-16 also solved 37/41 (2,261
  tokens on average) and 1/10 (15,271 tokens).

### Server-side default effort (`DEFAULT_EFFORT=high`)

**Correction (2026-10-18):** pi *does* send `reasoning_effort`: `"medium"` at its default thinking level (seen in its request
bodies against Qwen, [PI.md](PI.md)). GLM's template treats `medium` as `max`, and a request's explicit value beats
`DEFAULT_EFFORT`, so the server-side default below would *not* help pi. Inferred, not tested (pi was never pointed at GLM):
for GLM, set pi's thinking level to `high` so it sends `high`.

A client that sends no `reasoning_effort` gets `max` on this template, the slow setting above. The pinned build has vLLM's
`--default-chat-template-kwargs`, merged *under* each request's own kwargs, so a request's explicit value still wins
(`vllm/renderers/params.py`). `glmfp4/up.sh` exposes it as `DEFAULT_EFFORT`. Check on one boot with `SPEC_TOKENS=3 DEFAULT_EFFORT=high`:
the same prompt, greedy, with the field omitted / low / high / max
(`results/glmfp4/glmfp4-k3-default-effort-high-2026-10-17.json`):

| request | completion tokens | reasoning chars |
|---|---|---|
| omitted | 224 | 75 |
| low | 154 | 0 |
| high | 229 | 99 |
| max | 3,834 | 11,832 |

"Omitted" behaves as high: a few dozen reasoning characters, against 12K at max. It isn't token-for-token identical to the
explicit-high answer. Our best guess is that greedy decoding on this build isn't bit-reproducible between requests (MTP drafts,
b12x kernels), but we didn't confirm it: two identical explicit-high requests weren't compared. **For clients that send no effort, boot with
`DEFAULT_EFFORT=high`** (pi sends one; see the correction above). It's off by default so the script reproduces nvidia's recipe as written.

---

## Sampling temperature: 0.6 vs the checkpoint's 1.0 (2026-10-24)

GLM was reinstalled overnight to answer the question the Qwen and MiMo temperature studies raised: does temperature 0.6 change it,
compared with the checkpoint's 1.0 that the server applies when a request sends nothing (pi without `samplingParams`)? Every
thinking result above was already at 1.0; thinking off (effort `low`) had only been run greedy. Same server as above (K=3, LM-only,
`DEFAULT_EFFORT=high`), same harness, data, sandbox and per-problem seeds, top_p 0.95, no top_k, API key on, nothing else on the server.
The 1.0 thinking run was repeated the same night, as a noise floor against 2026-10-17. One run each, paired McNemar.

**Quality** (full sets, concurrency 8; effort `low` at `max_tokens` 4096, effort `high` at a 16K budget):

| Test | 10-17 (`low`: greedy; `high`: 1.0) | **0.6** | **1.0** | 0.6 vs 1.0 (0.6-only / 1.0-only) |
|---|---|---|---|---|
| HumanEval+, `low` | 91.5 | 92.1 | 91.5 | 5 / 4, p=1.0 |
| MBPP+, `low` | 81.7 | **83.6** | 83.1 | 7 / 5, p=0.77 (0.6 vs greedy: 2 / 9, p=0.065) |
| HumanEval+, `high` | 92.1 | 90.2 | 91.5 | 4 / 6, p=0.75 |
| **LiveCodeBench**, `high` | 55.0 | **57.5** | 53.8 | 5 / 2, p=0.45 |

- **No temperature effect on quality, in either mode.** Nothing reaches p < 0.05. Today's 1.0 against 10-17's 1.0 (same settings,
  same seeds) moved LiveCodeBench 55.0 → 53.8 (3 / 4 flips) and HumanEval+ 92.1 → 91.5. That ±2–3 points is the noise, and it covers
  every 0.6-vs-1.0 gap. GLM is unlike MiMo here (MBPP+ −4.8 at 1.0, p=0.011) and Qwen (LiveCodeBench 37.5 vs 43.8 at `medium`).
- 57.5 is the best LiveCodeBench on the cluster, but it isn't distinguishable from GLM's 55.0 / 53.8 at 1.0.
- Lengths: `low` HumanEval+ / MBPP+ mean 150 / 135 tokens at 0.6, 153 / 141 at 1.0 (greedy 144 / 128), none capped. `high`
  HumanEval+ / LiveCodeBench 410 / 7,847 at 0.6 (31/80 capped), 358 / 7,260 at 1.0 (28 capped); 10-17: 429 / 7,918 (31 capped).

**Time to solve** (`quality/run_tts.sh`, one problem at a time, 16K budget, the COMPARISON.md subsets; first-ever timing at `low`):

| Setting | HE+ solved / 41 | s per problem | s per solved | median s | mean tokens | LCB solved / 10 | s per problem | mean tokens |
|---|---|---|---|---|---|---|---|---|
| `low`, **0.6** | 35 | **4.2** | **4.9** | | 166 | 1 | 266 | 8,758 |
| `low`, 1.0 | 35 | 15.3 | 17.9 | | 541 | 2 | 272 | 8,503 |
| `high`, **0.6** | **38** | 16.2 | 17.5 | 7.0 | 703 | 2 | 344 | 11,788 |
| `high`, 1.0 | 37 | 23.7 | 26.3 | 5.9 | 874 | 2 | 378 | 12,618 |
| `high`, 1.0 (10-17, above) | 36 | 18.5 | 21.1 | | 728 | 2 | 323 | 10,680 |

**Read the HumanEval+ seconds with care: the gaps are one or two runaway answers.** At `low` 1.0, HumanEval/116 reasoned 53K
characters into the 16K cap without answering (502 s of the 626 s total; at 0.6 the same problem finished in 1,368 tokens).
Without it, both temperatures run about 3 s per problem (median answer 114 vs 124 tokens). At `high`, the *median* problem is
faster at 1.0 (5.9 vs 7.0 s), and 1.0 was slower on only 15 of 41 problems. The mean gap is the tail: one capped answer in each run,
plus a 236 s one at 1.0. Two such events in 41 problems can't separate the temperatures. The weak signal, consistent with Qwen and
MiMo, is that 1.0 runs away more often.

**`low` is not "no reasoning".** On easy problems it is, nearly: 6 of 41 HumanEval+ answers carried 7–85 characters of reasoning, and one
(HumanEval/116) 2.8K at 0.6 and 53K at 1.0.
On hard ones it thinks hard: all 10 LiveCodeBench problems reasoned, 5–6 of them for 20K–54K characters, and 4–5 of 10 ran into the
16K cap at both temperatures, so `low` there costs about as much time as `high` (266–272 vs 323–378 s per problem). The earlier
docs called `low` "no reasoning" from the HumanEval+/MBPP+ runs, where answers average ~150 tokens.

**What this means for pi.** For GLM, unlike Qwen and MiMo, leaving pi at the checkpoint's 1.0 costs no measured quality. 0.6 is
still the slightly safer choice (fewer runaways, the higher point estimates), and it is what pi already sends Qwen. Untested:
effort `max` at 0.6, tool calls and long context at 0.6.

Files: `results/quality/GF4-t{06,10}-{off,deep}/`, `results/quality/TTS-glmfp4-{low,high}-t{06,10}/`. Lowest free memory over the
night: 3.7 GiB (memory guard at 1.5 GiB never fired). Runs, on the worker, with `API=http://<head-link-ip>:8000 API_KEY=...
MODEL=glm53-flash-fp4local`: `TASKS="humaneval mbpp" NOTHINK_KWARGS='{"chat_template_kwargs":{"reasoning_effort":"low"}}'
NOTHINK_MAX_TOKENS=4096 NOTHINK_TEMP=0.6 NOTHINK_TOP_P=0.95 quality/run_quality.sh <label>`; `TASKS="humaneval lcb"
THINK_KWARGS='{"chat_template_kwargs":{"reasoning_effort":"high"}}' THINK_TEMP=0.6 THINK_TOP_P=0.95 THINK_TOP_K= quality/run_deep.sh
<label>`; the same `THINK_*` with `quality/run_tts.sh <label>` (effort `low` or `high`).

## How to run it again

```bash
glmfp4/setup_nodes.sh               # idempotent: 166 GiB download + rsync to the worker, image pull
SPEC_TOKENS=3 DEFAULT_EFFORT=high glmfp4/up.sh   # needs setup_nodes.sh first (removed 2026-10-17); stops Qwen; API_KEY lock-out if set; ~11 min cold, ~5 min warm
                                    # http://<head>:8000/v1, model glm53-flash-fp4local
glmfp4/down.sh --fp4local           # back to the live Qwen FP4-Local (LIVE=1 SPEC_TOKENS=2 fp4local/up.sh)
glmfp4/down.sh --qwen               # or back to config C
```

Knobs: `SPEC_TOKENS` (default 5, nvidia's; **use 3**), `DRAFT_MOE` (`humming`), `MAX_SEQS` (8), `MAX_BATCHED` (4096), `GMU` (0.87),
`MAX_LEN` (500000), `EP` (fails), `LM_ONLY` (1), `DEFAULT_EFFORT` (unset = template max; `high` for clients that send none; pi sends `medium`, which overrides it, see above).

On the nodes: checkpoint in the HF cache (`models--nvidia--GLM-5.3-Flash-FP4-Local` + the shared `hub/blobs/` store, 166 GiB per
node), caches in `~/glmfp4-cache` (root-owned files from the container; delete with a privileged container as in GLM.md).
**Removed from both nodes on 2026-10-17**: only the 44 GLM-only blobs (165 GiB) were deleted from the shared blob store, no other model's links broke, and the b12x image stays because Qwen FP4-Local uses it. Head disk afterwards: 508 GB free.

## Caveats / not measured

- One run per quality test; the time-to-solve subsets are small (41 and 10 problems), single runs.
- Thinking tests at effort `high`, the time-to-solve also at `max`; `max` was not run through the full LiveCodeBench, where its long
  thinking would hit the 16K cap even more often (it averaged 15,043 tokens on the 10-problem subset).
- Text only (`--language-model-only`): vision and video input not served. Which part of the vision path caused the post-load memory
  dip isn't isolated.
- 7–8 GiB free per node is less headroom than any other deployment here except config C. Long-running stability (hours of agent
  traffic, the 500K-token window) is untested.
- Not tried: both RoCE port twins (needs an IP on the second port, a host network change), PLE/`B12X_*` knobs from the RTX recipe,
  `draft_tensor_parallel_size` 1, `--performance-mode interactivity`, `max-num-seqs` above 8.
- The b12x build is a community nightly, pinned by digest.
- Harness bug, fixed: the first GLM time-to-solve run meant to send effort `high` but a quoting error sent the literal string
  `${EFFORT:-high}`, which the template treats as `max`. That run is reported as the effort-max column; effort high was re-run.
