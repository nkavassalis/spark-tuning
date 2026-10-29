# DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, TensorFold) on the 2× Spark cluster

**Status 2026-10-29:** served and measured 2026-10-28/29 (speed, quality, time to solve, the post-cutoff AtCoder set; below),
then stopped and its weights (`~/dsv41-hf`, 295 GiB on the head, 587 GiB on the worker) **removed from both nodes** on
2026-10-29. The engine image `tensorfold-dsv41:exl3-tp2-928adbd0e57d` and the recipe checkout stay on both;
`deepseek/setup_nodes.sh` re-downloads (~400 GB) and rebuilds the weight files. Current cluster state:
[AGENTS.md Purpose](AGENTS.md#purpose).

**Verdict:** the strongest hard-problem coder measured on the cluster (LiveCodeBench 76-79% vs ≤59%; post-cutoff AtCoder 70.1%
vs Qwen's 53.7%) and the fastest single-stream decode (~92 / 69 tok/s thinking off / on), with time to solve on hard problems
half Qwen's. Against that: 4 streams at once (aggregate stops at ~130 tok/s, Qwen keeps scaling), a 1M-position shared window,
a ~1 s tail recomputed on every warm prefix hit, worse long context than GLM, average thinking-off quality, and a 900 GB
footprint across the two nodes.

## What it is

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` (its card: a 552B backbone; the quant's metadata says 105B params; active params not stated) |
| Weights | [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) @ `45ee5d0f`, 210.7 GB in 39 shards; Engram's n-gram tables unquantized from the base model (shards 47/48, 203 GB) |
| Bit widths | routed experts 3 bit (2 bit in layers 18-22), attention 5, shared experts 4-5, indexer wk 8, head 6, MTP/DSpark drafter 4; body average 2.90 bpw |
| Engine | TensorFold native (Zig) engine, BertholomusAI's `dsv41` family (fork `bertholomus/TensorFold` @ `629c524`) + 27 recipe patches; TP=2 over RoCE, one rank a Spark; DSpark speculative decoding from the checkpoint's own draft blocks (drafted output equals plain output, per the recipe) |
| Recipe | [`MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) @ `27741e1` (2026-10-24), run unmodified; `deepseek/` wraps it |
| Served as | `DeepSeek-V4.1-Flash-1017` on `:8000` (recipe default 8888); 1,048,576-token window, 4 streams, thinking on by default, server sampling temperature 1.0 / top_p 0.95 / top_k 20 |

## Review (2026-10-28, before running anything)

**Recipe.** Read `README.md`, `start.sh`, `scripts/{config,prepare,nodes}.sh`, `zig/Dockerfile`. Findings:
- No torch.compile and no GMU: memory is the engine's fixed 12.2 GiB reserve plus three guards. `start.sh` refuses to start
  with under 112 GiB available on either node, and the recipe's memguard watchdog on each node kills its rank when `MemAvailable`
  stays under 2.5 GiB for 4 s, or at once under 1 GiB. The recipe says a rank that takes all the memory froze a Spark
  once (reserve lowered to 10 GiB); we leave the reserve and guards at their defaults.
- Containers run with `--network host --ipc host --cap-add IPC_LOCK --device /dev/infiniband`. They are not privileged.
  No sudo is needed, except for the recipe's NFS export.
- The API is **open by default**. `deepseek/up.sh` sets the recipe's `API_KEY=1` with the cluster.env key whenever that key is set.
- The recipe's only supported layout has the worker read the head's HF cache over **NFS**. That needs root on the head
  (`apt install nfs-kernel-server`, `/etc/exports`), and the head would need ~530 GB; it has 508 GB free. Not used (user's
  call, 2026-10-28). See "How it is set up here".
- No published image matches these pins: ghcr has `exl3-tp2-7c9bf5d36350`, and the pins hash to `928adbd0e57d`. The image is built
  locally from `zig/Dockerfile` on NVIDIA's `pytorch:26.07-py3` (Zig 0.17.0 by sha256, the engine by commit).
- The recipe's quality evidence is functional checks only: token-hash replay, needle 3/3 at 479K, tool and smoke checks. There are no task benchmarks.

**Quantization quality, from Mia's model card (not measured here).** Teacher-forced KL against the original weights on
100 conversations: 0.0137 nats, top-1 agreement 96.6%. On 8K-31K synthetic sessions: KL 0.005-0.006, top-1 98.8-99.1%.
Nothing above 31.2K tokens was evaluated, and neither were free generation, task accuracy or vision.

### The TensorFold "2x faster" alternative

[`Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster`](https://huggingface.co/Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster)
holds no weights. It is a recipe card for the **same engine family**, the author's fork of BertholomusAI's dsv41 TensorFold
engine, serving a **different 2.9 bpw EXL3 pack**:
[`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), 43 shards.
- Its "2x faster" compares TensorFold with **vLLM** (MiaAI-Lab's vLLM kit) on the same two Sparks:
  - decode, prose 56.8 vs 27.0 tok/s, code 92.2 vs 39.9;
  - 32K prefill 2,165 vs 925 tok/s.
  
  It is not faster than the Mia recipe, which uses the same TensorFold engine. The Mia recipe's own sparkDash numbers are
  higher still (prose 74.9, code 109.9 tok/s for one stream), but the two came from different tools and prompts and
  neither has been measured here.
- The packs are close to identical. Mia's card reports matching widths for all 46,702 main-model quantized tensors and
  byte-identical tokenizer files. The difference is the MTP drafter (4 bit for Mia, 6 for LibertAI) and the quantizer
  (Mia's "EAQ" patches to exllamav3). Mia's paired comparison against LibertAI on the same 100 conversations, measured by
  the Mia side:

  | | KL (nats) | top-1 agreement |
  |---|---|---|
  | Mia EAQ v1 | 0.0137 | 96.6% |
  | LibertAI 2.9 bpw | 0.0155 | 96.2% |

  The paired KL difference is -0.0017 [-0.0020, -0.0014]. The drafter's bit width changes how many drafted tokens are
  accepted, which is a speed question; drafting does not change the output.
- **Verdict:** no significant quality difference is expected between the two, and none our harness could detect. LCB-80
  moves about ±3-5 points between reruns of the same model (MIMO.md). Trying the second one would cost another ~400 GB
  download and a second engine build for what is close to the same model. Only worth it if the Mia pack misbehaves in
  some way that points at the quant or at its drafter. The real open quality question is **2.9 bpw against the full
  model**, with layers 18-22 at 2 bits. Our task benchmarks against the other models answer it from the outside.

## How it is set up here (no NFS)

The recipe documents a "no NFS at all" mode: each node gets a local tree, with `PREPARE=0`, `NFS_VOLUME` set to a host
path and `HEAD_ENGRAM_DIR`. `deepseek/setup_nodes.sh` builds those trees:
- **Worker** (the disk is there): the full download, the image build, and the recipe's `prepare.sh` steps 4, 5 and 7 as
  its own commands (kit and MANIFEST check, token map, the weight files of **both** ranks, the served folder). Engram's
  tables are hard-linked into rank 1's `/engram`.
- **Head**: only what rank 0 reads, copied from the worker over the CX7 link. That is the small files and the vision
  shard (`model-00039`), Engram's tables, rank 0's weight file, the kit and the token map.
- **The recipe's `local.sh`** on the head points at both trees.

Disk, measured 2026-10-28 after setup: worker 587 GiB in `~/dsv41-hf` (checkpoint 197, Engram 190, a weight file for each
rank ~99 each) plus the 24.9 GB image; head 295 GiB plus the image, which leaves the head 189 GiB free. The worker keeps rank
0's weight file as well, so the head can be re-synced without re-preparing.

Running a download or build next to the live model (2026-10-28):
- The `hf` CLI 2.x (Xet) reached **13.4 GB RSS** on the worker while the live Qwen was serving, and `MemAvailable` fell from
  12.7 to 5.2 GiB. It was killed; `MemAvailable` recovered to 17.6 GiB and Qwen kept answering.
- The download now runs in a container capped at 4 GiB (`dsv41-download`). The weight-file writes run under a 6 GiB cap
  (`MEM_CAP`).
- The image build was OOM-killed at 6 GiB: the Dockerfile compiles both executables at once (`-j$(nproc)`). It now runs
  under 8 GiB on one CPU (`BUILD_MEM`, `BUILD_CPUSET`), so `nproc` is 1 and they compile one after the other; the image is
  the same.

## How to run it

```bash
deepseek/setup_nodes.sh            # idempotent; does not stop the live model (--only N runs one step)
deepseek/up.sh                     # stops every other model, recipe ./start.sh restart (API_KEY from cluster.env locks it)
deepseek/down.sh --fp4local        # recipe ./stop.sh, then the live Qwen FP4-Local again
```
Knobs pass through to the recipe: `CONTEXT`, `PARALLEL`, `THINKING`, `REASONING_EFFORT`, `THINKING_BUDGET`, `MAX_TOKENS`,
`TEMPERATURE`/`TOP_P`/`TOP_K`, `DRAFTS`, `POOL`, `KV_SHARD`, `SPILL_GIB` (`deepseek/up.sh`).

## Speed (2026-10-28, recipe defaults: 4 streams, 1,048,576-token window, drafts on; API key locked, nothing else on the server)

Loaded in 36 s (rank cache read whole in 20 s) after `start.sh`'s checks. Memory available after load: head 7.2-7.9 GiB,
worker 8.9-9.7 GiB (memguard's floor is 2.5). The KV pool is the recipe's default: `CONTEXT` positions (1,048,576) shared by the
4 streams, against FP4-Local's 4.54M tokens in live mode. `tensorfold:requests_running` + `requests_waiting` read 0 before and after
every run. Unauthenticated requests (agents still pointed at :8000) got 401 throughout.

**Decode**, `scripts/bench_decode.py`: 500 tokens, `ignore_eos` (checked: 300 of 300 tokens decoded), T=0.6, median of 3, the same
prompts as every other model. Files `results/deepseek/deepseek-bench_decode-{thinkoff,default}-2026-10-28.json`. "Thinking
default" sends nothing, which is the template's `high` (effort 75). Acceptance is TensorFold's `mtp_accepted_total / mtp_drafted_total`
(no draft-step count is exported, so no mean acceptance length).

| ctx | task | FP4-Local live, off / default (`xhigh`) | **DeepSeek, off / default (`high`)** | DeepSeek accepted share, off / default |
|---|---|---|---|---|
| 1K | code | 65.0 / 50.9 | **86.4 / 61.0** | 0.74 / 0.52 |
| 1K | code_ts | 65.4 / 47.3 | **90.0 / 69.1** | 0.75 / 0.61 |
| 1K | edit | 67.7 / 58.4 | **98.8 / 81.5** | 0.79 / 0.70 |
| 32K | code / code_ts / edit | — | 87.2 / 78.3 / 96.0 ; 72.1 / 69.4 / 74.9 | |
| 64K | code / code_ts / edit | — | 80.1 / 82.0 / 94.6 ; 63.5 / 63.0 / 70.2 | |
| 128K | code | 62.4 / 49.3 | **88.1 / 60.9** | 0.76 / 0.53 |
| 128K | code_ts | 63.8 / 44.1 | **89.5 / 70.5** | 0.76 / 0.64 |
| 128K | edit | 61.4 / 52.5 | **96.9 / 69.1** | 0.81 / 0.60 |

Means of the six 1K/128K cells: **91.6 / 68.7 tok/s** against FP4-Local's 64.3 / 50.4, about +40%. Decode does not slow down with
context (MiMo fell 10.6 to 7.1 steps/s from 1K to 128K). DeepSeek's tokenizer is also denser: these prompts are 0.80× as many
tokens (825 vs 1,033 at "1K", 102K vs 128K at "128K"), so per character of output the gap is larger still. The FP4-Local column
is `results/fp4local/ple-device-bench_decode-*-2026-10-21.json` (live mode, K=2).

**Concurrency and cold prefill**, the MiMo recipe's `bench/mimobench.py` (T=0, thinking off, keyed copy, run on the head):
`results/deepseek/bench-deepseek-tp2.{md,json}`.

| C | FP4-Local live, aggregate / per-stream | **DeepSeek aggregate / per-stream** | DeepSeek mean TTFT (s) |
|---|---|---|---|
| 1 | 57.6 / 63.9 | **71.5 / 84.0** | 0.26 |
| 2 | 87.9 / 51.2 | **101.6 / 62.0** | 0.39 |
| 3 | 115.6 / 45.4 | **123.2 / 50.3** | 0.51 |
| 4 | **141.2** / 41.4 | 133.1 / 41.0 | 0.66 |
| 5 | **158.0** / 36.8 | 117.9 / 47.5 | 1.37 |
| 6 | **187.5** / 36.6 | 121.6 / 43.7 | 1.95 |

DeepSeek leads up to 3 streams. It serves 4 at once (`PARALLEL`), so at 5 and 6 the extra requests queue (TTFT 1.4-2 s) and the
aggregate stops rising; FP4-Local keeps scaling to 6.

| cold prefill target | FP4-Local tokens / TTFT / tok/s | DeepSeek tokens / TTFT / tok/s |
|---|---|---|
| 32K | 31,551 / 11.0 s / 2,859 | 18,304 / **8.9 s** / 2,060 |
| 64K | 63,221 / 22.5 s / 2,813 | 36,536 / **17.7 s** / 2,060 |

Per token DeepSeek prefills ~28% slower, but this (prose) prompt is only 0.58× as many DeepSeek tokens, so the same text is
ready ~20% sooner. A cold 88,641-token code prompt took 43.1 s (below).

**Prefix cache**, `scripts/prefix_probe.py` (TensorFold has no per-source counters; it reads the reply's `cached_tokens`):
`results/deepseek/deepseek-prefix-probe-{24000,96000}-2026-10-28.json`.

| step | 22,041-token prompt | 88,641-token prompt |
|---|---|---|
| cold | 22,041 computed, 12.3 s | 88,641 computed, 43.1 s |
| warm, immediately again | 1,561 recomputed (92.9% cached), 1.09 s | 577 recomputed (99.3%), 0.75 s |
| 1 token changed at the start | 22,042 recomputed, 10.1 s | — |
| 1 token changed at the end | 1,563 recomputed, 1.09 s | 579, 0.77 s |
| after 3 competing 22K prompts | 1,561 recomputed, 1.09 s | — |
| after 600 s idle | 1,561 recomputed, 1.09 s | — |

The cache works in 2,048-token blocks ("prompts in 2048-row chunks" in the engine's load line): a warm hit recomputes the
tail past the last full block (`prompt mod 2048`, under 2,048 tokens, ~1 s), unlike FP4-Local's zero. It survived competing
prompts and 10 minutes idle. One run each.

## Quality (2026-10-28/29, the harness every other model ran; API key locked)

`quality/run_quality.sh` (thinking off, greedy) and `quality/run_deep.sh` (thinking on at the template default `high`, 32K budget,
concurrency 8), the same problems, seeds and sandbox as COMPARISON.md. No NLL / bits-per-byte: TensorFold ignores
`prompt_logprobs`. Result folders `results/quality/{DS-off,DS-deep32-t10,DS-deep32-t06,TTS32-ds-t06,DS-atc-t06}`. McNemar exact
p on the same problems; one run per cell.

**Thinking off (greedy):**

| Test | **DeepSeek** | Qwen FP4-Local | GLM FP4-Local (`low`) | GLM E224 (`low`) | MiMo |
|---|---|---|---|---|---|
| HumanEval+ | 91.5 | 94.5 (p=0.23) | 91.5 (p=1.0) | 92.1 (p=1.0) | 88.4 (p=0.30) |
| MBPP+ | 78.0 | 79.9 (p=0.21) | **81.8 (p=0.007)** | **82.0 (p=0.011)** | 74.3 (DeepSeek better, p=0.034) |
| GSM8K | 97.0 | 96.4 (p=0.24) | 96.5 (p=0.35) | — | 96.4 (p=0.26) |

p is DeepSeek against that model on the same problems (folders `F4-fp4local`, `GF4-glmfp4`, `G224-low`, `M-mimo-tp2`). Thinking
off, DeepSeek is an average Spark model: it ties Qwen and loses MBPP+ to both GLMs (which always think; `low` is their lightest setting).

**Thinking on, 32K budget:**

| Test | **DeepSeek T=1.0** | **DeepSeek T=0.6** | best other Spark model at 32K | Flint int4 (Mac, T=0.6) |
|---|---|---|---|---|
| LiveCodeBench (80, 2025-01..04) | **78.8** (11 at the cap) | **76.2** (12) | 58.8 (MiMo 1.0, GLM FP4-Local 1.0) | 71.2 |
| HumanEval+ | 92.1 (base 97.6) | 94.5 (98.2) | 95.7 (Qwen `medium`, 16K) | 91.5 |
| long context (144, up to 128K) | 94.4 | — | 100.0 (GLM E224) | — |

- **LiveCodeBench is the outlier**: DeepSeek beats every Spark model at p ≤ 0.001 at both temperatures, and nearly as a
  superset (T=1.0: 0 problems lost against Qwen `medium` / `xhigh` and MiMo, 1 against GLM FP4-Local; 22-24 gained). 1.0 vs 0.6:
  4 / 6, p=0.75. Against Flint int4 on the Mac: 7 / 1 (T=1.0, p=0.070), 8 / 4 (T=0.6, p=0.39).
  The 80 problems date from 2025-01-11 to 2025-04-06, long before this checkpoint, so a gap this size could be contamination:
  that is what the AtCoder check below is for.
- HumanEval+ ties everything (p ≥ 0.07). Long context: 136/144, below GLM FP4-Local (p=0.039) and E224 (p=0.008), tie with
  Qwen FP4-Local (97.2, p=0.29) and MiMo (90.3, p=0.15).

**Time to solve at 32K** (`quality/run_tts.sh`, T=0.6, concurrency 1, nothing else on the server), against COMPARISON.md's
32K table:

| Model | HE+ solved / 41 | s per problem | LCB solved / 10 | s per problem | at the cap / 10 |
|---|---|---|---|---|---|
| Qwen FP4-Local `medium` | 36 | 14.3 | 1 | 564 | 9 |
| MiMo, thinking on | 37 | 22.1 | 1 | 717 | 8 |
| GLM FP4-Local `high` | 37 | **9.0** | 3 | 628 | 6 |
| **DeepSeek `high`** | **39** | 18.7 (median 8.7) | **6** | **287** | 3 |

On hard problems DeepSeek both solves more and finishes sooner: it reasons less (22.8K mean tokens against Qwen's 30.5K) and decodes
~40% faster. HE+ is slower on average than GLM because of a few long answers (max 118 s); the median is 8.7 s.

**Problems newer than the checkpoint: post-cutoff AtCoder** (2026-10-29, `LCB_DATA=atc147.jsonl`, T=0.6, 32K, concurrency 8;
the set and its sample-test scoring are mac-tuning's, FP4LOCAL.md "Problems newer than the training data"). The lcb80 runs
were rescored on sample tests only so both sides are scored the same way (`results/quality/samples-only/DS-deep32-*`):

| Model | lcb80, samples only | **atc147** | drop | atc147 paired vs DeepSeek |
|---|---|---|---|---|
| Qwen FP4-Local `medium` / `xhigh` | 53.8 / 53.8 | 53.7 / 53.1 | ~0 | 25 / 1 and 27 / 2, p<0.001 |
| Flint int4 (Mac, T=0.6) | 76.2 | 68.0 | −8 | 9 / 6, p=0.61 |
| **DeepSeek `high`** | 83.8 (both temperatures) | **70.1** (103/147; 42 at the cap) | **−14** | |

- **The old 80 overstate DeepSeek the most** (−14 points against Flint's −8 and Qwen's ~0), so part of its lcb80 lead is
  contamination or a better fit to older problems; which one can't be told from this.
- **What is left is still the best result on new problems**: +16 points over Qwen FP4-Local, nearly as a superset, and a tie with
  Flint int4 on the Mac.
