# DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, TensorFold) on the 2× Spark cluster

**Status 2026-10-28:** set up on both nodes (downloaded and prepared on the worker, rank 0's files copied to the head,
the image built and loaded on both, the recipe's `DRY_RUN=1 ./start.sh` checked). **Not served, nothing measured yet**:
starting it stops the live Qwen FP4-Local, and that waits for the user's go-ahead. Current cluster state:
[AGENTS.md Purpose](AGENTS.md#purpose).

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

## Test plan (after the go-ahead)

All runs locked with the API key, and nothing else on the server:
1. load, memory after load on both nodes (recipe warns within 3 GiB of memguard's floor);
2. `scripts/bench_decode.py` C1-C6 at 1K-128K, thinking off and on;
3. `scripts/prefix_probe.py`;
4. `quality/run_quality.sh` and `quality/run_deep.sh` at the 32K thinking budget (COMPARISON.md), at the server's
   temperature 1.0 and at 0.6;
5. time to solve (`quality/run_tts.sh`);
6. optionally the post-cutoff AtCoder set (`LCB_DATA=atc147.jsonl`, FP4LOCAL.md).

Then `deepseek/down.sh --fp4local`.
