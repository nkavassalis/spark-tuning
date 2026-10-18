# Can SGLang serve nvidia Qwen3.8-Flash-Next-FP4-Local, and would it be faster?

**Question (2026-10-18):** the live model (FP4-Local, [FP4LOCAL.md](FP4LOCAL.md)) needs the community b12x vLLM nightly. Could
SGLang serve the same checkpoint instead, and would that be faster? **This was researched, not measured.** The cluster stayed on
FP4-Local the whole time, and no SGLang server was started (the user's call, after the findings below).

## Answer

**Not today, in any SGLang build.** Checked against the SGLang image already on the head (`lmsysorg/sglang:qwen38flashnext`,
built 2026-09-03, commit `593134d`), upstream `main` as of 2026-10-18 (`python/sglang/srt/models/qwen4_exp.py`), the newest
Docker Hub tags (`v0.5.22-cu130` 2026-10-16, `nightly-dev-cu13-20261018-b234b5c3`) and open PRs. Two formats in the checkpoint
have no reader in SGLang:

| Checkpoint part | Size | Format | SGLang | Lossless way around it |
|---|---|---|---|---|
| Decoder routed experts (48 layers × 512) scales | 3.6 GiB (CSF) | NVFP4 block scales in nvidia's compressed **CSF** format (`weight_scale.nvfp4_csf_fixed` + `…_exceptions`) | no CSF code anywhere (code and PR search: `csf`, `weight_scale_encoding`, `nvfp4_csf`: nothing) | **Yes, offline.** Decode to plain E4M3 `weight_scale` `[rows, in/16]`: row base byte + 4-bit offset per block, with sparse exact-byte exceptions (the rule in b12x's `transform_csf_blocks`). Bit-exact; +3.4 GiB on disk (7.0 GiB plain). The MTP experts already ship in that plain layout. |
| PLE n-gram table (layer 2, 128 shards) | 50.6 GiB | **NVFP4** group-16: U8 `[2,500,012, 80]` + E4M3 `weight_scale [.., 10]` + FP32 `weight_scale_2` (`ple_embedding_dtype: "nvfp4"`) | **BF16 or per-tensor FP8 only** (`"PLE embedding offload requires bfloat16 or fp8 weights"`, in the image and on `main`) | **No conversion fits.** BF16 is ~100 GB (and still rounds the global scale); per-tensor FP8 can't hold e2m1 × E4M3-block × global exactly, so it would change the model. Only a code change works: dequantize NVFP4 rows when gathering (16 rows of 160 values per token, one PLE layer). |

Everything else SGLang already reads: the `qwen4_exp` architecture, ModelOpt `MIXED_PRECISION`, MXFP8 linears/attention,
`W4A16_NVFP4` MTP experts. That leaves SGLang's own SM121 issues (QSA decode path, MTP `!` collapse), which upstream reports as
fixed (#36845, #36811/#38290), but which weren't exercised here.

**What it would take (not done):** expand the CSF scales offline (lossless), add NVFP4 PLE gather to SGLang (a local patch: no
upstream PR exists), then check the output matches vLLM token for token before trusting any speed number. Estimated several hours,
with the cluster offline for 3–5 of them.

## Would it be faster? Probably not for this workload (published numbers only)

No SGLang number on this cluster for this checkpoint exists. The closest published data is SGLang's own verified 2× Spark recipe
(PR #37995, merged 2026-09-07, image `dev-qwen38-next-local`) on `nvidia/Qwen3.8-Flash-Next-NVFP4`: the same architecture, plain
NVFP4 experts, FP8 PLE table held in GPU memory and TP-sharded. Their bench is `bench_serving`, random dataset, ISL 1024 / OSL 256,
so it isn't comparable cell for cell with ours (random tokens draft badly with MTP; ours are code and edit prompts).

| | SGLang, 2× Spark, nvidia NVFP4 (PR #37995) | **FP4-Local on vLLM b12x, K=2 (ours, [FP4LOCAL.md](FP4LOCAL.md))** |
|---|---|---|
| Single stream, MTP | 47.4 tok/s (MTP 3/1/4, random prompts) | 56.5 (recipe bench C1) / 63.3 (thinking off) / 49.5 (thinking on) |
| Single stream, no MTP | 24.4 tok/s (TPOT 38.6 ms) | not measured (always ran MTP) |
| Concurrency | 166 tok/s at c=16, 198 at c=24 (MTP) | 137.9 at C4, 186.9 at C6 (`max-num-seqs` 8) |
| KV pool | 1.59M tokens (MTP cell) | 5.49M tokens |
| Free memory after boot | ~12 GB per node | 14–18 GiB per node |

Other 2× Spark SGLang reports for this model family: 64.4 / 116.8 / 114.1 tok/s at ×1 / ×2 / ×4 (mrexodia recipe, RadixArk
NVFP4), 47.5 / 87.6 tok/s at 1 / 4 streams (PixelML recipe). Both used older SGLang builds and SM121 patches.

Reading: single-stream SGLang on GB10 sits at or below what FP4-Local already does here. Its strength is wide batching (c=16–96),
which a few pi agents don't use. With FP4-Local it would also lose two things the user picked FP4-Local for: the NVMe-backed PLE
table, which is why vLLM b12x has a 5.5M-token KV pool (an SGLang NVMe PLE backend exists only as open PR #36567 and the
`qwen4-main-squashed` branch), and the measured zero-recompute warm prefix hits (SGLang's hybrid-model prefix cache is untested here).
One real difference: SGLang could run **expert parallelism**, which the b12x build refuses. That is the main unproven suspect for
FP4-Local's ~20% concurrency gap to config C. Whether SGLang's EP would close it is unmeasured.

## Context found on the nodes

- The head still has `lmsysorg/sglang:qwen38flashnext` (33 GB): left over from the SGLang deployment the Qwen deployment repo
  replaced with vLLM on 2026-08-31 (its root commit `6e08722`). That backup checkout no longer exists.
- Revisit when either lands upstream: CSF scales or NVFP4 PLE tables in SGLang (`gh search prs --repo sgl-project/sglang csf`),
  or when nvidia lists SGLang under "Supported Runtime Engine(s)" on the model card (vLLM only at revision `b9ce18a`).
