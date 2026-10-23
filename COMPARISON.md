# Every model and every reasoning setting we ran, side by side

All on this cluster: 2× DGX Spark, TP=2, same harness, data and sandbox. This page only collects numbers; each one comes from the
model's own doc, which has the method, the result file and the caveats: [FP8DENSE.md](FP8DENSE.md) / [QWEN.md](QWEN.md) (Qwen
A/B/C), [FP4LOCAL.md](FP4LOCAL.md) (Qwen FP4-Local), [MIMO.md](MIMO.md) / [QWEN_VS_MIMO.md](QWEN_VS_MIMO.md), [GLM.md](GLM.md)
(E192/E224), [GLMFP4.md](GLMFP4.md) (GLM FP4-Local). Collected 2026-10-18. **"—" = not measured.** One run per cell unless noted;
the noise floor (Qwen A run twice) is 2–3 problems each way on HumanEval+/MBPP+.

> [!WARNING]
> **Until 2026-10-19 pi did not run Qwen at the sampling settings this repo benchmarks.** pi sends no `temperature` by default, so vLLM applies the
> checkpoint's `generation_config.json`: **temperature 1.0 / top_p 0.95 / top_k 20**. Every Qwen thinking-on score in this repo was
> measured at **0.6** / 0.95 / 20 (thinking off: greedy). Re-measured at pi's real settings on 2026-10-19 (FP4-Local, `medium`, 1.0):
> HumanEval+ is unchanged (95.1 vs 95.7), but **LiveCodeBench drops to 37.5% from 43.8%**. All 6 lost problems are hard ones
> (hard: 4/46 vs 9/46) and 49 vs 43 of 80 hit the 16K cap. Paired: vs `medium` at 0.6, 6 lost / 1 gained, p=0.125; vs `xhigh` at 0.6
> p=0.016, vs `low` at 0.6 p=0.021. One run each. **Read the repo's Qwen thinking scores as "at 0.6", not "what pi gets".**
> **Fixed client-side 2026-10-19:** pi now sends 0.6 via `"samplingParams": {"temperature": 0.6}` on the model
> ([pi/models.json.example](pi/models.json.example)). A top-level `"temperature"` key on the model is **silently ignored** by pi.
> The server default is still 1.0 for any other client (server fix, not applied: `--override-generation-config '{"temperature":0.6}'`).
> Details: [FP4LOCAL.md](FP4LOCAL.md) "Sampling temperature".

## The reasoning controls, per model

| Model | Control | Levels | Default when the request sends nothing | What pi sends |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (A/B/C, FP4-Local) | `enable_thinking` + `reasoning_effort` (thinking on only) | off; `low`, `medium`, `xhigh` (`high` = HTTP 400) | thinking on, **`xhigh`** | `medium` |
| MiMo-V2.6-Flash-RL | `enable_thinking` only (no effort control in the template or model card) | off; on | template: on. **Our server: off** (recipe default, tool-call-storm mitigation) | — (not installed) |
| GLM-5.3-Flash (E192, E224, FP4-Local) | `reasoning_effort` only (always thinks) | `low` (= no reasoning), `high`, anything else = `max` | **`max`** | `medium`, which GLM treats as `max` (inferred, see GLMFP4.md) |

Before 2026-10-18 no Qwen run sent `reasoning_effort`, so every older Qwen "thinking on" number is at `xhigh`.

## Quality

Thinking off: greedy, `max_tokens` 1536 (GSM8K 1024; GLM 4096). Thinking on: seeded, 16K budget; sampling Qwen 0.6 / 0.95 / 20,
MiMo and GLM 1.0 / 0.95. HumanEval+ and MBPP+ are "plus" (base + extra tests). LiveCodeBench: 80 stdin problems; about half of
every model's runs hit the 16K cap, so it measures "solves it within 16K tokens".

| Model | Setting | HumanEval+ (164) | MBPP+ (378) | GSM8K (1319) | **LiveCodeBench** (80) | Long context 64K/128K (144) |
|---|---|---|---|---|---|---|
| Qwen A (nvidia NVFP4) | thinking off | 95.1 | 79.9 | 96.5 | — | — |
| Qwen B (RadixArk BF16-dense) | thinking off | 93.9 | 79.4 | 96.4 | — | 96.5% |
| | thinking on, `xhigh` | 92.7 | — | — | 43.8 | |
| Qwen C (FP8-dense) | thinking off | 93.3 | 79.9 | 96.4 | — | 95.1% |
| | thinking on, `xhigh` | 92.7 | — | — | 41.2 | |
| **Qwen FP4-Local** (live) | thinking off | 94.5 | 79.9 | 96.4 | — | 97.2% |
| | thinking on, `low` | 94.5 | — | — | 47.5 | |
| | thinking on, `medium`, T=0.6 | **95.7** | — | — | 43.8 | |
| | thinking on, `medium`, **T=1.0 (what pi gets)** | 95.1 | — | — | **37.5** | |
| | thinking on, `xhigh` | 92.7 | — | — | 46.2 | |
| MiMo | thinking off (server rep-pen 1.05) | 88.4 | 74.3 | 96.4 | — | 90.3% |
| | thinking off, T=0.6 (2026-10-23) | 84.8 | **75.4** | — | — | |
| | thinking off, **T=1.0 (server default, what pi gets)** | 86.0 | 70.6 | — | — | |
| | thinking on | 89.0 | — | — | 46.2 | |
| | thinking on, T=0.6 (2026-10-23) | 92.1 | — | — | 48.8 | |
| | thinking on, T=1.0 rerun (2026-10-23) | 92.1 | — | — | 45.0 | |
| GLM E192 | `low` (no reasoning) | 92.1 | 79.9 | 96.3 | — | 97.9% |
| | `high` | 92.7 | — | — | 47.5 | |
| GLM E224 | `low` | 92.1 | **82.0** | — | — | **100%** |
| **GLM FP4-Local** | `low` | 91.5 | 81.7 | **96.5** | — | 99.3% |
| | `high` | 92.1 | — | — | **55.0** | |

Not run anywhere: `max` on the full sets for any GLM; thinking-on MBPP+/GSM8K; GLM E224 thinking on. Long context is thinking off
(GLM: `low`) for everyone.

Significant differences (paired McNemar, p < 0.05): MBPP+ Qwen C over MiMo (p=0.001) and GLM E192/E224 over MiMo; long context GLM
E192 over MiMo (p=0.007) and E224 over Qwen C (p=0.016); **LiveCodeBench GLM FP4-Local `high` over Qwen C (p=0.003) and over Qwen
FP4-Local `medium` (p=0.012)**; MiMo thinking off MBPP+ 0.6 over 1.0 (p=0.011, [MIMO.md](MIMO.md) "Sampling temperature"). Against Qwen FP4-Local `xhigh` / `low` that LCB gap is p=0.065 / 0.15. Nothing else is
significant, including every Qwen effort level against every other.

Thinking-mode length (mean completion tokens, HumanEval+ / LiveCodeBench): Qwen B 1,942 / 10,990; Qwen C 1,877 / 10,929;
Qwen FP4-Local `xhigh` 1,826 / 10,443, `medium` 1,004 / 10,593, `low` 840 / 10,196; MiMo 833 / — (2026-10-23: 742 / 9,331 at 0.6, 792 / 9,855 at 1.0); GLM E192 `high` 485 / 8,380;
GLM FP4-Local `high` 429 / 7,918.

## Time to solve (one problem at a time, wall clock per request, 16K budget)

Every 4th HumanEval+ problem (41) and every 8th LiveCodeBench problem (10), nothing else on the server. The two FP4-Local
models were timed on 10-18/19, MiMo (DFlash 7) on 2026-10-23, with thinking off sampled too (MIMO.md).

| Model | Setting | HE+ solved / 41 | s per problem | s per solved | mean tokens | LCB solved / 10 | s per problem | mean tokens |
|---|---|---|---|---|---|---|---|---|
| Qwen FP4-Local (K=2) | thinking off (greedy) | 38 | **4.8** | **5.2** | 304 | **3** | **173** | 10,508 |
| | `low` | **39** | 13.4 | 14.1 | 854 | 1 | 291 | 15,340 |
| | `medium`, T=0.6 | 36 | 14.7 | 16.7 | 945 | 1 | 301.5 | 15,726 |
| | `medium`, **T=1.0 (what pi gets)** | 38 | 15.8 | 17.0 | 1,010 | 1 | 332.5 | 15,789 |
| | `xhigh` | 37 | 40.4 | 44.8 | 2,168 | 1 | 292 | 15,313 |
| GLM FP4-Local (K=3) | `high` | 36 | 18.5 | 21.1 | 728 | 2 | 323 | 10,680 |
| | `max` | 35 | 99.3 | 116.4 | 3,875 | 1 | 438 | 15,043 |
| | `low` (no reasoning) | **not measured** (GLM was removed before thinking-off timing was added); *estimate* ~3.5–4 s per problem from 144 mean tokens at 41 tok/s. Full HE+ at `low`, greedy: 91.5 | | | 144 | — | — | — |
| MiMo (DFlash 7) | thinking off, T=0.6 | 34 | 7.2 | 8.7 | 356 | 1 | 110 | 4,069 |
| | thinking off, **T=1.0 (what pi gets)** | 35 | 9.5 | 11.1 | 411 | 0 | 160 | 4,327 |
| | thinking on, T=0.6 | 37 | 21.8 | 24.1 | 904 | 1 | 414 | 14,199 |
| | thinking on, T=1.0 | 37 | 37.4 | 41.4 | 1,205 | 2 | 467 | 13,134 |

GLM `low` has no measured time to solve; its row is an estimate from the thinking-off decode rate and mean answer length,
not a timing. The Qwen thinking-off row is greedy and the rest sample, so its LCB 3/10 is a hint, not a like-for-like win. With thinking off
Qwen reasons in the visible answer on hard problems (10.5K tokens). n=10 can't separate LCB solve counts. MiMo at 1.0 takes
1.3× (off) / 1.7× (on) as long as at 0.6 for the same HumanEval+ solve count; even at 0.6 it is 1.5× Qwen's thinking-off time
and about Qwen `medium`'s with thinking on, at a lower pass rate.

## Speed

`scripts/bench_decode.py`: 500 tokens, T=0.6, median of 3, mean of 6 cells (code / code_ts / edit at 1K and 128K). Recipe bench:
`bench/mimobench.py`, T=0, thinking off (GLM `low`), aggregate tok/s. "Thinking on" decode was measured with nothing sent, so Qwen
= `xhigh`, GLM = `max`; decode speed per token doesn't depend on the effort level, only on how many tokens get written.

| Model (MTP) | Decode, thinking off / on, tok/s | Recipe bench C1 / C4 / C6 | Cold prefill 32K / 64K, tok/s | KV pool | Free per node |
|---|---|---|---|---|---|
| Qwen C (K=2) | 67.0 / 52.1 | 58.3 / 174.5 / 228.2 | 3,031 / 2,989 | 4.25–4.48M | ~4–7 GiB |
| **Qwen FP4-Local (K=2)** | 63.3 / 49.5 | 56.5 / 137.9 / 186.9 | 2,728 / 2,672 | **5.49M** | **14–18 GiB** |
| MiMo (DFlash 7) | 35.8 (our mean of the 6 cells) / — | 42.8 / 101.0 / 139.6 | 1,486 / 1,271 | 2.05M | ~6–9 GiB |
| GLM E192 (K=2) | 26.8 / 27.7 (our means of the 6 cells) | 21.0 / 73.1 / 88.0 | 1,954 / 2,009 | 2.11M | 19 GiB |
| GLM E224 (K=2) | 30.0 / 27.9 | 26.6 / 70.6 / 86.7 | 1,950 / 1,987 | 2.11M | 11–13 GiB |
| GLM FP4-Local (K=3) | 41.0 / 36.8 | 35.3 / 75.3 / 93.7 | 2,170 / 2,164 | 2.02M | 7–8 GiB |

Qwen A and B thinking-on decode (2026-09-27, nothing sent): 44.4 and 39.0 tok/s. Warm prefix hits recompute 0 tokens on both
FP4-Local builds and ~2.5K on Qwen C. Speed numbers are single runs with 6–15% spread on the recipe bench, and were valid only with
nothing else on the server.

## How to use it (from the above)

- **Send temperature 0.6 from the client** (pi: `samplingParams`, see [pi/models.json.example](pi/models.json.example)). The server default 1.0 measured worse on hard problems (warning above).
- **Default: Qwen FP4-Local, thinking on at `medium` (pi's default) or `low`.** Same quality as `xhigh`, ~3× faster on easy
  problems, and faster than GLM `high` there too. Don't pick `xhigh`.
- **Mechanical edits and small functions: Qwen with thinking off**, ~3× faster again at the same HumanEval+ pass rate.
- **A run of hard algorithmic problems: GLM FP4-Local at `high`** (LiveCodeBench 55.0, the only significant coding-quality win over Qwen; GLM E224 also beat Qwen C on long context, p=0.016).
  It's a model switch, not a per-request choice, and GLM is not installed now (`glmfp4/setup_nodes.sh`, 166 GiB). Never run GLM at
  `max` for interactive work: 99 s per easy problem.
- Keep one thinking level per session: changing it re-prefills the whole history ([PI.md](PI.md)).
- MiMo lost to Qwen on speed everywhere and on MBPP+ significantly; it is not installed. If it comes back, 0.6 beats the
  server's 1.0 on MBPP+ (thinking off) and on time to solve, but the recipe found 0.6 makes tool-call storms more frequent (MIMO.md).
