Project mode: REAL

# AGENTS.md: spark-tuning

## Purpose
Public notes, scripts and measured results from tuning LLM serving on a 2-node DGX Spark (GB10) cluster
(vLLM TP=2 over the CX7 200G link). The cluster serves one model at a time:
- Qwen3.8-Flash-Next (NVFP4 + FP8-dense, MTP K=2): `QWEN.md`, `FP8DENSE.md`. Tuned; the production model.
- MiMo-V2.6-Flash-RL (DFlash 7): `MIMO.md`. Set up, benchmarked, then **removed from both nodes** on 2026-10-07. Reinstalled
  for one day on 2026-10-23 for a temperature study (thinking off: 0.6 beats the server's 1.0 on MBPP+, p=0.011, and 1.0 is 1.3–1.7×
  slower to solve; thinking on: tie; MIMO.md "Sampling temperature"), then removed again.
- GLM-5.3-Flash, autotrust NAS-pruned NVFP4 builds E192 / E224 (NVFP4 MTP K=2): `GLM.md`. Benchmarked 2026-10-13:
  ties or beats Qwen C on every task benchmark (not on corpus NLL), 2–3× slower decode (~11.5 vs ~25.5 engine steps/s). E224 is the better GLM
  (same speed, 100% long context). **Removed from both nodes** on 2026-10-13 (checkpoints, image, caches); `GLM_VARIANT=E224 glm/setup_nodes.sh && GLM_VARIANT=E224 glm/up.sh` rebuilds it (158 GB download).
- nvidia Qwen3.8-Flash-Next-FP4-Local (QAD NVFP4 + MXFP8, community b12x vLLM build): `FP4LOCAL.md`. Benchmarked 2026-10-16 at TP=2:
  ties or beats config C on every quality test; 5% slower with thinking on (K=2) and ~20% less aggregate at 3–6 streams, but a
  ~30% bigger KV pool (5.5M tokens), 14–18 GiB free per node and zero-recompute warm cache hits. **The live model since 2026-10-16**
  (user's call), at MTP K=2: `LIVE=1 SPEC_TOKENS=2 fp4local/up.sh`. Since 2026-10-21 live mode holds the PLE n-gram table in GPU
  memory at GMU 0.84 instead of reading it from NVMe (+1.5% decode, +4–5% cold prefill, KV 4.54M tokens; FP4LOCAL.md). Reasoning effort (2026-10-18, FP4LOCAL.md): template accepts
  `low`/`medium`/`xhigh`, **omitted = `xhigh`** (every pre-10-18 Qwen thinking number), pi sends `medium`; `low`/`medium` tie `xhigh`
  on quality and are ~3× faster on easy problems. No EP on the current b12x build: a
  software limitation (b12x MoE backend + CSF scratch design), not the scale format; see FP4LOCAL.md "Why no expert parallelism".
- nvidia GLM-5.3-Flash-FP4-Local (QAD NVFP4 + MXFP8, all 288 experts, same b12x build): `GLMFP4.md`. Benchmarked 2026-10-17 at TP=2,
  MTP K=3 (nvidia's K=5 is the slowest K here), `--language-model-only` (the full recipe dipped the head to ~1.4 GiB free after
  weight load). Best LiveCodeBench on the cluster (55.0%, vs config C p=0.003), ties elsewhere; decode 41/37 tok/s (effort low/max)
  vs Qwen FP4-Local 63/50. **Removed from both nodes** on 2026-10-17; `glmfp4/setup_nodes.sh && DEFAULT_EFFORT=high SPEC_TOKENS=3 glmfp4/up.sh` rebuilds it (166 GiB), `glmfp4/down.sh --fp4local` goes back.
- Head-to-head: `QWEN_VS_MIMO.md` (and the three-way tables in `GLM.md`). **Current state: nvidia FP4-Local (K=2) serving on :8000 as
  `qwen3.8-flash-next`; config C is installed and comes back with `fp4local/down.sh --qwen`.**
  MiMo is not installed; `mimo/setup_nodes.sh && mimo/up.sh` rebuilds it (~30 min, 178 GB download).
Goal for all: fast, high-quality single-user code generation for coding agents, with no change that
alters output quality unless it is measured.

## THE PUBLIC RULE (read first)
**This repo is PUBLIC on GitHub.** It must never contain:
- IPs (LAN or link), hostnames/machine names, usernames, home-directory paths, router/DHCP/network details;
- tokens or credentials of any kind (HF token, GitHub token, etc.).
Use placeholders (`<head>`, `<head-cx7-ip>`, `user@head-host`). Real values live only in the untracked
`cluster.env` (git-ignored; template `cluster.env.example`) and on the nodes themselves. Scripts must
read hosts from env/`cluster.env` and never have real defaults. Results JSON written by benchmarks contain the
URL used: sed it to `<head>` before committing (`sed -i 's#http://[0-9.]*:#http://<head>:#' results/...`).
Before every commit run:
```bash
git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/home/|hf_[A-Za-z0-9]{10,}|gh[pousr]_[A-Za-z0-9]{10,}'
```
and inspect every hit. History was squashed on 2026-10-07 to remove earlier leaks. If a leak is pushed,
tell the user. Fixing it means scrubbing plus a history rewrite, and rotating any secret.

## Layout
| Path | Responsibility |
|---|---|
| `README.md` | human entry point: index, setup, switching models, tests |
| `QWEN.md`, `FP8DENSE.md` | Qwen tuning write-up (MTP K, NCCL/control plane, FP8-dense eval, failures, timeline) |
| `MIMO.md` | MiMo setup, **tool-call storm mitigations and caveats**, measurements, open items |
| `QWEN_VS_MIMO.md` | same-cluster comparison; Qwen quality numbers reused from 2026-09-27/28 runs, not rerun (user's call) |
| `PI.md` | the **client** side: how pi shapes load, what diverges a prefix cache, compaction staging and its revert |
| `GLM.md` | GLM-5.3-Flash E192/E224: setup, `reasoning_effort` semantics, speed + quality vs Qwen C and MiMo |
| `FP4LOCAL.md` | nvidia FP4-Local vs config C: K sweep, quality vs A/B/C, KV sizing, lock-out and live switch |
| `GLMFP4.md` | nvidia GLM FP4-Local: K sweep, speed, quality vs Qwen C / FP4-Local / E192 / E224, time-to-solve at C=1 |
| `COMPARISON.md` | 2026-10-18 roll-up: every model × reasoning setting (Qwen off/low/medium/xhigh, MiMo off/on, GLM low/high/max): quality, time to solve, speed, usage advice |
| `SGLANG.md` | 2026-10-18 research: no SGLang build reads FP4-Local (CSF expert scales: lossless offline fix; NVFP4 PLE table: needs an SGLang patch); published SGLang 2× Spark numbers not faster for single-user; nothing run |
| `pi/models.json.example` | IP-free copy of the pi provider config for the live Qwen: temperature 0.6 via `samplingParams`, image input on |
| `cluster.env.example` | template for the untracked `cluster.env` (HEAD, WORKER, API, REPO, MIMO_RECIPE) |
| `scripts/` | Qwen-era benchmark tools; `bench_decode.py` works for any model (`--model`, `--url`); `prefix_probe.py` measures prefix-cache behaviour; `watch_stalls.sh` is the 2 s stall sampler |
| `quality/` | quality-eval harness (runs on a node with docker; `API` env required). `NOTHINK_TEMP`/`NOTHINK_TOP_P` sample thinking-off runs (unset = greedy); `run_tts.sh` + `tts_score.py` time to solve at concurrency 1 |
| `configs/` | diffs applied to the Qwen deployment repo (`env.diff`, `start.sh.patch`, FP8 overlays) |
| `mimo/lib.sh` | shared vars: pinned recipe commit, image, iface/HCA defaults, ports, `on`, `drop_caches` |
| `mimo/setup_nodes.sh` | recipe checkout + render `mimo.env` + weights (head download, rsync to worker) + patches |
| `mimo/up.sh` / `mimo/down.sh [--qwen]` | switch cluster to MiMo / stop MiMo (optionally relaunch Qwen) |
| `mimo/mimo.env.example` | template rendered onto nodes as `<recipe>/launch/mimo.env` (placeholders filled on the nodes) |
| `mimo/toolcap/` | tool-call-cap proxy (upstream copy + `upstream.diff`), its node tests, `routes.json` |
| `mimo/test_mimo.py` | functional checks against the live MiMo deployment |
| `glm/lib.sh` | shared vars for GLM: `GLM_VARIANT` (E192/E224), image `vllm/vllm-openai:v0.31.0`, paths, `on`, `drop_caches` |
| `glm/setup_nodes.sh` | head download, MTP config flip (`num_nextn_predict_layers` 1), rsync to worker, image pull, patch script copy |
| `glm/up.sh` / `glm/down.sh [--qwen]` | switch cluster to GLM (writes `~/glm-cache/serve.sh` per node, container `vllm_glm`) / stop it |
| `glm/patch_vllm.py` | idempotent in-container patch of vLLM 0.31.0 so the standalone `mtp-nvfp4/` draft loads |
| `fp4local/lib.sh`, `setup_nodes.sh`, `up.sh`, `down.sh` | FP4-Local at TP=2 with nvidia's recipe (compile on, user-approved 2026-10-16); `API_KEY` locks it for benchmarks, `LIVE=1` serves it to agents |
| `glmfp4/lib.sh`, `setup_nodes.sh`, `up.sh`, `down.sh [--fp4local\|--qwen]` | GLM FP4-Local at TP=2 with nvidia's recipe (+ LM-only, max-num-seqs 8); knobs `SPEC_TOKENS`, `MAX_BATCHED`, `GMU`, `EP`, `LM_ONLY`; `API_KEY` lock-out |
| `LICENSE` | MIT (2026-10-17); vendored `mimo/toolcap/` keeps its upstream MIT notice |
| `results/` | raw outputs; `results/mimo/` for MiMo |

## How to run
```bash
cp cluster.env.example cluster.env   # fill in (never commit)
mimo/setup_nodes.sh                  # idempotent; ~178 GB weights (MiMo is currently NOT installed on the nodes)
mimo/up.sh                           # stops Qwen, starts MiMo (~13.5 min) + proxy
mimo/down.sh --qwen                  # back to Qwen
```
GLM: `GLM_VARIANT=E192 glm/up.sh` (or E224), `glm/down.sh --qwen`; API `http://<head>:8000/v1`, model `glm53-flash-e192`.
FP4-Local: `fp4local/up.sh` (needs `fp4local/setup_nodes.sh` once; checkpoint is in the HF cache, whose 2.x blob store lives at `~/.cache/huggingface/hub/blobs/`).
**Benchmark lock-out:** put `API_KEY=` in `cluster.env`; `fp4local/up.sh` serves with `--api-key`, and `bench_decode.py`, `quality/*` and the runners send it. In the b12x build `/metrics` is behind the key too.
GLM always thinks: send `chat_template_kwargs.reasoning_effort` = `low` (no reasoning) or `high`; **omitted = `max`**.
Endpoints (MiMo): agents `http://<head>:8000/mimo/v1` (proxy); raw vLLM `http://<head>:8888/v1`; model `mimo-v2.6-flash`.
On the nodes: recipe checkout `~/mimo-recipe`, weights `~/models/MiMo-V2.6-Flash-RL`, cache `~/mimo-cache`,
proxy files and log `~/mimo-toolcap/` (container `mimo-toolcap`), vLLM container `vllm_mimo`.

## How to run the tests
```bash
scripts/stage_pi_compaction.py --check                 # confirm the pi client staging (PI.md)
node --test mimo/toolcap/*.test.cjs                    # 13 proxy unit tests, offline
set -a; . ./cluster.env; set +a; mimo/test_mimo.py     # live: chat, tool-call parse, storm guard
bash -n mimo/*.sh glm/*.sh scripts/*.sh quality/*.sh    # syntax
python3 -m py_compile scripts/*.py quality/*.py mimo/*.py glm/*.py
```
Benchmarks (`scripts/bench_decode.py`, recipe `bench/mimobench.py` run on the head, `scripts/prefix_probe.py`) are
only valid when nothing else uses the server. **This is not a footnote**: live agent traffic made the same tool read
C1 aggregate 37 instead of 62 tok/s and 1K decode 52.8 instead of 63.9 on 2026-10-15. Before trusting a speed number
confirm `vllm:num_requests_running` <= your own load (or watch `scripts/watch_stalls.sh`), and re-run any C1 cell whose
TTFT exceeds ~0.3 s.

## Design decisions and why
- **Recipe used unmodified, pinned** (`RECIPE_COMMIT` in `mimo/lib.sh`). Our scripts wrap it, so upstream fixes come
  from bumping the pin instead of from local forks.
- **Weights/cache under `$HOME`, not `/var/tmp`**: systemd-tmpfiles cleans `/var/tmp` after 30 days.
- **Worker gets its own copy via rsync over CX7** instead of NFS: no root needed, and the worker disk has room.
- **Page-cache drop via privileged container**: no passwordless sudo. GMU 0.90 needs it.
- **Proxy runs on the head in a container on :8000** (the port Qwen used), so agents reach it on the LAN.
  The only change to upstream is the `TOOLCAP_HOST` bind address (`mimo/toolcap/upstream.diff`). Keep that diff minimal.
- `up.sh` ignores `serve.sh`'s exit status on the worker (the recipe script exits 1 on success for rank≠0)
  and checks the container instead.
- Server sampling defaults (temp 1.0, top_p 0.95, rep-pen 1.05) are a storm mitigation. **Don't lower temperature
  "for code"**: 0.6 makes tool-call storms worse (recipe measurement).

## Client (pi)
The user's agent is pi. It always streams and only sends `temperature` if configured, so both MiMo storm
mitigations apply. The local pi config (outside this repo) has a `qwen` provider at `<head>:8000/v1` and a `mimo`
provider at `<head>:8000/mimo/v1`; only the one matching the model currently served works (the `mimo` entry is inert while MiMo is not provisioned).
Qwen thinks by default, which costs it speed (see `QWEN_VS_MIMO.md`: 66–70 tok/s with thinking off vs 48–56 with it
on). If decode speed matters more than reasoning in pi, turn reasoning off — Qwen with thinking on still beat MiMo
with thinking off in every cell measured.

**Read `PI.md` before investigating a "slow/stalling cluster" report.** The 2026-10-15 investigation found the server
clean and the load shape client-side: pi re-sends whole histories (a third of turns >100K tokens), and a prefix-cache
miss costs ~50 s of exclusive prefill. Client compaction: 2026-10-15 staged `reserveTokens` 65536 / `keepRecentTokens` 20000 (config C, several agents);
**2026-10-21 relaxed to 40000 / 40000** (compacts above 222,144; reserve must stay >= maxTokens 32000 + 4096), applied and reversible with
`scripts/stage_pi_compaction.py --check|--revert`.

## Conventions
- Every number in the docs is measured, with the result file next to it. Label anything unmeasured as unmeasured.
- Shell: `set -euo pipefail`, source `mimo/lib.sh`, use `on "$HEAD" ...`. Python: stdlib only.
- Dates in docs: YYYY-MM-DD. New experiments get a section in the model's .md plus a result file.

## Known issues / TODOs
- **Server default for Qwen is temperature 1.0 (the checkpoint's `generation_config`), every Qwen thinking score here is at 0.6.**
  pi fixed client-side 2026-10-19 with `samplingParams` (`pi/models.json.example`; a top-level model `temperature` is silently ignored).
  Measured 2026-10-19 (FP4-Local, `medium`): HumanEval+ same, LiveCodeBench 37.5% vs 43.8%, all losses on hard problems (p=0.125
  same level; p=0.016/0.021 vs `xhigh`/`low` at 0.6). Server fix `--override-generation-config '{"temperature":0.6}'` is **not applied**
  (changes output for every client; user's call). FP4LOCAL.md "Sampling temperature".
- **Multi-agent stalls (config C, 2026-10-15): mechanism proven.** A captured 50 s freeze was one request
  re-prefilling **182,012 tokens** at 3,640 tok/s: GPU 100% SM on compute, no preemption, queue empty, PSI 0.00
  (reclaim ruled out), RoCE carrying NCCL only. Driver: pi re-sends whole histories (p50 ~54K, max ~172K tokens), so a
  prefix divergence re-crunches everything, and config C's hybrid mamba recurrent-state cache granularity also
  re-crunches ~2.5-3K tokens on every warm hit (`FP4LOCAL.md` explains the mechanism). The switch to FP4-Local removed
  the recurring cost - warm hits now recompute **0** (independently re-measured 2026-10-16: 96,559-token prompt, 0
  recomputed, 0.35 s end to end, `results/fp4local/fp4local-k2-mycheck-prefix-96000-2026-10-16.json`).
  **Residual risk:** a genuinely *cold* prompt is now ~36% slower per token (2,668 vs 3,640 tok/s), so a 180K cold miss
  would monopolise the engine ~68 s instead of 50 s. Client-side mitigation: pi compaction (65536 on 2026-10-15,
  relaxed to 40000 on 2026-10-21 for mostly single-user use; restore 65536 if multi-agent stalls return; see `PI.md`). The 2026-10-15 A/B plan (batched-tokens 4096 + `--long-prefill-token-threshold` + GMU 0.80) is
  **retired**: FP4-Local already runs 4096/0.8 and `FP4LOCAL.md` shows 8192 gains nothing. `--long-prefill-token-threshold`
  is untried on both models and is the only remaining server lever if giant-cold-prompt stalls show up.
  Async scheduling stays ON - demoted to untested hypothesis, no longer needed.
- **Warm hits are free on FP4-Local because of prefix-match granularity, not block size** (`FP4LOCAL.md`); config C's
  ~2.5-3K tail is exactly `(prompt mod 1648) + 1648`, i.e. no fine hits at all. **Untested**: config C would probably
  lose the tail with `--prefix-match-unit` set (a small `--block-size` alone is the weaker bet, since the old build's
  `enable_partial_hash_hits` predicate only looks at the Mamba group). Never tried - it needs a restart of the server
  this session runs on.
- **The pi status-line token speed is not a measurement.** `pi-token-speed` counts 1 token per streamed SSE chunk
  (default `direct` strategy) while MTP delivers ~2-2.6 tokens per chunk, ignores tool-call args except `edit`/`write`,
  and times from stream start (prefill included) - so it reads ~half the true rate, ~24-25 while a benchmark says 45+.
  Measured table and fix (`countStrategy: "estimate"`, staged 2026-10-16) in `PI.md`. Never conclude anything about
  speed from it.
- **Speed is NOT regressed** (checked 2026-10-15 against the recorded baselines: C1 62.4 vs 58.3, C6 218 vs 228, env
  byte-identical, same image, no throttling). Per-stream ~20 tok/s is 4 concurrent agents + thinking on; see the
  decomposition table in QWEN.md "Speed re-check" before re-investigating this.
- The proxy caps **streamed** responses only. Non-streaming requests can still storm.
- MiMo is 1.3–2.7× slower than Qwen config C for single-user code (both thinking off); the gap grows with context (MiMo step rate falls 10.6→7.1/s from 1K to 128K, Qwen flat ~25/s).
- MiMo quality: one run per test, so there's no noise floor for MiMo. Qwen's quality numbers are reused from 09-27/28.
- `quality_eval.py nll` per-token NLL is only comparable within one tokenizer; use `bpb` across models.
- GLM: `scripts/bench_decode.py --chat-kwargs '{"reasoning_effort":"low"}'`; quality harness `NOTHINK_KWARGS` / `THINK_KWARGS` /
  `NOTHINK_MAX_TOKENS` (defaults reproduce the Qwen/MiMo requests). GLM decode runs show an unexplained slow mode
  (~7.5 vs ~11.5 engine steps/s) in some effort-low runs; see GLM.md.
- `scripts/bench_decode.py --thinking {default,on,off}`: Qwen's template default is thinking ON, MiMo's server default OFF. Set it explicitly for cross-model comparisons.
- The recipe bench came in 6–15% under the recipe's published aggregate (single run).

## Testing limitations / unverified
- Storm guard verified only with a synthetic 12-parallel-call request, not a real long agent session.
- MiMo image/video/audio input, needle-in-a-haystack, 300K-context requests: not exercised here.
- No real pi agent session benchmark (task success / wall-clock) for any model.
- GLM: thinking-mode tests ran at `reasoning_effort=high`, not `max` (pi's default); GLM tool calling only smoke-tested;
  GLM vision input, 1M context and `max_num_seqs` > 8 not exercised.
- `mimo/up.sh`, `mimo/down.sh --qwen` and removing MiMo from the nodes were all exercised end to end on 2026-10-07.
- The cleanup chown at the end of `mimo/setup_nodes.sh` chowned everything to root until 2026-10-23 (ids expanded in the container);
  fixed and tested on the worker. Serving still writes root-owned files into `mimo-cache`, so remove MiMo with a container `rm -rf` (MIMO.md).
- MiMo at temperature 0.6: quality and speed measured 2026-10-23, tool-call storms **not** (the recipe found 0.6 makes them worse).
- Stall episodes WERE captured live with `scripts/watch_stalls.sh` on 2026-10-15 (`results/stall-watch/`), but the two
  columns that would prove the mechanism (`prefill_recomputed`, `prefix_cache_hit_alloc`) read 0 at the time: vLLM
  serves big counters in scientific notation (`4.68835e+06`) and the row parser truncated them. Fixed and re-verified
  (3 windows of 2.4-3.6K recomputed tokens at 92-95% hit rate seen on a live 32K prefill). The stall cause is still
  unproven; the next capture should settle cached-admission vs recompute.
- C6 re-check (218.4 tok/s, 2026-10-15) is a single run; spread on this bench was previously measured at 6-15%.
- Prefix-cache results (cold/warm/competing/idle) are one run each, no repeat spread.

## Do-not-touch (ask the user first)
- Don't enable torch.compile / compilation modes or raise GMU above the recipe's 0.90. A memory overcommit hung
  the head once and it needed a hard power cycle (QWEN.md).
- Don't remove the tool-call cap, change `toolCap`, or point agents at the raw :8888 port.
- Don't change repo visibility, push secrets/site details, or rewrite history without asking.
- Don't modify the Qwen deployment repo on the head beyond what `configs/` documents.
- Don't take a speed measurement while agents are using the server, and don't conclude "it regressed" from one.

## Guardrails in effect
- GitHub: `nkavassalis/spark-tuning`, **PUBLIC** (the user's choice; normally repos are private). Account must be `nkavassalis` (`gh auth status`).
- Before any push: run the tests above (proxy unit tests, syntax checks, live `mimo/test_mimo.py` if the cluster
  is on MiMo) and the leak scan. No web UI, so no Playwright.
