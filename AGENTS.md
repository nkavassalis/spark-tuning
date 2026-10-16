Project mode: REAL

# AGENTS.md: spark-tuning

## Purpose
Public notes, scripts and measured results from tuning LLM serving on a 2-node DGX Spark (GB10) cluster
(vLLM TP=2 over the CX7 200G link). The cluster serves one model at a time:
- Qwen3.8-Flash-Next (NVFP4 + FP8-dense, MTP K=2): `QWEN.md`, `FP8DENSE.md`. Tuned; the production model.
- MiMo-V2.6-Flash-RL (DFlash 7): `MIMO.md`. Set up, benchmarked, then **removed from both nodes** on 2026-10-07.
- GLM-5.3-Flash, autotrust NAS-pruned NVFP4 builds E192 / E224 (NVFP4 MTP K=2): `GLM.md`. Benchmarked 2026-10-13:
  ties or beats Qwen C on every task benchmark (not on corpus NLL), 2–3× slower decode (~11.5 vs ~25.5 engine steps/s). E224 is the better GLM
  (same speed, 100% long context). **Removed from both nodes** on 2026-10-13 (checkpoints, image, caches); `GLM_VARIANT=E224 glm/setup_nodes.sh && GLM_VARIANT=E224 glm/up.sh` rebuilds it (158 GB download).
- nvidia Qwen3.8-Flash-Next-FP4-Local (QAD NVFP4 + MXFP8, community b12x vLLM build): `FP4LOCAL.md`. Benchmarked 2026-10-16 at TP=2:
  ties or beats config C on every quality test, but 5–14% slower with thinking on and ~20% less aggregate at 3–6 streams, so
  config C stayed live. Checkpoint + image still on both nodes: `LIVE=1 SPEC_TOKENS=2 fp4local/up.sh` makes it the live model.
- Head-to-head: `QWEN_VS_MIMO.md` (and the three-way tables in `GLM.md`). **Current state: Qwen config C serving on :8000.**
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
| `cluster.env.example` | template for the untracked `cluster.env` (HEAD, WORKER, API, REPO, MIMO_RECIPE) |
| `scripts/` | Qwen-era benchmark tools; `bench_decode.py` works for any model (`--model`, `--url`); `prefix_probe.py` measures prefix-cache behaviour; `watch_stalls.sh` is the 2 s stall sampler |
| `quality/` | quality-eval harness (runs on a node with docker; `API` env required) |
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
miss costs ~50 s of exclusive prefill. Staged there on 2026-10-15: pi `compaction.reserveTokens` 65536 /
`keepRecentTokens` 20000 (compacts above 196,608 instead of 245,760), applied and reversible with
`scripts/stage_pi_compaction.py --check|--revert`.

## Conventions
- Every number in the docs is measured, with the result file next to it. Label anything unmeasured as unmeasured.
- Shell: `set -euo pipefail`, source `mimo/lib.sh`, use `on "$HEAD" ...`. Python: stdlib only.
- Dates in docs: YYYY-MM-DD. New experiments get a section in the model's .md plus a result file.

## Known issues / TODOs
- **Qwen stalls with 3-4 concurrent agents** (high load, no tokens, ~10-50 s episodes). Cause not proven. Captured
  episodes (2026-10-15) show `gen_delta~0` with KV creeping and RoCE at 1.3-1.6 GB/s and **zero** memory pressure
  (PSI 0.00, swapout 0), so reclaim is ruled out for those; leading explanation is admission/prefill of huge
  mostly-cached prompts (pi resends the whole history: p50 ~54K, max ~172K tokens). See QWEN.md "Multi-agent stalls".
  Async scheduling is confirmed ON with MTP (nothing disables it); turning it off is the cheap A/B, and both that and
  GMU 0.80 need a restart.
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
- The cleanup chown step at the end of `mimo/setup_nodes.sh` has not been run (added during cleanup); verify it on the next provision.
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
