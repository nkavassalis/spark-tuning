Project mode: REAL

# AGENTS.md: spark-tuning

## Purpose
Public notes, scripts and measured results from tuning LLM serving on a 2-node DGX Spark (GB10) cluster
(vLLM TP=2 over the CX7 200G link). The cluster serves one model at a time:
- Qwen3.8-Flash-Next (NVFP4 + FP8-dense, MTP K=2): `QWEN.md`, `FP8DENSE.md`. Tuned, currently stopped.
- MiMo-V2.6-Flash-RL (DFlash 7): `MIMO.md`. Set up, benchmarked, then **removed from both nodes** on 2026-10-07.
- Head-to-head: `QWEN_VS_MIMO.md`. **Current state: Qwen config C serving on :8000.** MiMo is not installed; `mimo/setup_nodes.sh && mimo/up.sh` rebuilds it (~30 min, 178 GB download).
Goal for both: fast, high-quality single-user code generation for coding agents, with no change that
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
| `cluster.env.example` | template for the untracked `cluster.env` (HEAD, WORKER, API, REPO, MIMO_RECIPE) |
| `scripts/` | Qwen-era benchmark tools; `bench_decode.py` works for any model (`--model`, `--url`) |
| `quality/` | quality-eval harness (runs on a node with docker; `API` env required) |
| `configs/` | diffs applied to the Qwen deployment repo (`env.diff`, `start.sh.patch`, FP8 overlays) |
| `mimo/lib.sh` | shared vars: pinned recipe commit, image, iface/HCA defaults, ports, `on`, `drop_caches` |
| `mimo/setup_nodes.sh` | recipe checkout + render `mimo.env` + weights (head download, rsync to worker) + patches |
| `mimo/up.sh` / `mimo/down.sh [--qwen]` | switch cluster to MiMo / stop MiMo (optionally relaunch Qwen) |
| `mimo/mimo.env.example` | template rendered onto nodes as `<recipe>/launch/mimo.env` (placeholders filled on the nodes) |
| `mimo/toolcap/` | tool-call-cap proxy (upstream copy + `upstream.diff`), its node tests, `routes.json` |
| `mimo/test_mimo.py` | functional checks against the live MiMo deployment |
| `results/` | raw outputs; `results/mimo/` for MiMo |

## How to run
```bash
cp cluster.env.example cluster.env   # fill in (never commit)
mimo/setup_nodes.sh                  # idempotent; ~178 GB weights (MiMo is currently NOT installed on the nodes)
mimo/up.sh                           # stops Qwen, starts MiMo (~13.5 min) + proxy
mimo/down.sh --qwen                  # back to Qwen
```
Endpoints: agents `http://<head>:8000/mimo/v1` (proxy); raw vLLM `http://<head>:8888/v1`; model `mimo-v2.6-flash`.
On the nodes: recipe checkout `~/mimo-recipe`, weights `~/models/MiMo-V2.6-Flash-RL`, cache `~/mimo-cache`,
proxy files and log `~/mimo-toolcap/` (container `mimo-toolcap`), vLLM container `vllm_mimo`.

## How to run the tests
```bash
node --test mimo/toolcap/*.test.cjs                    # 13 proxy unit tests, offline
set -a; . ./cluster.env; set +a; mimo/test_mimo.py     # live: chat, tool-call parse, storm guard
bash -n mimo/*.sh scripts/*.sh quality/*.sh            # syntax
python3 -m py_compile scripts/*.py quality/*.py mimo/*.py
```
Benchmarks (`scripts/bench_decode.py`, recipe `bench/mimobench.py` run on the head) are only valid when
nothing else uses the server.

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

## Conventions
- Every number in the docs is measured, with the result file next to it. Label anything unmeasured as unmeasured.
- Shell: `set -euo pipefail`, source `mimo/lib.sh`, use `on "$HEAD" ...`. Python: stdlib only.
- Dates in docs: YYYY-MM-DD. New experiments get a section in the model's .md plus a result file.

## Known issues / TODOs
- The proxy caps **streamed** responses only. Non-streaming requests can still storm.
- MiMo is 1.3–2.7× slower than Qwen config C for single-user code (both thinking off); the gap grows with context (MiMo step rate falls 10.6→7.1/s from 1K to 128K, Qwen flat ~25/s).
- MiMo quality: one run per test, so there's no noise floor for MiMo. Qwen's quality numbers are reused from 09-27/28.
- `quality_eval.py nll` per-token NLL is only comparable within one tokenizer; use `bpb` across models.
- `scripts/bench_decode.py --thinking {default,on,off}`: Qwen's template default is thinking ON, MiMo's server default OFF. Set it explicitly for cross-model comparisons.
- The recipe bench came in 6–15% under the recipe's published aggregate (single run).

## Testing limitations / unverified
- Storm guard verified only with a synthetic 12-parallel-call request, not a real long agent session.
- MiMo image/video/audio input, needle-in-a-haystack, 300K-context requests: not exercised here.
- No real pi agent session benchmark (task success / wall-clock) for either model.
- `mimo/up.sh`, `mimo/down.sh --qwen` and removing MiMo from the nodes were all exercised end to end on 2026-10-07.
- The cleanup chown step at the end of `mimo/setup_nodes.sh` has not been run (added during cleanup); verify it on the next provision.

## Do-not-touch (ask the user first)
- Don't enable torch.compile / compilation modes or raise GMU above the recipe's 0.90. A memory overcommit hung
  the head once and it needed a hard power cycle (QWEN.md).
- Don't remove the tool-call cap, change `toolCap`, or point agents at the raw :8888 port.
- Don't change repo visibility, push secrets/site details, or rewrite history without asking.
- Don't modify the Qwen deployment repo on the head beyond what `configs/` documents.

## Guardrails in effect
- GitHub: `nkavassalis/spark-tuning`, **PUBLIC** (the user's choice; normally repos are private). Account must be `nkavassalis` (`gh auth status`).
- Before any push: run the tests above (proxy unit tests, syntax checks, live `mimo/test_mimo.py` if the cluster
  is on MiMo) and the leak scan. No web UI, so no Playwright.
