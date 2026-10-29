# Shared helpers for flint/*.sh. Sourced, not executed.
# Recursive Flint (39.8B MoE, ~2.5B active), single-GPU model: vLLM's Flint kernels are TP=1 only, so each variant runs on
# one Spark and the two run side by side, never two on one node (the overcommit rule, scripts/models.sh):
#   nvfp4: RecursiveSI/flint-nvfp4-gguf (one 22.9 GB GGUF) on the head, config + tokenizer from RecursiveSI/flint
#   bf16:  RecursiveSI/flint (79.6 GB, 15 shards) on the worker
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/cluster.env" ] && . "$ROOT/cluster.env"   # untracked; see cluster.env.example
: "${HEAD:?set HEAD=user@head-host (cluster.env)}" "${WORKER:?set WORKER=user@worker-host (cluster.env)}"
REPO=${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}
BF16_REPO=RecursiveSI/flint;            BF16_REV=2a247be0c0aec895ccb0ac7ca92229ec6df37288   # 2026-10-28
GGUF_REPO=RecursiveSI/flint-nvfp4-gguf; GGUF_REV=6ec43e19b565fc8d560e2d04985d7724f20673bf
GGUF_FILE=flint-nvfp4.gguf; GGUF_SHA=97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde
BF16_DIR=models/flint                   # relative to $HOME: config/tokenizer on both nodes, the shards on the worker only
GGUF_DIR=models/flint-nvfp4-gguf        # head only
# vLLM nightly (main at edad8b03, arm64, CUDA 13.0): Flint support merged 2026-10-22 after v0.32.0; its Flint kernels are
# gated to SM100, so GB10 (sm_121) runs the stock layers. The cu134 nightlies need a newer driver than the nodes have.
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:be65c903b77f61ff2c514ce9a4c98d21f1de1d09a13948f064677b8f1970a369}
PORT=${PORT:-8000}
CACHE_NAME=flint-cache                  # ~/flint-cache -> /root/.cache (vLLM/Triton/FlashInfer caches, serve.sh, api_key)
# variant -> node, container, served name
node_of()  { case $1 in nvfp4) echo "$HEAD" ;; bf16) echo "$WORKER" ;; *) echo "unknown variant $1" >&2; return 1 ;; esac; }
cname_of() { echo "vllm_flint_$1"; }
served_of() { echo "flint-$1"; }
on() { local n=$1; shift; ssh -o BatchMode=yes "$n" "$@"; }
# vm.drop_caches without sudo (SSH user is in the docker group): a privileged container.
drop_caches() { on "$1" "sync; docker run --rm --privileged --entrypoint sh $IMAGE -c 'echo 3 > /proc/sys/vm/drop_caches' && echo 'page cache dropped'"; }
