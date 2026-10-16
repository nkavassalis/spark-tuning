# Shared helpers for fp4local/*.sh. Sourced, not executed.
# nvidia/Qwen3.8-Flash-Next-FP4-Local (QAD NVFP4 experts + MXFP8 attention, ~96 GiB) served TP=2 across both Sparks
# with the community b12x vLLM build that nvidia's model card requires.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/cluster.env" ] && . "$ROOT/cluster.env"   # untracked; see cluster.env.example
: "${HEAD:?set HEAD=user@head-host (cluster.env)}" "${WORKER:?set WORKER=user@worker-host (cluster.env)}"
REPO=${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}
HF_REPO=nvidia/Qwen3.8-Flash-Next-FP4-Local
HF_REV=${HF_REV:-b9ce18a6887e18c089fdea8509a33e63429a3300}   # tested revision (2026-10-15 upload)
HUB_DIR=.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local   # relative to $HOME, both nodes
SERVED_NAME=qwen3.8-flash-next-fp4local
# Pinned digest of eugr/spark-vllm-b12x:nightly-20261015 (vLLM 0.1.dev21830+g8244d1c44, torch 2.13.0+cu130)
IMAGE=${IMAGE:-eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595}
IFACE=${IFACE:-enp1s0f0np0}; HCA=${HCA:-rocep1s0f0}
PORT=${PORT:-8000}; MASTER_PORT=${MASTER_PORT:-29500}
CNAME=vllm_fp4local
CACHE_NAME=fp4local-cache                            # ~/fp4local-cache: vllm/flashinfer/b12x/triton caches, serve.sh, api_key
on() { local n=$1; shift; ssh -o BatchMode=yes "$n" "$@"; }
link_ip() { on "$1" "ip -4 -o addr show $IFACE | awk '{print \$4}' | cut -d/ -f1"; }
# vm.drop_caches without sudo: privileged container (SSH user is in the docker group).
drop_caches() { on "$1" "sync; docker run --rm --privileged --entrypoint sh $IMAGE -c 'echo 3 > /proc/sys/vm/drop_caches' && echo 'page cache dropped'"; }
