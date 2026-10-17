# Shared helpers for glmfp4/*.sh. Sourced, not executed.
# nvidia/GLM-5.3-Flash-FP4-Local (QAD NVFP4 experts + MXFP8 attention, all 288 experts, ~166 GiB) served TP=2 across both
# Sparks with the community b12x vLLM build that nvidia's model card requires (same image as fp4local/).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/cluster.env" ] && . "$ROOT/cluster.env"   # untracked; see cluster.env.example
: "${HEAD:?set HEAD=user@head-host (cluster.env)}" "${WORKER:?set WORKER=user@worker-host (cluster.env)}"
REPO=${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}
HF_REPO=nvidia/GLM-5.3-Flash-FP4-Local
HF_REV=${HF_REV:-76348ddb2be2e35e39ecfc00e75dc07703f4f3d5}   # tested revision
HUB_DIR=.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local   # relative to $HOME, both nodes
SERVED_NAME=glm53-flash-fp4local
# Pinned digest of eugr/spark-vllm-b12x:nightly-20261015 (= :latest on 2026-10-17; vLLM 0.1.dev21830+g8244d1c44)
IMAGE=${IMAGE:-eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595}
IFACE=${IFACE:-enp1s0f0np0}; HCA=${HCA:-rocep1s0f0}
PORT=${PORT:-8000}; MASTER_PORT=${MASTER_PORT:-29501}
CNAME=vllm_glmfp4
CACHE_NAME=glmfp4-cache                              # ~/glmfp4-cache: vllm/flashinfer/b12x/triton caches, serve.sh, api_key
on() { local n=$1; shift; ssh -o BatchMode=yes "$n" "$@"; }
link_ip() { on "$1" "ip -4 -o addr show $IFACE | awk '{print \$4}' | cut -d/ -f1"; }
# vm.drop_caches without sudo: privileged container (SSH user is in the docker group).
drop_caches() { on "$1" "sync; docker run --rm --privileged --entrypoint sh $IMAGE -c 'echo 3 > /proc/sys/vm/drop_caches' && echo 'page cache dropped'"; }
