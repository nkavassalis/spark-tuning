# Shared helpers for glm/*.sh. Sourced, not executed.
# GLM-5.3-Flash NAS-pruned NVFP4 builds by autotrust (E192 = 192 of 288 experts kept, E224 = 224).
#   GLM_VARIANT=E192 (default) | E224
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/cluster.env" ] && . "$ROOT/cluster.env"   # untracked; see cluster.env.example
: "${HEAD:?set HEAD=user@head-host (cluster.env)}" "${WORKER:?set WORKER=user@worker-host (cluster.env)}"
REPO=${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}
GLM_VARIANT=${GLM_VARIANT:-E192}
HF_REPO=autotrust/GLM5.3-Flash-$GLM_VARIANT-DGX-Spark
HF_REV=${HF_REV:-}                                   # pin with HF_REV=<sha>; see GLM.md for the tested revisions
MODEL_DIR=models/GLM5.3-Flash-$GLM_VARIANT-DGX-Spark # relative to $HOME, same path on both nodes
SERVED_NAME=glm53-flash-$(echo "$GLM_VARIANT" | tr A-Z a-z)
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}             # stock release image; patches applied at start (patch_vllm.py)
IFACE=${IFACE:-enp1s0f0np0}; HCA=${HCA:-rocep1s0f0}
PORT=${PORT:-8000}; MASTER_PORT=${MASTER_PORT:-29500}
CNAME=vllm_glm
CACHE_NAME=glm-cache                                 # ~/glm-cache -> /root/.cache (vLLM + FlashInfer JIT caches)
on() { local n=$1; shift; ssh -o BatchMode=yes "$n" "$@"; }
link_ip() { on "$1" "ip -4 -o addr show $IFACE | awk '{print \$4}' | cut -d/ -f1"; }
# vm.drop_caches without sudo (SSH user is in the docker group): a privileged container.
# vLLM's startup check counts page cache as used memory on the unified-memory GB10.
drop_caches() { on "$1" "sync; docker run --rm --privileged --entrypoint sh $IMAGE -c 'echo 3 > /proc/sys/vm/drop_caches' && echo 'page cache dropped'"; }
