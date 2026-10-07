# Shared helpers for mimo/*.sh. Sourced, not executed.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/cluster.env" ] && . "$ROOT/cluster.env"   # untracked; see cluster.env.example
: "${HEAD:?set HEAD=user@head-host (cluster.env)}" "${WORKER:?set WORKER=user@worker-host (cluster.env)}"
MIMO_RECIPE=${MIMO_RECIPE:-mimo-recipe}
REPO=${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}
RECIPE_URL=https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe
RECIPE_COMMIT=${RECIPE_COMMIT:-9c699a2d027c2bf072d130f935acdbeacb06f208}   # pinned; bump deliberately
IMAGE=${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}
IFACE=${IFACE:-enp1s0f0np0}; HCA=${HCA:-rocep1s0f0}
PROXY_PORT=${PROXY_PORT:-8000}; VLLM_PORT=${VLLM_PORT:-8888}
PROXY_IMAGE=${PROXY_IMAGE:-node:22-alpine}
on() { local n=$1; shift; ssh -o BatchMode=yes "$n" "$@"; }
# vm.drop_caches without sudo: the SSH user is in the docker group, so use a privileged container.
# Needed because vLLM's startup probe at GMU 0.90 does not count reclaimable page cache.
drop_caches() { on "$1" "sync; docker run --rm --privileged --entrypoint sh $IMAGE -c 'echo 3 > /proc/sys/vm/drop_caches' && echo 'page cache dropped'"; }
