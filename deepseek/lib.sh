# Shared helpers for deepseek/*.sh. Sourced, not executed.
# DeepSeek-V4.1-Flash-1017 as Mia's AI Lab's EAQ v1 EXL3 2.9 bpw pack (196 GiB, 39 shards) on TensorFold's native Zig
# engine (BertholomusAI's dsv41 family), TP=2, one rank a Spark, served by MiaAI-Lab's recipe, pinned and unmodified.
# The recipe's only mode for the worker is NFS from the head (needs root; the head would also need ~530 GB). We use its
# documented "no NFS at all" mode instead: everything is downloaded and prepared on the worker (it has the disk), and the
# head gets only what rank 0 reads (small files + vision shard, Engram's tables, rank 0's weight file, kit, token map).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/cluster.env" ] && . "$ROOT/cluster.env"   # untracked; see cluster.env.example
: "${HEAD:?set HEAD=user@head-host (cluster.env)}" "${WORKER:?set WORKER=user@worker-host (cluster.env)}"
REPO=${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}
RECIPE_URL=https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git
RECIPE_COMMIT=${RECIPE_COMMIT:-27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6}   # 2026-10-24, reviewed 2026-10-28
RECIPE=dsv41-recipe                      # recipe checkout, relative to $HOME, both nodes
DS_HF=dsv41-hf                           # the recipe's HF_CACHE, relative to $HOME, both nodes (full on the worker, partial on the head)
# The recipe's own pins (scripts/config.sh at RECIPE_COMMIT); setup_nodes.sh checks they still match.
MODEL_ID=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw
MODEL_REV=45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b
ENGRAM_ID=deepseek-ai/DeepSeek-V4.1-Flash-1017
ENGRAM_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_SHARDS="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"   # 101.5 GB each
VISION_SHARD=model-00039-of-00039.safetensors   # the only checkpoint shard rank 0 reads at run time (recipe README)
KIT_SUB=tensorfold-dsv41/kit-abe07c7d
DATA_SUB=tensorfold-dsv41/${MODEL_REV:0:8}
SNAP=hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REV
ESNAP=hub/models--${ENGRAM_ID//\//--}/snapshots/$ENGRAM_REV
CNAME=dsv41-1017-exl3-tf                 # the recipe's container name, same on both nodes
SERVED_NAME=DeepSeek-V4.1-Flash-1017
CACHE_NAME=dsv41-cache                   # ~/dsv41-cache on the head: api_key (benchmark lock-out)
IFACE=${IFACE:-enp1s0f0np0}
PORT=${PORT:-8000}
on() { local n=$1; shift; ssh -o BatchMode=yes "$n" "$@"; }
link_ip() { on "$1" "ip -4 -o addr show $IFACE | awk '{print \$4}' | cut -d/ -f1"; }
# The engine image's tag, as the recipe computes it from its patches and pins (tensorfold-dsv41:exl3-tp2-<hash>).
image_tag() { on "$1" "cd ~/$RECIPE && bash -c 'source scripts/config.sh >/dev/null && echo \$IMAGE'"; }
# vm.drop_caches without sudo: privileged container (SSH user is in the docker group).
drop_caches() { on "$1" "sync; docker run --rm --privileged --entrypoint sh $(image_tag "$1") -c 'echo 3 > /proc/sys/vm/drop_caches' && echo 'page cache dropped'"; }
