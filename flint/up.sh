#!/usr/bin/env bash
# up.sh: serve Recursive Flint with vLLM at TP=1: NVFP4 GGUF on the head, BF16 on the worker (lib.sh), each on its node's
# :8000. Stops every other model (stop_other_models, scripts/models.sh; the other Flint variant is kept, it is on the other
# node), drops the node's page cache, starts the container, waits for the API.
#   flint/up.sh [nvfp4|bf16|both]      (default both)
# Flags: vLLM's Flint recipe (2026-10-22 blog: glm47 parsers, --exclude-tools-when-tool-choice-none; the GGUF folder, not the file
#   (a .gguf path would take vLLM's generic GGUF route; the Flint model reads its own GGUF, a lossless repack to NVFP4), config
# and tokenizer from the BF16 repo) + the cluster's rules: torch.compile off (mode 0, AGENTS.md), API key if set in cluster.env.
# Knobs (env): MAX_LEN (262144), MAX_SEQS (8), GMU (nvfp4 0.50, bf16 0.75: the worker also runs the quality harness),
#   CUDAGRAPH (FULL_DECODE_ONLY; NONE = eager).
# Endpoints afterwards: http://<head>:8000/v1 (flint-nvfp4), http://<worker>:8000/v1 (flint-bf16); both also answer to `flint`.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
VARIANTS=${1:-both}; [ "$VARIANTS" = both ] && VARIANTS="nvfp4 bf16"
MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-8}; CUDAGRAPH=${CUDAGRAPH:-FULL_DECODE_ONLY}

. "$ROOT/scripts/models.sh"
stop_other_models "$(cname_of nvfp4)" "$(cname_of bf16)"   # one Flint per node, nothing else

start() {   # start <variant>
  local v=$1 n c gmu model extra
  n=$(node_of "$v"); c=$(cname_of "$v")
  case $v in
    nvfp4) gmu=${GMU:-0.50}; model=/gguf; extra="--hf-config-path /model --tokenizer /model"
           on "$n" "[ -s ~/$GGUF_DIR/$GGUF_FILE ]" || { echo "GGUF missing on $n; run flint/setup_nodes.sh"; exit 1; } ;;
    bf16)  gmu=${GMU:-0.75}; model=/model; extra=""
           on "$n" "[ -s ~/$BF16_DIR/model-00015-of-00015.safetensors ]" || { echo "shards missing on $n; run flint/setup_nodes.sh"; exit 1; } ;;
  esac
  on "$n" "docker rm -f $c >/dev/null 2>&1 || true"
  drop_caches "$n"
  on "$n" "mkdir -p ~/$CACHE_NAME; rm -f ~/$CACHE_NAME/api_key"
  [ -n "${API_KEY:-}" ] && printf '%s' "$API_KEY" | on "$n" "umask 077; cat > ~/$CACHE_NAME/api_key"
  # The serve command goes to the node as a script so the JSON args survive quoting; the key is read inside the container.
  on "$n" "cat > ~/$CACHE_NAME/serve.sh" <<EOS
set -e
KEY=""; [ -s /root/.cache/api_key ] && KEY="--api-key \$(cat /root/.cache/api_key)"
exec vllm serve $model $extra --served-model-name $(served_of "$v") flint \\
  --host 0.0.0.0 --port $PORT --dtype bfloat16 \\
  --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS --gpu-memory-utilization $gmu \\
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 --exclude-tools-when-tool-choice-none \\
  --compilation-config '{"mode":0,"cudagraph_mode":"$CUDAGRAPH"}' \$KEY
EOS
  on "$n" "docker run -d --name $c --gpus all --network host --ipc host --shm-size 16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v \$HOME/$BF16_DIR:/model:ro -v \$HOME/$GGUF_DIR:/gguf:ro -v \$HOME/$CACHE_NAME:/root/.cache \
    -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    --entrypoint bash $IMAGE /root/.cache/serve.sh >/dev/null && echo 'started $c'"
}
for v in $VARIANTS; do start "$v"; done

AUTH=""; [ -n "${API_KEY:-}" ] && AUTH="-H 'Authorization: Bearer $API_KEY'"
for v in $VARIANTS; do
  n=$(node_of "$v"); c=$(cname_of "$v"); t0=$(date +%s)
  echo "== waiting for $c (docker logs -f $c to watch)"
  for i in $(seq 1 180); do
    if on "$n" "curl -sf -m 5 $AUTH localhost:$PORT/v1/models" 2>/dev/null | grep -q "$(served_of "$v")"; then
      echo "$c ready after $(( $(date +%s) - t0 )) s"; break; fi
    if ! on "$n" "docker ps -q -f name=^$c\$" | grep -q .; then
      echo "$c exited:"; on "$n" "docker logs --tail 60 $c 2>&1"; exit 1; fi
    [ "$i" = 180 ] && { echo "$c not ready after 30 min"; exit 1; }
    sleep 10
  done
  on "$n" "docker logs $c 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Model loading took' | tail -3"
  on "$n" "free -g | awk '/Mem:/{print \"$v node mem used/avail GiB:\", \$3, \$7}'"
done
