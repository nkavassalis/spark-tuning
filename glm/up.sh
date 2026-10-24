#!/usr/bin/env bash
# up.sh: switch the cluster to GLM-5.3-Flash. Stops any other model (stop_other_models, scripts/models.sh), starts the head
# container, waits for its rendezvous port, starts the headless worker, waits until the API is ready.
#   GLM_VARIANT=E192 glm/up.sh
# Knobs (env): SPEC_TOKENS (MTP draft length, 0 = off; default 2), MAX_LEN (262144), MAX_SEQS (8),
#   KV_BYTES (fixed KV pool per node, default 20 GiB), GMU (startup free-memory gate, default 0.80),
#   CUDAGRAPH (FULL_DECODE_ONLY; NONE = eager).
# Endpoint afterwards: http://<head>:8000/v1, model glm53-flash-e192 / -e224. Logs: docker logs -f vllm_glm
set -euo pipefail
. "$(dirname "$0")/lib.sh"
SPEC_TOKENS=${SPEC_TOKENS:-2}; MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-8}
KV_BYTES=${KV_BYTES:-21474836480}; GMU=${GMU:-0.80}; CUDAGRAPH=${CUDAGRAPH:-FULL_DECODE_ONLY}

. "$ROOT/scripts/models.sh"
stop_other_models "$CNAME"   # never two models in memory at once
for n in "$HEAD" "$WORKER"; do on "$n" "docker rm -f $CNAME >/dev/null 2>&1 || true"; done
for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done

head_ip=$(link_ip "$HEAD")
# The serve command goes to the node as a script (~/glm-cache/serve.sh) so the JSON args survive quoting.
# torch.compile stays off (mode 0): a compile-time memory spike hung the head once (QWEN.md).
SPEC=""
[ "$SPEC_TOKENS" -gt 0 ] && SPEC="--speculative-config '{\"method\":\"mtp\",\"model\":\"/model/mtp-nvfp4\",\"num_speculative_tokens\":$SPEC_TOKENS}'"
serve_script() {   # serve_script <rank> <extra args>
  cat <<EOS
set -e
python3 /root/.cache/patch_vllm.py
exec vllm serve /model --served-model-name $SERVED_NAME --tensor-parallel-size 2 --nnodes 2 --node-rank $1 \\
  --master-addr $head_ip --master-port $MASTER_PORT \\
  --kv-cache-dtype fp8 --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS --max-num-batched-tokens 8192 \\
  --kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization $GMU --no-enable-flashinfer-autotune \\
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \\
  --compilation-config '{"mode":0,"cudagraph_mode":"$CUDAGRAPH"}' $SPEC $2
EOS
}

start() {   # start <node> <rank> <extra args>
  local n=$1 rank=$2 extra=$3 ip; ip=$(link_ip "$n")
  serve_script "$rank" "$extra" | on "$n" "cat > ~/$CACHE_NAME/serve.sh"
  on "$n" "docker run -d --name $CNAME --gpus all --network host --ipc host --shm-size 64g \
    --device /dev/infiniband --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    -v \$HOME/$MODEL_DIR:/model:ro -v \$HOME/$CACHE_NAME:/root/.cache \
    -e VLLM_HOST_IP=$ip -e VLLM_USE_DEEP_GEMM=0 -e HF_HUB_OFFLINE=1 -e MAX_JOBS=2 -e TOKENIZERS_PARALLELISM=false \
    -e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE -e NCCL_IB_HCA==$HCA -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
    --entrypoint bash $IMAGE /root/.cache/serve.sh >/dev/null && echo started rank $rank"
}
echo "== head (rank 0)"; start "$HEAD" 0 "--host 0.0.0.0 --port $PORT"
for i in $(seq 1 60); do on "$HEAD" "timeout 1 bash -c '</dev/tcp/$head_ip/$MASTER_PORT'" 2>/dev/null && break; sleep 5; done
echo "== worker (rank 1)"; start "$WORKER" 1 "--headless"

echo "== waiting for the API (docker logs -f $CNAME on the head to watch)"
t0=$(date +%s)
for i in $(seq 1 240); do
  if on "$HEAD" "curl -sf -m 5 localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "ready after $(( $(date +%s) - t0 )) s"; break; fi
  for n in "$HEAD" "$WORKER"; do
    if ! on "$n" "docker ps -q -f name=^$CNAME\$" | grep -q .; then
      echo "$CNAME exited on $n:"; on "$n" "docker logs --tail 60 $CNAME 2>&1"; exit 1; fi
  done
  sleep 10
done
on "$HEAD" "docker logs $CNAME 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Model loading took' | tail -3"
for n in "$HEAD" "$WORKER"; do on "$n" "free -g | awk '/Mem:/{print \"$n mem used/avail GiB:\", \$3, \$7}'"; done
