#!/usr/bin/env bash
# up.sh: switch the cluster to nvidia/GLM-5.3-Flash-FP4-Local at TP=2. Stops whatever vLLM is serving (Qwen config C
# or FP4-Local), drops the page cache, starts the head, waits for its rendezvous port, starts the headless worker,
# waits for the API.
#   glmfp4/up.sh
# The vLLM env and flags are nvidia's 2 x DGX Spark recipe from the model card, verbatim, plus the CX7 NCCL settings
# every other deployment here uses. That includes torch.compile (VLLM_USE_AOT_COMPILE), which AGENTS.md otherwise
# forbids: same terms as fp4local/ (nvidia's recipe as is, everything else stopped first), approved by the user.
# Deviations from the card, all knobs: MAX_SEQS 8 (card: 4), so the C1-C6 bench and the quality harness's
# concurrency 8 run as for every other model; one RoCE HCA (the card lists both "twins"; ours has an IP on one only);
# --language-model-only (LM_ONLY=1): text only, no vision tower or encoder profiling. With the vision path the head
# dropped to <1.5 GiB available right after weight loading, twice, and the memory guard killed it (GLMFP4.md).
# If API_KEY is set (cluster.env), the head serves with --api-key and every client must send it (benchmark lock-out).
# Knobs (env): SPEC_TOKENS (5), DRAFT_MOE (humming), MAX_SEQS (8), MAX_BATCHED (4096), GMU (0.87), MAX_LEN (500000),
#   EP (0; 1 = --enable-expert-parallel), LM_ONLY (1), DEFAULT_EFFORT (unset = the template's max; low|high sets the
#   server-side default reasoning_effort via --default-chat-template-kwargs; a request's own value still wins).
set -euo pipefail
. "$(dirname "$0")/lib.sh"
SPEC_TOKENS=${SPEC_TOKENS:-5}; DRAFT_MOE=${DRAFT_MOE:-humming}; MAX_SEQS=${MAX_SEQS:-8}; MAX_BATCHED=${MAX_BATCHED:-4096}
GMU=${GMU:-0.87}; MAX_LEN=${MAX_LEN:-500000}; EP=${EP:-0}; LM_ONLY=${LM_ONLY:-1}
EPARG=""; [ "$EP" = 1 ] && EPARG="--enable-expert-parallel"
[ "$LM_ONLY" = 1 ] && EPARG="$EPARG --language-model-only"
[ -n "${DEFAULT_EFFORT:-}" ] && EPARG="$EPARG --default-chat-template-kwargs '{\"reasoning_effort\":\"$DEFAULT_EFFORT\"}'"
SPEC=""
[ "$SPEC_TOKENS" -gt 0 ] && SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS,\"moe_backend\":\"$DRAFT_MOE\",\"attention_backend\":\"B12X\"}'"

if on "$HEAD" "docker ps -q -f name=^vllm-fn\$" | grep -q .; then
  echo "== stopping Qwen deployment (vllm-fn)"; on "$HEAD" "cd $REPO && ./stop.sh"
fi
for n in "$HEAD" "$WORKER"; do on "$n" "docker rm -f $CNAME vllm_fp4local >/dev/null 2>&1 || true"; done
for n in "$HEAD" "$WORKER"; do
  on "$n" "docker ps -q | grep -q . && { echo 'refusing: other containers are running on $n'; docker ps; exit 1; } || true"
done
for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done

head_ip=$(link_ip "$HEAD")
snap=snapshots/$HF_REV
on "$HEAD" "[ -f ~/$HUB_DIR/$snap/config.json ]" || { echo "checkpoint missing on head; run glmfp4/setup_nodes.sh"; exit 1; }
on "$WORKER" "[ -f ~/$HUB_DIR/$snap/config.json ]" || { echo "checkpoint missing on worker; run glmfp4/setup_nodes.sh"; exit 1; }
on "$HEAD" "mkdir -p ~/$CACHE_NAME; rm -f ~/$CACHE_NAME/api_key"
if [ -n "${API_KEY:-}" ]; then
  printf '%s' "$API_KEY" | on "$HEAD" "umask 077; cat > ~/$CACHE_NAME/api_key"
fi

serve_script() {   # serve_script <rank> <extra args>; nvidia's DGX Spark env, verbatim
  cat <<EOS
set -e
export CUTE_DSL_ARCH="sm_121a"
export SAFETENSORS_FAST_GPU="1"
export VLLM_ENABLE_ROCE_ALLREDUCE="1"
export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_CTAS=8
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SSM_CONV_STATE_LAYOUT="DS"
export VLLM_USE_AOT_COMPILE="1"
export VLLM_USE_MEGA_AOT_ARTIFACT="1"
export VLLM_USE_V2_MODEL_RUNNER="1"
export VLLM_ENABLE_PCIE_ALLREDUCE="0"
export B12X_POLICY_MODE="auto"
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export B12X_W4A16_A4_PREFILL=1
KEY=""; [ -s /root/.cache/api_key ] && KEY="--api-key \$(cat /root/.cache/api_key)"
exec vllm serve /root/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/$snap \\
  --served-model-name $SERVED_NAME \\
  --tensor-parallel-size 2 --nnodes 2 --node-rank $1 \\
  --master-addr $head_ip --master-port $MASTER_PORT \\
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \\
  --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \\
  --attention-backend B12X --block-size 256 --moe-backend b12x --linear-backend b12x \\
  --no-enable-flashinfer-autotune --load-format b12x \\
  --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS --max-num-batched-tokens $MAX_BATCHED \\
  $SPEC \\
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \\
  --gpu-memory-utilization $GMU $EPARG \$KEY $2
EOS
}

start() {   # start <node> <rank> <extra args>
  local n=$1 rank=$2 extra=$3 ip; ip=$(link_ip "$n")
  serve_script "$rank" "$extra" | on "$n" "mkdir -p ~/$CACHE_NAME/{vllm,flashinfer,b12x,triton}; cat > ~/$CACHE_NAME/serve.sh"
  on "$n" "docker run -d --name $CNAME --privileged --gpus all --network host --ipc host --shm-size 16g \
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1 --ulimit stack=67108864 \
    -v \$HOME/.cache/huggingface:/root/.cache/huggingface:ro \
    -v \$HOME/$CACHE_NAME/vllm:/root/.cache/vllm -v \$HOME/$CACHE_NAME/flashinfer:/root/.cache/flashinfer \
    -v \$HOME/$CACHE_NAME/b12x:/root/.cache/b12x -v \$HOME/$CACHE_NAME/triton:/root/.triton \
    -v \$HOME/$CACHE_NAME/serve.sh:/root/.cache/serve.sh:ro \
    \$( [ -f \$HOME/$CACHE_NAME/api_key ] && echo -v \$HOME/$CACHE_NAME/api_key:/root/.cache/api_key:ro ) \
    -e HF_HUB_OFFLINE=1 -e VLLM_HOST_IP=$ip -e MN_IF_NAME=$IFACE -e UCX_NET_DEVICES=$IFACE -e TP_SOCKET_IFNAME=$IFACE \
    -e OMPI_MCA_btl_tcp_if_include=$IFACE \
    -e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE -e NCCL_IB_HCA==$HCA -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
    --entrypoint bash $IMAGE /root/.cache/serve.sh >/dev/null && echo started rank $rank"
}
echo "== head (rank 0)"; start "$HEAD" 0 "--host 0.0.0.0 --port $PORT"
for i in $(seq 1 60); do on "$HEAD" "timeout 1 bash -c '</dev/tcp/$head_ip/$MASTER_PORT'" 2>/dev/null && break; sleep 5; done
echo "== worker (rank 1)"; start "$WORKER" 1 "--headless"

echo "== waiting for the API (docker logs -f $CNAME on the head to watch)"
AUTH=""; [ -n "${API_KEY:-}" ] && AUTH="-H 'Authorization: Bearer $API_KEY'"
t0=$(date +%s); ready=0
for i in $(seq 1 360); do
  if on "$HEAD" "curl -sf -m 5 $AUTH localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "ready after $(( $(date +%s) - t0 )) s"; ready=1; break; fi
  for n in "$HEAD" "$WORKER"; do
    if ! on "$n" "docker ps -q -f name=^$CNAME\$" | grep -q .; then
      echo "$CNAME exited on $n:"; on "$n" "docker logs --tail 60 $CNAME 2>&1"; exit 1; fi
  done
  sleep 10
done
[ $ready = 1 ] || { echo "not ready after an hour"; exit 1; }
on "$HEAD" "docker logs $CNAME 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Model loading took' | tail -3"
for n in "$HEAD" "$WORKER"; do on "$n" "free -g | awk '/Mem:/{print \"$n mem used/avail GiB:\", \$3, \$7}'"; done
