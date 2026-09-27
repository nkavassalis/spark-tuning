#!/usr/bin/env bash
# Read-only health/config check of the dual-Spark vLLM deployment. Changes nothing.
#   ./check_cluster.sh
# Verifies: API health, served args, NCCL env on both nodes, that the control plane
# is on the 200G CX7 link, and that a request actually moves bytes over RoCE.
HEAD="${HEAD:-nick@10.1.13.99}"; WORKER="${WORKER:-nick@10.1.13.98}"
API="${API:-http://10.1.13.99:8000}"

echo "== API"; curl -sf -m 5 "$API/health" && echo "healthy" || echo "NOT healthy"
curl -s -m 5 "$API/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print("model", d["id"], "max_model_len", d["max_model_len"])' 2>/dev/null

for n in "$HEAD" "$WORKER"; do
    echo "== $n"
    ssh "$n" 'docker ps --format "{{.Names}} {{.Status}}" -f name=vllm-fn
      docker inspect vllm-fn --format "{{json .Config.Env}}" | tr , "\n" | grep -E "VLLM_HOST_IP|NCCL_(NET|IB_HCA|IB_ROCE|CUMEM|NVLS|SOCKET)" | tr -d "\"[]"
      docker inspect vllm-fn --format "{{join .Config.Cmd \" \"}}" | grep -oE -- "--(master-addr|max-model-len|speculative-config|compilation-config|performance-mode) [^ ]+"
      free -g | awk "/Mem:/{print \"mem used/avail GiB:\", \$3, \$7}"'
done

echo "== RoCE traffic during one 64-token request (should be non-zero)"
C=/sys/class/infiniband/rocep1s0f0/ports/1/counters/port_xmit_data
a=$(ssh "$HEAD" cat $C)
curl -s -m 60 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":64}' >/dev/null
b=$(ssh "$HEAD" cat $C)
echo "rocep1s0f0 xmit: $(( (b - a) * 4 / 1024 )) KiB"
