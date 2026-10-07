#!/usr/bin/env bash
# up.sh: switch the cluster to MiMo. Stops the Qwen deployment (if running), starts the MiMo worker
# then head (recipe launch/serve.sh), starts the tool-call-cap proxy on the head, waits until ready.
#   mimo/up.sh              # ~11-15 min cold start
# Endpoints afterwards:
#   agents:     http://<head>:8000/mimo/v1   (toolcap proxy: aborts a streamed response at the 7th tool call)
#   raw vLLM:   http://<head>:8888/v1        (no guard; benchmarks only)
set -euo pipefail
. "$(dirname "$0")/lib.sh"

if on "$HEAD" "docker ps -q -f name=^vllm-fn\$" | grep -q .; then
  echo "== stopping Qwen deployment (vllm-fn)"; on "$HEAD" "cd $REPO && ./stop.sh"
fi
on "$HEAD" "docker rm -f mimo-toolcap >/dev/null 2>&1 || true"

for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done
# serve.sh ends with `[ "$R" = 0 ] && echo ...`, so it exits 1 on the worker even on success:
# ignore its status and check the container instead.
echo "== worker (rank 1)"; on "$WORKER" "cd $MIMO_RECIPE && bash launch/serve.sh 1" | tail -1 || true
on "$WORKER" "docker ps -q -f name=^vllm_mimo\$" | grep -q . || { echo "worker container not running"; exit 1; }
echo "== head (rank 0)";   on "$HEAD"   "cd $MIMO_RECIPE && bash launch/serve.sh 0" | tail -2

echo "== toolcap proxy on head :$PROXY_PORT -> 127.0.0.1:$VLLM_PORT"
on "$HEAD" "mkdir -p mimo-toolcap"
tar -C "$ROOT/mimo/toolcap" -cf - toolcap-proxy.cjs routes.json | on "$HEAD" "tar -C mimo-toolcap -xf -"
on "$HEAD" "docker run -d --name mimo-toolcap --restart unless-stopped --network host --user \$(id -u):\$(id -g) \
  -v \$HOME/mimo-toolcap:/app -w /app -e TOOLCAP_HOST=0.0.0.0 -e IMAGE_CAP_PORT=$PROXY_PORT \
  $PROXY_IMAGE node toolcap-proxy.cjs >/dev/null && echo started"

echo "== waiting for vLLM (docker logs -f vllm_mimo on the head to watch)"
for i in $(seq 1 180); do
  if on "$HEAD" "curl -sf -m 5 localhost:$VLLM_PORT/v1/models" 2>/dev/null | grep -q mimo-v2.6-flash; then
    echo "ready after ~$((i*10)) s"; break; fi
  if ! on "$HEAD" "docker ps -q -f name=^vllm_mimo\$" | grep -q .; then
    echo "vllm_mimo exited:"; on "$HEAD" "docker logs --tail 40 vllm_mimo"; exit 1; fi
  sleep 10
done
on "$HEAD" "curl -sf -m 5 localhost:$PROXY_PORT/mimo/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print("via proxy:", d["id"], "max_model_len", d["max_model_len"])'
