# Shared by every <model>/up.sh: the cluster holds one model at a time. Sourced after the model's lib.sh (needs on, HEAD,
# WORKER, REPO). Two models in unified memory at once is the overcommit that hung the head once (QWEN.md), so this lives in
# one place instead of a hand-written stop list per up.sh (glm/up.sh and fp4local/up.sh each missed a model, 2026-10-24).
# Adding a model: put its container name(s) here; scripts/check_repo.py fails until you do.
MODEL_CONTAINERS="vllm_fp4local vllm_glmfp4 vllm_glm vllm_mimo mimo-toolcap"   # config C (vllm-fn) is stopped via its repo

# stop_other_models <container to keep>: stop config C and every other model container on both nodes, then refuse to go on
# if anything else is still running (an unknown model must not end up sharing memory with this one).
stop_other_models() {
  local keep=$1 n c running
  if [ "$keep" != vllm-fn ] && on "$HEAD" "docker ps -q -f name=^vllm-fn\$" | grep -q .; then
    echo "== stopping Qwen config C (vllm-fn)"; on "$HEAD" "cd $REPO && ./stop.sh"
  fi
  for n in "$HEAD" "$WORKER"; do
    running=$(on "$n" "docker ps --format '{{.Names}}'")
    for c in $MODEL_CONTAINERS; do
      [ "$c" = "$keep" ] && continue
      if grep -qx "$c" <<<"$running"; then echo "== stopping $c on $n"; on "$n" "docker rm -f $c >/dev/null"; fi
    done
    # The quality harness (quality/*.sh, image spark-quality:*) runs on the worker and holds no model: let it be
    # (2026-10-24: a harness container for another server made mimo/up.sh refuse after config C was already stopped).
    running=$(on "$n" "docker ps --format '{{.Names}} {{.Image}}'" | awk '$2 !~ /^spark-quality:/ {print $1}' | grep -vx "$keep" || true)
    if [ -n "$running" ]; then echo "refusing: other containers still running on $n:"; echo "$running"; exit 1; fi
  done
}
