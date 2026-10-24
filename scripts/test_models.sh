#!/usr/bin/env bash
# Offline test of scripts/models.sh stop_other_models: a fake `on` simulates both nodes' running containers.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/models.sh"
HEAD=h; WORKER=w; REPO=repo
declare -A RUN   # RUN[node]="space-separated running containers"
on() {   # fake ssh: understands the three commands stop_other_models sends
  local n=$1 cmd=$2 c
  case "$cmd" in
    *"name=^vllm-fn"*) for c in ${RUN[$n]}; do [ "$c" = vllm-fn ] && echo id; done; true ;;
    *"./stop.sh"*) RUN[$n]=$(tr ' ' '\n' <<<"${RUN[$n]}" | grep -vx vllm-fn | tr '\n' ' ') ;;
    *"docker ps --format"*) tr ' ' '\n' <<<"${RUN[$n]}" | grep . || true ;;
    *"docker rm -f "*) c=${cmd#docker rm -f }; c=${c%% *}; RUN[$n]=$(tr ' ' '\n' <<<"${RUN[$n]}" | grep -vx "$c" | tr '\n' ' ') ;;
    *) echo "unexpected: $cmd" >&2; return 1 ;;
  esac
}
fail=0
check() {   # check <name> <keep> <head before> <worker before> <expect: ok|refuse> <head after> <worker after>
  local out rc=0; RUN[h]=$3; RUN[w]=$4
  out=$(stop_other_models "$2"; declare -p RUN) || rc=$?
  eval "$(grep '^declare -A RUN' <<<"$out" || true)"
  local got=ok; [ $rc -ne 0 ] && got=refuse
  local ha wa; ha=$(echo ${RUN[h]}); wa=$(echo ${RUN[w]})
  if [ "$got" = "$5" ] && { [ "$got" = refuse ] || { [ "$ha" = "$6" ] && [ "$wa" = "$7" ]; }; }; then echo "ok   $1"
  else echo "FAIL $1: got $got head='$ha' worker='$wa'"; fail=1; fi
}
check "fp4local up over mimo + proxy"      vllm_fp4local "vllm_mimo mimo-toolcap" "vllm_mimo" ok "" ""
check "glm up over the live fp4local"      vllm_glm      "vllm_fp4local" "vllm_fp4local" ok "" ""
check "mimo up over config C"              vllm_mimo     "vllm-fn" "" ok "" ""
check "keeps its own container"            vllm_glm      "vllm_glm vllm_glmfp4" "vllm_glm" ok "vllm_glm" "vllm_glm"
check "idle cluster"                       vllm_fp4local "" "" ok "" ""
check "refuses an unknown container"       vllm_fp4local "vllm_newmodel" "" refuse "" ""
check "refuses unknown on the worker only" vllm_glm      "vllm_fp4local" "something_else" refuse "" ""
check "empty keep stops every model"       ""            "vllm_glmfp4 vllm_fp4local" "vllm_glmfp4" ok "" ""
exit $fail
