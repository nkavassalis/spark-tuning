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
    *"name=^dsv41-1017-exl3-tf"*) for c in ${RUN[$n]}; do [ "$c" = dsv41-1017-exl3-tf ] && echo id; done; true ;;
    *"dsv41-recipe && ./stop.sh"*) for c in h w; do RUN[$c]=$(tr ' ' '\n' <<<"${RUN[$c]}" | grep -vx dsv41-1017-exl3-tf | tr '\n' ' '); done ;;
    *"name=^vllm-fn"*) for c in ${RUN[$n]}; do [ "$c" = vllm-fn ] && echo id; done; true ;;
    *"./stop.sh"*) RUN[$n]=$(tr ' ' '\n' <<<"${RUN[$n]}" | grep -vx vllm-fn | tr '\n' ' ') ;;
    *"docker ps --format '{{.Names}} {{.Image}}'"*)   # harness-* containers run the quality image
      for c in ${RUN[$n]}; do case $c in harness*) echo "$c spark-quality:evalplus-0.3.1" ;; *) echo "$c some/image:tag" ;; esac; done; true ;;
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
check "fp4local up over deepseek (stop.sh)" vllm_fp4local "dsv41-1017-exl3-tf" "dsv41-1017-exl3-tf" ok "" ""
check "deepseek up keeps its own pair"     dsv41-1017-exl3-tf "vllm_fp4local dsv41-1017-exl3-tf" "dsv41-1017-exl3-tf" ok "dsv41-1017-exl3-tf" "dsv41-1017-exl3-tf"
check "ignores a quality-harness container" vllm_mimo    "vllm_fp4local" "vllm_fp4local harness_x" ok "" "harness_x"
check "flint keeps both variants (one per node)" "vllm_flint_nvfp4 vllm_flint_bf16" "vllm_flint_nvfp4 vllm_fp4local" "vllm_flint_bf16 vllm_fp4local" ok "vllm_flint_nvfp4" "vllm_flint_bf16"
check "fp4local up over both flints"       vllm_fp4local "vllm_flint_nvfp4" "vllm_flint_bf16" ok "" ""
check "flint up over deepseek"             "vllm_flint_nvfp4 vllm_flint_bf16" "dsv41-1017-exl3-tf" "dsv41-1017-exl3-tf harness_y" ok "" "harness_y"
exit $fail
