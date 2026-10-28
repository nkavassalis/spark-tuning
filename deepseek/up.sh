#!/usr/bin/env bash
# up.sh: switch the cluster to DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, TensorFold, TP=2). Stops every other model on both
# nodes, drops the page cache, then runs the recipe's own `./start.sh restart` on the head (it checks memory, starts rank 1
# on the worker and rank 0 on the head, waits for the API, smoke-tests it and starts memguard on both nodes).
#   deepseek/up.sh
# If API_KEY is set (cluster.env), the head serves with the recipe's API_KEY=1 and that key (benchmark lock-out).
# Knobs (env, passed to the recipe unchanged; defaults are the recipe's, scripts/config.sh): CONTEXT (1048576),
#   PARALLEL (4), THINKING (1), REASONING_EFFORT (template default), THINKING_BUDGET, MAX_TOKENS (32768),
#   TEMPERATURE / TOP_P / TOP_K (1.0 / 0.95 / 20), DRAFTS (1), POOL, KV_SHARD, SPILL_GIB.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
on "$HEAD" "[ -f ~/$RECIPE/scripts/local.sh ]" || { echo "not set up; run deepseek/setup_nodes.sh"; exit 1; }

. "$ROOT/scripts/models.sh"
stop_other_models "$CNAME"   # an old DeepSeek pair is left to the recipe's restart (stop.sh saves its logs)
for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done

on "$HEAD" "mkdir -p ~/$CACHE_NAME; rm -f ~/$CACHE_NAME/api_key"
lock=0
if [ -n "${API_KEY:-}" ]; then
  printf '%s\n' "$API_KEY" | on "$HEAD" "umask 077; cat > ~/$CACHE_NAME/api_key"; lock=1
fi
knobs="API_KEY=$lock"
for v in CONTEXT PARALLEL THINKING REASONING_EFFORT THINKING_BUDGET MAX_TOKENS TEMPERATURE TOP_P TOP_K DRAFTS POOL KV_SHARD SPILL_GIB; do
  [ -n "${!v:-}" ] && knobs="$knobs $v=${!v}"
done
echo "== head: ./start.sh restart ($knobs)"
on "$HEAD" "cd ~/$RECIPE && env $knobs ./start.sh restart < /dev/null"
