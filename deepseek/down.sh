#!/usr/bin/env bash
# down.sh: stop DeepSeek (both ranks, memguard) with the recipe's ./stop.sh, which saves each rank's log first.
#   deepseek/down.sh               # stop only
#   deepseek/down.sh --fp4local    # stop, then bring back the live model: LIVE=1 SPEC_TOKENS=2 fp4local/up.sh (no API key)
#   deepseek/down.sh --qwen        # stop, then relaunch the Qwen config C deployment (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
on "$HEAD" "cd ~/$RECIPE && ./stop.sh < /dev/null"
on "$HEAD" "rm -f ~/$CACHE_NAME/api_key"
echo "DeepSeek stopped"
case "${1:-}" in
  --fp4local) LIVE=1 SPEC_TOKENS=2 "$ROOT/fp4local/up.sh" ;;
  --qwen) for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done; on "$HEAD" "cd $REPO && ./start.sh --launch" ;;
esac
