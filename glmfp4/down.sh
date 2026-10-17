#!/usr/bin/env bash
# down.sh: stop the GLM FP4-Local server (both ranks).
#   glmfp4/down.sh               # stop only
#   glmfp4/down.sh --fp4local    # stop, then bring back the live model: LIVE=1 SPEC_TOKENS=2 fp4local/up.sh (no API key)
#   glmfp4/down.sh --qwen        # stop, then relaunch the Qwen config C deployment (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
for n in "$HEAD" "$WORKER"; do on "$n" "docker rm -f $CNAME >/dev/null 2>&1 || true"; done
on "$HEAD" "rm -f ~/$CACHE_NAME/api_key"
echo "GLM FP4-Local stopped"
case "${1:-}" in
  --fp4local) LIVE=1 SPEC_TOKENS=2 "$ROOT/fp4local/up.sh" ;;
  --qwen) for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done; on "$HEAD" "cd $REPO && ./start.sh --launch" ;;
esac
