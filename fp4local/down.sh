#!/usr/bin/env bash
# down.sh: stop the FP4-Local server (both ranks).
#   fp4local/down.sh            # stop only
#   fp4local/down.sh --qwen     # stop, then relaunch the Qwen deployment (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
for n in "$HEAD" "$WORKER"; do on "$n" "docker rm -f $CNAME >/dev/null 2>&1 || true"; done
on "$HEAD" "rm -f ~/$CACHE_NAME/api_key"
echo "FP4-Local stopped"
if [ "${1:-}" = --qwen ]; then
  for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done
  on "$HEAD" "cd $REPO && ./start.sh --launch"
fi
