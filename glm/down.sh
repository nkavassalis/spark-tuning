#!/usr/bin/env bash
# down.sh: stop GLM (both ranks).
#   glm/down.sh            # stop only
#   glm/down.sh --qwen     # stop, then relaunch the Qwen deployment (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
for n in "$HEAD" "$WORKER"; do on "$n" "docker rm -f $CNAME >/dev/null 2>&1 || true"; done
echo "GLM stopped"
if [ "${1:-}" = --qwen ]; then
  for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done
  on "$HEAD" "cd $REPO && ./start.sh --launch"
fi
