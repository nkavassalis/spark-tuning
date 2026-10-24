#!/usr/bin/env bash
# down.sh: stop GLM (both ranks).
#   glm/down.sh            # stop only
#   glm/down.sh --fp4local # stop, then bring back the live model: LIVE=1 SPEC_TOKENS=2 fp4local/up.sh (no API key)
#   glm/down.sh --qwen     # stop, then relaunch Qwen config C (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
for n in "$HEAD" "$WORKER"; do on "$n" "docker rm -f $CNAME >/dev/null 2>&1 || true"; done
echo "GLM stopped"
if [ "${1:-}" = --qwen ]; then
  for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done
  on "$HEAD" "cd $REPO && ./start.sh --launch"
fi
if [ "${1:-}" = --fp4local ]; then LIVE=1 SPEC_TOKENS=2 "$ROOT/fp4local/up.sh"; fi
