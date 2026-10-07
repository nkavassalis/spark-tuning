#!/usr/bin/env bash
# down.sh: stop MiMo (both ranks) and the toolcap proxy.
#   mimo/down.sh            # stop only
#   mimo/down.sh --qwen     # stop, then relaunch the Qwen deployment (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
on "$HEAD" "docker rm -f mimo-toolcap vllm_mimo 2>/dev/null || true"
on "$WORKER" "docker rm -f vllm_mimo 2>/dev/null || true"
echo "MiMo stopped"
if [ "${1:-}" = --qwen ]; then
  for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done
  on "$HEAD" "cd $REPO && ./start.sh --launch"
fi
