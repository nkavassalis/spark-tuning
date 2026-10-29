#!/usr/bin/env bash
# down.sh: stop both Flint variants (head NVFP4, worker BF16).
#   flint/down.sh               # stop only
#   flint/down.sh --fp4local    # stop, then bring back the live model: LIVE=1 SPEC_TOKENS=2 fp4local/up.sh (no API key)
#   flint/down.sh --qwen        # stop, then relaunch the Qwen config C deployment (./start.sh --launch, ~11 min)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
for v in nvfp4 bf16; do
  n=$(node_of "$v"); on "$n" "docker rm -f $(cname_of "$v") >/dev/null 2>&1 || true; rm -f ~/$CACHE_NAME/api_key"
done
echo "Flint stopped"
case "${1:-}" in
  --fp4local) LIVE=1 SPEC_TOKENS=2 "$ROOT/fp4local/up.sh" ;;
  --qwen) for n in "$WORKER" "$HEAD"; do drop_caches "$n"; done; on "$HEAD" "cd $REPO && ./start.sh --launch" ;;
esac
