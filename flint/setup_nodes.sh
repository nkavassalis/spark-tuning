#!/usr/bin/env bash
# setup_nodes.sh: pull the vLLM image on both nodes and download Flint: config + tokenizer on both, the BF16 shards on the
# worker, the NVFP4 GGUF on the head. Does not stop anything: the downloads run in a container capped at DL_MEM (page cache
# included), so a node that serves a model keeps its memory. Every LFS file is checked against the sha256 in the HF tree.
# Idempotent: finished files are re-hashed and skipped, partial ones resumed (curl -C -).
#   flint/setup_nodes.sh
set -euo pipefail
. "$(dirname "$0")/lib.sh"
DL_MEM=${DL_MEM:-2g}
SMALL="config.json generation_config.json chat_template.jinja tokenizer_config.json model.safetensors.index.json LICENSE"

for n in "$HEAD" "$WORKER"; do
  echo "== $n: image"
  on "$n" "docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE"
done

# fetch <node> <repo> <rev> <dir> <file sha256|-> ...: one capped container per node, files fetched one at a time.
fetch() {
  local n=$1 repo=$2 rev=$3 dir=$4; shift 4
  local list; list=$(printf '%s\n' "$@")
  echo "== $n: $repo @ ${rev:0:8} -> ~/$dir ($(( $# )) files)"
  on "$n" "mkdir -p ~/$dir && docker run --rm -i --name flint-download --memory $DL_MEM --memory-swap $DL_MEM \
    --user \$(id -u):\$(id -g) -v \$HOME/$dir:/d -w /d --entrypoint bash $IMAGE -c '
      set -euo pipefail; c=0
      while read -r f sum; do c=\$((c+1))
        if [ -s \"\$f\" ] && { [ \"\$sum\" = - ] || [ \"\$(sha256sum \"\$f\" | cut -d\" \" -f1)\" = \"\$sum\" ]; }; then continue; fi
        curl -fsSL --retry 5 --retry-all-errors -C - -o \"\$f\" https://huggingface.co/$repo/resolve/$rev/\$f ||
          curl -fsSL --retry 5 -o \"\$f\" https://huggingface.co/$repo/resolve/$rev/\$f
        [ \"\$sum\" = - ] || [ \"\$(sha256sum \"\$f\" | cut -d\" \" -f1)\" = \"\$sum\" ] || { echo \"\$f: sha256 mismatch\"; rm -f \"\$f\"; exit 1; }
        echo \"  \$f ok\"
      done
      [ \$c = $# ] || { echo \"read \$c of $# file names\"; exit 1; }' <<<\"$list\""
}
small=(); for f in $SMALL; do small+=("$f -"); done
small+=("tokenizer.json a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68")
fetch "$HEAD" "$BF16_REPO" "$BF16_REV" "$BF16_DIR" "${small[@]}"
fetch "$HEAD" "$GGUF_REPO" "$GGUF_REV" "$GGUF_DIR" "$GGUF_FILE $GGUF_SHA"
# The shard names and hashes come from the HF tree API at BF16_REV (LFS oid = sha256 of the file).
shards=$(curl -fsSL "https://huggingface.co/api/models/$BF16_REPO/tree/$BF16_REV" | python3 -I -c '
import json, sys
for f in json.load(sys.stdin):
    if f["path"].endswith(".safetensors"): print(f["path"], f["lfs"]["oid"])')
[ "$(wc -l <<<"$shards")" = 15 ] || { echo "expected 15 shards, got: $shards"; exit 1; }
mapfile -t sh <<<"$shards"
fetch "$WORKER" "$BF16_REPO" "$BF16_REV" "$BF16_DIR" "${small[@]}" "${sh[@]}"
on "$HEAD" "du -sh ~/$BF16_DIR ~/$GGUF_DIR; df -h --output=avail ~ | tail -1"
on "$WORKER" "du -sh ~/$BF16_DIR; df -h --output=avail ~ | tail -1"
echo "setup done. Next (stops every other model): flint/up.sh [nvfp4|bf16|both]"
