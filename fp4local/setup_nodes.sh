#!/usr/bin/env bash
# setup_nodes.sh: prepare both Sparks for nvidia/Qwen3.8-Flash-Next-FP4-Local (does NOT stop anything that is serving).
#   1. head: hf download into the HF cache at the pinned revision (~102 GB)
#   2. worker: rsync the HF cache entry from the head over the CX7 link
#   3. both: pull the pinned b12x image, create ~/fp4local-cache
# Idempotent: hf download resumes, rsync only copies what is missing.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

echo "== head: download $HF_REPO@${HF_REV:0:7} (log ~/fp4local-download.log)"
on "$HEAD" "[ -x ~/.venv-hf/bin/hf ] || { python3 -m venv ~/.venv-hf && ~/.venv-hf/bin/pip install -q -U huggingface_hub; }
  ~/.venv-hf/bin/hf download $HF_REPO --revision $HF_REV > ~/fp4local-download.log 2>&1 < /dev/null && tail -1 ~/fp4local-download.log"
worker_ip=$(link_ip "$WORKER")
echo "== worker: rsync ~/$HUB_DIR from the head over $IFACE"
# huggingface_hub >= 2 keeps file contents in a hub-wide store (~/.cache/huggingface/hub/blobs/xx/<sha256>) behind
# relative symlinks from models--*/blobs, so both trees go over. rsync keeps the links relative; -H keeps hard links.
on "$WORKER" "mkdir -p ~/$HUB_DIR ~/.cache/huggingface/hub/blobs"
on "$HEAD" "[ -d ~/.cache/huggingface/hub/blobs ] && rsync -aH --info=progress2 --no-inc-recursive ~/.cache/huggingface/hub/blobs/ $worker_ip:.cache/huggingface/hub/blobs/" | tr '\r' '\n' | tail -1
on "$HEAD" "rsync -aH ~/$HUB_DIR/ $worker_ip:$HUB_DIR/"
on "$WORKER" "test -r \$(readlink -f ~/$HUB_DIR/snapshots/$HF_REV/tensors/model-00001-of-00021.safetensors) && echo 'worker: snapshot resolves'"
for n in "$HEAD" "$WORKER"; do
  echo "== $n: image + cache dir"
  on "$n" "docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE; mkdir -p ~/$CACHE_NAME/{vllm,flashinfer,b12x,triton}"
done
echo "setup done. Next: fp4local/up.sh"
