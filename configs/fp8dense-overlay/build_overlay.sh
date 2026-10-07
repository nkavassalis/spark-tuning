#!/usr/bin/env bash
# Reconstruct files/overlay/{model,hyperconnection,mtp,modelopt}.py for FP8_DENSE=true.
#
# Upstream (MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks @ d2f54b7) ships only the four .diff
# files; start.sh calls files/overlay/apply_patches.py, which was never committed, so
# FP8_DENSE=true dies at "Step 4c". Run this on the head, from the repo root, AFTER at least one
# normal launch (so files/modelopt_patched.py exists).
#
# modelopt.diff is applied on top of files/modelopt_patched.py (MXFP8 fallback + FP8_BLOCK_SCALES)
# rather than the image original, so those patches are kept. Pair with configs/start.sh.patch,
# which stops start.sh from ALSO mounting modelopt_patched.py on the same path (docker
# "duplicate mount point").
set -euo pipefail
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
P=/usr/local/lib/python3.12/dist-packages/vllm
cd files/overlay
c=$(docker create "$IMAGE" /bin/true)
for f in model hyperconnection mtp; do
  docker cp "$c:$P/models/qwen3_8_flash_next/nvidia/$f.py" "$f.py.orig"
done
docker rm "$c" >/dev/null
cp ../modelopt_patched.py modelopt.py.orig
for f in model hyperconnection mtp modelopt; do
  patch -s -o "$f.py" "$f.py.orig" < "$f.diff"
  echo "$f.py: $(grep -c 'fp8dense overlay' "$f.py") overlay markers"
done
python3 -m py_compile model.py hyperconnection.py mtp.py modelopt.py && echo "overlay OK"
