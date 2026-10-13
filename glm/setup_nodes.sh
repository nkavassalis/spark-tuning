#!/usr/bin/env bash
# setup_nodes.sh: prepare both Sparks for GLM-5.3-Flash (does NOT stop anything that is serving).
#   GLM_VARIANT=E192 glm/setup_nodes.sh      # or E224
#   1. head: hf download of the checkpoint (incl. the bundled NVFP4 MTP draft in mtp-nvfp4/)
#   2. head: enable the MTP code path in the target config (text_config.num_nextn_predict_layers 0 -> 1,
#      backup config.json.bak-nomtp). Inert without --speculative-config.
#   3. worker: rsync the checkpoint from the head over the CX7 link (same path on both nodes)
#   4. both: pull the vLLM image, copy patch_vllm.py to ~/glm-cache/
# Idempotent: hf download resumes, rsync only copies what is missing.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

echo "== head: download $HF_REPO${HF_REV:+@$HF_REV} -> ~/$MODEL_DIR (log ~/glm-download.log)"
on "$HEAD" "[ -x ~/.venv-hf/bin/hf ] || { python3 -m venv ~/.venv-hf && ~/.venv-hf/bin/pip install -q -U huggingface_hub; }
  ~/.venv-hf/bin/hf download $HF_REPO ${HF_REV:+--revision $HF_REV} --local-dir ~/$MODEL_DIR > ~/glm-download.log 2>&1 && tail -1 ~/glm-download.log"
on "$HEAD" "cd ~/$MODEL_DIR && [ -f config.json.bak-nomtp ] || cp config.json config.json.bak-nomtp
  python3 -c 'import json; c=json.load(open(\"config.json\")); c[\"text_config\"][\"num_nextn_predict_layers\"]=1; json.dump(c, open(\"config.json\",\"w\"), indent=2)'
  echo \"num_nextn_predict_layers: \$(grep -o '\"num_nextn_predict_layers\": [0-9]*' config.json)\""

worker_ip=$(link_ip "$WORKER")
echo "== worker: rsync ~/$MODEL_DIR from the head over $IFACE"
on "$WORKER" "mkdir -p ~/$MODEL_DIR"
on "$HEAD" "rsync -a --info=progress2 --no-inc-recursive ~/$MODEL_DIR/ $worker_ip:$MODEL_DIR/ --exclude .cache" | tr '\r' '\n' | tail -1

for n in "$HEAD" "$WORKER"; do
  echo "== $n: image + patch script"
  on "$n" "docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE; mkdir -p ~/$CACHE_NAME"
  on "$n" "cat > ~/$CACHE_NAME/patch_vllm.py" < "$ROOT/glm/patch_vllm.py"
done
echo "setup done. Next: GLM_VARIANT=$GLM_VARIANT glm/up.sh"
