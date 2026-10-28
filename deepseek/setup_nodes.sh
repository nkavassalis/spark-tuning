#!/usr/bin/env bash
# setup_nodes.sh: prepare both Sparks for DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, TensorFold) without NFS.
# Does NOT stop anything that is serving: the memory-hungry steps run in capped containers (download DL_MEM 4g, image build
# BUILD_MEM 8g, weight files MEM_CAP 6g), so an overrun kills that container instead of freezing a Spark that also holds a
# live model.
#   1. both: the recipe checkout at RECIPE_COMMIT; its pins must equal lib.sh's
#   2. worker: hf download of the checkpoint (196 GiB) and Engram's two shards (~203 GB) into ~/dsv41-hf/hub
#   3. worker: the engine image (BUILD_MEM 8g, one CPU), built from the recipe's zig/Dockerfile with the recipe's build args (no published image
#      matches these pins); then copied to the head over the CX7 link
#   4. worker: the kit (RoPE tables, image routing bias, MANIFEST check), Engram's token map, both ranks' weight files
#      (~106 GB each), the served folder, Engram's tables hard-linked into rank 1's /engram folder: the recipe's
#      prepare.sh steps 4, 5 and 7, as its commands, with local paths instead of the NFS volume
#   5. head: rsync from the worker over the CX7 link of only what rank 0 reads (~320 GB), the served folder relinked
#   6. head: the recipe's scripts/local.sh (WORKER over the CX7 link, PREPARE=0, the worker's tree as a host path)
# Idempotent: hf download resumes, rsync copies what is missing, every step skips work that is done.
#   deepseek/setup_nodes.sh [--only N]
set -euo pipefail
. "$(dirname "$0")/lib.sh"
MEM_CAP=${MEM_CAP:-6g}; DL_MEM=${DL_MEM:-4g}
# The image build: the Dockerfile compiles its two executables in parallel (-j$(nproc)) and that was OOM-killed at 6g
# (2026-10-28). One CPU makes nproc 1 inside the build, so they compile one at a time; it does not change the image.
BUILD_MEM=${BUILD_MEM:-8g}; BUILD_CPUSET=${BUILD_CPUSET:-0}
DL_IMAGE=python:3.12-slim; HF_HUB_VERSION=2.3.0   # the hf CLI the other setup scripts use
ONLY=""; [ "${1:-}" = --only ] && ONLY=$2
step() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
worker_ip=$(link_ip "$WORKER"); user=${WORKER%@*}; worker_home=$(on "$WORKER" 'echo $HOME')

if step 1; then
  for n in "$HEAD" "$WORKER"; do
    echo "== $n: recipe @ ${RECIPE_COMMIT:0:7}"
    on "$n" "set -e; [ -d ~/$RECIPE/.git ] || git clone -q $RECIPE_URL ~/$RECIPE
      cd ~/$RECIPE; git fetch -q origin $RECIPE_COMMIT 2>/dev/null || git fetch -q origin
      git checkout -q --detach $RECIPE_COMMIT; test \"\$(git rev-parse HEAD)\" = $RECIPE_COMMIT
      git status --porcelain --untracked-files=no | grep -q . && { echo 'local changes in the recipe checkout'; exit 1; } || true"
    on "$n" "cd ~/$RECIPE && bash -c 'source scripts/config.sh >/dev/null; echo \$MODEL_ID \$MODEL_REVISION \$ENGRAM_ID \$ENGRAM_REVISION \$KIT_SUB \$DATA_SUB'" |
      grep -qx "$MODEL_ID $MODEL_REV $ENGRAM_ID $ENGRAM_REV $KIT_SUB $DATA_SUB" || { echo "recipe pins differ from deepseek/lib.sh"; exit 1; }
  done
fi

if step 2; then
  # hf 2.x (Xet) grew to 13.4 GB RSS on the worker next to a live model (2026-10-28), so the download runs in a detached
  # container capped at DL_MEM (container dsv41-download; stop_other_models refuses to start a model while it runs).
  echo "== worker: download in container dsv41-download, capped at $DL_MEM (docker logs dsv41-download)"
  inc=$(for f in model.safetensors.index.json config.json $ENGRAM_SHARDS; do printf -- '--include %s ' "$f"; done)
  on "$WORKER" "set -e; mkdir -p ~/$DS_HF/hub; rm -f ~/$DS_HF/.download.done
    docker ps -a --format '{{.Names}}' | grep -qx dsv41-download && docker rm -f dsv41-download >/dev/null
    docker run -d --name dsv41-download --memory $DL_MEM --memory-swap $DL_MEM --user \$(id -u):\$(id -g) -e HOME=/tmp \
      -v \$HOME/$DS_HF:/hf $DL_IMAGE sh -c 'pip install -q --user huggingface_hub==$HF_HUB_VERSION && PATH=/tmp/.local/bin:\$PATH &&
        hf download $MODEL_ID --revision $MODEL_REV --cache-dir /hf/hub &&
        hf download $ENGRAM_ID --revision $ENGRAM_REV --cache-dir /hf/hub $inc && touch /hf/.download.done' >/dev/null"
  while on "$WORKER" "docker ps -q -f name=^dsv41-download\$" | grep -q .; do sleep 60; done
  on "$WORKER" "[ -f ~/$DS_HF/.download.done ]" || { echo "download failed:"; on "$WORKER" "docker logs --tail 20 dsv41-download"; exit 1; }
  on "$WORKER" "docker rm dsv41-download >/dev/null; set -e; test -f ~/$DS_HF/$SNAP/model.safetensors.index.json
    for f in $ENGRAM_SHARDS; do s=\$(stat -L -c %s ~/$DS_HF/$ESNAP/\$f); case \$f in
      *47-of-*) [ \$s = 101535150936 ] ;; *48-of-*) [ \$s = 101537926640 ] ;; esac || { echo \"\$f: \$s bytes, not the 1017 release's\"; exit 1; }; done
    echo \"worker: checkpoint \$(du -shL ~/$DS_HF/$SNAP | cut -f1), Engram \$(du -shL ~/$DS_HF/$ESNAP | cut -f1)\""
fi

IMAGE=$(image_tag "$WORKER")
if step 3; then
  echo "== worker: image $IMAGE"
  if ! on "$WORKER" "docker image inspect $IMAGE >/dev/null 2>&1"; then
    # The recipe's own build (scripts/prepare.sh step 2): same context, same build args, the legacy builder so that
    # --memory caps every RUN step (BuildKit ignores it).
    on "$WORKER" "set -e; cd ~/$RECIPE; source scripts/config.sh >/dev/null
      docker image inspect \$BASE_IMAGE >/dev/null 2>&1 || docker pull -q \$BASE_IMAGE
      ctx=\$(mktemp -d ~/dsv41-logs/build.XXXXXX); mkdir -p \$ctx/zig
      cp -r zig/patches zig/patches-inputs \$ctx/zig/; cp zig/Dockerfile \$ctx/Dockerfile
      DOCKER_BUILDKIT=0 docker build --memory $BUILD_MEM --memory-swap $BUILD_MEM --cpuset-cpus $BUILD_CPUSET -t \$IMAGE -f \$ctx/Dockerfile \
        --build-arg BASE_IMAGE=\$BASE_IMAGE --build-arg TF_REPO=\$TF_REPO --build-arg TF_REF=\$TF_REF \
        --build-arg INPUTS_REF=\$INPUTS_REF --build-arg ZIG_VERSION=\$ZIG_VERSION --build-arg ZIG_SHA256=\$ZIG_SHA256 \
        --build-arg ZIG_MIRRORS=\"\$ZIG_MIRRORS\" --build-arg PATCHES_HASH=\$(image_hash) \$ctx > ~/dsv41-logs/build.log 2>&1 < /dev/null ||
        { rm -rf \$ctx; grep -E 'error|returned a non-zero' ~/dsv41-logs/build.log | tail -4; echo 'image build failed (~/dsv41-logs/build.log)'; exit 1; }
      rm -rf \$ctx; tail -2 ~/dsv41-logs/build.log"
  fi
  [ "$(on "$WORKER" "docker image inspect -f '{{index .Config.Labels \"tf.patches\"}}' $IMAGE")" = "${IMAGE##*-}" ] ||
    { echo "image label does not match the recipe's hash"; exit 1; }
  if ! on "$HEAD" "docker image inspect $IMAGE >/dev/null 2>&1"; then
    echo "== head: docker load from the worker over $IFACE"
    on "$HEAD" "ssh -o BatchMode=yes $user@$worker_ip docker save $IMAGE | docker load | tail -1"
  fi
fi

if step 4; then
  echo "== worker: kit, token map, weight files, served folder "
  # in_hf: the recipe's throwaway container over HF_CACHE (prepare.sh), plus the memory cap
  on "$WORKER" "set -euo pipefail; cd ~/$RECIPE; HF=\$HOME/$DS_HF
    in_hf() { docker run --rm --network none --memory $MEM_CAP --memory-swap $MEM_CAP --user \$(id -u):\$(id -g) -e HOME=/tmp -v \$HF:/hf \"\$@\"; }
    source scripts/config.sh >/dev/null
    KIT_DIR=\$HF/$KIT_SUB; KIT_IN=/hf/$KIT_SUB/kit; SNAP_C=/hf/$SNAP; TOOLS=/opt/tensorfold/tools/inputs; DATA=\$HF/$DATA_SUB
    if [ ! -d \$KIT_DIR/.git ]; then
      rm -rf \$KIT_DIR; git init -q \$KIT_DIR; git -C \$KIT_DIR remote add origin \$KIT_REPO
      git -C \$KIT_DIR fetch -q --depth 1 origin \$KIT_REF; git -C \$KIT_DIR checkout -q --detach FETCH_HEAD
    fi
    [ \"\$(git -C \$KIT_DIR rev-parse HEAD)\" = \$KIT_REF ]
    [ -s \$KIT_DIR/kit/rope-plain-cos.f32 ] && [ -s \$KIT_DIR/kit/rope-compressed-sin.f32 ] ||
      in_hf --entrypoint python3 $IMAGE \$TOOLS/make_rope.py \$SNAP_C/config.json \$KIT_IN >/dev/null
    [ -s \$KIT_DIR/kit/vision/gate_bias_vl.safetensors ] ||
      in_hf --entrypoint python3 $IMAGE \$TOOLS/make_bias_vl.py \$SNAP_C \$KIT_IN/vision/gate_bias_vl.safetensors >/dev/null
    bad=0; n=0
    while read -r sum path; do n=\$((n+1))
      [ \"\$(sha256sum \$KIT_DIR/kit/\$path 2>/dev/null | cut -d' ' -f1)\" = \$sum ] || [ \$path = vision/gate_bias_vl.safetensors ] ||
        { echo \"kit: \$path missing or differs\"; bad=1; }
    done < \$KIT_DIR/kit/MANIFEST
    [ \$bad = 0 ]; echo \"kit: \$n files checked against MANIFEST (the image bias is 1017's own)\"
    mkdir -p \$DATA/rank-cache
    [ -s \$DATA/dsv41_token_map.json ] ||
      in_hf --entrypoint python3 $IMAGE \$TOOLS/make_token_map.py \$SNAP_C /hf/$DATA_SUB/dsv41_token_map.json >/dev/null
    tm=\$(sha256sum \$DATA/dsv41_token_map.json | cut -d' ' -f1)
    [ \$tm = \$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"token_map_sha256\"])' \$KIT_DIR/kit/engram.json) ] &&
      echo 'token map: the bytes the kit was recorded with' || echo \"WARN token map \$tm differs from the kit's\"
    for r in 0 1; do
      want=\$(in_hf --entrypoint bash $IMAGE -c \"tf-dsv41-rank-cache \$SNAP_C /tmp/x \$r 2 --index-only | sha256sum | cut -c1-20\")
      if [ \$r = 0 ]; then out=\$DATA/rank-cache; out_c=/hf/$DATA_SUB/rank-cache; else out=\$HF/rank-cache-1; out_c=/hf/rank-cache-1; fi
      mkdir -p \$out
      if [ -f \$out/rank\${r}of2-\$want.bin ]; then echo \"weight file of rank \$r: ready\"; continue; fi
      find \$out -maxdepth 1 -name \"rank\${r}of2-*.bin\" -delete; rm -f \$out/.rank\${r}of2.partial
      echo \"weight file of rank \$r (~106 GB)\"
      in_hf --entrypoint tf-dsv41-rank-cache $IMAGE \$SNAP_C \$out_c \$r 2 >/dev/null
    done
    VIEW=\$DATA/model; rm -rf \$VIEW.new; mkdir -p \$VIEW.new
    for f in \$HF/$SNAP/*; do ln -s \"\$(realpath -s --relative-to=\$VIEW.new \$f)\" \$VIEW.new/; done
    rm -f \$VIEW.new/chat_template.jinja; cp model/chat_template.jinja \$VIEW.new/; rm -rf \$VIEW; mv \$VIEW.new \$VIEW
    E=\$HF/engram; mkdir -p \$E
    for f in model.safetensors.index.json config.json $ENGRAM_SHARDS; do [ -f \$E/\$f ] || ln \$(readlink -f \$HF/$ESNAP/\$f) \$E/\$f; done
    echo 'worker: ready'"
fi

if step 5; then
  echo "== head: rsync rank 0's files from the worker over $IFACE (~320 GB)"
  src="$user@$worker_ip:$DS_HF"
  on "$HEAD" "set -e; mkdir -p ~/$DS_HF/$SNAP ~/$DS_HF/engram ~/$DS_HF/$DATA_SUB/rank-cache ~/$DS_HF/$KIT_SUB
    r() { rsync -aL --partial --info=progress2 --no-inc-recursive \"\$@\" | tr '\r' '\n' | tail -1; }
    r --exclude 'model-000*' $src/$SNAP/ ~/$DS_HF/$SNAP/
    r $src/$SNAP/$VISION_SHARD ~/$DS_HF/$SNAP/
    r $src/engram/ ~/$DS_HF/engram/
    rsync -a $src/$KIT_SUB/ ~/$DS_HF/$KIT_SUB/
    r $src/$DATA_SUB/dsv41_token_map.json $src/$DATA_SUB/rank-cache ~/$DS_HF/$DATA_SUB/
    cd ~/$DS_HF/$DATA_SUB; rm -rf model.new; mkdir model.new
    for f in ~/$DS_HF/$SNAP/*; do ln -s \"\$(realpath -s --relative-to=model.new \$f)\" model.new/; done
    rm -f model.new/chat_template.jinja; cp ~/$RECIPE/model/chat_template.jinja model.new/; rm -rf model; mv model.new model
    du -sh ~/$DS_HF; df -h --output=avail ~ | tail -1"
fi

if step 6; then
  echo "== head: ~/$RECIPE/scripts/local.sh"
  # The worker's own full tree as a host path (NFS_VOLUME), PREPARE=0 (prepare.sh assumes the NFS layout), Engram's
  # tables on each node's disk (HEAD_ENGRAM_DIR, WORKER_ENGRAM_DIR), rank 1's weight file beside them.
  on "$HEAD" "cat > ~/$RECIPE/scripts/local.sh" <<EOF
# Written by spark-tuning deepseek/setup_nodes.sh: no NFS, each Spark reads its own copy.
WORKER=$user@$worker_ip
HF_CACHE=\$HOME/$DS_HF
NFS_VOLUME=$worker_home/$DS_HF
HEAD_ENGRAM_DIR=\$HOME/$DS_HF/engram
WORKER_ENGRAM_DIR='\$HOME/$DS_HF/engram'
WORKER_CACHE_DIR='\$HOME/$DS_HF/rank-cache-1'
PREPARE=0
PORT=$PORT
API_KEY_FILE=\$HOME/$CACHE_NAME/api_key
EOF
  on "$HEAD" "cd ~/$RECIPE && DRY_RUN=1 ./start.sh 2>&1 | tail -4"
fi
echo "setup done. Next (stops the live model): deepseek/up.sh"
