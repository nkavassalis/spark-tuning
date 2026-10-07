#!/usr/bin/env bash
# setup_nodes.sh: prepare both Sparks for MiMo-V2.6-Flash-RL (does NOT stop anything that is serving).
#   1. clone the recipe at the pinned commit on both nodes
#   2. render mimo.env.example -> <recipe>/launch/mimo.env on both (site values discovered on the nodes)
#   3. head: recipe setup.sh (pull image, download ~178 GB weights, stage patches, audio libs)
#   4. worker: copy the weights from the head over the CX7 link (faster than a 2nd internet download),
#      then setup.sh with SKIP_DOWNLOAD=1
# Idempotent: hf download resumes, rsync only copies what is missing. Run from any machine with SSH to both.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

for n in "$HEAD" "$WORKER"; do
  echo "== $n: recipe @ ${RECIPE_COMMIT:0:7}"
  on "$n" "[ -d $MIMO_RECIPE/.git ] || git clone -q $RECIPE_URL $MIMO_RECIPE; cd $MIMO_RECIPE && git fetch -q origin && git checkout -q $RECIPE_COMMIT && git log -1 --oneline"
done

cidr=$(on "$HEAD" "ip -4 -o addr show $IFACE | awk '{print \$4}'")
[ -n "$cidr" ] || { echo "no IPv4 on $IFACE on the head"; exit 3; }
head_ip=${cidr%/*}
range=$(python3 -c "import ipaddress,sys;print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
for n in "$HEAD" "$WORKER"; do
  home=$(on "$n" 'echo $HOME')
  sed -e "s#__HEAD_LINK_IP__#$head_ip#; s#__HOME__#$home#g; s#__IFACE__#$IFACE#; s#__HCA__#$HCA#; s#__ADDR_RANGE__#$range#" \
    "$ROOT/mimo/mimo.env.example" | on "$n" "cat > $MIMO_RECIPE/launch/mimo.env"
  echo "== $n: wrote $MIMO_RECIPE/launch/mimo.env"
done

echo "== head: setup.sh (download can take a while; log: ~/mimo-setup.log)"
on "$HEAD" "command -v hf >/dev/null || [ -x ~/.venv-hf/bin/hf ] || { python3 -m venv ~/.venv-hf && ~/.venv-hf/bin/pip install -q -U huggingface_hub; }
  cd $MIMO_RECIPE && PATH=~/.venv-hf/bin:\$PATH bash setup.sh 2>&1 | tee ~/mimo-setup.log | grep -E '^==|ok|done|rror'"

worker_ip=$(on "$WORKER" "ip -4 -o addr show $IFACE | awk '{print \$4}' | cut -d/ -f1")
echo "== worker: rsync weights from head over $IFACE"
on "$WORKER" "mkdir -p models/MiMo-V2.6-Flash-RL"
on "$HEAD" "rsync -a --info=progress2 --no-inc-recursive models/MiMo-V2.6-Flash-RL/ $worker_ip:models/MiMo-V2.6-Flash-RL/ --exclude .cache" | tr '\r' '\n' | tail -1
on "$WORKER" "cd $MIMO_RECIPE && SKIP_DOWNLOAD=1 bash setup.sh 2>&1 | tee ~/mimo-setup.log | grep -E '^==|ok|done|rror'"
echo "setup done. Next: mimo/up.sh"
