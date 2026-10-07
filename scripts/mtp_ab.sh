#!/usr/bin/env bash
# A/B the MTP draft length (MTP_NUM_SPECULATIVE_TOKENS) with everything else fixed.
#
#   ./mtp_ab.sh --yes                 # tests K=2 and K=3 (default)
#   ./mtp_ab.sh --yes 1 2 3           # any list of K values
#   BENCH_ARGS="--contexts 1000,128000 --repeats 3" ./mtp_ab.sh --yes 2 3
#
# Each K = one full cluster relaunch (~11 min cold start) + the benchmark.
# THIS TAKES THE SERVER DOWN while it runs. At the end the original K is restored
# and the server is relaunched and health-checked.
#
# Results: ../results/mtp_ab-<timestamp>/K<k>.json + K<k>.txt
# Only MTP_NUM_SPECULATIVE_TOKENS in .env is touched; nothing else is changed.
set -euo pipefail

[ -f "$(dirname "$0")/../cluster.env" ] && . "$(dirname "$0")/../cluster.env"   # untracked; see cluster.env.example
HEAD="${HEAD:?set HEAD=user@head-host (or cluster.env)}"
REPO="${REPO:-Qwen3.8-Flash-Next-Dual-DGX-Sparks}"   # deployment repo path on the head (relative to the SSH home dir)
API="${API:-http://localhost:8000}"
BENCH_ARGS="${BENCH_ARGS:---tasks code,code_ts,edit,prose --contexts 1000,64000 --temps 0.0,0.6 --repeats 2}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[[ "${1:-}" == "--yes" ]] || { sed -n 2,14p "$0"; echo; echo "Refusing to run without --yes (it restarts the cluster)."; exit 1; }
shift
KS=("$@"); [[ ${#KS[@]} -gt 0 ]] || KS=(2 3)

OUT="$HERE/../results/mtp_ab-$(date +%Y%m%d-%H%M)"; mkdir -p "$OUT"
ORIG_K=$(ssh "$HEAD" "grep -oP '^MTP_NUM_SPECULATIVE_TOKENS=\K[0-9]+' $REPO/.env")
echo "original K=$ORIG_K ; testing K in: ${KS[*]} ; results -> $OUT"

set_k_and_launch() {
    local k=$1
    echo "[$(date +%T)] relaunching with MTP_NUM_SPECULATIVE_TOKENS=$k"
    ssh "$HEAD" "cd $REPO && sed -i -E 's/^MTP_NUM_SPECULATIVE_TOKENS=[0-9]+/MTP_NUM_SPECULATIVE_TOKENS=$k/' .env \
        && ./stop.sh >/dev/null 2>&1; nohup ./start.sh --launch > /tmp/mtp_ab-K$k.log 2>&1 < /dev/null &"
    sleep 60
    for _ in $(seq 1 120); do                      # up to 20 more minutes
        if curl -sf -m 5 "$API/health" >/dev/null; then echo "[$(date +%T)] healthy"; return 0; fi
        if ! ssh "$HEAD" 'docker ps -q -f name=vllm-fn | grep -q .'; then
            echo "container died - see /tmp/mtp_ab-K$k.log on head"; return 1; fi
        sleep 10
    done
    echo "timed out waiting for health"; return 1
}

restore() {
    echo "[$(date +%T)] restoring original K=$ORIG_K"
    set_k_and_launch "$ORIG_K" || echo "!!! restore relaunch FAILED - check the head manually"
}
trap restore EXIT

for k in "${KS[@]}"; do
    set_k_and_launch "$k"
    # shellcheck disable=SC2086
    python3 "$HERE/bench_decode.py" --url "$API" --label "K=$k" --json "$OUT/K$k.json" $BENCH_ARGS | tee "$OUT/K$k.txt"
done

python3 "$HERE/compare_results.py" "$OUT"/K*.json | tee "$OUT/summary.txt"
