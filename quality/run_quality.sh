#!/usr/bin/env bash
# Run the full quality suite against the currently served config.   ./run_quality.sh <label>
# Runs on the WORKER node (CPU only; the GPUs are busy serving). Needs docker + internet once for setup.
#   generation : container with host network (talks to the API only)
#   execution  : container with --network none, memory/pids limits (runs model-written code)
# Output: ~/spark-quality/results/<label>/
set -euo pipefail
LABEL="${1:?usage: run_quality.sh <label>}"
W="$HOME/spark-quality"; OUT="$W/results/$LABEL"; mkdir -p "$OUT" "$W/cache" "$W/data"
API="${API:?set API=http://<head>:8000}"
IMG=spark-quality:evalplus-0.3.1
cd "$W"
docker image inspect $IMG >/dev/null 2>&1 || docker build -q -t $IMG -f Dockerfile . >/dev/null
[[ -s data/gsm8k_test.jsonl ]] || curl -sfL -o data/gsm8k_test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
RUN_NET=(docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/cache
         -e API="$API" -e MODEL="${MODEL:-qwen3.8-flash-next}" -e CONCURRENCY="${CONCURRENCY:-8}" -e GSM8K=/data/gsm8k_test.jsonl
         -v "$W":/work -v "$W/cache":/cache -v "$W/data":/data $IMG)
RUN_SANDBOX=(docker run --rm --network none --memory 6g --pids-limit 512 --cpus 8 -u "$(id -u):$(id -g)"
         -e HOME=/tmp -e XDG_CACHE_HOME=/cache -v "$W/cache":/cache -v "$OUT":/out $IMG)
[[ -s data/corpus.jsonl ]] || "${RUN_NET[@]}" python3 quality_eval.py corpus /data/corpus.jsonl
# pre-fetch datasets (network) so the sandbox never needs it
"${RUN_NET[@]}" python3 -c "from evalplus.data import get_human_eval_plus as h, get_mbpp_plus as m; print(len(h()), len(m()))"

TASKS="${TASKS:-nll humaneval mbpp gsm8k}"      # e.g. TASKS="humaneval mbpp" for a noise-floor rerun
has() { [[ " $TASKS " == *" $1 "* ]]; }
if has nll; then
  echo "== [$LABEL] NLL"; "${RUN_NET[@]}" python3 quality_eval.py nll --corpus /data/corpus.jsonl --out "results/$LABEL/nll.jsonl" | tee "$OUT/nll.txt"
fi
for t in humaneval mbpp gsm8k; do
  has $t || continue
  echo "== [$LABEL] gen $t"; "${RUN_NET[@]}" python3 quality_eval.py gen --task $t --out "results/$LABEL/$t.jsonl"
done
echo "== [$LABEL] execute (sandboxed)"
for t in humaneval mbpp; do
  has $t || continue
  rm -f "$OUT/${t}_eval_results.json"
  "${RUN_SANDBOX[@]}" evalplus.evaluate --dataset $t --samples "/out/$t.jsonl" --parallel 8 --i-just-wanna-run 2>&1 \
    | grep -E "pass@|base|plus" | tee "$OUT/${t}_score.txt"
done
has gsm8k && "${RUN_NET[@]}" python3 quality_eval.py gsm8k-score "results/$LABEL/gsm8k.jsonl" | tee "$OUT/gsm8k_score.txt"
echo "== [$LABEL] done -> $OUT"
