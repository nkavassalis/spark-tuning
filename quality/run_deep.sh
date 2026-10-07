#!/usr/bin/env bash
# Thinking-mode + long-context quality suite.   ./run_deep.sh <label>
# Same container/sandbox arrangement as run_quality.sh. Output: ~/spark-quality/results/<label>/
set -euo pipefail
LABEL="${1:?usage: run_deep.sh <label>}"
W="$HOME/spark-quality"; OUT="$W/results/$LABEL"; mkdir -p "$OUT" "$W/cache" "$W/data"
API="${API:?set API=http://<head>:8000}"
IMG=spark-quality:evalplus-0.3.1
cd "$W"
RUN_NET=(docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/cache
         -e API="$API" -e MODEL="${MODEL:-qwen3.8-flash-next}" -e CONCURRENCY="${CONCURRENCY:-8}" -e LONGCTX_CONC="${LONGCTX_CONC:-4}"
         -e THINK_TEMP="${THINK_TEMP-0.6}" -e THINK_TOP_P="${THINK_TOP_P-0.95}" -e THINK_TOP_K="${THINK_TOP_K-20}"
         -v "$W":/work -v "$W/cache":/cache -v "$W/data":/data $IMG)
RUN_SANDBOX=(docker run --rm --network none --memory 6g --pids-limit 512 --cpus 8 -u "$(id -u):$(id -g)"
         -e HOME=/tmp -e XDG_CACHE_HOME=/cache -v "$W":/work:ro -v "$W/cache":/cache -v "$W/data":/data -v "$OUT":/out $IMG)
TASKS="${TASKS:-longctx humaneval lcb}"
has() { [[ " $TASKS " == *" $1 "* ]]; }

# one-time data prep (identical inputs for every config)
if [[ ! -s data/lcb80.jsonl ]]; then
  [[ -s data/lcb_test6.jsonl ]] || curl -sfL -o data/lcb_test6.jsonl \
    https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl
  "${RUN_SANDBOX[@]}" python3 deep_eval.py lcb-prep --src /data/lcb_test6.jsonl --out /data/lcb80.jsonl --n 80
fi
[[ -s data/longctx.jsonl ]] || "${RUN_SANDBOX[@]}" python3 deep_eval.py longctx-prep --out /data/longctx.jsonl

if has longctx; then
  echo "== [$LABEL] long context"
  "${RUN_NET[@]}" python3 deep_eval.py gen-longctx --data /data/longctx.jsonl --out "results/$LABEL/longctx.jsonl" | tee "$OUT/longctx_score.txt"
fi
if has humaneval; then
  echo "== [$LABEL] HumanEval+ thinking"
  "${RUN_NET[@]}" python3 deep_eval.py gen-think --task humaneval --max-tokens 16384 --out "results/$LABEL/he_think.jsonl"
  rm -f "$OUT/he_think_eval_results.json"
  "${RUN_SANDBOX[@]}" evalplus.evaluate --dataset humaneval --samples /out/he_think.jsonl --parallel 8 --i-just-wanna-run 2>&1 \
    | grep -E "pass@|base|plus" | tee "$OUT/he_think_score.txt"
fi
if has lcb; then
  echo "== [$LABEL] LiveCodeBench thinking"
  "${RUN_NET[@]}" python3 deep_eval.py gen-think --task lcb --data /data/lcb80.jsonl --max-tokens 16384 --out "results/$LABEL/lcb_think.jsonl"
  "${RUN_SANDBOX[@]}" python3 deep_eval.py lcb-exec --data /data/lcb80.jsonl --gen /out/lcb_think.jsonl | tee "$OUT/lcb_score.txt"
fi
echo "== [$LABEL] done -> $OUT"
