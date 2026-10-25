#!/usr/bin/env bash
# Time to solve, one problem at a time (CONCURRENCY=1), 16K budget (THINK_MAX_TOKENS overrides).   ./run_tts.sh <label>   (on the worker, from ~/spark-quality)
# HE+ every 4th problem (41), LCB every 8th (10): the subsets in COMPARISON.md "Time to solve". Wall clock per request.
# Env: API (required), MODEL, API_KEY (optional), THINK_KWARGS / THINK_TEMP / THINK_TOP_P / THINK_TOP_K as in run_deep.sh.
# Thinking off: THINK_KWARGS='{"chat_template_kwargs":{"enable_thinking":false}}'. Nothing else may use the server meanwhile.
set -uo pipefail
L="${1:?usage: run_tts.sh <label>}"
W="$HOME/spark-quality"; OUT="$W/results/$L"; mkdir -p "$OUT"; cd "$W"
IMG=spark-quality:evalplus-0.3.1
NET=(docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/cache
     -e API="${API:?set API=http://<head>:8000}" -e API_KEY="${API_KEY:-}" -e MODEL="${MODEL:-qwen3.8-flash-next}" -e CONCURRENCY=1
     -e THINK_KWARGS="${THINK_KWARGS:-}" -e THINK_TEMP="${THINK_TEMP-0.6}" -e THINK_TOP_P="${THINK_TOP_P-0.95}" -e THINK_TOP_K="${THINK_TOP_K-20}"
     -v "$W":/work -v "$W/cache":/cache -v "$W/data":/data $IMG)
BOX=(docker run --rm --network none --memory 6g --pids-limit 512 --cpus 8 -u "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/cache
     -v "$W":/work:ro -v "$W/cache":/cache -v "$W/data":/data -v "$OUT":/out $IMG)
echo "=== $(date -u +%T) $L HE+ (every 4th)"
"${NET[@]}" python3 deep_eval.py gen-think --task humaneval --stride 4 --max-tokens "${THINK_MAX_TOKENS:-16384}" --out "results/$L/he_tts.jsonl"
"${NET[@]}" python3 tts_score.py pad "results/$L/he_tts.jsonl" "results/$L/he_tts_padded.jsonl"
rm -f "$OUT/he_tts_padded_eval_results.json"
"${BOX[@]}" evalplus.evaluate --dataset humaneval --samples /out/he_tts_padded.jsonl --parallel 8 --i-just-wanna-run >/dev/null 2>&1
"${NET[@]}" python3 tts_score.py report "results/$L/he_tts.jsonl" "results/$L/he_tts_padded_eval_results.json" | tee "$OUT/he_tts_score.txt"
echo "=== $(date -u +%T) $L LCB (every 8th)"
"${NET[@]}" python3 deep_eval.py gen-think --task lcb --data /data/lcb80.jsonl --stride 8 --max-tokens "${THINK_MAX_TOKENS:-16384}" --out "results/$L/lcb_tts.jsonl"
"${BOX[@]}" python3 deep_eval.py lcb-exec --data /data/lcb80.jsonl --gen /out/lcb_tts.jsonl | tee "$OUT/lcb_tts_score.txt"
"${NET[@]}" python3 -c "import json;r=[json.loads(l) for l in open('results/$L/lcb_tts.jsonl')];T=sum(x['elapsed_s'] for x in r);print(f'LCB subset wall {T:.0f}s, {T/len(r):.1f}s/problem, mean tokens {sum(x[\"completion_tokens\"] for x in r)/len(r):.0f}')" | tee -a "$OUT/lcb_tts_score.txt"
echo "=== $(date -u +%T) $L tts done"
