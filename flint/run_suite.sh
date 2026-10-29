#!/usr/bin/env bash
# The Flint quality suite against one variant, on the worker from ~/spark-quality (copy it there; quality/ is the harness).
#   ./flint_suite.sh <nvfp4|bf16> <api base, e.g. http://<head>:8000> <key file>
# The settings of mac-tuning's run_flint.sh at T=0.6 (the Mac int4 runs FLINT-t06 / FLINT-atc-t06), so the results pair
# problem by problem: 32768 tokens for every task (Flint always reasons; the card asks for >= 32K), top_p 0.95, top_k 20,
# THINK_KWARGS / NOTHINK_KWARGS '{}' (no reasoning switch), NLL skipped. Two differences, both from serving with vLLM:
# CONCURRENCY 8 instead of 1 (time to solve is measured separately, run_tts.sh at C=1), and long context added
# (run_deep.sh longctx, greedy, as for every Spark model). Labels: FL-<variant>-t06, FL-<variant>-atc-t06, TTS32-fl-<variant>-t06.
# Steps run in order of how much they tell apart; GSM8K last (every model here scores ~96%). Each one is resumable.
set -uo pipefail
V=${1:?variant}; cd ~/spark-quality
export API="${2:?api}" API_KEY="$(cat "${3:?key file}")" MODEL=flint-$V CONCURRENCY=${CONCURRENCY:-8}
export THINK_MAX_TOKENS=32768 THINK_TEMP=0.6 THINK_TOP_P=0.95 THINK_TOP_K=20 THINK_KWARGS='{}'
export NOTHINK_KWARGS='{}' NOTHINK_MAX_TOKENS=32768 NOTHINK_TEMP=0.6 NOTHINK_TOP_P=0.95
L=FL-$V-t06
step() { echo "=== $(date -u +%FT%TZ) $V start $1"; }
step "HE+ LCB80";  TASKS="humaneval lcb" ./run_deep.sh $L
step atc147;       TASKS=lcb LCB_DATA=atc147.jsonl ./run_deep.sh FL-$V-atc-t06
step MBPP+;        TASKS=mbpp ./run_quality.sh $L
step longctx;      TASKS=longctx ./run_deep.sh $L
step TTS;          ./run_tts.sh TTS32-fl-$V-t06
step GSM8K;        TASKS=gsm8k ./run_quality.sh $L
echo "=== $(date -u +%FT%TZ) $V suite done"
