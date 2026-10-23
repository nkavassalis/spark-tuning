#!/usr/bin/env python3
# Score a strided HumanEval+ time-to-solve run: pad missing tasks (evalplus wants all 164), then report the subset.
#   python3 tts_score.py pad <he.jsonl> <padded.jsonl>   |   python3 tts_score.py report <he.jsonl> <eval_results.json>
import json, sys
if sys.argv[1] == "pad":
    from evalplus.data import get_human_eval_plus
    rows = [json.loads(l) for l in open(sys.argv[2])]; have = {r["task_id"] for r in rows}
    with open(sys.argv[3], "w") as f:
        for r in rows: f.write(json.dumps({"task_id": r["task_id"], "solution": r["solution"]}) + "\n")
        for t in get_human_eval_plus():
            if t not in have: f.write(json.dumps({"task_id": t, "solution": ""}) + "\n")
else:
    rows = [json.loads(l) for l in open(sys.argv[2])]
    ev = json.load(open(sys.argv[3]))["eval"]
    ok = {t: v[0]["base_status"] == "pass" and v[0]["plus_status"] == "pass" for t, v in ev.items()}
    n = len(rows); k = sum(ok[r["task_id"]] for r in rows); T = sum(r["elapsed_s"] for r in rows)
    print(f"HE+ subset n={n} plus-pass {k}/{n}, wall {T:.0f}s, {T/n:.1f}s/problem, {T/max(k,1):.1f}s per solved, "
          f"mean tokens {sum(r['completion_tokens'] for r in rows)/n:.0f}")
