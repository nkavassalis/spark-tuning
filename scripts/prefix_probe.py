#!/usr/bin/env python3
"""prefix_probe.py: where does the vLLM prefix cache actually stand — does a prompt's KV survive?

Reads the server's own counters (vllm:prompt_tokens_by_source{source="local_compute"|"local_cache_hit"})
around each request, so "did this prompt get recomputed?" is measured, not inferred from TTFT.

  scripts/prefix_probe.py                       # cold, warm, after-competing-traffic, after idle
  scripts/prefix_probe.py --tokens 40000 --load 3 --idle 900
  scripts/prefix_probe.py --steps cold,warm     # just the two quick ones

Why each step:
  cold    first sight of a brand-new prompt (nonce at the front) -> everything is recomputed; the baseline
  warm    same bytes again, nothing else in between               -> proves the mechanism works at all
  load    same prompt after N unique prompts of similar size      -> LRU eviction under real pressure
  idle    same prompt after sitting unused                        -> is there any time-based flush?
Read-only apart from the inference requests. Hosts come from cluster.env / API. Stdlib only.
"""
import argparse, json, os, random, time, urllib.request, sys
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Optional bearer key (vLLM --api-key); the b12x build also guards /metrics with it.
AUTH = {"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}

FILLER = ("def transform_record_{i}(row, opts):\n"
          "    out = {{}}\n"
          "    for key, spec in opts.items():\n"
          "        if key not in row:\n"
          "            continue\n"
          "        out[key] = spec(row[key]) if callable(spec) else row[key]\n"
          "    return out\n\n")


def counters(api):
    txt = urllib.request.urlopen(urllib.request.Request(api.rstrip('/') + '/metrics', headers=AUTH), timeout=10).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith('vllm:prompt_tokens_by_source_total') and 'local_' in line:
            src = line.split('source="')[1].split('"')[0]
            out[src] = float(line.rsplit(' ', 1)[1])
        elif line.startswith('vllm:kv_cache_usage_perc') or line.startswith('vllm:gpu_cache_usage_perc'):
            out['kv'] = float(line.rsplit(' ', 1)[1])
        elif line.startswith('vllm:request_success_total{'):
            out['done'] = out.get('done', 0) + float(line.rsplit(' ', 1)[1])
        elif line.startswith('vllm:prefix_cache_queries_total{'):
            out['queries'] = float(line.rsplit(' ', 1)[1])
    return out


def post(api, prompt, max_tokens):
    body = json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next"),
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        api.rstrip('/') + '/v1/chat/completions', body, {'Content-Type': 'application/json', **AUTH}), timeout=1800).read())
    return time.time() - t0, r['usage']['prompt_tokens']


def run(api, label, prompt, max_tokens, quiet=False):
    a, b = counters(api), None
    el, ptok = post(api, prompt, max_tokens)
    b = counters(api)
    # The counters are server-global: if another request finished during the probe, the delta is not ours alone.
    others = int(b.get('done', 0) - a.get('done', 0)) - 1
    qdelta = b.get('queries', 0) - a.get('queries', 0)
    comp = b.get('local_compute', 0) - a.get('local_compute', 0)
    hit = b.get('local_cache_hit', 0) - a.get('local_cache_hit', 0)
    row = {"step": label, "prompt_tokens": ptok, "recomputed": round(comp), "cached": round(hit),
           "cached_pct": round(100 * hit / max(1, hit + comp), 1), "e2e_s": round(el, 2),
           "kv_after": round(b.get('kv', 0) * 100, 1), "overlap_requests": max(0, others),
           "queries_delta": round(qdelta), "clean": others <= 0 and abs(qdelta - ptok) < ptok * 0.2}
    if not quiet:
        flag = "" if row["clean"] else f"   [contaminated: {row['overlap_requests']} other request(s) finished]"
        print(f"  {label:<26} prompt={ptok:>7,}  recomputed={row['recomputed']:>7,}  "
              f"cached={row['cached']:>7,} ({row['cached_pct']}%)  e2e={el:5.2f}s  kv={row['kv_after']}%{flag}")
    return row


def make_prompt(tokens, nonce):
    n = max(1, int(tokens / 60))                      # measured: one unit of this filler = ~60 model tokens
    # (calibrated 2026-10-15: 923 units -> 55,329 prompt tokens on this checkpoint/chat template)
    return nonce + "\n" + "".join(FILLER.format(i=i) for i in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("API", "http://localhost:8000"))
    ap.add_argument("--tokens", type=int, default=24000, help="probe prompt size in tokens")
    ap.add_argument("--load", type=int, default=3, help="unique competing prompts for the 'load' step")
    ap.add_argument("--idle", type=int, default=600, help="seconds to sit unused for the 'idle' step")
    ap.add_argument("--steps", default="cold,warm,load,idle")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    steps = a.steps.split(',')
    random.seed(7)
    probe = make_prompt(a.tokens, f"PROBE-{random.randint(10**8, 10**9-1)}: write 'ok'")
    rows = []
    print(f"probe prompt: {a.api} model={os.environ.get('MODEL','qwen3.8-flash-next')} target={a.tokens} tokens")
    if 'cold' in steps:
        rows.append(run(a.api, 'cold (first sight)', probe, 8))
    if 'warm' in steps:
        rows.append(run(a.api, 'warm (immediately again)', probe, 8))
    if 'load' in steps:
        for i in range(a.load):                       # same size class as the probe, nothing in common with it
            run(a.api, f'  competing prompt {i+1}', make_prompt(a.tokens, f"FILLER-{i}-{random.randint(10**8,10**9-1)}"), 8, quiet=True)
        rows.append(run(a.api, f'after {a.load} competing prompts', probe, 8))
    if 'idle' in steps:
        print(f"  sitting idle {a.idle} s (nothing sent to the server)...", flush=True)
        time.sleep(a.idle)
        rows.append(run(a.api, f'after {a.idle} s idle', probe, 8))
    if a.json:
        json.dump({"api": "<head>", "model": os.environ.get("MODEL", ""), "args": vars(a), "rows": rows},
                  open(a.json, 'w'), indent=1)
        print(f"wrote {a.json}")


if __name__ == '__main__':
    main()
