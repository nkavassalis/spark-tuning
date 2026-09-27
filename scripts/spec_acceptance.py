#!/usr/bin/env python3
"""MTP / speculative-decoding acceptance from vLLM /metrics, plus a draft-length estimate.

Modes
  snapshot:  ./spec_acceptance.py --save before.json
  delta:     ./spec_acceptance.py --since before.json      # acceptance since the snapshot
  watch:     ./spec_acceptance.py --watch 60               # rolling delta every 60 s (your real traffic)
  lifetime:  ./spec_acceptance.py                          # since server start

What the numbers mean
  per-pos p_i   fraction of draft steps in which draft token i was accepted
                (unconditional: p_2 already includes "p_0 and p_1 accepted").
  accept len    1 + sum(p_i) = tokens emitted per target forward pass.

Draft-length estimate (--draft-cost)
  Step time is modelled as  T(K) = 1 + c*K  (target pass = 1, each draft pass = c).
  Throughput(K) ~ (1 + sum_{i<K} p_i) / (1 + c*K).
  The last draft position pays for itself only if  p_{K-1} > c * (accept len at K-1 ... ) —
  i.e. roughly  p_last > c * E(K-1) / (1 + c*(K-1)).
  c ~= 0.12 is derived from the repo's byte model (docs/CLAUDE/fable5-1-report.md:
  drafter ~0.8 GiB/pass vs ~6.8 GiB for the target step, 47k draft vocab). It is an
  ESTIMATE — K only ever gets decided by an A/B run (scripts/mtp_ab.sh). This model can
  only evaluate K values <= the one currently running (it can drop positions, not invent them).
"""
import argparse, json, re, time, urllib.request

WANTED = ("vllm:spec_decode_num_drafts_total",
          "vllm:spec_decode_num_draft_tokens_total",
          "vllm:spec_decode_num_accepted_tokens_total",
          "vllm:spec_decode_num_accepted_tokens_per_pos_total",
          "vllm:generation_tokens_total")
LINE = re.compile(r"^([a-z_:]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)$")


def scrape(url):
    body = urllib.request.urlopen(url, timeout=20).read().decode()
    out = {}
    for ln in body.splitlines():
        m = LINE.match(ln.strip())
        if not m or m.group(1) not in WANTED:
            continue
        name, labels, val = m.groups()
        pos = re.search(r'position="(\d+)"', labels or "")
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] = out.get(key, 0.0) + float(val)
    return out


def report(d, draft_cost):
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    if drafts <= 0:
        print("no speculative steps in this window"); return
    p, i = [], 0
    while f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]" in d:
        p.append(d[f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]"] / drafts); i += 1
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    gen = d.get("vllm:generation_tokens_total", 0)
    print(f"draft steps {int(drafts):,}   generated tokens {int(gen):,}   K={len(p)}")
    print("per-position acceptance: " + "  ".join(f"p{i}={x:.3f}" for i, x in enumerate(p)))
    print(f"mean acceptance length: {1 + acc / drafts:.2f} tokens / target step")
    print(f"\nestimated relative throughput (draft cost c={draft_cost}):")
    best = max(range(1, len(p) + 1), key=lambda k: (1 + sum(p[:k])) / (1 + draft_cost * k))
    for k in range(1, len(p) + 1):
        e = 1 + sum(p[:k]); thr = e / (1 + draft_cost * k)
        print(f"  K={k}: {e:.2f} tok/step / {1 + draft_cost * k:.2f} cost = {thr:.3f}"
              + ("   <- best of these" if k == best else ""))
    print("  (K=1/K=2 rows assume the earlier positions' acceptance is unchanged when fewer"
          " tokens are drafted - approximately true; confirm with mtp_ab.sh)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://10.1.13.99:8000/metrics")
    ap.add_argument("--save"); ap.add_argument("--since")
    ap.add_argument("--watch", type=int, metavar="SECONDS")
    ap.add_argument("--draft-cost", type=float, default=0.12)
    a = ap.parse_args()
    now = scrape(a.url)
    if a.save:
        json.dump(now, open(a.save, "w"), indent=1); print(f"saved {a.save}"); return
    if a.watch:
        prev = now
        while True:
            time.sleep(a.watch); cur = scrape(a.url)
            print(f"\n=== {time.strftime('%T')} last {a.watch}s ===")
            report({k: cur[k] - prev.get(k, 0) for k in cur}, a.draft_cost); prev = cur
    base = json.load(open(a.since)) if a.since else {}
    report({k: now[k] - base.get(k, 0) for k in now}, a.draft_cost)


if __name__ == "__main__":
    main()
