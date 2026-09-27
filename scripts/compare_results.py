#!/usr/bin/env python3
"""Side-by-side comparison of bench_decode.py --json outputs.

  ./compare_results.py ../results/mtp_ab-*/K2.json ../results/mtp_ab-*/K3.json
The first file is the reference; other columns show tok/s and % change vs it.
"""
import json, sys

files = sys.argv[1:]
if len(files) < 2:
    sys.exit(__doc__)
runs = [json.load(open(f)) for f in files]
labels = [r.get("label") or f for r, f in zip(runs, files)]
key = lambda row: (row["context"], row["temp"], row["task"])
tables = [{key(r): r for r in run["rows"]} for run in runs]

print(f"{'ctx':>8} {'temp':>4} {'task':<8} " + " ".join(f"{l:>18}" for l in labels))
for k in tables[0]:
    ref = tables[0][k]["decode_tok_s"]
    cells = []
    for t in tables:
        if k not in t:
            cells.append(f"{'-':>18}"); continue
        v = t[k]["decode_tok_s"]; acc = t[k].get("acceptance") or {}
        al = acc.get("mean_accept_len", 0)
        cells.append(f"{v:6.1f} ({(v / ref - 1) * 100:+5.1f}%) a{al:.2f}" if t is not tables[0]
                     else f"{v:6.1f}         a{al:.2f}")
    print(f"{k[0]:>8,} {k[1]:>4.1f} {k[2]:<8} " + " ".join(cells))
print("\n(aX.XX = mean acceptance length: tokens emitted per target forward pass)")
