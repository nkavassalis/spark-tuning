#!/usr/bin/env python3
"""Decode-speed benchmark with per-run MTP (speculative decoding) acceptance.

Derived from the repo's bench/decodebench.py + bench/mtp_accept.py, merged so
every row reports tok/s AND the draft acceptance measured during that row
(delta of the server's /metrics counters around the request).

Every run decodes exactly --decode tokens (ignore_eos), so tok/s is comparable
across tasks. Run it from anywhere that can reach the API:

  ./bench_decode.py                                   # defaults: code-focused
  ./bench_decode.py --contexts 1000,64000,128000 --temps 0.0,0.6 --repeats 3
  ./bench_decode.py --json ../results/mtp2-$(date +%F).json

NOTE: acceptance counters are server-global. If another client is using the
server during a run, that row's acceptance (and tok/s) is contaminated.
"""
import argparse, json, os, re, statistics, sys, time, urllib.request

FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")

# A realistic "agentic edit" context: real-looking source code the model must modify.
SRC_UNIT = '''
def parse_record_{i}(line: str) -> dict:
    """Parse one CSV-ish record into a dict; raises ValueError on bad input."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        raise ValueError(f"expected 4 fields, got {{len(parts)}}")
    ident, name, qty, price = parts
    return {{"id": int(ident), "name": name, "qty": int(qty), "price": float(price)}}
'''

TASKS = {
    "prose":  "Write a flowing, continuous essay about the history of maritime "
              "navigation. Use ordinary narrative prose, no lists, no headings.",
    "code":   "Write a complete, heavily-commented Python implementation of a "
              "red-black tree with insert, delete and search.",
    "code_ts": "Write a complete TypeScript implementation of an LRU cache with TTL "
               "expiry, generics, and a full Jest test suite.",
    "edit":   "Refactor every parse_record_N function above into a single generic "
              "parse_record(line, schema) function with type hints and docstrings, "
              "then rewrite each original as a thin wrapper. Output the full code.",
    "copy":   "Reproduce verbatim, in order, entries 000005 through 000034 from "
              "the log above. Output the lines exactly as they appear.",
}
# which context builder each task uses
CTX_KIND = {"edit": "src"}

METRICS = ("vllm:spec_decode_num_drafts_total",
           "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_per_pos_total")
LINE = re.compile(r"^([a-z_:]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)$")


def scrape(base):
    try:
        body = urllib.request.urlopen(base + "/metrics", timeout=20).read().decode()
    except Exception:
        return {}
    out = {}
    for ln in body.splitlines():
        m = LINE.match(ln.strip())
        if not m or m.group(1) not in METRICS:
            continue
        name, labels, val = m.groups()
        pos = re.search(r'position="(\d+)"', labels or "")
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] = out.get(key, 0.0) + float(val)
    return out


def accept_delta(a, b):
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    if drafts <= 0:
        return None
    per_pos, i = [], 0
    while f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]" in d:
        per_pos.append(d[f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]"] / drafts)
        i += 1
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    return {"per_pos": per_pos, "mean_accept_len": 1 + acc / drafts,
            "draft_accept_rate": acc / max(1, d.get("vllm:spec_decode_num_draft_tokens_total", 1))}


def build_ctx(kind, tokens):
    if kind == "src":
        n = max(1, int(tokens / 110))            # ~110 tokens per unit
        return "".join(SRC_UNIT.format(i=i) for i in range(n))
    return "".join(FILLER.format(i=i) for i in range(max(1, int(tokens / 25))))


def run(base, model, ctx, task, n, temp, thinking=None):
    payload = {"model": model,
               "messages": [{"role": "user", "content": ctx + "\n\n" + TASKS[task]}],
               "max_tokens": n, "min_tokens": n, "ignore_eos": True,
               "temperature": temp, "stream": True,
               "stream_options": {"include_usage": True}}
    if thinking is not None:                 # None = server/template default
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = t_last = usage = None
    for raw in urllib.request.urlopen(req, timeout=3600):
        s = raw.decode().strip()
        if not s.startswith("data: ") or s == "data: [DONE]":
            continue
        o = json.loads(s[6:])
        if o.get("usage"):
            usage = o["usage"]
        for ch in o.get("choices", []):
            if ch.get("delta") is None:
                continue
            if ttft is None:
                ttft = time.time() - t0
            t_last = time.time()
    ctok = (usage or {}).get("completion_tokens", n)
    ptok = (usage or {}).get("prompt_tokens", 0)
    dec = (ctok - 1) / (t_last - t0 - ttft) if t_last and ttft and t_last - t0 > ttft else 0.0
    return ptok, ctok, ttft or 0.0, dec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("API", "http://localhost:8000"))
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--tasks", default="code,code_ts,edit,prose")
    ap.add_argument("--contexts", default="1000,64000")
    ap.add_argument("--temps", default="0.0,0.6")
    ap.add_argument("--decode", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=1, help="median of N runs per cell")
    ap.add_argument("--label", default="", help="free-form tag stored in --json output")
    ap.add_argument("--thinking", choices=["default", "on", "off"], default="default",
                    help="send chat_template_kwargs.enable_thinking (default: don't send, server default applies)")
    ap.add_argument("--json", help="write results to this file")
    a = ap.parse_args()

    # warm-up (first request after launch triggers FlashInfer autotune)
    think = {"default": None, "on": True, "off": False}[a.thinking]
    run(a.url, a.model, "", "code", 32, 0.0, think)

    rows = []
    hdr = f"{'ctx':>8} {'temp':>4} {'task':<8} {'ptok':>7} {'TTFT s':>7} {'tok/s':>6} {'acc.len':>7}  per-pos acceptance"
    print(hdr); print("-" * len(hdr))
    for c in [int(x) for x in a.contexts.split(",")]:
        for t in [float(x) for x in a.temps.split(",")]:
            for task in a.tasks.split(","):
                ctx = build_ctx(CTX_KIND.get(task, "log"), c)
                decs, ttfts, accs = [], [], []
                for _ in range(a.repeats):
                    m0 = scrape(a.url)
                    ptok, ctok, ttft, dec = run(a.url, a.model, ctx, task, a.decode, t, think)
                    acc = accept_delta(m0, scrape(a.url))
                    decs.append(dec); ttfts.append(ttft); accs.append(acc)
                dec = statistics.median(decs)
                acc = accs[decs.index(sorted(decs)[len(decs) // 2])]
                pp = " / ".join(f"{p:.2f}" for p in acc["per_pos"]) if acc else "n/a"
                al = f"{acc['mean_accept_len']:.2f}" if acc else "n/a"
                print(f"{c:>8,} {t:>4.1f} {task:<8} {ptok:>7,} {statistics.median(ttfts):>7.2f} {dec:>6.1f} {al:>7}  {pp}")
                sys.stdout.flush()
                rows.append({"context": c, "temp": t, "task": task, "prompt_tokens": ptok,
                             "decode_tok_s": dec, "all_decode_tok_s": decs,
                             "ttft_s": statistics.median(ttfts), "acceptance": acc})
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"label": a.label, "thinking": a.thinking, "url": a.url, "time": time.strftime("%F %T"),
                       "decode_tokens": a.decode, "repeats": a.repeats, "rows": rows}, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
