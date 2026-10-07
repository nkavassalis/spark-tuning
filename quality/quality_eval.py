#!/usr/bin/env python3
"""Quality evaluation for the dual-Spark vLLM server (run inside quality/run_quality.sh's container).

Subcommands
  corpus  OUT.txt                     build the fixed NLL corpus (Python stdlib source + license prose)
  nll     --corpus C --out O.jsonl    per-token logprobs of the corpus under the served model (prefill only,
                                      deterministic; the most sensitive detector of weight-quantization damage)
  gen     --task {humaneval,mbpp,gsm8k} --out O.jsonl
                                      greedy generation, thinking disabled, CONCURRENCY parallel requests
  gsm8k-score  O.jsonl                exact-match accuracy
  compare-nll  REF.jsonl X.jsonl ...  mean NLL, paired delta, top-1 agreement vs REF
  compare-pass REF.json X.json ...    paired pass/fail comparison of evalplus result files / gsm8k scored files

Everything is greedy (temperature 0) with thinking off so that differences between configs are
attributable to the weights rather than to sampling. Greedy is still not bit-deterministic under
concurrent batching - the noise floor is measured by running the reference config twice.
"""
import argparse, concurrent.futures as cf, glob, json, math, os, re, sys, time, urllib.request

URL = os.environ.get("API", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
CONC = int(os.environ.get("CONCURRENCY", "8"))


def post(path, payload, timeout=1800, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"retry {i+1}: {e}", file=sys.stderr); time.sleep(5)


def chat(content, max_tokens):
    r = post("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": content}],
        "temperature": 0.0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}})
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), r["choices"][0].get("finish_reason"), r["usage"]["completion_tokens"]


# ---------------------------------------------------------------- corpus / NLL
CORPUS_FILES = ["argparse.py", "asyncio/base_events.py", "collections/__init__.py", "csv.py",
                "dataclasses.py", "difflib.py", "email/message.py", "enum.py", "functools.py",
                "heapq.py", "http/client.py", "inspect.py", "json/decoder.py", "logging/__init__.py",
                "pathlib.py", "pprint.py", "shutil.py", "statistics.py", "string.py", "textwrap.py",
                "threading.py", "typing.py", "urllib/parse.py", "zipfile/__init__.py", "LICENSE.txt"]
CHUNK_CHARS = 9000     # ~2.5k tokens per request
MAX_CHUNKS_PER_FILE = 4


def cmd_corpus(a):
    import sysconfig
    root = sysconfig.get_paths()["stdlib"]
    docs = []
    for f in CORPUS_FILES:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            print("missing", p, file=sys.stderr); continue
        t = open(p, encoding="utf-8", errors="replace").read()
        for i in range(0, min(len(t), CHUNK_CHARS * MAX_CHUNKS_PER_FILE), CHUNK_CHARS):
            docs.append({"source": f"{f}#{i // CHUNK_CHARS}", "text": t[i:i + CHUNK_CHARS]})
    with open(a.out, "w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    print(f"{len(docs)} chunks, {sum(len(d['text']) for d in docs):,} chars -> {a.out} (python {sys.version.split()[0]})")


def nll_one(doc):
    r = post("/v1/completions", {"model": MODEL, "prompt": doc["text"], "max_tokens": 1,
                                 "temperature": 0.0, "prompt_logprobs": 5})
    pl = r["choices"][0]["prompt_logprobs"]
    toks = []
    for entry in pl[1:]:                       # first token has no logprob
        # the actual prompt token is included even if outside top-5; rank tells us which it is
        actual = None; top = []
        for tid, v in entry.items():
            top.append((v["rank"], int(tid), v["logprob"]))
        top.sort()
        # vLLM returns the actual token first when it is outside the top-k; otherwise it is one of top-k.
        first_tid = int(next(iter(entry)))
        actual = (first_tid, entry[str(first_tid)]["logprob"], entry[str(first_tid)]["rank"])
        toks.append({"t": actual[0], "lp": round(actual[1], 5), "r": actual[2],
                     "top1": top[0][1]})
    return {"source": doc["source"], "tokens": toks}


def cmd_nll(a):
    docs = [json.loads(l) for l in open(a.corpus)]
    out = []
    with cf.ThreadPoolExecutor(1) as ex:      # sequential: deterministic, no batching effects
        for i, res in enumerate(ex.map(nll_one, docs)):
            out.append(res)
            if i % 10 == 0:
                print(f"  nll {i+1}/{len(docs)}", file=sys.stderr)
    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    n = sum(len(r["tokens"]) for r in out)
    s = sum(t["lp"] for r in out for t in r["tokens"])
    print(f"{n:,} tokens, mean NLL {-s / n:.5f} nats, ppl {math.exp(-s / n):.4f} -> {a.out}")


def cmd_bpb(a):
    """Bits per byte of the corpus: tokenizer-independent, so it compares DIFFERENT models (per-token NLL
    does not). Sum of token NLLs (tokens 2..n of each chunk) / UTF-8 bytes of the chunk text. The first
    token of each chunk has no logprob but its bytes are counted: a ~1e-4 bias, the same for all models.
    Files may be .jsonl or .jsonl.xz."""
    text = {d["source"]: d["text"] for d in map(json.loads, open(a.corpus))}
    print(f"{'file':<40} {'chunks':>6} {'tokens':>8} {'bytes':>9} {'bytes/tok':>9} {'bits/byte':>9}")
    for f in a.files:
        nll = load_nll(f); s = 0.0; n = 0; b = 0
        for src, toks in nll.items():
            s += -sum(t["lp"] for t in toks); n += len(toks); b += len(text[src].encode())
        print(f"{f[-40:]:<40} {len(nll):>6} {n:>8,} {b:>9,} {b / n:>9.3f} {s / math.log(2) / b:>9.5f}")


def load_nll(p):
    if p.endswith(".xz"):
        import lzma
        return {r["source"]: r["tokens"] for r in map(json.loads, lzma.open(p, "rt"))}
    return {r["source"]: r["tokens"] for r in map(json.loads, open(p))}


def cmd_compare_nll(a):
    ref = load_nll(a.files[0])
    print(f"{'file':<46} {'tokens':>8} {'NLL':>8} {'dNLL vs ref':>12} {'|dlp| mean':>10} {'top1 agree':>10}")
    for f in a.files:
        x = load_nll(f); n = 0; s = 0.0; ds = 0.0; dabs = 0.0; agree = 0; aligned = 0
        for src, toks in x.items():
            rt = ref.get(src)
            for i, t in enumerate(toks):
                n += 1; s += -t["lp"]
                if rt and i < len(rt) and rt[i]["t"] == t["t"]:
                    aligned += 1
                    d = (-t["lp"]) - (-rt[i]["lp"]); ds += d; dabs += abs(d)
                    agree += (rt[i]["top1"] == t["top1"])
        print(f"{os.path.basename(f):<46} {n:>8,} {s / n:>8.5f} {ds / max(1, aligned):>+12.5f} "
              f"{dabs / max(1, aligned):>10.5f} {agree / max(1, aligned):>10.4f}")


# ---------------------------------------------------------------- code / math generation
EVALPLUS_PROMPT = ("Please provide a self-contained Python script that solves the following problem "
                   "in a markdown code block:\n```\n{prompt}\n```\n")
GSM_PROMPT = ("{q}\n\nSolve the problem step by step. Then give the final numeric answer on the last "
              "line in the form '#### <number>'.")


def load_tasks(task):
    if task in ("humaneval", "mbpp"):
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        ds = get_human_eval_plus() if task == "humaneval" else get_mbpp_plus()
        return [{"task_id": k, "prompt": EVALPLUS_PROMPT.format(prompt=v["prompt"].strip()),
                 "entry_point": v["entry_point"]} for k, v in ds.items()]
    rows = [json.loads(l) for l in open(os.environ.get("GSM8K", "/data/gsm8k_test.jsonl"))]
    return [{"task_id": f"gsm8k/{i}", "prompt": GSM_PROMPT.format(q=r["question"]),
             "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(rows)]


def cmd_gen(a):
    tasks = load_tasks(a.task)
    if a.limit:
        tasks = tasks[:a.limit]
    done = {}
    if os.path.exists(a.out):                  # resumable
        done = {r["task_id"]: r for r in map(json.loads, open(a.out))}
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"{a.task}: {len(tasks)} tasks, {len(todo)} to do, concurrency {CONC}", file=sys.stderr)
    max_tokens = 1024 if a.task == "gsm8k" else 1536

    def work(t):
        raw, fin, ntok = chat(t["prompt"], max_tokens)
        rec = {"task_id": t["task_id"], "raw": raw, "finish": fin, "completion_tokens": ntok}
        if a.task == "gsm8k":
            rec["gold"] = t["gold"]
        else:
            from evalplus.sanitize import sanitize
            rec["solution"] = sanitize(raw, t["entry_point"])
        return rec

    t0 = time.time()
    with open(a.out, "a") as fh, cf.ThreadPoolExecutor(CONC) as ex:
        for i, rec in enumerate(ex.map(work, todo)):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 25 == 0:
                print(f"  {i+1}/{len(todo)}  {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"{a.task}: wrote {a.out} in {time.time() - t0:.0f}s", file=sys.stderr)


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm_pred(text):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    s = m.group(1) if m else (NUM.findall(text) or [""])[-1]
    return s.replace(",", "").rstrip(".")


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def cmd_gsm8k_score(a):
    rows = [json.loads(l) for l in open(a.file)]
    res = {r["task_id"]: num_eq(gsm_pred(r["raw"]), r["gold"]) for r in rows}
    out = a.file.replace(".jsonl", "_scored.json")
    json.dump({"per_task": res, "accuracy": sum(res.values()) / len(res), "n": len(res)}, open(out, "w"))
    print(f"GSM8K accuracy {sum(res.values())}/{len(res)} = {100 * sum(res.values()) / len(res):.2f}%  -> {out}")


def load_pass(p):
    d = json.load(open(p))
    if "per_task" in d:                               # gsm8k scored file
        return {k: bool(v) for k, v in d["per_task"].items()}, {k: bool(v) for k, v in d["per_task"].items()}
    base, plus = {}, {}
    for tid, rs in d["eval"].items():                  # evalplus results
        r = rs[0]
        base[tid] = r["base_status"] == "pass"
        plus[tid] = base[tid] and r["plus_status"] == "pass"
    return base, plus


def cmd_compare_pass(a):
    ref_b, ref_p = load_pass(a.files[0])
    print(f"{'file':<46} {'n':>4} {'base %':>7} {'plus %':>7} {'plus: ref-only':>14} {'x-only':>7}  (McNemar exact p)")
    for f in a.files:
        b, p = load_pass(f)
        ks = sorted(set(ref_p) & set(p))
        lost = sum(1 for k in ks if ref_p[k] and not p[k]); gained = sum(1 for k in ks if p[k] and not ref_p[k])
        n = lost + gained
        pval = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(lost, gained) + 1)) / 2 ** n) if n else 1.0
        print(f"{os.path.basename(f):<46} {len(ks):>4} {100 * sum(b[k] for k in ks) / len(ks):>7.2f} "
              f"{100 * sum(p[k] for k in ks) / len(ks):>7.2f} {lost:>14} {gained:>7}  p={pval:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("corpus"); s.add_argument("out")
    s = sp.add_parser("nll"); s.add_argument("--corpus", required=True); s.add_argument("--out", required=True)
    s = sp.add_parser("gen"); s.add_argument("--task", required=True, choices=["humaneval", "mbpp", "gsm8k"])
    s.add_argument("--out", required=True); s.add_argument("--limit", type=int)
    s = sp.add_parser("gsm8k-score"); s.add_argument("file")
    s = sp.add_parser("compare-nll"); s.add_argument("files", nargs="+")
    s = sp.add_parser("bpb"); s.add_argument("--corpus", required=True); s.add_argument("files", nargs="+")
    s = sp.add_parser("compare-pass"); s.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"corpus": cmd_corpus, "nll": cmd_nll, "gen": cmd_gen, "gsm8k-score": cmd_gsm8k_score,
     "compare-nll": cmd_compare_nll, "compare-pass": cmd_compare_pass, "bpb": cmd_bpb}[a.cmd](a)


if __name__ == "__main__":
    main()
