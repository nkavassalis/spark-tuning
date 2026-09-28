#!/usr/bin/env python3
"""Thinking-mode and long-context quality tests (companion to quality_eval.py).

Subcommands (run inside the spark-quality container, see run_deep.sh)
  lcb-prep     --src test6.jsonl --out lcb.jsonl [--n 80]   pick stdin-type LiveCodeBench problems, decode tests
  longctx-prep --out longctx.jsonl                           build planted-function long-context samples
  gen-think    --task {humaneval,lcb} --data F --out O       thinking ON, T=0.6/top_p=0.95/top_k=20, fixed per-task seed
  lcb-exec     --data lcb.jsonl --gen O.jsonl                run solutions against tests (SANDBOX: --network none)
  gen-longctx  --data longctx.jsonl --out O.jsonl            greedy, thinking off; scored inline
  compare      REF X ...                                     paired comparison of *_scored.json files

Pairing: every request carries seed=crc32(task_id), so two configs draw the same random stream; any
divergence comes from the logits, not from sampling luck.
"""
import argparse, base64, concurrent.futures as cf, json, math, os, pickle, random, re, subprocess, sys, time, zlib
import urllib.request

URL = os.environ.get("API", "http://192.168.100.10:8000")
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
CONC = int(os.environ.get("CONCURRENCY", "8"))


def post(path, payload, timeout=7200, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"retry {i+1}: {e}", file=sys.stderr); time.sleep(10)


def resumable_map(fn, items, out, conc):
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["task_id"] for l in open(out)}
    todo = [t for t in items if t["task_id"] not in done]
    print(f"{len(items)} items, {len(todo)} to do, concurrency {conc}", file=sys.stderr)
    t0 = time.time()
    with open(out, "a") as fh, cf.ThreadPoolExecutor(conc) as ex:
        for i, rec in enumerate(ex.map(fn, todo)):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 10 == 0:
                print(f"  {i+1}/{len(todo)} {time.time()-t0:.0f}s", file=sys.stderr)


# ------------------------------------------------------------------ thinking-mode generation
LCB_PROMPT = ("You will be given a question (problem specification) and will generate a correct Python "
              "program that matches the specification and passes all tests.\n\nQuestion: {q}\n\n"
              "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
              "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
              "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
              "```python\n# YOUR CODE HERE\n```")
HE_PROMPT = ("Please provide a self-contained Python script that solves the following problem "
             "in a markdown code block:\n```\n{prompt}\n```\n")


def think_chat(task_id, content, max_tokens):
    r = post("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": content}],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens,
        "seed": zlib.crc32(task_id.encode()),
        "chat_template_kwargs": {"enable_thinking": True}})
    c = r["choices"][0]; m = c["message"]
    return {"task_id": task_id, "content": m.get("content") or "",
            "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
            "completion_tokens": r["usage"]["completion_tokens"], "finish": c.get("finish_reason")}


def last_code_block(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else ""


def cmd_gen_think(a):
    if a.task == "humaneval":
        from evalplus.data import get_human_eval_plus
        from evalplus.sanitize import sanitize
        ds = get_human_eval_plus()
        items = [{"task_id": k, "prompt": HE_PROMPT.format(prompt=v["prompt"].strip()),
                  "entry_point": v["entry_point"]} for k, v in ds.items()]

        def fn(t):
            rec = think_chat(t["task_id"], t["prompt"], a.max_tokens)
            rec["solution"] = sanitize(rec["content"], t["entry_point"]); return rec
    else:
        items = [json.loads(l) for l in open(a.data)]

        def fn(t):
            rec = think_chat(t["task_id"], LCB_PROMPT.format(q=t["question"]), a.max_tokens)
            rec["code"] = last_code_block(rec["content"]); return rec
    resumable_map(fn, items, a.out, CONC)


# ------------------------------------------------------------------ LiveCodeBench
def decode_tests(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def cmd_lcb_prep(a):
    rows = [json.loads(l) for l in open(a.src)]
    stdin = [r for r in rows if all(t.get("testtype") == "stdin" for t in json.loads(r["public_test_cases"]))]
    random.Random(1234).shuffle(stdin)
    pick = stdin[:a.n]
    with open(a.out, "w") as fh:
        for r in pick:
            tests = json.loads(r["public_test_cases"]) + decode_tests(r["private_test_cases"])
            tests = [{"input": t["input"], "output": t["output"]} for t in tests][:a.max_tests]
            fh.write(json.dumps({"task_id": f"lcb/{r['question_id']}", "question": r["question_content"],
                                 "difficulty": r.get("difficulty"), "platform": r.get("platform"),
                                 "contest_date": r.get("contest_date"), "tests": tests}) + "\n")
    diff = {}
    for r in pick:
        diff[r.get("difficulty")] = diff.get(r.get("difficulty"), 0) + 1
    print(f"{len(stdin)} stdin problems in {a.src}; picked {len(pick)} {diff} -> {a.out}")


def norm_out(s):
    return [ln.split() for ln in s.strip().splitlines() if ln.strip()]


def run_one(args):
    task_id, code, tests = args
    if not code.strip():
        return task_id, False, "no code"
    path = f"/tmp/sol_{zlib.crc32(task_id.encode())}.py"
    open(path, "w").write(code)
    t_total = time.time()
    for i, t in enumerate(tests):
        if time.time() - t_total > 120:
            return task_id, False, "total timeout"
        try:
            p = subprocess.run([sys.executable, path], input=t["input"], capture_output=True,
                               text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return task_id, False, f"timeout on test {i}"
        if p.returncode != 0:
            return task_id, False, f"runtime error on test {i}"
        if norm_out(p.stdout) != norm_out(t["output"]):
            return task_id, False, f"wrong answer on test {i}"
    return task_id, True, "pass"


def cmd_lcb_exec(a):
    data = {json.loads(l)["task_id"]: json.loads(l) for l in open(a.data)}
    gens = [json.loads(l) for l in open(a.gen)]
    jobs = [(g["task_id"], g.get("code", ""), data[g["task_id"]]["tests"]) for g in gens]
    res = {}
    with cf.ProcessPoolExecutor(8) as ex:
        for tid, ok, why in ex.map(run_one, jobs):
            res[tid] = {"pass": ok, "why": why}
    out = a.gen.replace(".jsonl", "_scored.json")
    n = len(res); k = sum(v["pass"] for v in res.values())
    json.dump({"per_task": {t: v["pass"] for t, v in res.items()}, "detail": res,
               "accuracy": k / n, "n": n}, open(out, "w"), indent=1)
    print(f"LCB pass@1 {k}/{n} = {100*k/n:.1f}% -> {out}")


# ------------------------------------------------------------------ long context
WORDS = ["offset", "scale", "bias", "gain", "limit", "stride", "shift", "decay"]


def cmd_longctx_prep(a):
    import sysconfig
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
                   if f.endswith(".py") and "test" not in dp and "idlelib" not in dp)
    rng = random.Random(42)
    rng.shuffle(files)
    code = []
    for f in files:
        try:
            code.append(f"# ===== file: {os.path.relpath(f, root)} =====\n" + open(f, encoding="utf-8").read())
        except Exception:
            pass
    big = "\n".join(code)
    CPT = 4.1                                  # measured chars/token for this corpus
    samples = []
    for length in a.lengths:
        for s in range(a.per_length):
            srng = random.Random(length * 1000 + s)
            start = srng.randrange(0, len(big) - int(length * CPT) - 1)
            ctx = big[start:start + int(length * CPT)]
            lines = ctx.split("\n")
            # 24 planted functions with near-duplicate names; 6 of them are two-hop
            word = srng.choice(WORDS); prefix = f"_calib_{word}_"
            ids = srng.sample(range(10, 99), 24)
            consts = {i: srng.randrange(10000, 99999) for i in ids}
            planted = {}
            for i in ids:
                planted[i] = f"def {prefix}{i}():\n    return {consts[i]}\n"
            hop_ids = srng.sample(ids, 6); hops = {}
            for j, i in enumerate(hop_ids):
                tgt = srng.choice([x for x in ids if x != i]); mul = srng.randrange(2, 9); add = srng.randrange(1, 999)
                name = f"_derive_{word}_{i}"
                planted[f"h{i}"] = f"def {name}(x):\n    return {prefix}{tgt}() * {mul} + x + {add}\n"
                hops[name] = (tgt, mul, add)
            # insert each planted def before a random top-level line
            anchors = [k for k, ln in enumerate(lines) if ln.startswith(("def ", "class "))] or [len(lines) // 2]
            spots = sorted(((srng.choice(anchors), p) for p in planted.values()), key=lambda x: -x[0])
            for k, p in spots:                 # descending, so earlier indices stay valid
                lines.insert(k, p)
            ctx = "\n".join(lines)
            q_ret = srng.choice(ids)
            q_hop = srng.choice(list(hops)); hx = srng.randrange(1, 50)
            tgt, mul, add = hops[q_hop]
            q_rev = srng.choice([i for i in ids if i != q_ret])
            question = (f"The code above contains several functions whose names start with `{prefix}` and "
                        f"`_derive_{word}_`. Answer precisely, using only the code above.\n"
                        f"Q1: What integer does `{prefix}{q_ret}()` return?\n"
                        f"Q2: `{q_hop}` calls exactly one `{prefix}*` function. What integer does that called function return?\n"
                        f"Q3: Which function named `{prefix}*` returns {consts[q_rev]}?\n"
                        "Reply with exactly three lines and nothing else:\nA1: <integer>\nA2: <integer>\nA3: <function name>")
            samples.append({"task_id": f"longctx/{length}/{s}", "length": length, "context": ctx,
                            "question": question,
                            "answers": {"A1": str(consts[q_ret]), "A2": str(consts[tgt]),   # v2: pure two-hop retrieval (v1 asked for q_hop(hx): mental arithmetic)
                                        "A3": f"{prefix}{q_rev}"}})
    with open(a.out, "w") as fh:
        for x in samples:
            fh.write(json.dumps(x) + "\n")
    print(f"{len(samples)} samples ({a.per_length} x {a.lengths}) -> {a.out}")


def cmd_gen_longctx(a):
    items = [json.loads(l) for l in open(a.data)]

    def fn(t):
        r = post("/v1/chat/completions", {
            "model": MODEL, "messages": [{"role": "user", "content": t["context"] + "\n\n" + t["question"]}],
            "temperature": 0.0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}})
        text = r["choices"][0]["message"]["content"] or ""
        got = {k: (re.search(rf"{k}:\s*`?([A-Za-z0-9_\-]+)", text) or [None, ""])[1] for k in ("A1", "A2", "A3")}
        ok = {k: got[k].strip("`() ") == v for k, v in t["answers"].items()}
        return {"task_id": t["task_id"], "length": t["length"], "prompt_tokens": r["usage"]["prompt_tokens"],
                "raw": text, "got": got, "ok": ok}
    resumable_map(fn, items, a.out, int(os.environ.get("LONGCTX_CONC", "4")))
    rows = [json.loads(l) for l in open(a.out)]
    per = {}
    for r in rows:
        for k, v in r["ok"].items():
            per[f"{r['task_id']}/{k}"] = v
    json.dump({"per_task": per, "accuracy": sum(per.values()) / len(per), "n": len(per)},
              open(a.out.replace(".jsonl", "_scored.json"), "w"), indent=1)
    for L in sorted({r["length"] for r in rows}):
        for k in ("A1", "A2", "A3"):
            sub = [r["ok"][k] for r in rows if r["length"] == L]
            print(f"  ctx {L:>6}  {k} ({ {'A1':'retrieve','A2':'two-hop','A3':'reverse'}[k] }): {sum(sub)}/{len(sub)}")
    print(f"long-context accuracy {sum(per.values())}/{len(per)} = {100*sum(per.values())/len(per):.1f}%  "
          f"(mean prompt {sum(r['prompt_tokens'] for r in rows)//len(rows):,} tokens)")


# ------------------------------------------------------------------ comparison
def cmd_compare(a):
    ref = json.load(open(a.files[0]))["per_task"]
    print(f"{'file':<58} {'n':>4} {'acc %':>6} {'ref-only':>8} {'x-only':>7}  McNemar p")
    for f in a.files:
        x = json.load(open(f))["per_task"]; ks = sorted(set(ref) & set(x))
        lost = sum(1 for k in ks if ref[k] and not x[k]); gained = sum(1 for k in ks if x[k] and not ref[k])
        n = lost + gained
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n) if n else 1.0
        print(f"{f:<58} {len(ks):>4} {100*sum(x[k] for k in ks)/len(ks):>6.1f} {lost:>8} {gained:>7}  p={p:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("lcb-prep"); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=80); s.add_argument("--max-tests", type=int, default=30)
    s = sp.add_parser("longctx-prep"); s.add_argument("--out", required=True)
    s.add_argument("--lengths", type=int, nargs="+", default=[64000, 128000]); s.add_argument("--per-length", type=int, default=24)
    s = sp.add_parser("gen-think"); s.add_argument("--task", choices=["humaneval", "lcb"], required=True)
    s.add_argument("--data"); s.add_argument("--out", required=True); s.add_argument("--max-tokens", type=int, default=16384)
    s = sp.add_parser("lcb-exec"); s.add_argument("--data", required=True); s.add_argument("--gen", required=True)
    s = sp.add_parser("gen-longctx"); s.add_argument("--data", required=True); s.add_argument("--out", required=True)
    s = sp.add_parser("compare"); s.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"lcb-prep": cmd_lcb_prep, "longctx-prep": cmd_longctx_prep, "gen-think": cmd_gen_think,
     "lcb-exec": cmd_lcb_exec, "gen-longctx": cmd_gen_longctx, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
