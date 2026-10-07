#!/usr/bin/env python3
"""Functional checks for the MiMo deployment, at the level an agent sees it. Stdlib only.

  mimo/test_mimo.py                       # uses $API (cluster.env) -> proxy at $API/mimo/v1, raw at :8888
  mimo/test_mimo.py --proxy URL --raw URL

Checks:
  1. plain chat via the proxy: answer lands in `content` (thinking off at the server), not in reasoning
  2. single tool call via the proxy: parsed into `tool_calls` with valid JSON arguments
  3. tool-call storm guard: a streamed request that asks for 12 parallel calls gets at most 6 back,
     ending with finish_reason=tool_calls (the proxy cut it). The raw endpoint is also asked, to show
     what the model does unguarded (informational: sampling is random, so a given run may stay under 6;
     the proxy logs 'tool-call storm guard tripped' in ~/mimo-toolcap/proxy.log on the head when it cuts).
Exit code 0 only if 1-3 pass. Requests send no sampling params, like the agents do (server defaults apply).
"""
import argparse, json, os, sys, time, urllib.request
from urllib.parse import urlsplit

MODEL = "mimo-v2.6-flash"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Delhi", "Quito", "Rome", "Seoul", "Accra", "Hanoi", "Bern"]


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(url, body):
    with post(url, body) as r:
        return json.load(r)


def stream_tool_calls(url, body):
    """Return (distinct tool-call indices, final finish_reason, args by index) from an SSE response."""
    body = dict(body, stream=True)
    calls, finish = {}, None
    with post(url, body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                for tc in (c.get("delta") or {}).get("tool_calls") or []:
                    calls.setdefault(tc["index"], "")
                    calls[tc["index"]] += (tc.get("function") or {}).get("arguments") or ""
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    return calls, finish


def main():
    api = os.environ.get("API", "http://localhost:8000")
    host = urlsplit(api).hostname or "localhost"
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=api.rstrip("/") + "/mimo/v1")
    ap.add_argument("--raw", default=f"http://{host}:8888/v1")
    a = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    t = time.time()
    r = chat(a.proxy, {"model": MODEL, "max_tokens": 64,
                       "messages": [{"role": "user", "content": "What is 17*23? Reply with just the number."}]})
    m = r["choices"][0]["message"]
    check("1 plain chat via proxy", "391" in (m.get("content") or ""),
          f"content={m.get('content')!r} reasoning={'yes' if m.get('reasoning_content') or m.get('reasoning') else 'no'} {time.time()-t:.1f}s")

    r = chat(a.proxy, {"model": MODEL, "max_tokens": 256, "tools": TOOLS,
                       "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]})
    tcs = r["choices"][0]["message"].get("tool_calls") or []
    args_ok = False
    if tcs:
        try:
            args_ok = "paris" in json.loads(tcs[0]["function"]["arguments"]).get("city", "").lower()
        except Exception:
            pass
    check("2 single tool call parsed", len(tcs) == 1 and args_ok,
          f"n={len(tcs)} finish={r['choices'][0]['finish_reason']} args={tcs[0]['function']['arguments'] if tcs else None}")

    storm = {"model": MODEL, "max_tokens": 2048, "tools": TOOLS, "messages": [{"role": "user", "content":
             "Call get_weather once for EACH of these cities, all in parallel in this single response, "
             "one tool call per city: " + ", ".join(CITIES) + "."}]}
    calls, finish = stream_tool_calls(a.raw, storm)
    print(f"[info] raw endpoint (unguarded): {len(calls)} tool calls, finish_reason={finish}")
    calls, finish = stream_tool_calls(a.proxy, storm)
    valid = sum(1 for v in calls.values() if _json_ok(v))
    check("3 storm guard via proxy", len(calls) <= 6 and valid == len(calls) and
          finish == "tool_calls", f"{len(calls)} calls ({valid} with complete JSON args), finish_reason={finish}")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


def _json_ok(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
