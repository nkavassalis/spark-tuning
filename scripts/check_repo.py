#!/usr/bin/env python3
"""Offline drift check for this repo's docs and switch scripts. Run before every commit (AGENTS.md "Keeping the docs current").

Catches the kinds of drift the 2026-10-24 audit found by hand:
  1. a <model>/up.sh that starts a container without stop_other_models, or a container missing from MODEL_CONTAINERS
     (scripts/models.sh), i.e. a model that could load next to another one;
  2. docs naming a repo path that doesn't exist, or a link/anchor to a .md heading that doesn't exist;
  3. docs naming a script flag (`glm/down.sh --fp4local`) or env knob (`LIVE=1 fp4local/up.sh`) the script doesn't have;
  4. undated statements of current state ("currently", "today", "(current)", ...): every claim about what is live or
     installed must carry a YYYY-MM-DD in the same paragraph, so a reader can tell when it stops being true.
Stdlib only. Exit 1 on any finding.
"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
problems = []
def bad(where, msg): problems.append(f"{where}: {msg}")

# ---- 1. one model at a time ------------------------------------------------------------------------------------------
known = set(re.search(r'MODEL_CONTAINERS="([^"]*)"', open("scripts/models.sh").read()).group(1).split())
for lib in glob.glob("*/lib.sh"):
    m = re.search(r"^CNAME=(\S+)", open(lib).read(), re.M)
    if m and m.group(1) not in known:
        bad(lib, f"container {m.group(1)} is not in MODEL_CONTAINERS (scripts/models.sh)")
for up in glob.glob("*/up.sh"):
    s = open(up).read()
    if "docker run" not in s:
        continue
    if 'scripts/models.sh"' not in s or not re.search(r"^stop_other_models\b", s, re.M):
        bad(up, "starts a container but does not source scripts/models.sh and call stop_other_models")
    for name in re.findall(r"--name ([A-Za-z][\w.-]+)", s):
        if name not in known:
            bad(up, f"starts container {name}, which is not in MODEL_CONTAINERS")

# ---- docs ------------------------------------------------------------------------------------------------------------
docs = sorted(glob.glob("*.md"))
TOPDIRS = {d for d in os.listdir(".") if os.path.isdir(d) and not d.startswith(".")}

def slug(h):   # GitHub heading anchor
    h = re.sub(r"[`*_\[\]()]", lambda m: "_" if m.group(0) == "_" else "", h.strip().lower())
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")
anchors = {d: {slug(h) for h in re.findall(r"^#+ (.+)$", open(d).read(), re.M)} for d in docs}

def lineno(text, pos): return text.count("\n", 0, pos) + 1

DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b(0[1-9]|1[0-2])-[0-3]\d\b")
NOW = re.compile(r"\b(currently|right now|now running|now serving|nowadays|today|at the moment|not installed now|"
                 r"installed now|is the production|in production now|what pi sends)\b|\((current|now)\)", re.I)

for d in docs:
    text = open(d).read()
    # 2a. repo paths in backticks
    for m in re.finditer(r"`([^`\s]+)", text):
        tok = m.group(1).rstrip(".,;:)")
        first = tok.split("/")[0]
        if "/" not in tok or first not in TOPDIRS or re.search(r"[*<>{}$\[|]", tok):
            continue
        if not os.path.exists(tok):
            bad(f"{d}:{lineno(text, m.start())}", f"path `{tok}` does not exist")
    # 2b. links to local .md files and anchors
    for m in re.finditer(r"\]\(([^)\s]+)\)", text):
        target = m.group(1)
        if re.match(r"[a-z]+://", target) or target.startswith("mailto:"):
            continue
        path, _, anchor = target.partition("#")
        path = path or d
        if not os.path.exists(path):
            bad(f"{d}:{lineno(text, m.start())}", f"link to missing {path}")
        elif anchor and path.endswith(".md") and anchor not in anchors.get(path, set()):
            bad(f"{d}:{lineno(text, m.start())}", f"link to missing heading {path}#{anchor}")
    # 3. script flags and env knobs
    for m in re.finditer(r"((?:[A-Z_]+=\S+ +)*)\b((?:%s)/[\w-]+\.(?:sh|py))((?: +--?[\w-]+(?:\|--?[\w-]+)*)*)"
                         % "|".join(sorted(TOPDIRS)), text):
        envs, script, flags = m.groups()
        if not os.path.exists(script):
            continue   # reported by 2a when backticked
        src = open(script).read()
        lib = os.path.join(os.path.dirname(script), "lib.sh")
        src += open(lib).read() if os.path.exists(lib) else ""
        where = f"{d}:{lineno(text, m.start())}"
        for f in re.findall(r"--?[\w-]+", flags or ""):
            if f not in src:
                bad(where, f"`{script} {f}`: flag not found in {script}")
        for v in re.findall(r"([A-Z_]+)=", envs or ""):
            if not re.search(r"\b%s\b" % v, src):
                bad(where, f"`{v}=... {script}`: knob not read by {script} or its lib.sh")
    # 4. undated current-state wording, judged per paragraph (a fenced code block counts as one paragraph)
    pos = 0
    for para in text.split("\n\n"):
        m = NOW.search(para)
        if m and not DATE.search(para):
            bad(f"{d}:{lineno(text, pos + m.start())}", f"undated current-state wording \"{m.group(0)}\": add the date it was true")
        pos += len(para) + 2

if problems:
    print("\n".join(problems))
    print(f"\n{len(problems)} problem(s); see AGENTS.md \"Keeping the docs current\"")
    sys.exit(1)
print(f"check_repo: ok ({len(docs)} docs, {len(glob.glob('*/up.sh'))} up.sh)")
