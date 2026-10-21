#!/usr/bin/env python3
"""stage_pi_compaction.py: apply / verify / revert the client-side compaction staging documented in PI.md.

Why this exists: pi compacts when context > contextWindow - reserveTokens and then keeps only the last
keepRecentTokens verbatim. 2026-10-15 (config C, several agents) staged 65536 / 20000 to cap cold-prefill stalls.
2026-10-21 (FP4-Local: warm hits recompute 0, mostly one user) moved to 40000 / 40000: compacts at ~222K, keeps twice
the recent work, and the reserve still covers pi's maxTokens 32000 + its 4096 safety margin (see PI.md). This touches
only the pi client config: no server change, reversible.

  scripts/stage_pi_compaction.py --check            # report current values and the resulting threshold
  scripts/stage_pi_compaction.py                    # apply (backup first, idempotent)
  scripts/stage_pi_compaction.py --revert           # restore the newest backup matching settings.json.bak-*
  scripts/stage_pi_compaction.py --reserve 65536 --keep 20000   # the 2026-10-15 values

Stdlib only. Refuses to touch a file it cannot parse, and never rewrites the file if the values already match.
"""
import argparse, glob, json, os, shutil, sys
from pathlib import Path

TARGET_RESERVE = 40000           # >= pi maxTokens 32000 + CONTEXT_SAFETY_TOKENS 4096, so a full response always fits
TARGET_KEEP = 40000
DEFAULT_WINDOW = 262144          # --max-model-len of the served model (FP4-Local and config C); pi/models.json.example uses it too


def settings_path():
    return Path(os.environ.get("PI_AGENT_DIR", Path.home() / ".pi" / "agent")) / "settings.json"


def load(p):
    if not p.exists():
        sys.exit(f"settings file not found: {p}")
    data = json.loads(p.read_text())          # raise rather than clobber a file we cannot parse
    if not isinstance(data, dict):
        sys.exit(f"{p} is not a JSON object; refusing to edit")
    return data


def report(data, window):
    c = data.get("compaction", {})
    reserve = c.get("reserveTokens", 16384)   # pi's defaults if unset
    keep = c.get("keepRecentTokens", 20000)
    print(f"  enabled={c.get('enabled', True)}  reserveTokens={reserve}  keepRecentTokens={keep}")
    print(f"  compacts above {window - reserve} context tokens (window {window})")
    return reserve, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve", type=int, default=TARGET_RESERVE)
    ap.add_argument("--keep", type=int, default=TARGET_KEEP)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="context window of the served model")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    p = settings_path()
    backs = sorted(glob.glob(str(p) + ".bak-*"), key=os.path.getmtime)   # newest last, whatever the suffix

    if a.revert:
        if not backs:
            sys.exit("no settings.json.bak-* to revert to")
        shutil.copyfile(backs[-1], p)
        print(f"reverted from {backs[-1]}")
        report(load(p), a.window)
        return

    data = load(p)
    print(f"{p}")
    reserve, keep = report(data, a.window)
    if a.check:
        return
    c = data.setdefault("compaction", {})
    if c.get("reserveTokens") == a.reserve and c.get("keepRecentTokens") == a.keep:
        print("already staged; nothing written")
        return
    if backs:
        print(f"note: existing backup {backs[-1]}")
    backup = Path(str(p) + ".bak-" + __import__("time").strftime("%Y%m%d-%H%M%S"))
    shutil.copyfile(p, backup)
    c["enabled"] = True
    c["reserveTokens"] = a.reserve
    c["keepRecentTokens"] = a.keep
    p.write_text(json.dumps(data, indent=2) + "\n")
    print(f"backup: {backup.name}")
    after = report(load(p), a.window)
    ok = after == (a.reserve, a.keep)
    print("VERIFIED" if ok else "VERIFY FAILED — check the file")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
