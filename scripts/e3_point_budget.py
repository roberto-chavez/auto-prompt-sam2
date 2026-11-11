#!/usr/bin/env python
"""Total the ACTUAL points used per stage-7 row from persisted result_*.json (run in Colab or locally).

Usage: python scripts/e3_point_budget.py DIR [DIR...]
DIR contains result_a1_points.json, result_a1_assisted.json, ... (e.g. the ones inside evidence.zip or the Drive
stage7 generations). Counts come from the per-chip 'counts' blocks written by the experiment (never nominal k*chips).
Prints a table and writes e3_point_budget.json to the current directory. Reads JSON only; no model, no labels.
"""
import json, sys
from pathlib import Path

KEYS = ("components", "positive_points", "negative_points", "positive_shortfall", "negative_shortfall")


def walk(o, found):
    if isinstance(o, dict):
        c = o.get("counts")
        if isinstance(c, dict) and "positive_points" in c:
            found.append(c)
            return
        for v in o.values():
            walk(v, found)
    elif isinstance(o, list):
        for v in o:
            walk(v, found)


def main(dirs):
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("result_*.json")):
            found = []
            walk(json.loads(f.read_text("utf-8")), found)
            if found:
                out[f"{d}/{f.stem[7:]}"] = {"chips": len(found), **{k: sum(c.get(k, 0) for c in found) for k in KEYS}}
    for k, v in out.items():
        print(k, v)
    Path("e3_point_budget.json").write_text(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
