#!/usr/bin/env python
"""Local, model-free checks for stage 8: prints the protocol hash to freeze and shows the gate verdict.
Usage: python scripts/run_stage8_local_test.py [--print-hash] [revision_stage7_*.json]
Never opens reserved data. Exit 0 always; read the printed verdict."""
import json, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]
from auto_prompt_sam2.evaluation import gate_problems, protocol_sha256

def main():
    proto = json.loads((Path(__file__).resolve().parents[1] / "configs/stage8_protocol.json").read_text("utf-8"))
    print("protocol_sha256 =", protocol_sha256(proto))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        with open(args[0], encoding="utf-8") as stream:
            bad = gate_problems(json.load(stream))
        print("GATE:", "CERRADA" if bad else "ABIERTA")
        for problem in bad:
            print(" -", problem)


if __name__ == "__main__":
    main()
