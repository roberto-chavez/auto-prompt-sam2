"""Verify a locally downloaded Stage 8 report against the published SHA-256."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

EXPECTED = "aa95eee0fb110e4e94d39728d5d86ac1fcba06881fa8ea6c940e2c4e4e771f76"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Downloaded stage8_report.json")
    args = parser.parse_args()
    if not args.report.is_file():
        parser.error(f"file does not exist: {args.report}")
    actual = hashlib.sha256(args.report.read_bytes()).hexdigest()
    print(f"actual:   {actual}")
    print(f"expected: {EXPECTED}")
    if actual != EXPECTED:
        print("RESULT: MISMATCH")
        return 1
    print("RESULT: SHA-256 verified")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
