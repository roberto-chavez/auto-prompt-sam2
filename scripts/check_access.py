"""Token, if any, is passed through stdin, never argv or a stored login."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.access import check_access
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json
from auto_prompt_sam2.runtime import load_config, run_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-stdin", action="store_true")
    args = parser.parse_args()
    token = sys.stdin.read().strip() if args.token_stdin else None
    config = load_config(REPO)
    report = check_access(config, token)
    token = None
    ArtifactStore(args.output).publish("access", canonical_json(report), run_identity(REPO, config))
    print(json.dumps(report, indent=2))  # Allowlisted metadata only; no headers or exception text.


if __name__ == "__main__":
    main()
