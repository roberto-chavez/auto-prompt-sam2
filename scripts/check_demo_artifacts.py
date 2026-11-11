"""Read-only preflight. Safe on PC: no imports of torch, SAM or Gradio."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_prompt_sam2.demo_config import inspect_artifacts, load_demo_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON; relative paths are resolved from repository root")
    args = parser.parse_args()
    try:
        result = inspect_artifacts(load_demo_config(args.config))
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "invalid", "error": str(error), "models_executed": False}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
