"""Run the preview locally; real inference is restricted to remote GPU runtimes."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auto_prompt_sam2.demo_config import load_demo_config
from auto_prompt_sam2.demo_ui import build_demo


if __name__ == "__main__":
    config = load_demo_config()
    build_demo(config).launch(
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False, max_file_size=f"{config['max_upload_mb']}mb",
        show_error=False,
    )
