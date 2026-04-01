"""
Surveillance Pipeline — CLI entry point.

Usage examples:
  python run.py
  python run.py --config config/default.yaml
  python run.py --input assets/sample_scene.mp4 --no-display
  python run.py --loitering-threshold 10 --crowd-threshold 5
  python run.py --confidence 0.5 --output-dir output/run2
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from the surveillance-pipeline/ directory without installing
sys.path.insert(0, str(Path(__file__).parent))

from surveillance_pipeline.config import load_config
from surveillance_pipeline.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Multi-Target Surveillance Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to YAML config file (defaults to config/default.yaml if it exists)",
    )
    parser.add_argument("--input",  metavar="PATH", help="Override video input path")
    parser.add_argument("--output-dir", metavar="PATH", help="Override output directory")
    parser.add_argument("--confidence", type=float, metavar="F", help="Detection confidence threshold (0–1)")
    parser.add_argument("--loitering-threshold", type=float, metavar="S", help="Loitering alert threshold in seconds")
    parser.add_argument("--crowd-threshold", type=int, metavar="N", help="Crowd alert object count threshold")
    parser.add_argument("--no-display", action="store_true", help="Disable live display window")
    parser.add_argument("--no-save", action="store_true", help="Disable saving annotated video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve config path: explicit arg → default file → empty (all defaults)
    config_path = args.config
    if config_path is None:
        default_cfg = Path(__file__).parent / "config" / "default.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    # Build overrides dict from CLI args
    overrides = {
        "input":               args.input,
        "output_dir":          args.output_dir,
        "confidence":          args.confidence,
        "loitering_threshold": args.loitering_threshold,
        "crowd_threshold":     args.crowd_threshold,
        "no_display":          args.no_display or None,
        "no_save":             args.no_save or None,
    }
    # Remove None values so they don't shadow YAML values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config = load_config(config_path, overrides)
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
