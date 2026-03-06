#!/usr/bin/env python3
"""
Evaluate trained RobustLMP-GAN models.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --config my_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.config import get_cfg
from robustlmp_gan.main import run_pipeline
from robustlmp_gan.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RobustLMP-GAN models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    import logging
    setup_logging(level=getattr(logging, args.log_level))
    cfg = get_cfg(args.config)
    run_pipeline(cfg, stage="evaluate")


if __name__ == "__main__":
    main()
