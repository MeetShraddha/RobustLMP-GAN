#!/usr/bin/env python3
"""
Train RobustLMP-GAN models.

Usage::

    python scripts/train.py                          # full pipeline
    python scripts/train.py --stage wgan            # WGAN only
    python scripts/train.py --stage lstm            # LSTM only
    python scripts/train.py --config my_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.config import get_cfg
from robustlmp_gan.main import run_pipeline
from robustlmp_gan.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RobustLMP-GAN")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "data", "wgan", "lstm"],
        default="all",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    import logging
    setup_logging(level=getattr(logging, args.log_level))
    cfg = get_cfg(args.config)
    run_pipeline(cfg, stage=args.stage)


if __name__ == "__main__":
    main()
