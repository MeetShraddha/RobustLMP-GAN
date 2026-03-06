#!/usr/bin/env python3
"""
Download EIA 930 hourly interchange data for PJM (2019–2023).

Usage::

    python scripts/download_interchange.py --api-key YOUR_EIA_KEY
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.config import get_cfg
from robustlmp_gan.data.loader import download_eia_interchange
from robustlmp_gan.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EIA 930 interchange data")
    parser.add_argument("--api-key", required=True, help="EIA v2 API key")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    cfg = get_cfg(args.config)
    icfg = cfg["eia_api"]

    download_eia_interchange(
        api_key=args.api_key,
        output_dir=cfg["data"]["eia_interchange_dir"],
        base_url=icfg["base_url"],
        page_size=icfg["page_size"],
        sleep_seconds=icfg["sleep_seconds"],
    )
    logging.getLogger(__name__).info("Interchange download complete.")


if __name__ == "__main__":
    main()
