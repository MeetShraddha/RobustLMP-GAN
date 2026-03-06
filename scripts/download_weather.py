#!/usr/bin/env python3
"""
Download NOAA GHCND daily weather data for PJM-footprint cities (2019–2023)
and expand to hourly resolution.

Usage::

    python scripts/download_weather.py --token YOUR_NOAA_TOKEN
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.config import get_cfg
from robustlmp_gan.data.loader import download_noaa_weather, process_weather_to_hourly
from robustlmp_gan.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NOAA weather data")
    parser.add_argument("--token", required=True, help="NOAA CDO API token")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    cfg = get_cfg(args.config)
    ncfg = cfg["noaa_api"]

    df_raw = download_noaa_weather(
        token=args.token,
        stations=ncfg["stations"],
        output_dir=cfg["data"]["noaa_weather_dir"],
        base_url=ncfg["base_url"],
        datatypes=ncfg["datatypes"],
        page_size=ncfg["page_size"],
        sleep_seconds=ncfg["sleep_seconds"],
    )
    process_weather_to_hourly(
        df_raw,
        base_temp=ncfg["base_temp_celsius"],
        output_dir=cfg["data"]["noaa_weather_dir"],
    )
    logging.getLogger(__name__).info("Weather download and processing complete.")


if __name__ == "__main__":
    main()
