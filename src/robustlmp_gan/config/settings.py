"""
Configuration loader for RobustLMP-GAN.

Loads YAML config and exposes a typed settings object used
throughout the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Default config location (can be overridden via env var)
_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Path to YAML config file. Defaults to the bundled
              ``config.yaml`` or the ``ROBUSTLMP_CONFIG`` env var.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    if path is None:
        path = os.environ.get("ROBUSTLMP_CONFIG", _DEFAULT_CONFIG)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_cfg(path: str | Path | None = None) -> dict[str, Any]:
    """Convenience alias for :func:`load_config`."""
    return load_config(path)
