"""
Shared utility functions for RobustLMP-GAN.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (both CPU
    and CUDA).

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger with console and optional file handler.

    Args:
        level: Logging level (default: ``logging.INFO``).
        log_file: Optional path to write logs. If ``None``, logs only
            to the console.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def get_device() -> torch.device:
    """Return the best available torch device.

    Returns:
        ``torch.device("cuda")`` if a GPU is available, else
        ``torch.device("cpu")``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_temporal_splits(
    df,
    train_end: str = "2023-01-01",
    val_end: str = "2023-07-01",
    dt_col: str = "datetime_beginning_ept",
):
    """Split a DataFrame into train, validation, and test subsets.

    Uses strictly temporal splitting (no shuffling) to prevent
    look-ahead bias.

    Args:
        df: DataFrame with a datetime column.
        train_end: Exclusive upper bound for training set.
        val_end: Exclusive upper bound for validation set.
        dt_col: Name of the datetime column.

    Returns:
        Tuple of ``(df_train, df_val, df_test)``.
    """
    train_mask = df[dt_col] < train_end
    val_mask = (df[dt_col] >= train_end) & (df[dt_col] < val_end)
    test_mask = df[dt_col] >= val_end
    return (
        df[train_mask].copy(),
        df[val_mask].copy(),
        df[test_mask].copy(),
    )
