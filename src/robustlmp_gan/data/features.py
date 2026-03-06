"""
Feature engineering for RobustLMP-GAN.

Adds calendar features, autoregressive lags, rolling statistics,
DA-RT spreads, fuel price features, and weather-derived demand features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical time features.

    Adds:
        - ``hour_of_day``, ``day_of_week``, ``month``
        - ``is_weekend``, ``is_peak_hour``
        - Cyclical sine/cosine encodings for hour, day-of-week, and month

    Args:
        df: DataFrame with a ``datetime_beginning_ept`` column.

    Returns:
        DataFrame with calendar features appended (in-place copy).
    """
    df = df.copy()
    dt = df["datetime_beginning_ept"]

    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek   # 0 = Monday, 6 = Sunday
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].isin(range(7, 23)).astype(int)

    # Cyclical encoding — prevents the model treating hour 23 and 0 as far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.debug("Calendar features added")
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str = "total_lmp_rt",
    lag_hours: list[int] | None = None,
    seasonal_lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add autoregressive lag features grouped by node.

    Lag features are computed per-node to avoid mixing prices across
    different pricing nodes.

    Args:
        df: DataFrame sorted by ``[pnode_name, datetime_beginning_ept]``.
        target: Column to lag.
        lag_hours: Hourly lags (t-1 to t-24 by default).
        seasonal_lags: Additional seasonal lags (default: 48, 72, 168).

    Returns:
        DataFrame with lag columns appended.
    """
    if lag_hours is None:
        lag_hours = list(range(1, 25))
    if seasonal_lags is None:
        seasonal_lags = [48, 72, 168]

    df = df.copy()
    all_lags = lag_hours + seasonal_lags
    for lag in all_lags:
        df[f"lmp_lag_{lag}h"] = df.groupby("pnode_name")[target].shift(lag)

    logger.debug("Lag features added: %d lags", len(all_lags))
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "total_lmp_rt",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and standard deviation per node.

    Uses a 1-step shift before rolling to avoid look-ahead bias.

    Args:
        df: DataFrame sorted by ``[pnode_name, datetime_beginning_ept]``.
        target: Column to compute rolling statistics on.
        windows: Rolling window sizes in hours. Defaults to [6, 24, 168].

    Returns:
        DataFrame with rolling feature columns appended.
    """
    if windows is None:
        windows = [6, 24, 168]

    df = df.copy()
    for window in windows:
        df[f"lmp_rollmean_{window}h"] = df.groupby("pnode_name")[target].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
        df[f"lmp_rollstd_{window}h"] = df.groupby("pnode_name")[target].transform(
            lambda x: x.shift(1).rolling(window).std()
        )

    logger.debug("Rolling features added for windows: %s", windows)
    return df


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-ahead vs real-time price spread features.

    These capture the difference between forward price expectations and
    real-time settlement — a key signal for detecting adversarial
    manipulation patterns.

    Args:
        df: DataFrame containing both DA and RT LMP component columns.

    Returns:
        DataFrame with ``da_rt_spread``, ``congestion_spread``, and
        ``energy_spread`` columns appended.
    """
    df = df.copy()
    df["da_rt_spread"] = df["total_lmp_da"] - df["total_lmp_rt"]
    df["congestion_spread"] = df["congestion_price_da"] - df["congestion_price_rt"]
    df["energy_spread"] = (
        df["system_energy_price_da"] - df["system_energy_price_rt"]
    )
    logger.debug("Spread features added")
    return df


def add_fuel_price_features(
    df: pd.DataFrame,
    df_ng: pd.DataFrame,
    heat_rate: float = 7.5,
) -> pd.DataFrame:
    """Merge natural gas prices and compute implied gas LMP.

    Args:
        df: Main feature DataFrame with a ``datetime_beginning_ept`` column.
        df_ng: Natural gas price DataFrame (output of
            :func:`~robustlmp_gan.data.loader.load_eia_natural_gas`).
        heat_rate: Heat rate in MMBtu/MWh (standard CCGT assumption).

    Returns:
        DataFrame with ``ng_price_mmbtu``, ``implied_gas_lmp``, and
        ``fuel_price_spread`` columns merged in.
    """
    df = df.copy()
    df["year_month"] = df["datetime_beginning_ept"].dt.to_period("M")
    df = df.merge(df_ng[["year_month", "ng_price_mmbtu"]], on="year_month", how="left")
    df["implied_gas_lmp"] = df["ng_price_mmbtu"] * heat_rate
    df["fuel_price_spread"] = df["system_energy_price_rt"] - df["implied_gas_lmp"]
    logger.debug("Fuel price features added")
    return df


def add_interchange_features(
    df: pd.DataFrame,
    df_interchange: pd.DataFrame,
) -> pd.DataFrame:
    """Merge hourly net interchange data into the main DataFrame.

    Args:
        df: Main feature DataFrame.
        df_interchange: Hourly interchange DataFrame (output of
            :func:`~robustlmp_gan.data.loader.load_interchange_csv`).

    Returns:
        DataFrame with net interchange and its lag features merged in.
        Remaining NaN values are forward-filled.
    """
    df = df.copy()
    df = df.merge(df_interchange, on="datetime_beginning_ept", how="left")

    interchange_cols = [
        "net_interchange_mw",
        "net_interchange_lag1h",
        "net_interchange_lag24h",
    ]
    df[interchange_cols] = df[interchange_cols].ffill()
    logger.debug("Interchange features merged")
    return df


def add_weather_features(
    df: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge hourly weather features into the main DataFrame.

    Args:
        df: Main feature DataFrame.
        df_weather: Hourly weather DataFrame (output of
            :func:`~robustlmp_gan.data.loader.process_weather_to_hourly`).

    Returns:
        DataFrame with temperature, wind, precipitation, HDD, and CDD
        columns merged in and forward-filled.
    """
    df = df.copy()
    df = df.merge(df_weather, on="datetime_beginning_ept", how="left")

    weather_cols = ["temp_avg", "temp_max", "temp_min", "wind_speed", "precip", "HDD", "CDD"]
    df[weather_cols] = df[weather_cols].ffill()
    logger.debug("Weather features merged")
    return df


def clip_lmp_outliers(
    df: pd.DataFrame,
    low_pct: float = 0.01,
    high_pct: float = 0.99,
) -> pd.DataFrame:
    """Clip extreme LMP values to percentile bounds.

    Prevents Winter Storm Uri-type spikes from dominating training loss.

    Args:
        df: DataFrame containing ``total_lmp_rt`` and ``total_lmp_da``.
        low_pct: Lower clipping percentile.
        high_pct: Upper clipping percentile.

    Returns:
        DataFrame with clipped target columns.
    """
    df = df.copy()
    for col in ["total_lmp_rt", "total_lmp_da"]:
        lo = df[col].quantile(low_pct)
        hi = df[col].quantile(high_pct)
        df[col] = df[col].clip(lo, hi)
        logger.debug("Clipped %s to [%.2f, %.2f]", col, lo, hi)
    return df


def drop_lag_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any lag or rolling feature is NaN.

    The first 168 rows per node will always have NaN lags — this
    function removes those warm-up rows.

    Args:
        df: Feature DataFrame.

    Returns:
        DataFrame with NaN lag rows removed and index reset.
    """
    lag_cols = [
        c for c in df.columns
        if any(x in c for x in ["lmp_lag", "rollmean", "rollstd"])
    ]
    before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    logger.info("Dropped %d rows with NaN lags (kept %d)", before - len(df), len(df))
    return df


def build_feature_columns(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """Return the list of feature columns (excludes targets and IDs).

    Args:
        df: Feature DataFrame.
        exclude_cols: Columns to exclude. Defaults to a standard set of
            identifier and target columns.

    Returns:
        Ordered list of feature column names.
    """
    if exclude_cols is None:
        exclude_cols = [
            "datetime_beginning_utc",
            "datetime_beginning_ept",
            "pnode_id",
            "pnode_name",
            "type",
            "zone",
            "year_month",
            "total_lmp_rt",
            "total_lmp_da",
        ]
    return [c for c in df.columns if c not in exclude_cols]
