"""Unit tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.data.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_spread_features,
    clip_lmp_outliers,
    drop_lag_nans,
    build_feature_columns,
)


@pytest.fixture
def sample_df():
    """Minimal two-node DataFrame for testing."""
    dates = pd.date_range("2021-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "datetime_beginning_ept": list(dates) * 2,
        "pnode_name": ["A"] * 200 + ["B"] * 200,
        "total_lmp_rt": np.random.uniform(20, 60, 400),
        "total_lmp_da": np.random.uniform(20, 60, 400),
        "congestion_price_rt": np.random.uniform(-5, 5, 400),
        "congestion_price_da": np.random.uniform(-5, 5, 400),
        "system_energy_price_rt": np.random.uniform(20, 50, 400),
        "system_energy_price_da": np.random.uniform(20, 50, 400),
    })
    return df.sort_values(["pnode_name", "datetime_beginning_ept"]).reset_index(drop=True)


def test_add_calendar_features_columns(sample_df):
    out = add_calendar_features(sample_df)
    expected = [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "is_peak_hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    ]
    for col in expected:
        assert col in out.columns, f"Missing column: {col}"


def test_cyclical_encoding_range(sample_df):
    out = add_calendar_features(sample_df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert out[col].between(-1, 1).all(), f"{col} out of [-1, 1]"


def test_is_weekend_binary(sample_df):
    out = add_calendar_features(sample_df)
    assert set(out["is_weekend"].unique()).issubset({0, 1})


def test_add_lag_features_creates_columns(sample_df):
    out = add_lag_features(sample_df, lag_hours=[1, 2], seasonal_lags=[24])
    for col in ["lmp_lag_1h", "lmp_lag_2h", "lmp_lag_24h"]:
        assert col in out.columns


def test_lag_features_grouped_by_node(sample_df):
    out = add_lag_features(sample_df, lag_hours=[1], seasonal_lags=[])
    # First row of each node should be NaN
    for node in ["A", "B"]:
        first_idx = out[out["pnode_name"] == node].index[0]
        assert pd.isna(out.loc[first_idx, "lmp_lag_1h"])


def test_add_rolling_features(sample_df):
    df2 = add_lag_features(sample_df, lag_hours=[1], seasonal_lags=[])
    out = add_rolling_features(df2, windows=[6])
    assert "lmp_rollmean_6h" in out.columns
    assert "lmp_rollstd_6h" in out.columns


def test_add_spread_features(sample_df):
    out = add_spread_features(sample_df)
    assert "da_rt_spread" in out.columns
    assert "congestion_spread" in out.columns
    assert "energy_spread" in out.columns
    # Verify arithmetic
    expected = sample_df["total_lmp_da"] - sample_df["total_lmp_rt"]
    pd.testing.assert_series_equal(out["da_rt_spread"].reset_index(drop=True),
                                   expected.reset_index(drop=True))


def test_clip_lmp_outliers(sample_df):
    # Inject extreme outliers
    df = sample_df.copy()
    df.loc[0, "total_lmp_rt"] = 99999
    df.loc[1, "total_lmp_rt"] = -99999
    out = clip_lmp_outliers(df, low_pct=0.01, high_pct=0.99)
    assert out["total_lmp_rt"].max() < 99999
    assert out["total_lmp_rt"].min() > -99999


def test_drop_lag_nans(sample_df):
    df2 = add_lag_features(sample_df, lag_hours=[1], seasonal_lags=[])
    before = len(df2)
    out = drop_lag_nans(df2)
    assert len(out) < before
    assert out["lmp_lag_1h"].isna().sum() == 0


def test_build_feature_columns_excludes_targets(sample_df):
    cols = build_feature_columns(sample_df)
    assert "total_lmp_rt" not in cols
    assert "total_lmp_da" not in cols
    assert "pnode_name" not in cols
