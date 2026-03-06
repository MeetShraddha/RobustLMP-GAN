"""
Data loading utilities for RobustLMP-GAN.

Handles loading PJM LMP CSVs, EIA natural gas prices,
EIA interchange data, and NOAA weather data.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PJM LMP
# ─────────────────────────────────────────────────────────────────────────────

def load_pjm_lmp(
    raw_dir: str,
    pattern: str = "rt_da_monthly_lmps_20*.csv",
    date_start: str = "2019-01-01",
    date_end: str = "2024-01-01",
) -> pd.DataFrame:
    """Load and concatenate all PJM monthly LMP CSV files.

    Args:
        raw_dir: Directory containing the monthly CSV files.
        pattern: Glob pattern to match CSV filenames.
        date_start: Inclusive start date for filtering (ISO 8601).
        date_end: Exclusive end date for filtering (ISO 8601).

    Returns:
        Concatenated DataFrame filtered to the requested date range,
        with datetime parsed and spurious columns removed.

    Raises:
        FileNotFoundError: If no files match the pattern.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files found matching: {os.path.join(raw_dir, pattern)}"
        )
    logger.info("Found %d LMP files", len(files))

    df = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in files],
        ignore_index=True,
    )
    logger.info("Combined shape: %s", df.shape)

    # Drop columns that are always null in this dataset
    cols_to_drop = [c for c in ["voltage", "equipment"] if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info("Dropped null columns: %s", cols_to_drop)

    # Parse datetime (pandas 2.0+ compatible)
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Temporal filter
    df = df[
        (df["datetime_beginning_ept"] >= date_start)
        & (df["datetime_beginning_ept"] < date_end)
    ].copy()
    logger.info("Shape after date filter: %s", df.shape)
    return df


def filter_top_congestion_nodes(
    df: pd.DataFrame,
    node_types: list[str] | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Filter DataFrame to top-N high-congestion ZONE/HUB nodes.

    Nodes are ranked by mean absolute real-time congestion price.

    Args:
        df: Full LMP DataFrame (output of :func:`load_pjm_lmp`).
        node_types: List of allowed ``type`` values. Defaults to
            ``["ZONE", "HUB"]``.
        top_n: Number of highest-congestion nodes to keep.

    Returns:
        Filtered DataFrame containing only the selected nodes.
    """
    if node_types is None:
        node_types = ["ZONE", "HUB"]

    df = df[df["type"].isin(node_types)].copy()
    logger.info("Shape after node-type filter: %s", df.shape)

    node_congestion = (
        df.groupby("pnode_name")["congestion_price_rt"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"congestion_price_rt": "mean_abs_congestion"})
    )

    selected_n = min(top_n, len(node_congestion))
    top_nodes = node_congestion.head(selected_n)["pnode_name"].tolist()
    logger.info("Selected %d nodes by congestion rank", selected_n)

    return df[df["pnode_name"].isin(top_nodes)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# EIA Natural Gas Prices
# ─────────────────────────────────────────────────────────────────────────────

def load_eia_natural_gas(
    path: str,
    sheet_name: str = "Data 1",
    skiprows: int = 2,
    date_start: str = "2019-01-01",
    date_end: str = "2024-01-01",
) -> pd.DataFrame:
    """Load EIA natural gas prices and resample to monthly averages.

    Args:
        path: Path to the EIA ``.xls`` or ``.xlsx`` file.
        sheet_name: Excel sheet containing the price series.
        skiprows: Number of header rows to skip.
        date_start: Inclusive start date.
        date_end: Exclusive end date.

    Returns:
        DataFrame with columns ``period`` (month-start datetime) and
        ``ng_price_mmbtu``, plus a ``year_month`` Period column for
        merging.
    """
    df_ng = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows, header=0)
    df_ng.columns = ["date", "ng_price_mmbtu"]
    df_ng["date"] = pd.to_datetime(df_ng["date"])
    df_ng = df_ng[
        (df_ng["date"] >= date_start) & (df_ng["date"] < date_end)
    ].copy()

    # Resample daily → monthly average
    df_ng = (
        df_ng.set_index("date")
        .resample("MS")["ng_price_mmbtu"]
        .mean()
        .reset_index()
        .rename(columns={"date": "period"})
    )
    df_ng["year_month"] = df_ng["period"].dt.to_period("M")
    logger.info("Natural gas prices loaded: %d months", len(df_ng))
    return df_ng


# ─────────────────────────────────────────────────────────────────────────────
# EIA 930 Interchange (API download)
# ─────────────────────────────────────────────────────────────────────────────

_EIA_DATE_CHUNKS = [
    ("2019-01-01T00", "2019-03-31T23"),
    ("2019-04-01T00", "2019-06-30T23"),
    ("2019-07-01T00", "2019-09-30T23"),
    ("2019-10-01T00", "2019-12-31T23"),
    ("2020-01-01T00", "2020-03-31T23"),
    ("2020-04-01T00", "2020-06-30T23"),
    ("2020-07-01T00", "2020-09-30T23"),
    ("2020-10-01T00", "2020-12-31T23"),
    ("2021-01-01T00", "2021-03-31T23"),
    ("2021-04-01T00", "2021-06-30T23"),
    ("2021-07-01T00", "2021-09-30T23"),
    ("2021-10-01T00", "2021-12-31T23"),
    ("2022-01-01T00", "2022-03-31T23"),
    ("2022-04-01T00", "2022-06-30T23"),
    ("2022-07-01T00", "2022-09-30T23"),
    ("2022-10-01T00", "2022-12-31T23"),
    ("2023-01-01T00", "2023-03-31T23"),
    ("2023-04-01T00", "2023-06-30T23"),
    ("2023-07-01T00", "2023-09-30T23"),
    ("2023-10-01T00", "2023-12-31T23"),
]


def download_eia_interchange(
    api_key: str,
    output_dir: str = "eia930_interchange",
    base_url: str = "https://api.eia.gov/v2/electricity/rto/interchange-data/data/",
    page_size: int = 5000,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """Download PJM hourly interchange data from the EIA v2 API.

    Iterates over quarterly date chunks (2019–2023) and paginates
    within each chunk, collecting all rows.

    Args:
        api_key: EIA API key.
        output_dir: Directory to write the combined CSV.
        base_url: EIA API endpoint.
        page_size: Records per API request.
        sleep_seconds: Delay between paginated requests.

    Returns:
        Raw DataFrame of interchange rows.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_chunks: list[pd.DataFrame] = []

    for start, end in _EIA_DATE_CHUNKS:
        logger.info("Fetching interchange %s to %s", start, end)
        offset = 0

        while True:
            params: dict[str, Any] = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[fromba][]": "PJM",
                "start": start,
                "end": end,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": page_size,
            }
            try:
                resp = requests.get(base_url, params=params, timeout=60)
                data = resp.json()
                if "response" not in data:
                    logger.error("Unexpected API response: %s", data)
                    break
                rows = data["response"]["data"]
                if not rows:
                    break
                all_chunks.append(pd.DataFrame(rows))
                logger.debug("offset %d: fetched %d rows", offset, len(rows))
                if len(rows) < page_size:
                    break
                offset += page_size
                time.sleep(sleep_seconds)
            except Exception as exc:
                logger.error("Exception at offset %d: %s", offset, exc)
                break

    if not all_chunks:
        logger.warning("No interchange data downloaded")
        return pd.DataFrame()

    df_interchange = pd.concat(all_chunks, ignore_index=True)
    output_path = os.path.join(output_dir, "pjm_interchange_2019_2023.csv")
    df_interchange.to_csv(output_path, index=False)
    logger.info(
        "Interchange saved: %s (%d rows)", output_path, len(df_interchange)
    )
    return df_interchange


def load_interchange_csv(path: str) -> pd.DataFrame:
    """Load previously downloaded interchange CSV.

    Args:
        path: Path to the interchange CSV file.

    Returns:
        Processed DataFrame with hourly net interchange and lag features.
    """
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime_beginning_ept"] = pd.to_datetime(
        df["period"].str.replace("T", " ") + ":00:00"
    )
    hourly = (
        df.groupby("datetime_beginning_ept")["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "net_interchange_mw"})
    )
    hourly = hourly.sort_values("datetime_beginning_ept")
    hourly["net_interchange_lag1h"] = hourly["net_interchange_mw"].shift(1)
    hourly["net_interchange_lag24h"] = hourly["net_interchange_mw"].shift(24)
    logger.info("Hourly interchange shape: %s", hourly.shape)
    return hourly


# ─────────────────────────────────────────────────────────────────────────────
# NOAA Weather (API download)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_STATIONS = {
    "Philadelphia": "GHCND:USW00013739",
    "Pittsburgh": "GHCND:USW00094823",
    "Chicago": "GHCND:USW00094846",
    "Columbus_OH": "GHCND:USW00014821",
    "Washington_DC": "GHCND:USW00013743",
}


def download_noaa_weather(
    token: str,
    stations: dict[str, str] | None = None,
    output_dir: str = "noaa_weather",
    base_url: str = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
    datatypes: list[str] | None = None,
    page_size: int = 1000,
    sleep_seconds: float = 0.3,
    years: range | None = None,
) -> pd.DataFrame:
    """Download daily GHCND weather records from the NOAA CDO API.

    Args:
        token: NOAA CDO web services API token.
        stations: Mapping of city name → GHCND station ID. Defaults
            to five major PJM-footprint cities.
        output_dir: Directory to write the raw weather CSV.
        base_url: NOAA CDO API endpoint.
        datatypes: List of GHCND data type IDs to request.
        page_size: Records per API request (max 1000).
        sleep_seconds: Delay between paginated requests.
        years: Year range to download. Defaults to range(2019, 2024).

    Returns:
        Raw multi-city weather DataFrame.
    """
    if stations is None:
        stations = _DEFAULT_STATIONS
    if datatypes is None:
        datatypes = ["TMAX", "TMIN", "AWND", "PRCP"]
    if years is None:
        years = range(2019, 2024)

    os.makedirs(output_dir, exist_ok=True)
    all_weather: list[pd.DataFrame] = []
    headers = {"token": token}

    for city, station_id in stations.items():
        logger.info("Downloading weather for %s", city)
        for year in years:
            offset = 0
            while True:
                params: dict[str, Any] = {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "datatypeid": datatypes,
                    "startdate": f"{year}-01-01",
                    "enddate": f"{year}-12-31",
                    "limit": page_size,
                    "offset": offset,
                    "units": "metric",
                }
                try:
                    resp = requests.get(
                        base_url, headers=headers, params=params, timeout=30
                    )
                    data = resp.json()
                    if "results" not in data:
                        break
                    rows = data["results"]
                    df_chunk = pd.DataFrame(rows)
                    df_chunk["city"] = city
                    all_weather.append(df_chunk)
                    total = data["metadata"]["resultset"]["count"]
                    fetched = offset + len(rows)
                    if fetched >= total:
                        break
                    offset += page_size
                    time.sleep(sleep_seconds)
                except Exception as exc:
                    logger.error(
                        "%s %d offset %d: %s", city, year, offset, exc
                    )
                    break

    if not all_weather:
        logger.warning("No weather data downloaded")
        return pd.DataFrame()

    df_raw = pd.concat(all_weather, ignore_index=True)
    raw_path = os.path.join(output_dir, "pjm_weather_raw_2019_2023.csv")
    df_raw.to_csv(raw_path, index=False)
    logger.info("Raw weather saved: %s", raw_path)
    return df_raw


def process_weather_to_hourly(
    df_raw: pd.DataFrame,
    base_temp: float = 18.3,
    output_dir: str = "noaa_weather",
) -> pd.DataFrame:
    """Pivot, aggregate, and expand daily weather to hourly resolution.

    Steps:
        1. Pivot data type rows to columns.
        2. Average across all cities to create a single PJM daily series.
        3. Compute HDD and CDD from average temperature.
        4. Vectorised repeat of daily rows × 24 to produce hourly rows.

    Args:
        df_raw: Raw NOAA weather DataFrame (output of
            :func:`download_noaa_weather` or loaded from CSV).
        base_temp: Base temperature in °C for HDD/CDD (default 18.3 °C).
        output_dir: Directory to write the hourly weather CSV.

    Returns:
        Hourly weather DataFrame with a ``datetime_beginning_ept`` column.
    """
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.normalize()

    df_pivot = df_raw.pivot_table(
        index=["date", "city"],
        columns="datatype",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df_pivot.columns.name = None

    df_daily = (
        df_pivot.groupby("date")
        .agg(
            temp_max=("TMAX", "mean"),
            temp_min=("TMIN", "mean"),
            wind_speed=("AWND", "mean"),
            precip=("PRCP", "mean"),
        )
        .reset_index()
    )
    df_daily["temp_avg"] = (df_daily["temp_max"] + df_daily["temp_min"]) / 2
    df_daily["HDD"] = (base_temp - df_daily["temp_avg"]).clip(lower=0)
    df_daily["CDD"] = (df_daily["temp_avg"] - base_temp).clip(lower=0)

    # Vectorised expand: daily → hourly
    df_rep = df_daily.loc[df_daily.index.repeat(24)].reset_index(drop=True)
    df_rep["hour"] = list(range(24)) * len(df_daily)
    df_rep["datetime_beginning_ept"] = df_rep["date"] + pd.to_timedelta(
        df_rep["hour"], unit="h"
    )
    df_hourly = df_rep.drop(columns=["date", "hour"])

    hourly_path = os.path.join(output_dir, "pjm_weather_hourly_2019_2023.csv")
    df_hourly.to_csv(hourly_path, index=False)
    logger.info("Hourly weather saved: %s  shape=%s", hourly_path, df_hourly.shape)
    return df_hourly
