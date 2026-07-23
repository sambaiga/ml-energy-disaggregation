"""Feature engineering for clustering of consumption data.

This module extracts named, interpretable features (profiles, magnitude,
volatility, trend, shape, autocorrelation, calendar/peak, distribution)
from already-cleaned meter time series. Cleaning, normalisation, and season
labelling live in `ark.cluster.preprocessing` instead, a different concern:
this module only extracts features from data that module has already
prepared. The final output is a feature matrix (meters × features) ready
for clustering (e.g., K-Means).

All functions are fully vectorised and use pandas/numpy operations in C.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from ark.cluster.preprocessing import clean_time_series, map_to_seasons, normalize_by_daily_max

__all__ = [
    "FeatureConfig",
    "create_profile_features",
    "create_shape_features",
    "create_autocorrelation_features",
    "create_calendar_peak_features",
    "create_distribution_features",
    "create_magnitude_features",
    "create_volatility_features",
    "create_trend_features",
    "extract_features",
]

log = logging.getLogger(__name__)


class FeatureConfig(BaseModel):
    """Configuration for `extract_features`.

    Replaces a previous 10-parameter call signature: construct one
    `FeatureConfig` per experiment (never inside a k-loop), so a bad
    `hemisphere` value raises here, at construction, rather than deep
    inside `map_to_seasons`.
    """

    include_profile: bool = True
    profile_type: Literal["weekly", "seasonal", "both"] = "both"
    include_magnitude: bool = False
    include_volatility: bool = False
    include_trend: bool = False
    hemisphere: Literal["north", "south"] = "south"
    include_shape: bool = False
    include_autocorr: bool = False
    include_calendar_peak: bool = False
    include_distribution: bool = False

    @model_validator(mode="after")
    def _at_least_one_feature_group(self) -> "FeatureConfig":
        flags = (
            self.include_profile,
            self.include_magnitude,
            self.include_volatility,
            self.include_trend,
            self.include_shape,
            self.include_autocorr,
            self.include_calendar_peak,
            self.include_distribution,
        )
        if not any(flags):
            raise ValueError("at least one include_* feature group must be enabled")
        return self


def create_profile_features(
    normalized_data: pd.DataFrame,
    hemisphere: str = "south",
    timestamp_column: str = "timestamp",
    profile_type: Literal["weekly", "seasonal", "both"] = "both",
) -> pd.DataFrame:
    """Create weekly (7×24) and/or seasonal (4×24) hourly average profiles.

    Args:
        normalized_data: Normalised DataFrame with a DatetimeIndex or a timestamp column.
        hemisphere: "north" or "south" (default "south").
        timestamp_column: Name of the timestamp column.
        profile_type: Which profile summaries to return. Options are "weekly",
            "seasonal", or "both" (default).
    """
    df = normalized_data.copy()

    if timestamp_column not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            index_name = df.columns[0]
            if index_name != timestamp_column:
                df = df.rename(columns={index_name: timestamp_column})
        else:
            raise KeyError(f"{timestamp_column} not found in normalized_data columns and index is not a DatetimeIndex")

    df["hour"] = df[timestamp_column].dt.hour
    df["dayofweek"] = df[timestamp_column].dt.dayofweek
    df["season"] = map_to_seasons(df[timestamp_column], hemisphere=hemisphere)

    value_vars = [c for c in df.columns if c not in {timestamp_column, "hour", "dayofweek", "season"}]

    long = df.melt(
        id_vars=["hour", "dayofweek", "season"],
        value_vars=value_vars,
        var_name="meter",
        value_name="value",
    )

    profile_type = profile_type.lower()
    if profile_type not in {"weekly", "seasonal", "both"}:
        raise ValueError("profile_type must be 'weekly', 'seasonal', or 'both'")

    results = []

    if profile_type in {"weekly", "both"}:
        weekly = long.groupby(["meter", "dayofweek", "hour"])["value"].mean().unstack(["dayofweek", "hour"])
        weekly.columns = [f"w_{d}_{h}" for d, h in weekly.columns]
        results.append(weekly)

    if profile_type in {"seasonal", "both"}:
        seasonal = long.groupby(["meter", "season", "hour"])["value"].mean().unstack(["season", "hour"])
        seasonal.columns = [f"s_{s}_{h}" for s, h in seasonal.columns]
        results.append(seasonal)

    if len(results) == 1:
        return results[0]

    return pd.concat(results, axis=1)


def create_shape_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create shape-based features such as peak hour and load factor.

    Args:
        data: Input DataFrame with datetime index and meters as columns.

    Returns:
        DataFrame of shape features (meters × features).
    """
    if data.empty:
        return pd.DataFrame(index=data.columns)

    data = data.loc[:, ~data.columns.duplicated()]

    hourly_profile = data.groupby(data.index.hour).mean().T

    peak_hour = hourly_profile.idxmax(axis=1)
    morning = hourly_profile.loc[:, 6:10].mean(axis=1)
    evening = hourly_profile.loc[:, 17:21].mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        morning_evening_ratio = (morning / evening).replace([np.inf, -np.inf], 0).fillna(0)

    daily_totals = data.resample("D").sum()
    daily_max = data.resample("D").max()
    with np.errstate(divide="ignore", invalid="ignore"):
        load_factor = (daily_totals / (daily_max * 24)).fillna(0).mean(axis=0)

    features = pd.DataFrame(
        {
            "peak_hour": peak_hour,
            "morning_evening_ratio": morning_evening_ratio,
            "load_factor": load_factor,
        },
        index=data.columns,
    )
    return features[~features.index.duplicated(keep="first")]


def create_autocorrelation_features(
    data: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """Create autocorrelation-based features for each meter.

    Args:
        data: Input DataFrame with datetime index and meters as columns.
        lags: Lags to compute autocorrelation for.

    Returns:
        DataFrame of autocorrelation features (meters × lags).
    """
    rows = []

    for meter in data.columns:
        series = data[meter].dropna()
        row = {}
        for lag in lags:
            if len(series) > lag:
                row[f"autocorr_lag{lag}"] = float(series.autocorr(lag=lag))
            else:
                row[f"autocorr_lag{lag}"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows, index=data.columns)


def create_calendar_peak_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and peak-demand features from meter data.

    Args:
        data: Input DataFrame with datetime index and meters as columns.

    Returns:
        DataFrame of calendar/peak features (meters × features).
    """
    if data.empty:
        return pd.DataFrame(index=data.columns)

    daily = data.resample("D").sum()
    daily_max = data.resample("D").max()

    weekday_mean = daily[daily.index.dayofweek < 5].mean(axis=0)
    weekend_mean = daily[daily.index.dayofweek >= 5].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        weekday_weekend_ratio = (weekday_mean / weekend_mean).replace([np.inf, -np.inf], 0).fillna(0)

    peak_value = daily_max.max(axis=0)
    peak_to_avg = (peak_value / daily.mean(axis=0)).replace([np.inf, -np.inf], 0).fillna(0)

    morning_window = data.between_time("06:00", "10:00").resample("D").sum()
    evening_window = data.between_time("18:00", "22:00").resample("D").sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        morning_evening_ratio = (
            (morning_window.mean(axis=0) / evening_window.mean(axis=0)).replace([np.inf, -np.inf], 0).fillna(0)
        )

    return pd.DataFrame(
        {
            "daily_peak": peak_value,
            "peak_to_avg_ratio": peak_to_avg,
            "weekday_weekend_ratio": weekday_weekend_ratio,
            "morning_evening_ratio": morning_evening_ratio,
        },
        index=data.columns,
    )


def create_distribution_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create distributional features such as skewness and quantiles.

    Args:
        data: Input DataFrame with datetime index and meters as columns.

    Returns:
        DataFrame of distributional features (meters × features).
    """
    if data.empty:
        return pd.DataFrame(index=data.columns)

    daily = data.resample("D").sum()

    skewness = daily.skew(axis=0)
    kurtosis = daily.kurt(axis=0)
    q25 = daily.quantile(0.25, axis=0)
    q50 = daily.quantile(0.50, axis=0)
    q75 = daily.quantile(0.75, axis=0)
    iqr = q75 - q25

    return pd.DataFrame(
        {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "iqr": iqr,
        },
        index=data.columns,
    )


def create_magnitude_features(daily_totals: pd.DataFrame) -> pd.DataFrame:
    """Compute basic statistical magnitudes per meter from daily totals.

    Args:
        daily_totals: DataFrame of daily totals (meters × days).

    Returns:
        DataFrame of magnitude features (meters × features).
    """
    return pd.DataFrame(
        {
            "mean": daily_totals.mean(axis=0),
            "std": daily_totals.std(axis=0),
            "max": daily_totals.max(axis=0),
            "min": daily_totals.min(axis=0),
            "total": daily_totals.sum(axis=0),
        },
        index=daily_totals.columns,
    )


def create_volatility_features(daily_totals: pd.DataFrame) -> pd.DataFrame:
    """Compute CV and rolling std.

    Args:
        daily_totals: DataFrame of daily totals (meters × days).

    Returns:
        DataFrame of volatility features (meters × features).
    """
    vol_df = pd.DataFrame(index=daily_totals.columns)

    mean_vals = daily_totals.mean(axis=0)
    std_vals = daily_totals.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (std_vals / mean_vals).fillna(0).replace([np.inf, -np.inf], 0)
    vol_df["cv"] = cv

    roll_std = daily_totals.rolling(30).std().mean(axis=0) if len(daily_totals) >= 30 else daily_totals.std(axis=0)
    vol_df["rolling_std_30d"] = roll_std

    return vol_df


def create_trend_features(daily_totals: pd.DataFrame) -> pd.DataFrame:
    """Compute recent changes and linear slope.

    Args:
        daily_totals: DataFrame of daily totals (meters × days).

    Returns:
        DataFrame of trend features (meters × features).
    """
    n = len(daily_totals)
    trend_df = pd.DataFrame(index=daily_totals.columns)

    for lag in (1, 2, 3, 7):
        if n > lag:
            trend_df[f"change_lag{lag}"] = daily_totals.diff(lag).iloc[-1]
        else:
            trend_df[f"change_lag{lag}"] = 0.0

    if n > 1:
        x_centered = np.arange(n) - np.arange(n).mean()
        y_centered = daily_totals - daily_totals.mean(axis=0)
        slopes = (x_centered @ y_centered) / np.sum(x_centered**2)
        trend_df["slope"] = slopes
    else:
        trend_df["slope"] = 0.0

    return trend_df


def extract_features(data: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Extract a clustered-feature matrix from meter time-series data.

    Args:
        data: Input DataFrame with a datetime index and meters as columns.
        config: Which feature groups to include and how (see `FeatureConfig`).
            Defaults to `FeatureConfig()` (profile features only, "both"
            weekly+seasonal, southern-hemisphere season labels).

    Returns:
        A feature matrix with one row per meter and one column per extracted
        feature.
    """
    config = config or FeatureConfig()
    log.info("extracting features: %s", config.model_dump())

    cleaned = clean_time_series(data)
    daily = cleaned.resample("D").sum()

    feature_dfs = []

    if config.include_profile:
        normalized = normalize_by_daily_max(cleaned)
        profile_df = create_profile_features(
            normalized,
            hemisphere=config.hemisphere,
            profile_type=config.profile_type,
        )
        feature_dfs.append(profile_df)

    if config.include_magnitude:
        feature_dfs.append(create_magnitude_features(daily))
    if config.include_volatility:
        feature_dfs.append(create_volatility_features(daily))
    if config.include_trend:
        feature_dfs.append(create_trend_features(daily))
    if config.include_shape:
        feature_dfs.append(create_shape_features(cleaned))
    if config.include_autocorr:
        feature_dfs.append(create_autocorrelation_features(cleaned))
    if config.include_calendar_peak:
        feature_dfs.append(create_calendar_peak_features(cleaned))
    if config.include_distribution:
        feature_dfs.append(create_distribution_features(cleaned))

    return pd.concat(feature_dfs, axis=1)
