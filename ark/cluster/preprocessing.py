"""Cleaning, normalisation, and season-labelling for meter time series, ahead of feature extraction.

This is a different concern from `ark.cluster.feature`: everything here
prepares already-loaded meter data (dropping bad meters, filling gaps,
normalising magnitude, labelling calendar season) for either direct
clustering input or for `ark.cluster.feature.extract_features` to build
named features on top of. Nothing here extracts a named, interpretable
feature itself.
"""

import logging

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pandera.pandas as pa

__all__ = [
    "clean_time_series",
    "normalize_by_daily_max",
    "normalize_shape",
    "map_to_seasons",
]

log = logging.getLogger(__name__)

_METER_TIMESERIES_SCHEMA = pa.DataFrameSchema(
    columns={
        r".*": pa.Column(
            float,
            checks=pa.Check(
                lambda s: not s.isna().all(),
                error="a meter column is entirely null",
            ),
            nullable=True,
            regex=True,
        )
    },
    index=pa.Index("datetime64[ns]"),
    strict=False,
)


def _validate_meter_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the input contract every preprocessing function assumes.

    Checked at the boundary, once, rather than re-checked deeper in the
    pipeline: a `DatetimeIndex`, at least one meter column, and no meter
    column that is entirely null (a real, silent orientation/data bug this
    contract would have caught immediately, instead of surfacing later as a
    confusing pandas error).

    Args:
        data: DataFrame of meter readings to validate.

    Returns:
        `data`, unchanged, once validation passes.

    Raises:
        ValueError: if `data` has zero columns.
        pandera.errors.SchemaErrors: every index/column violation collected
            together (`lazy=True`), not just the first one found.
    """
    if data.shape[1] == 0:
        raise ValueError("data must have at least one meter column")
    _METER_TIMESERIES_SCHEMA.validate(data, lazy=True)
    return data


def clean_time_series(
    data: pd.DataFrame,
    max_missing_ratio: float = 0.5,
) -> pd.DataFrame:
    """Drop meters with too many missing values, then interpolate remaining gaps.

    Args:
        data: DataFrame of hourly (or sub-daily) meter readings.
        max_missing_ratio: Maximum allowed ratio of missing values per meter.

    Returns:
        DataFrame with cleaned time series data.
    """
    _validate_meter_timeseries(data)

    thresh = int((1 - max_missing_ratio) * len(data))
    cleaned = data.dropna(axis=1, thresh=thresh)
    dropped = data.columns.difference(cleaned.columns)
    if len(dropped) > 0:
        log.warning(
            "clean_time_series dropped %d meter(s) exceeding max_missing_ratio=%.2f: %s",
            len(dropped),
            max_missing_ratio,
            list(dropped),
        )

    cleaned = cleaned.interpolate(method="time", limit_area="inside")
    cleaned = cleaned.fillna(cleaned.median())
    return cleaned


def normalize_by_daily_max(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise each day's consumption by its daily maximum.

    Args:
        data: DataFrame of hourly (or sub-daily) meter readings.

    Returns:
        DataFrame of normalised readings (meters × timestamps).
    """
    daily_max = data.resample("D").max()
    daily_max_repeated = daily_max.reindex(data.index, method="ffill")
    return (data / daily_max_repeated).fillna(0)


def normalize_shape(profiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalise each profile by its own peak, preserving shape while removing magnitude.

    Deduplicated from eight identical copies previously pasted verbatim
    across `book/05-clustering/`, `book/06-ranking-recommendation/`,
    `book/07-anomaly-detection/`, and `book/08-forecasting/` notebooks.

    Args:
        profiles: Array of shape `(..., n_timesteps)`; the trailing axis is
            the one normalized (e.g. a `(n_customers, n_timesteps)` season-
            mean shape, or a single `(n_timesteps,)` profile).

    Returns:
        `profiles` divided by its own per-row peak along the trailing axis.
        A row whose peak is zero is left at zero rather than producing NaN.
    """
    peak = profiles.max(axis=-1, keepdims=True)
    peak = np.where(peak == 0, 1, peak)
    return profiles / peak


def map_to_seasons(dt_index: pd.DatetimeIndex | pd.Series, hemisphere: str = "south") -> np.ndarray:
    """Map a datetime index or timestamp series to seasons.

    Args:
        dt_index: Pandas DatetimeIndex or datetime Series.
        hemisphere: "north" or "south" (default: "south" for Australia).

    Returns:
        NumPy array of strings: 'spring', 'summer', 'autumn', or 'winter'.
    """
    if hemisphere.lower() not in ("north", "south"):
        raise ValueError("hemisphere must be 'north' or 'south'")

    if isinstance(dt_index, pd.Series):
        dt_index = dt_index.dt

    doy = dt_index.dayofyear
    seasons = np.full(len(doy), "winter", dtype=object)

    if hemisphere.lower() == "south":
        # Southern Hemisphere (AusNet default)
        seasons[(doy >= 335) | (doy <= 59)] = "summer"  # Dec-Feb
        seasons[(doy >= 60) & (doy < 152)] = "autumn"  # Mar-May
        seasons[(doy >= 152) & (doy < 244)] = "winter"  # Jun-Aug
        seasons[(doy >= 244) & (doy < 335)] = "spring"  # Sep-Nov
    else:
        # Northern Hemisphere (London, GoiEner)
        seasons[(doy >= 80) & (doy < 172)] = "spring"  # Mar-May
        seasons[(doy >= 172) & (doy < 264)] = "summer"  # Jun-Aug
        seasons[(doy >= 264) & (doy < 355)] = "autumn"  # Sep-Nov
        # Rest remains winter (Dec-Feb)

    return seasons
