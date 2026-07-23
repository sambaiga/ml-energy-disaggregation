import numpy as np
import pandas as pd
import pandera.errors
import pytest

from ark.cluster.preprocessing import clean_time_series, map_to_seasons, normalize_by_daily_max, normalize_shape


def test_clean_time_series_drops_meters_above_missing_threshold():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    # one meter mostly missing, one meter fully present: only the second should survive.
    mostly_missing = pd.Series([np.nan] * 8 + [1.0, 2.0], index=idx)
    clean = pd.Series(range(10), index=idx, dtype=float)
    df = pd.DataFrame({"bad": mostly_missing, "good": clean})

    cleaned = clean_time_series(df, max_missing_ratio=0.5)

    assert list(cleaned.columns) == ["good"]


def test_clean_time_series_interpolates_interior_gaps_only():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    # leading NaN, one interior gap, trailing NaN: interpolation must only
    # touch the interior gap; the rest falls back to the column median.
    series = pd.Series([np.nan, np.nan, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, np.nan], index=idx)
    df = pd.DataFrame({"m": series})

    cleaned = clean_time_series(df, max_missing_ratio=0.9)

    assert not cleaned["m"].isna().any()
    assert cleaned["m"].iloc[4] == 4.0  # interior gap, linearly interpolated between 3 and 5


def test_normalize_by_daily_max_handles_all_zero_day():
    idx = pd.date_range("2023-01-01", periods=48, freq="30min")
    values = np.zeros(48)
    df = pd.DataFrame({"m": values}, index=idx)

    normalized = normalize_by_daily_max(df)

    # a day with no consumption at all must not produce inf/NaN from a 0/0 division.
    assert not normalized["m"].isna().any()
    assert np.isfinite(normalized["m"]).all()
    assert (normalized["m"] == 0).all()


def test_map_to_seasons_south_matches_known_dates():
    dates = pd.to_datetime(["2023-01-15", "2023-07-15"])

    seasons = map_to_seasons(dates, hemisphere="south")

    assert list(seasons) == ["summer", "winter"]


def test_map_to_seasons_north_matches_known_dates():
    dates = pd.to_datetime(["2023-01-15", "2023-07-15"])

    seasons = map_to_seasons(dates, hemisphere="north")

    assert list(seasons) == ["winter", "summer"]


def test_map_to_seasons_invalid_hemisphere_raises():
    dates = pd.to_datetime(["2023-01-15"])

    with pytest.raises(ValueError, match="hemisphere"):
        map_to_seasons(dates, hemisphere="east")


def test_normalize_shape_zero_peak_returns_zero_not_nan():
    profiles = np.zeros((1, 5))

    normalized = normalize_shape(profiles)

    # the np.where(peak == 0, 1, peak) guard exists exactly for this case.
    assert not np.isnan(normalized).any()
    assert (normalized == 0).all()


def test_normalize_shape_peak_becomes_one():
    profiles = np.array([[1.0, 2.0, 4.0, 2.0]])

    normalized = normalize_shape(profiles)

    assert normalized.max() == pytest.approx(1.0)


def test_clean_time_series_rejects_frame_with_no_columns():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    empty = pd.DataFrame(index=idx)

    with pytest.raises(ValueError, match="at least one meter column"):
        clean_time_series(empty)


def test_clean_time_series_rejects_non_datetime_index():
    df = pd.DataFrame({"m": [1.0, 2.0, 3.0]})

    with pytest.raises(pandera.errors.SchemaErrors):
        clean_time_series(df)
