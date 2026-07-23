import numpy as np
import pandas as pd
import pytest

from ark.cluster.feature import (
    FeatureConfig,
    create_autocorrelation_features,
    create_calendar_peak_features,
    create_distribution_features,
    create_magnitude_features,
    create_profile_features,
    create_shape_features,
    create_trend_features,
    create_volatility_features,
    extract_features,
)
from ark.cluster.preprocessing import clean_time_series, normalize_by_daily_max


def _normalized_two_meter_frame(periods_days: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=24 * periods_days, freq="h")
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {"meter_A": rng.random(len(idx)), "meter_B": rng.random(len(idx))},
        index=idx,
    )
    return normalize_by_daily_max(clean_time_series(data))


def test_create_profile_features_shape_and_orientation():
    # Regression test: an earlier version of this function returned an
    # hour-indexed frame with scrambled (meter, dayofweek) column tuples
    # instead of a meter-indexed frame with real (dayofweek, hour) columns.
    normalized = _normalized_two_meter_frame()

    profiles = create_profile_features(normalized, hemisphere="north")

    assert profiles.shape == (2, 192)
    assert list(profiles.index) == ["meter_A", "meter_B"]


def test_create_profile_features_weekly_only_and_seasonal_only():
    normalized = _normalized_two_meter_frame()

    weekly = create_profile_features(normalized, hemisphere="north", profile_type="weekly")
    seasonal = create_profile_features(normalized, hemisphere="north", profile_type="seasonal")

    assert all(c.startswith("w_") for c in weekly.columns)
    assert all(c.startswith("s_") for c in seasonal.columns)


def test_create_profile_features_missing_timestamp_column_raises_keyerror():
    normalized = _normalized_two_meter_frame().reset_index(drop=True)

    with pytest.raises(KeyError, match="timestamp"):
        create_profile_features(normalized)


def test_create_shape_features_empty_input_returns_empty_frame_indexed_by_columns():
    empty = pd.DataFrame(columns=["a", "b"])
    empty.index = pd.DatetimeIndex([])

    result = create_shape_features(empty)

    assert list(result.index) == ["a", "b"]
    assert result.empty


def test_create_shape_features_deduplicates_columns():
    idx = pd.date_range("2023-01-01", periods=48 * 3, freq="30min")
    rng = np.random.default_rng(0)
    duplicated = pd.DataFrame(rng.random((len(idx), 2)), index=idx, columns=["m", "m"])

    result = create_shape_features(duplicated)

    assert list(result.index) == ["m"]


def test_create_autocorrelation_features_short_series_falls_back_to_zero():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    df = pd.DataFrame({"m": [1.0, 2.0, 3.0]}, index=idx)

    result = create_autocorrelation_features(df, lags=(6, 12))

    # len(series) == 3 is not > either lag, so both must hit the 0.0 fallback
    # rather than raising or returning NaN from an unstable short-series autocorr.
    assert result.loc["m", "autocorr_lag6"] == 0.0
    assert result.loc["m", "autocorr_lag12"] == 0.0


def test_create_calendar_peak_features_no_inf_on_zero_denominator():
    idx = pd.date_range("2023-01-01", periods=48 * 5, freq="30min")
    df = pd.DataFrame({"m": np.zeros(len(idx))}, index=idx)

    result = create_calendar_peak_features(df)

    assert np.isfinite(result.to_numpy()).all()


def test_create_distribution_features_basic_stats():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"m": range(10)}, index=idx, dtype=float)

    result = create_distribution_features(df)

    assert result.loc["m", "q50"] == pytest.approx(df.resample("D").sum()["m"].median())


def test_create_magnitude_features_basic_stats():
    daily = pd.DataFrame({"m": [1.0, 2.0, 3.0]})

    result = create_magnitude_features(daily)

    assert result.loc["m", "mean"] == pytest.approx(2.0)
    assert result.loc["m", "total"] == pytest.approx(6.0)


def test_create_volatility_features_short_series_falls_back_to_plain_std():
    daily = pd.DataFrame({"m": [10.0, 12.0, 14.0, 16.0, 18.0]})

    result = create_volatility_features(daily)

    # fewer than 30 days: rolling_std_30d must equal the plain std, not NaN
    # from an under-populated rolling window.
    assert result.loc["m", "rolling_std_30d"] == pytest.approx(daily["m"].std())


def test_create_trend_features_short_series_zero_fallback():
    daily = pd.DataFrame({"m": [10.0, 12.0, 14.0, 16.0, 18.0]})

    result = create_trend_features(daily)

    # only 5 days of data: lag7 has no valid diff and must fall back to 0.0.
    assert result.loc["m", "change_lag7"] == 0.0


def test_create_trend_features_slope_sign_matches_trend_direction():
    daily = pd.DataFrame({"m": [10.0, 12.0, 14.0, 16.0, 18.0]})

    result = create_trend_features(daily)

    assert result.loc["m", "slope"] > 0


def test_feature_config_requires_at_least_one_include_flag():
    with pytest.raises(ValueError, match="at least one"):
        FeatureConfig(
            include_profile=False,
            include_magnitude=False,
            include_volatility=False,
            include_trend=False,
            include_shape=False,
            include_autocorr=False,
            include_calendar_peak=False,
            include_distribution=False,
        )


def test_feature_config_rejects_invalid_hemisphere():
    with pytest.raises(ValueError):
        FeatureConfig(hemisphere="east")


def test_extract_features_hemisphere_threads_through_to_profile_features():
    # Jan disagrees between hemispheres (summer vs winter): if hemisphere
    # weren't actually threaded through to create_profile_features, both
    # configs would silently produce identical season columns.
    idx = pd.date_range("2023-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"meter_A": rng.random(len(idx))}, index=idx)

    south = extract_features(data, FeatureConfig(hemisphere="south"))
    north = extract_features(data, FeatureConfig(hemisphere="north"))

    assert not south.columns.equals(north.columns) or not south.equals(north)
