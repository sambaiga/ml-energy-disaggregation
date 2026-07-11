import numpy as np
import pytest

from ark.anomaly.features import extract_features, fft_magnitude_features, rolling_statistics


def test_rolling_statistics_matches_hand_computed_values():
    profile = np.array([1.0, 2.0, 1.0, 2.0])

    stats = rolling_statistics(profile)

    assert stats == pytest.approx([1.5, 0.5, 1.0, 1.0])


def test_rolling_statistics_zero_for_a_flat_profile():
    profile = np.full(10, 3.0)

    stats = rolling_statistics(profile)

    assert stats == pytest.approx([3.0, 0.0, 0.0, 0.0])


def test_fft_magnitude_features_ranks_the_dominant_frequency_first():
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    profile = np.sin(t) + 0.1 * np.sin(4 * t)

    top = fft_magnitude_features(profile, n_features=2)

    assert top[0] > top[1] > 0


def test_fft_magnitude_features_pads_when_the_signal_is_too_short():
    profile = np.array([1.0, 2.0])

    top = fft_magnitude_features(profile, n_features=5)

    assert len(top) == 5


def test_extract_features_concatenates_shape_stats_and_fft():
    profile = np.array([1.0, 2.0, 1.0, 2.0])

    features = extract_features(profile, n_fft_features=1)

    assert features.shape == (9,)
    # first len(profile) entries are the peak-normalized shape
    assert features[:4] == pytest.approx(profile / profile.max())


def test_extract_features_handles_an_all_zero_profile_without_dividing_by_zero():
    profile = np.zeros(6)

    features = extract_features(profile, n_fft_features=1)

    assert np.all(np.isfinite(features))
