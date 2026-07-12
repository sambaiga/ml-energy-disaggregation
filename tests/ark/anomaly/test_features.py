import numpy as np
import pytest

from ark.anomaly.features import (
    distance_matrix_features,
    event_features,
    extract_features,
    fft_magnitude_features,
    rolling_statistics,
)


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


def test_distance_matrix_features_matches_a_hand_computed_matrix():
    profile = np.array([1.0, 2.0, 3.0, 4.0])

    dist = distance_matrix_features(profile, n_bins=2)

    # bin 0 = mean(1,2) = 1.5, bin 1 = mean(3,4) = 3.5, |1.5-3.5| = 2.0
    assert dist == pytest.approx([2.0])


def test_distance_matrix_features_returns_only_the_upper_triangle():
    # a symmetric matrix with a zero diagonal has no more real degrees of
    # freedom than its own upper triangle; the full flattened matrix would
    # be a redundant, rank-deficient feature vector
    profile = np.arange(48).astype(float)

    dist = distance_matrix_features(profile, n_bins=12)

    assert dist.shape == (12 * 11 // 2,)


def test_distance_matrix_features_is_zero_for_a_flat_profile():
    profile = np.full(48, 5.0)

    dist = distance_matrix_features(profile, n_bins=8)

    assert dist == pytest.approx(np.zeros(8 * 7 // 2))


def test_event_features_matches_a_hand_computed_example():
    mask = np.array([False, True, True, False, True])
    severity = np.array([0.0, 0.02, 0.05, 0.0, 0.01])

    features = event_features(mask, severity, step_hours=1.0)

    assert features == pytest.approx([3.0, 2.0, 0.08 / 3, 0.05, 0.08, 7.0 / 3, 1.247], abs=1e-3)


def test_event_features_is_all_zero_when_no_event_occurs():
    mask = np.zeros(48, dtype=bool)
    severity = np.zeros(48)

    features = event_features(mask, severity)

    assert features == pytest.approx(np.zeros(7))


def test_event_features_counts_a_single_contiguous_episode_as_one():
    mask = np.array([False, True, True, True, False])
    severity = np.array([0.0, 0.1, 0.1, 0.1, 0.0])

    features = event_features(mask, severity)

    assert features[1] == pytest.approx(1.0)  # one episode, not three separate events


def test_event_features_severity_reflects_a_real_worse_violation():
    mask = np.array([True, True])
    mild_severity = np.array([0.01, 0.01])
    severe_severity = np.array([0.1, 0.1])

    mild = event_features(mask, mild_severity)
    severe = event_features(mask, severe_severity)

    assert severe[2] > mild[2]  # mean severity
    assert severe[3] > mild[3]  # max severity
