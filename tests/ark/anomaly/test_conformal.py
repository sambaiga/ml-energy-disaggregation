import numpy as np
import pytest

from ark.anomaly.conformal import calibrate_anomaly_threshold, is_anomalous


def test_calibrate_anomaly_threshold_covers_the_requested_fraction_of_calibration_scores():
    rng = np.random.default_rng(0)
    calibration_scores = rng.normal(size=500)

    threshold = calibrate_anomaly_threshold(calibration_scores, alpha=0.1)
    coverage = (calibration_scores <= threshold).mean()

    assert coverage == pytest.approx(0.90, abs=0.02)


def test_calibrate_anomaly_threshold_caps_at_the_maximum_score():
    calibration_scores = np.array([0.1, 0.2, 0.3])

    threshold = calibrate_anomaly_threshold(calibration_scores, alpha=0.01)

    assert threshold == pytest.approx(calibration_scores.max())


def test_is_anomalous_flags_only_scores_above_the_threshold():
    scores = np.array([0.1, 0.5, 0.9, 1.5])

    flagged = is_anomalous(scores, threshold=0.5)

    assert flagged.tolist() == [False, False, True, True]
