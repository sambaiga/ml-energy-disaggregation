"""Split-conformal threshold calibration for anomaly scores.

Generalizes the same split-conformal quantile calibration Chapters 3-5
already built (distance to a cluster centroid, then distance to a nearest
labeled neighbor) a fourth time: here the nonconformity score is an anomaly
score already computed by `ark.anomaly.detectors`, not a distance.
Calibrating this way turns "pick a percentile and hope" into a real,
checkable false-positive-rate guarantee on held-back, known-normal data.
"""

from __future__ import annotations

import numpy as np


def calibrate_anomaly_threshold(calibration_scores: np.ndarray, *, alpha: float = 0.1) -> float:
    """Calibrate a split-conformal anomaly threshold from held-out, known-normal scores.

    Args:
        calibration_scores: Anomaly scores for a held-back, known-normal
            calibration set (e.g. `ark.anomaly.detectors.ensemble_score`'s
            own output), not the training set the detectors themselves fit
            on.
        alpha: Target miscoverage rate (0.1 for a 90% coverage guarantee:
            at most 10% of genuinely normal points should be flagged).

    Returns:
        The calibrated threshold. A query point scoring above this is
        flagged as anomalous.

    Examples:
        >>> calibration_scores = np.array([0.1, 0.2, 0.15, 0.9, 0.3, 0.25, 0.18, 0.22, 0.12, 0.4])
        >>> round(calibrate_anomaly_threshold(calibration_scores, alpha=0.1), 2)
        0.9
    """
    n_calibration = len(calibration_scores)
    quantile_level = min(np.ceil((n_calibration + 1) * (1 - alpha)) / n_calibration, 1.0)
    return float(np.quantile(calibration_scores, quantile_level))


def is_anomalous(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Flag which query scores exceed the calibrated threshold.

    Args:
        scores: Anomaly scores to check.
        threshold: A threshold from `calibrate_anomaly_threshold`.

    Returns:
        One boolean per score: True if it exceeds the threshold.

    Examples:
        >>> is_anomalous(np.array([0.1, 0.95]), 0.9)
        array([False,  True])
    """
    return scores > threshold
