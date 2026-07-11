import numpy as np
import pytest

from ark.anomaly.detectors import (
    ecod_score,
    ensemble_score,
    isolation_forest_score,
    kde_score,
    lof_score,
    mahalanobis_score,
)


def _normal_training_data(seed: int = 0, n: int = 100, n_features: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_features))


def _query_with_one_real_outlier(n_features: int = 4) -> np.ndarray:
    inlier = np.zeros((1, n_features))
    outlier = np.full((1, n_features), 20.0)
    return np.vstack([inlier, outlier])


@pytest.mark.parametrize(
    "scorer",
    [mahalanobis_score, kde_score, isolation_forest_score, lof_score, ecod_score],
)
def test_every_detector_scores_the_real_outlier_higher_than_the_inlier(scorer):
    X_train = _normal_training_data()
    X_query = _query_with_one_real_outlier()

    scores = scorer(X_train, X_query)

    assert scores[1] > scores[0]


def test_ecod_score_matches_a_hand_computed_tail_probability():
    X_train = np.arange(1, 101).reshape(-1, 1).astype(float)  # uniform 1..100
    X_query = np.array([[50.0]])  # dead center: right at the median

    scores = ecod_score(X_train, X_query)

    # median sits at roughly the 50th percentile either way, tail ~0.5,
    # so -log(0.5) is the right order of magnitude, not near zero
    assert scores[0] == pytest.approx(-np.log(0.5), abs=0.05)


def test_ensemble_score_averages_rescaled_component_scores():
    a = np.array([0.0, 10.0])
    b = np.array([0.0, 1.0])

    combined = ensemble_score([a, b])

    # both components already span [0, 1] identically after rescaling
    assert combined == pytest.approx([0.0, 1.0])


def test_ensemble_score_handles_a_constant_component_without_dividing_by_zero():
    constant = np.array([5.0, 5.0, 5.0])
    varying = np.array([0.0, 1.0, 2.0])

    combined = ensemble_score([constant, varying])

    assert np.all(np.isfinite(combined))
    assert combined[0] < combined[2]
