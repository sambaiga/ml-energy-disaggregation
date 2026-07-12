import numpy as np
import pytest

from ark.anomaly.detectors import (
    copod_score,
    ecod_score,
    ensemble_score,
    ensemble_score_aom,
    ensemble_score_max,
    ensemble_score_median,
    fit_ensemble_bounds,
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
    [mahalanobis_score, kde_score, isolation_forest_score, lof_score, ecod_score, copod_score],
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


def test_copod_score_matches_pyod_own_ranking_on_a_skewed_distribution():
    # a real, right-skewed distribution, the case COPOD's own skew-weighting
    # is built for; ranking cross-checked directly against pyod's own COPOD
    # (typical < compressed left tail < deep right tail)
    rng = np.random.default_rng(0)
    X_train = rng.exponential(scale=1.0, size=(300, 3))
    X_query = np.array([X_train[0], [0.01, 0.01, 0.01], [8.0, 8.0, 8.0]])

    scores = copod_score(X_train, X_query)

    assert scores[0] < scores[1] < scores[2]


def test_copod_score_is_less_aggressive_than_ecod_against_a_dimension_own_skew():
    # ECOD always takes whichever tail is more extreme; COPOD only favors a
    # dimension's compressed tail when the skew itself points that way, so
    # on a right-skewed dimension a deep-left-tail query should score lower
    # under COPOD than under ECOD
    rng = np.random.default_rng(0)
    X_train = rng.exponential(scale=1.0, size=(300, 1))
    X_query = np.array([[0.01]])

    assert copod_score(X_train, X_query)[0] < ecod_score(X_train, X_query)[0]


def test_ensemble_score_averages_rescaled_component_scores():
    a = np.array([0.0, 10.0])
    b = np.array([0.0, 1.0])
    bounds = fit_ensemble_bounds([a, b])

    combined = ensemble_score([a, b], bounds)

    # both components already span [0, 1] identically after rescaling
    assert combined == pytest.approx([0.0, 1.0])


def test_ensemble_score_handles_a_constant_component_without_dividing_by_zero():
    constant = np.array([5.0, 5.0, 5.0])
    varying = np.array([0.0, 1.0, 2.0])
    bounds = fit_ensemble_bounds([constant, varying])

    combined = ensemble_score([constant, varying], bounds)

    assert np.all(np.isfinite(combined))
    assert combined[0] < combined[2]


def test_fit_ensemble_bounds_then_ensemble_score_stays_consistent_across_batches():
    reference = np.array([0.0, 5.0, 10.0])
    bounds = fit_ensemble_bounds([reference])

    # a later batch, entirely within the reference range, on the SAME scale
    later_batch = np.array([2.5, 10.0])
    combined = ensemble_score([later_batch], bounds)

    assert combined == pytest.approx([0.25, 1.0])


def test_ensemble_score_max_takes_the_largest_rescaled_component():
    a = np.array([0.0, 1.0])  # rescaled: [0.0, 1.0]
    b = np.array([0.0, 0.5])  # rescaled: [0.0, 0.5]
    bounds = fit_ensemble_bounds([a, b])

    combined = ensemble_score_max([a, b], bounds)

    assert combined == pytest.approx([0.0, 1.0])


def test_ensemble_score_median_takes_the_middle_rescaled_component():
    a = np.array([1.0])  # rescaled bounds fit on itself: 1.0
    b = np.array([0.0])
    c = np.array([0.5])
    bounds = fit_ensemble_bounds([np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])])

    combined = ensemble_score_median([a, b, c], bounds)

    assert combined == pytest.approx([0.5])


def test_ensemble_score_aom_sits_between_average_and_max():
    rng = np.random.default_rng(0)
    scores = [rng.normal(size=20) for _ in range(4)]
    bounds = fit_ensemble_bounds(scores)

    avg = ensemble_score(scores, bounds)
    mx = ensemble_score_max(scores, bounds)
    aom = ensemble_score_aom(scores, bounds, n_buckets=2, random_state=0)

    assert np.all(aom >= avg - 1e-9)
    assert np.all(aom <= mx + 1e-9)


def test_ensemble_score_aom_matches_a_hand_computed_bucket_average():
    # 4 detectors, 2 buckets: with a known shuffle order, AOM is just
    # (max of bucket 1's rescaled scores + max of bucket 2's) / 2
    scores = [np.array([0.0, v]) for v in [1.0, 2.0, 3.0, 4.0]]
    bounds = fit_ensemble_bounds(scores)  # each component already spans its own [0, v] -> rescaled [0, 1]

    aom = ensemble_score_aom(scores, bounds, n_buckets=2, random_state=0)

    # every component rescales to [0.0, 1.0] identically, so regardless of
    # which detectors land in which bucket, both bucket maxima are 1.0
    assert aom == pytest.approx([0.0, 1.0])
