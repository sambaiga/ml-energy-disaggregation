import numpy as np

from ark.recommend.retrieval import (
    calibrate_retrieval_threshold,
    is_within_retrieval_confidence,
    knn_predict,
)


def test_knn_predict_recovers_a_simple_linear_relationship():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 2))
    y_train = X_train[:, 0] * 2.0
    X_query = rng.normal(size=(10, 2))

    predicted = knn_predict(X_train, y_train, X_query, k=5)

    true_values = X_query[:, 0] * 2.0
    assert np.abs(predicted - true_values).mean() < 1.0


def test_knn_predict_handles_fewer_training_points_than_k():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 2.0])
    X_query = np.array([[1.5]])

    predicted = knn_predict(X_train, y_train, X_query, k=10)

    assert predicted.shape == (1,)


def test_calibrate_retrieval_threshold_is_positive_and_finite():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 2))
    X_calibration = rng.normal(size=(20, 2))

    tau = calibrate_retrieval_threshold(X_train, X_calibration, alpha=0.1)

    assert tau > 0
    assert np.isfinite(tau)


def test_is_within_retrieval_confidence_flags_a_far_outlier_as_untrusted():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 2))
    X_calibration = rng.normal(size=(20, 2))
    tau = calibrate_retrieval_threshold(X_train, X_calibration, alpha=0.1)

    near_query = X_train[[0]]
    far_query = np.array([[1000.0, 1000.0]])

    trusted = is_within_retrieval_confidence(X_train, np.vstack([near_query, far_query]), tau)

    assert trusted[0]
    assert not trusted[1]
