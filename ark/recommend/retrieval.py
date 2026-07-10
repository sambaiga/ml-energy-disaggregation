"""Content-based retrieval: predict an outcome from nearest labeled neighbors.

Also builds a split-conformal confidence set on top of that prediction.
Borrowed from the general recommender-systems literature (content-based
filtering, the same family Chapter 4's k-means archetypes already sit in),
not invented for this book. The conformal piece generalizes the centroid-
distance calibration Chapter 3 introduced and Chapter 4 already reused
once: there, the nonconformity score was distance to a cluster's own
centroid; here it is distance to a query point's nearest labeled neighbor,
the same split-conformal quantile calibration underneath.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    *,
    k: int = 5,
) -> np.ndarray:
    """Predict each query point's outcome as the mean of its k nearest labeled neighbors.

    Args:
        X_train: Labeled points' own feature embedding, shape `(n_train, n_features)`.
        y_train: Each labeled point's real outcome, shape `(n_train,)`.
        X_query: Points to predict for, shape `(n_query, n_features)`.
        k: Number of nearest neighbors to average.

    Returns:
        One predicted value per query point, shape `(n_query,)`.
    """
    n_neighbors = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
    _distances, neighbor_idx = nn.kneighbors(X_query)
    return y_train[neighbor_idx].mean(axis=1)


def calibrate_retrieval_threshold(
    X_train: np.ndarray,
    X_calibration: np.ndarray,
    *,
    alpha: float = 0.1,
) -> float:
    """Calibrate a split-conformal distance threshold for trusting a nearest neighbor.

    For each held-out calibration point, the nonconformity score is its
    distance to its own single nearest labeled (training) point. The
    threshold is the smallest distance that covers `1 - alpha` of those
    calibration scores, the same finite-sample quantile correction Chapter
    4's own archetype-membership conformal set used.

    Args:
        X_train: Labeled training points' embedding, shape `(n_train, n_features)`.
        X_calibration: A held-out split of labeled points, not in `X_train`.
        alpha: Miscoverage rate (0.1 for 90% confidence).

    Returns:
        The calibrated distance threshold `tau`.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    calibration_scores, _ = nn.kneighbors(X_calibration)
    calibration_scores = calibration_scores[:, 0]
    n_calibration = len(calibration_scores)
    quantile_level = min(np.ceil((n_calibration + 1) * (1 - alpha)) / n_calibration, 1.0)
    return float(np.quantile(calibration_scores, quantile_level))


def is_within_retrieval_confidence(
    X_train: np.ndarray,
    X_query: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Flag which query points have a labeled neighbor close enough to trust at the calibrated threshold.

    Args:
        X_train: Labeled training points' embedding.
        X_query: Points to check, shape `(n_query, n_features)`.
        tau: A threshold from :func:`calibrate_retrieval_threshold`.

    Returns:
        One boolean per query point: True if its nearest labeled neighbor sits within `tau`.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    distances, _ = nn.kneighbors(X_query)
    return distances[:, 0] <= tau
