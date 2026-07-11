"""Parametric and non-parametric anomaly scorers, and a simple ensemble.

Every scorer returns a real-valued score where *higher means more
anomalous*, the opposite of scikit-learn's own `decision_function`/
`score_samples` convention (high there means "more normal"), so scores from
different detectors can be compared and averaged directly without every
caller re-deriving the sign flip. No new dependency: every parametric and
non-parametric method here is either already in `scikit-learn` or, for
ECOD, simple enough to implement directly from its own published algorithm.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity, LocalOutlierFactor


def mahalanobis_score(X_train: np.ndarray, X_query: np.ndarray, *, random_state: int = 0) -> np.ndarray:
    """Parametric anomaly score from a Minimum Covariance Determinant fit, a real Mahalanobis distance.

    Args:
        X_train: Training feature matrix, shape `(n_train, n_features)`.
        X_query: Points to score, shape `(n_query, n_features)`.
        random_state: Seed for MCD's own random subsampling.

    Returns:
        One score per query point, higher meaning more anomalous.
    """
    model = EllipticEnvelope(random_state=random_state).fit(X_train)
    return -model.score_samples(X_query)


def kde_score(X_train: np.ndarray, X_query: np.ndarray, *, bandwidth: float = 1.0) -> np.ndarray:
    """Parametric anomaly score from a kernel density estimate: how far into a low-density region a point sits.

    Args:
        X_train: Training feature matrix.
        X_query: Points to score.
        bandwidth: KDE smoothing bandwidth, a real, honest choice to set
            rather than default blindly.

    Returns:
        One score per query point, higher meaning more anomalous.
    """
    model = KernelDensity(bandwidth=bandwidth).fit(X_train)
    return -model.score_samples(X_query)


def isolation_forest_score(X_train: np.ndarray, X_query: np.ndarray, *, random_state: int = 0) -> np.ndarray:
    """Non-parametric anomaly score: how few random splits it takes to isolate a point.

    Args:
        X_train: Training feature matrix.
        X_query: Points to score.
        random_state: Seed for the forest's own random splits.

    Returns:
        One score per query point, higher meaning more anomalous.
    """
    model = IsolationForest(random_state=random_state).fit(X_train)
    return -model.decision_function(X_query)


def lof_score(X_train: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    """Non-parametric anomaly score from local density relative to a point's own nearest neighbors.

    Args:
        X_train: Training feature matrix.
        X_query: Points to score.
        n_neighbors: Neighborhood size for the local density estimate.

    Returns:
        One score per query point, higher meaning more anomalous.
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(X_train)
    return -model.score_samples(X_query)


def ecod_score(X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Non-parametric, parameter-free anomaly score via per-dimension empirical-CDF tail probabilities.

    Implements Li, Zhao, Hu, Botta, Ionescu & Chen (2023), "ECOD:
    Unsupervised Outlier Detection Using Empirical Cumulative Distribution
    Functions," *IEEE Transactions on Knowledge and Data Engineering*,
    35(12):12181-12193, directly from the paper's own algorithm rather than
    a library: per feature dimension, estimate the training data's own
    empirical left- and right-tail probability for each query value, keep
    the smaller (more extreme) tail, then sum the negative log of that tail
    probability across dimensions. No bandwidth, no covariance estimate,
    no fitted parameters at all, the source of the method's own
    "parameter-free" claim.

    Args:
        X_train: Training feature matrix, shape `(n_train, n_features)`.
        X_query: Points to score, shape `(n_query, n_features)`.

    Returns:
        One score per query point, higher meaning more anomalous.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> X_train = rng.normal(size=(200, 1))
        >>> X_query = np.array([[0.0], [10.0]])
        >>> scores = ecod_score(X_train, X_query)
        >>> bool(scores[1] > scores[0])
        True
    """
    n_train = X_train.shape[0]
    scores = np.zeros(X_query.shape[0])
    for dim in range(X_train.shape[1]):
        train_col = np.sort(X_train[:, dim])
        query_col = X_query[:, dim]
        left_tail = np.searchsorted(train_col, query_col, side="right") / n_train
        right_tail = 1.0 - left_tail
        tail = np.minimum(left_tail, right_tail)
        tail = np.clip(tail, 1.0 / (n_train + 1), 1.0)  # avoid log(0) for an extreme point
        scores += -np.log(tail)
    return scores


def ensemble_score(scores: list[np.ndarray]) -> np.ndarray:
    """Combine several anomaly scores into one, each rescaled to `[0, 1]` first so no raw scale dominates.

    Args:
        scores: Same-length score arrays, one per detector, higher meaning
            more anomalous in each.

    Returns:
        The mean of the rescaled scores, one per query point.
    """
    rescaled = []
    for s in scores:
        lo, hi = s.min(), s.max()
        rescaled.append((s - lo) / (hi - lo) if hi > lo else np.zeros_like(s))
    return np.mean(rescaled, axis=0)
