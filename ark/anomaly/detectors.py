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


def copod_score(X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Non-parametric, parameter-free anomaly score via a skew-weighted empirical copula.

    Implements Li, Zhao, Hu, Botta, Ionescu & Chen (2020), "COPOD: Copula-
    Based Outlier Detection," *IEEE International Conference on Data
    Mining* (ICDM), ECOD's own older sibling from the same authors. Per
    feature dimension, this keeps *both* tail probabilities `ecod_score`
    collapses into one via `min`, then picks the tail a real, checkable
    per-dimension skewness sign favors (a right-skewed dimension flags its
    own right tail more readily, and vice versa), taking the max of that
    skew-favored term and the plain two-tail average so a dimension with
    near-zero skew still falls back to catching either tail. Matches
    `pyod.models.copod`'s own combination logic; reimplemented directly
    rather than pulled in, the same "simple enough to hand-roll" bar
    `ecod_score` was held to, using this module's own tail-probability
    convention rather than `pyod`'s own rank-based empirical CDF.

    Args:
        X_train: Training feature matrix, shape `(n_train, n_features)`.
        X_query: Points to score, shape `(n_query, n_features)`.

    Returns:
        One score per query point, higher meaning more anomalous.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> X_train = rng.normal(size=(200, 1))
        >>> X_query = np.array([[0.0], [10.0]])
        >>> scores = copod_score(X_train, X_query)
        >>> bool(scores[1] > scores[0])
        True
    """
    n_train = X_train.shape[0]
    eps = 1.0 / (n_train + 1)
    total = np.zeros(X_query.shape[0])
    for dim in range(X_train.shape[1]):
        train_col = X_train[:, dim]
        query_col = X_query[:, dim]
        left_tail = np.clip(np.searchsorted(np.sort(train_col), query_col, side="right") / n_train, eps, 1.0)
        right_tail = np.clip(1.0 - left_tail, eps, 1.0)
        u_left, u_right = -np.log(left_tail), -np.log(right_tail)

        centered = train_col - train_col.mean()
        std = train_col.std()
        skewness = np.mean(centered**3) / std**3 if std > 0 else 0.0
        skew_sign = np.sign(skewness)
        # matches pyod's own U_skew construction: skew_sign=+1 selects the
        # right tail, skew_sign=-1 the left tail, 0 sums both
        u_skew = u_left * -np.sign(skew_sign - 1) + u_right * np.sign(skew_sign + 1)

        total += np.maximum(u_skew, (u_left + u_right) / 2)
    return total


def fit_ensemble_bounds(reference_scores: list[np.ndarray]) -> list[tuple[float, float]]:
    """Fit fixed min/max rescaling bounds for each detector from one reference score distribution.

    Fit this once, on the same reference set the detectors themselves were
    trained on (or a held-out split of it), and reuse the same bounds for
    every later call to `ensemble_score`. Rescaling bounds computed fresh
    per batch would make a calibration-set score and a query-set score not
    actually comparable, silently breaking conformal calibration, since the
    "same" rescaled value would mean something different in each batch.

    Args:
        reference_scores: Same-length score arrays, one per detector,
            computed on a single reference batch.

    Returns:
        One `(min, max)` pair per detector, in the same order.
    """
    return [(float(s.min()), float(s.max())) for s in reference_scores]


def _rescale(scores: list[np.ndarray], bounds: list[tuple[float, float]]) -> np.ndarray:
    """Stack several score arrays into one `(n_query, n_detectors)` matrix, each column rescaled to `[0, 1]`."""
    rescaled = []
    for s, (lo, hi) in zip(scores, bounds, strict=True):
        rescaled.append((s - lo) / (hi - lo) if hi > lo else np.zeros_like(s))
    return np.column_stack(rescaled)


def ensemble_score(scores: list[np.ndarray], bounds: list[tuple[float, float]]) -> np.ndarray:
    """Combine several anomaly scores into one, each rescaled to `[0, 1]` by fixed, pre-fit bounds.

    The "average" combination strategy (Aggarwal & Sathe, 2015, "Theoretical
    Foundations and Algorithms for Outlier Ensembles"), the same one
    `combo.models.score_comb.average` implements. Reimplemented directly
    here rather than pulling in `combo`/`pyod`: both packages declare each
    other as a hard dependency (`combo` requires `pyod`, confirmed directly
    against `combo`'s own package metadata), dragging in `numba`,
    `llvmlite`, and `matplotlib` for what the source of each combination
    function turns out to be a single-line `numpy` reduction, the same
    "earn its complexity" bar ECOD was held to above.

    Args:
        scores: Same-length score arrays, one per detector, higher meaning
            more anomalous in each.
        bounds: `(min, max)` pairs from `fit_ensemble_bounds`, one per
            detector, in the same order as `scores`. Always the bounds
            fit once on a reference batch, never refit on `scores` itself,
            so scores from different calls stay on the same scale.

    Returns:
        The mean of the rescaled scores, one per query point.
    """
    return _rescale(scores, bounds).mean(axis=1)


def ensemble_score_max(scores: list[np.ndarray], bounds: list[tuple[float, float]]) -> np.ndarray:
    """Combine several anomaly scores by taking the maximum rescaled score, not the mean.

    The "maximization" strategy: flags a point the moment any single
    detector is confident, at the cost of amplifying whichever detector is
    the noisiest. See `ensemble_score` for the rescaling convention.

    Args:
        scores: Same-length score arrays, one per detector.
        bounds: `(min, max)` pairs from `fit_ensemble_bounds`, matching `scores`.

    Returns:
        The max of the rescaled scores, one per query point.
    """
    return _rescale(scores, bounds).max(axis=1)


def ensemble_score_median(scores: list[np.ndarray], bounds: list[tuple[float, float]]) -> np.ndarray:
    """Combine several anomaly scores by taking the median rescaled score.

    More robust than the mean to one detector's own miscalibration pulling
    the combined score off, at the cost of ignoring how extreme the other
    detectors' scores are. See `ensemble_score` for the rescaling convention.

    Args:
        scores: Same-length score arrays, one per detector.
        bounds: `(min, max)` pairs from `fit_ensemble_bounds`, matching `scores`.

    Returns:
        The median of the rescaled scores, one per query point.
    """
    return np.median(_rescale(scores, bounds), axis=1)


def ensemble_score_aom(
    scores: list[np.ndarray], bounds: list[tuple[float, float]], *, n_buckets: int = 2, random_state: int = 0
) -> np.ndarray:
    """Combine several anomaly scores via Average of Maximum (Aggarwal & Sathe, 2015).

    Splits detectors into `n_buckets` random subgroups, takes the max
    rescaled score within each subgroup, then averages across subgroups.
    Matches `combo.models.score_comb.aom`'s own static-bucketing algorithm
    exactly (shuffle detectors, split into equal-sized groups, max within,
    mean across), reimplemented rather than imported for the same reason
    given in `ensemble_score`'s own docstring. Intended to dampen the noise
    of any one weak detector while still letting a confidently-anomalous
    subgroup dominate, the combination Aggarwal & Sathe found most
    consistently effective across their own benchmarks.

    Args:
        scores: Same-length score arrays, one per detector. `len(scores)`
            must be evenly divisible by `n_buckets`.
        bounds: `(min, max)` pairs from `fit_ensemble_bounds`, matching `scores`.
        n_buckets: Number of subgroups to split detectors into.
        random_state: Seed for the detector shuffle.

    Returns:
        The average-of-subgroup-maxima, one per query point.
    """
    rescaled = _rescale(scores, bounds)
    n_detectors = rescaled.shape[1]
    per_bucket = n_detectors // n_buckets
    order = np.random.default_rng(random_state).permutation(n_detectors)
    bucket_maxima = np.column_stack(
        [rescaled[:, order[i : i + per_bucket]].max(axis=1) for i in range(0, n_detectors, per_bucket)]
    )
    return bucket_maxima.mean(axis=1)
