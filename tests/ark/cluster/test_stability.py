import numpy as np
from sklearn.cluster import KMeans

from ark.cluster.stability import cluster_stability, prediction_strength


def _kmeans_fn(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


def _three_blobs(rng: np.random.Generator) -> np.ndarray:
    blob_a = rng.normal(loc=-8, scale=0.5, size=(100, 2))
    blob_b = rng.normal(loc=0, scale=0.5, size=(100, 2))
    blob_c = rng.normal(loc=8, scale=0.5, size=(100, 2))
    return np.vstack([blob_a, blob_b, blob_c])


def test_prediction_strength_favors_the_true_k():
    # Three real, well-separated blobs: k=3 should replicate reliably across
    # independent halves (Tibshirani & Walther's own >0.8-0.9 bar), while an
    # over-fragmented k=6 (splitting each real blob into two arbitrary
    # halves) should not, since which arbitrary half a point lands in isn't
    # stable across resamples.
    X = _three_blobs(np.random.default_rng(0))

    scores = prediction_strength(X, k_range=[3, 6], cluster_fn=_kmeans_fn, n_splits=10, random_state=0)

    ps_3 = scores.loc[scores["k"] == 3, "prediction_strength"].iloc[0]
    ps_6 = scores.loc[scores["k"] == 6, "prediction_strength"].iloc[0]
    assert ps_3 > 0.8
    assert ps_3 > ps_6


def test_cluster_stability_is_low_for_an_outlier_isolated_cluster_and_high_for_real_ones():
    # The exact failure mode found in london-cluster.ipynb: k=4 where 3
    # clusters are real, balanced blobs and the 4th is a single extreme
    # outlier point isolated by k-means purely because it sits far from
    # everything else. Internal validity metrics (silhouette etc.) rate
    # this outlier cluster as perfectly separated; cluster_stability should
    # not. In practice the outlier's own score does not collapse to ~0 -
    # whenever that one point IS drawn in a bootstrap resample it reliably
    # gets isolated again, so its score settles near Hennig's "patterned
    # but not stable" band (0.6-0.75), well below a real cluster's ~0.9+,
    # rather than in his "stable" band above 0.75.
    rng = np.random.default_rng(0)
    blobs = _three_blobs(rng)
    outlier = np.array([[500.0, 500.0]])
    X = np.vstack([blobs, outlier])

    labels = _kmeans_fn(X, 4)
    outlier_cluster_id = labels[-1]
    real_cluster_ids = {c for c in np.unique(labels) if c != outlier_cluster_id}
    assert len(real_cluster_ids) == 3  # confirms k-means did isolate the outlier alone, matching the scenario

    stability = cluster_stability(X, labels, cluster_fn=_kmeans_fn, n_bootstrap=50, random_state=0)

    assert stability[outlier_cluster_id] < 0.75
    for cid in real_cluster_ids:
        assert stability[cid] > 0.9
        assert stability[cid] - stability[outlier_cluster_id] > 0.2


def test_cluster_stability_scores_are_bounded():
    X = _three_blobs(np.random.default_rng(1))
    labels = _kmeans_fn(X, 3)

    stability = cluster_stability(X, labels, cluster_fn=_kmeans_fn, n_bootstrap=20, random_state=1)

    assert all(0.0 <= score <= 1.0 for score in stability.values())
    assert set(stability.keys()) == set(np.unique(labels).tolist())


def test_prediction_strength_scores_are_bounded():
    X = _three_blobs(np.random.default_rng(2))

    scores = prediction_strength(X, k_range=[2, 3, 4], cluster_fn=_kmeans_fn, n_splits=5, random_state=2)

    assert scores["prediction_strength"].between(0.0, 1.0).all()
    assert list(scores["k"]) == [2, 3, 4]
