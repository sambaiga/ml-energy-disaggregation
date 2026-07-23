"""Ground-truth-free trustworthiness checks for a clustering result.

A different question again from `ark.cluster.cluster_validation` (choosing k
in an already-reduced space, given no labels) and `ark.cluster.dimensionality`
(choosing how many components to keep). Both of those internal validity
metrics answer "how separated are these groups," which a single extreme
outlier isolated into its own cluster satisfies trivially, by construction,
without finding any real structure. This module asks a genuinely different
question instead: does this specific clustering hold up under resampling,
the closest thing to an out-of-sample check available without labels.

Two complementary, independent methods, not one:
    - `prediction_strength` (Tibshirani & Walther 2005,
      `tibshiranwalther2005predictionstrength`) validates *k* itself: cluster
      two independent halves of the data, then check whether pairs the
      second half's own clustering puts together are still predicted
      together using the first half's cluster centroids. This is a global
      "is this k well-supported" signal; a singleton cluster contributes no
      pairs and is excluded from its own minimum rather than penalized
      directly, so this method alone will not catch an outlier-isolation
      cluster sitting inside an otherwise well-supported k.
    - `cluster_stability` (Hennig 2007, `hennig2007clusterwise`) is the one
      that does catch it: bootstrap-resample the data, fit centroids on
      each resample, predict cluster assignment back onto the *full*
      original dataset from those centroids, and measure the Jaccard
      overlap between each original cluster and its best match under that
      prediction. A real archetype's centroid is robustly recovered even
      from a partial resample and reclaims essentially all of its original
      members every time (stability near 1.0). A cluster built around one
      extreme household only has a centroid to speak of in draws where
      that household was actually sampled, and gets silently absorbed into
      whichever real cluster is nearest otherwise, landing in Hennig's own
      "patterned but not stable" band (0.6-0.75) rather than his "stable"
      band above 0.75, clearly below any real cluster nearby.
"""

from collections.abc import Callable, Sequence
import logging

import numpy as np
import pandas as pd

__all__ = ["cluster_stability", "prediction_strength"]

log = logging.getLogger(__name__)

ClusterFn = Callable[[np.ndarray, int], np.ndarray]


def _co_membership_agreement(test_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Minimum, over the test clustering's own clusters, of within-cluster co-membership agreement.

    For each cluster in `test_labels`, this is the probability that two
    of its members, picked at random, also share the same `predicted_labels`
    value. Clusters with fewer than 2 members contribute no pairs and are
    skipped, per Tibshirani & Walther's own convention (a singleton cannot
    disagree with itself either way, so it neither helps nor penalizes k).
    """
    agreements = []
    for cluster_id in np.unique(test_labels):
        members = np.where(test_labels == cluster_id)[0]
        n = len(members)
        if n < 2:
            continue
        _, counts = np.unique(predicted_labels[members], return_counts=True)
        agree_pairs = np.sum(counts * (counts - 1) / 2)
        total_pairs = n * (n - 1) / 2
        agreements.append(agree_pairs / total_pairs)
    if not agreements:
        # Every cluster in this split was a singleton: no pairs anywhere to
        # check. Not informative either way, so it doesn't drag down k.
        return 1.0
    return float(min(agreements))


def _prediction_strength_one_direction(x_train: np.ndarray, x_test: np.ndarray, k: int, cluster_fn: ClusterFn) -> float:
    train_labels = cluster_fn(x_train, k)
    test_labels = cluster_fn(x_test, k)
    train_cluster_ids = np.unique(train_labels)
    if len(train_cluster_ids) < 2 or len(np.unique(test_labels)) < 2:
        return 0.0

    centroids = np.array([x_train[train_labels == cid].mean(axis=0) for cid in train_cluster_ids])
    dists = np.linalg.norm(x_test[:, None, :] - centroids[None, :, :], axis=2)
    predicted_labels = train_cluster_ids[dists.argmin(axis=1)]

    return _co_membership_agreement(test_labels, predicted_labels)


def prediction_strength(
    X: np.ndarray,
    k_range: Sequence[int],
    cluster_fn: ClusterFn,
    n_splits: int = 10,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Validate k by whether it replicates across independent random halves of the data.

    Args:
        X: (n_samples, n_features) input data.
        k_range: Candidate cluster counts to evaluate.
        cluster_fn: A callable `cluster_fn(X, k) -> labels`, the same
            convention used throughout `ark.cluster`.
        n_splits: Number of random 50/50 splits to average over. Each split
            is evaluated in both directions (train on A, test on B, and
            vice versa) and averaged, so the effective sample size is
            `2 * n_splits`.
        random_state: Seed for the random splits.

    Returns:
        DataFrame with columns `k`, `prediction_strength` (mean over
        splits), `prediction_strength_std`. Per Tibshirani & Walther's own
        recommendation, the largest k with `prediction_strength` above
        0.8-0.9 is a reasonable, well-supported choice.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    half = n // 2

    rows = []
    for k in k_range:
        split_scores = []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            idx_a, idx_b = perm[:half], perm[half:]
            score_ab = _prediction_strength_one_direction(X[idx_a], X[idx_b], k, cluster_fn)
            score_ba = _prediction_strength_one_direction(X[idx_b], X[idx_a], k, cluster_fn)
            split_scores.append((score_ab + score_ba) / 2)
        rows.append(
            {
                "k": int(k),
                "prediction_strength": float(np.mean(split_scores)),
                "prediction_strength_std": float(np.std(split_scores)),
            }
        )
        log.info("prediction_strength: k=%d -> %.3f", k, rows[-1]["prediction_strength"])

    return pd.DataFrame(rows)


def cluster_stability(
    X: np.ndarray,
    labels: np.ndarray,
    cluster_fn: ClusterFn,
    n_bootstrap: int = 100,
    random_state: int | None = None,
) -> dict[int, float]:
    """Audit each cluster in an existing clustering for bootstrap stability.

    Args:
        X: (n_samples, n_features) input data.
        labels: The existing cluster assignment to audit, e.g. the output of
            `cluster_fn(X, k)` for whatever k is under consideration.
        cluster_fn: A callable `cluster_fn(X, k) -> labels`.
        n_bootstrap: Number of bootstrap resamples (sampling with
            replacement) to average over.
        random_state: Seed for the bootstrap draws.

    Returns:
        Dict mapping each original cluster id to its mean Jaccard stability
        score across bootstrap resamples. Per Hennig's own suggested
        interpretation bands: below ~0.5 the cluster should be considered
        dissolved (not real, e.g. exactly the outlier-isolation failure
        mode this module exists to catch), 0.6-0.75 is "patterned but not
        stable," and above ~0.75 is a stable, trustworthy cluster.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    labels = np.asarray(labels)
    cluster_ids = np.unique(labels)
    k = len(cluster_ids)
    original_members = {cid: set(np.where(labels == cid)[0].tolist()) for cid in cluster_ids}

    best_jaccards: dict[int, list[float]] = {cid: [] for cid in cluster_ids}

    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)
        x_boot = X[boot_idx]
        boot_labels = cluster_fn(x_boot, k)
        boot_cluster_ids = np.unique(boot_labels)
        if len(boot_cluster_ids) < 2:
            for cid in cluster_ids:
                best_jaccards[cid].append(0.0)
            continue

        # Predict cluster assignment for the FULL original dataset from
        # centroids estimated on just the bootstrap sample, so both sides of
        # the Jaccard comparison range over the same fixed n-point universe
        # instead of being confounded by which points this particular draw
        # happened to include. A real cluster's centroid is robustly
        # estimated even from a partial resample and correctly reclaims
        # essentially all of its original members; a lone outlier only has
        # a centroid to speak of in draws where that one point was sampled,
        # and gets silently absorbed into whichever real cluster is nearest
        # otherwise.
        centroids = np.array([x_boot[boot_labels == bid].mean(axis=0) for bid in boot_cluster_ids])
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        predicted_labels = boot_cluster_ids[dists.argmin(axis=1)]
        predicted_sets = {bid: set(np.where(predicted_labels == bid)[0].tolist()) for bid in boot_cluster_ids}

        for cid, orig_set in original_members.items():
            jaccards = [
                len(orig_set & pred_set) / len(orig_set | pred_set) if (orig_set | pred_set) else 1.0
                for pred_set in predicted_sets.values()
            ]
            best_jaccards[cid].append(max(jaccards))

    stability = {int(cid): float(np.mean(scores)) for cid, scores in best_jaccards.items()}
    for cid, score in stability.items():
        log.info("cluster_stability: cluster=%d -> %.3f", cid, score)
    return stability
