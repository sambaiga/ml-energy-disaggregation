from collections.abc import Sequence
import logging
from typing import Literal

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from ark.cluster.processing import cluster_dictionary

log = logging.getLogger(__name__)


def _cluster_count_param(model: BaseEstimator) -> str:
    """Return whichever of 'n_clusters'/'n_components' this estimator uses for its target cluster count.

    KMeans/AgglomerativeClustering/SpectralClustering use `n_clusters`;
    `GaussianMixture` uses `n_components` instead for the same concept, so a
    single hardcoded parameter name would reject GMM outright.
    """
    params = model.get_params()
    if "n_clusters" in params:
        return "n_clusters"
    if "n_components" in params:
        return "n_components"
    raise TypeError("model does not accept a 'n_clusters' or 'n_components' parameter (checked via get_params()).")


def clustering_validity_scores(
    X: np.ndarray,
    k_range: Sequence[int],
    model: BaseEstimator | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute multiple internal clustering validity scores for a range of k.

    Metrics computed:
        - inertia (if model has `inertia_` attribute) - lower is better
        - bic (if model has a `.bic(X)` method, e.g. `GaussianMixture`) - lower is better
        - silhouette - higher is better
        - calinski_harabasz - higher is better
        - davies_bouldin - lower is better

    Args:
        X: (n_samples, n_features) input data.
        k_range: Iterable of candidate cluster counts (will be cast to int).
        model: A scikit-learn clustering estimator instance with `fit_predict`.
            It must have a `n_clusters` parameter (e.g. KMeans,
            AgglomerativeClustering) or a `n_components` parameter (e.g.
            `GaussianMixture`) - whichever is present is set to each `k`. If
            None, uses KMeans with `n_init=50`, `init='k-means++'`,
            `max_iter=500`.
        random_state: Reproducibility seed (only used if model is None).
        n_jobs: Number of parallel jobs (-1 = all CPUs).

    Returns:
        DataFrame with columns: k, inertia, bic, silhouette,
        calinski_harabasz, davies_bouldin. Inertia/bic are NaN if the model
        does not provide them. A row whose model failed to fit for that k
        (e.g. k larger than n_samples) has NaN for every metric and a
        non-null `error` column describing what went wrong, rather than
        raising and losing every other k's results.

    Example:
        >>> scores = clustering_validity_scores(X, range(2, 11))
        >>> print(scores)
    """
    if model is None:
        model = KMeans(
            n_init=50,
            init="k-means++",
            random_state=random_state,
            max_iter=500,
        )
    elif not isinstance(model, BaseEstimator):
        raise TypeError("model must be a scikit-learn estimator or None.")

    # Check that the model can be cloned and has n_clusters or n_components
    if not hasattr(model, "set_params") or not hasattr(model, "get_params"):
        raise AttributeError("model must have get_params and set_params.")
    cluster_count_param = _cluster_count_param(model)

    def _evaluate_one(k: int) -> dict[str, int | float | str]:
        # Clone the original model and set its cluster-count parameter
        _model = clone(model)
        _model.set_params(**{cluster_count_param: int(k)})

        row: dict[str, int | float | str] = {"k": int(k)}

        try:
            labels = _model.fit_predict(X)
        except (ValueError, RuntimeError) as e:
            # A specific k can be infeasible for a specific model (e.g. k
            # larger than n_samples, or a non-converging fit). Report it as
            # a NaN row instead of crashing the whole k_range sweep.
            log.warning("fit_predict failed for k=%d: %s", k, e)
            row["inertia"] = np.nan
            row["bic"] = np.nan
            row["silhouette"] = np.nan
            row["calinski_harabasz"] = np.nan
            row["davies_bouldin"] = np.nan
            row["error"] = str(e)
            return row

        # Inertia (if available)
        row["inertia"] = _model.inertia_ if hasattr(_model, "inertia_") else np.nan

        # BIC (if available, e.g. GaussianMixture) - lower is better, feeds recommend_k(strategy="bic")
        row["bic"] = _model.bic(X) if hasattr(_model, "bic") else np.nan

        # Silhouette
        try:
            row["silhouette"] = silhouette_score(X, labels)
        except ValueError:
            row["silhouette"] = np.nan

        # Calinski-Harabasz
        try:
            row["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except ValueError:
            row["calinski_harabasz"] = np.nan

        # Davies-Bouldin
        try:
            row["davies_bouldin"] = davies_bouldin_score(X, labels)
        except ValueError:
            row["davies_bouldin"] = np.nan

        return row

    # Convert to list to allow multiple iterations if needed
    k_list = [int(k) for k in k_range]
    # If only one or two k's, sequential might be faster
    if len(k_list) <= 2 or n_jobs == 1:
        results = [_evaluate_one(k) for k in k_list]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_evaluate_one)(k) for k in k_list)

    return pd.DataFrame(results)


def _diversity_weights(score_matrix: pd.DataFrame) -> np.ndarray:
    """Compute diversity weights based on Spearman correlation.

    Metrics that are highly correlated with others receive lower weight,
    promoting a balanced assessment. Handles the read-only array issue
    by copying the correlation matrix before modifying it.
    """
    n = score_matrix.shape[1]
    if n == 1:
        return np.ones(1)
    corr = score_matrix.corr(method="spearman").abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    # A zero-variance metric (or one perfectly redundant in a degenerate way)
    # can make pandas/numpy report NaN for that pairwise correlation; treat
    # it as uncorrelated (0.0) rather than letting a single NaN propagate
    # through every row's mean and silently zero out every metric's weight.
    corr = np.nan_to_num(corr, nan=0.0)
    uniqueness = 1.0 - corr.mean(axis=1)
    weights = np.clip(uniqueness, 0.05, None)
    return weights / weights.sum()


def recommend_k(
    scores: pd.DataFrame,
    strategy: Literal[
        "davies_bouldin", "silhouette", "calinski_harabasz", "bic", "elbow", "consensus", "rank"
    ] = "davies_bouldin",
    tie_breaker: str = "silhouette",
    return_ranking: bool = False,
) -> int | tuple[int, pd.DataFrame]:
    """Recommend the best number of clusters from validity scores.

    Args:
        scores: DataFrame returned by `clustering_validity_scores`.
        strategy: Primary criterion:
            - 'davies_bouldin'  -> lowest Davies-Bouldin.
            - 'silhouette'      -> highest silhouette.
            - 'calinski_harabasz' -> highest Calinski-Harabasz.
            - 'bic'             -> lowest BIC (requires a model with a
               `.bic(X)` method, e.g. `GaussianMixture`; Murphy §21.3.7.2).
            - 'elbow'           -> "elbow" from inertia (heuristic; see the
               caveat below before reaching for this one).
            - 'consensus'       -> majority vote across the three main metrics.
            - 'rank'            -> weighted Borda ranking over **all** metrics
               (uses diversity weights to avoid redundancy; see notes below).
        tie_breaker: Only used for 'consensus' if there is a tie.
        return_ranking: If True and strategy is 'rank', returns a tuple
            (best_k, ranking_df) where ranking_df contains the full
            ranking table. Otherwise returns only best_k.

    Returns:
        Recommended k (int), or (int, pd.DataFrame) if `return_ranking=True`.

    Caveat on the 'elbow' strategy: Murphy (*Probabilistic Machine Learning:
    An Introduction*, §21.3.7) explicitly cautions against picking k from
    distortion/inertia alone, since inertia decreases monotonically with k
    and has no validation-set analogue to say when it should stop dropping
    ("unlike supervised learning, we cannot use reconstruction error on a
    validation set" to pick k the way one would pick model complexity
    elsewhere). The strategy is kept for exploratory use (and its underlying
    `inertia` column still participates in 'rank'), not because the
    second-derivative heuristic itself is a reliable standalone criterion -
    prefer 'bic', 'silhouette', or 'consensus'/'rank' over 'elbow' alone.

    Notes on the 'rank' strategy:
        - All metrics are oriented so that higher is better
          (lower-is-better metrics are negated).
        - A metric with no usable signal (e.g. `inertia` when the model has
          no `inertia_` attribute, so every value is NaN) is dropped before
          ranking rather than included, since a single all-NaN metric would
          otherwise zero out every metric's diversity weight.
        - For each k, we compute the rank per metric.
        - Diversity weights are derived from Spearman correlation
          among metrics, reducing influence of redundant ones.
        - A weighted Borda count (sum of ranks x weights) is computed;
          the k with the highest score is selected.
    """
    # ------------------------------------------------------------------
    # Direction mapping (lower is better or higher is better)
    # ------------------------------------------------------------------
    metric_direction = {
        "davies_bouldin": "min",
        "silhouette": "max",
        "calinski_harabasz": "max",
        "inertia": "min",
        "bic": "min",
    }

    def _best_by_metric(metric: str, frame: pd.DataFrame | None = None) -> int:
        frame = scores if frame is None else frame
        if metric not in frame.columns:
            raise ValueError(f"Metric '{metric}' not in scores.")
        series = frame[metric]
        if series.isna().all():
            raise ValueError(f"All values for '{metric}' are NaN.")
        if metric_direction[metric] == "min":
            return frame.loc[series.idxmin(), "k"]
        else:
            return frame.loc[series.idxmax(), "k"]

    # ------------------------------------------------------------------
    # Single-metric strategies
    # ------------------------------------------------------------------
    if strategy in metric_direction:
        best = _best_by_metric(strategy)
        return int(best)

    # ------------------------------------------------------------------
    # Elbow (heuristic)
    # ------------------------------------------------------------------
    if strategy == "elbow":
        if "inertia" not in scores.columns or scores["inertia"].isna().all():
            raise ValueError("Inertia not available for elbow method.")
        # The second-derivative heuristic below assumes ascending, gap-free
        # k values; `scores` is not guaranteed to already be in that order
        # (e.g. joblib preserves whatever order k_range was supplied in).
        sorted_scores = scores.sort_values("k").reset_index(drop=True)
        y = sorted_scores["inertia"].to_numpy()
        if len(y) < 3:
            raise ValueError("Need at least 3 k values for elbow detection.")
        if np.isnan(y).any():
            raise ValueError("Elbow method requires inertia for every k (no NaN gaps).")
        d1 = np.diff(y)
        d2 = np.diff(d1)
        elbow_idx = int(np.argmin(d2)) + 1
        return int(sorted_scores.iloc[elbow_idx]["k"])

    # ------------------------------------------------------------------
    # Consensus (majority vote)
    # ------------------------------------------------------------------
    if strategy == "consensus":
        metric_list = ["davies_bouldin", "silhouette", "calinski_harabasz"]
        valid_metrics = [m for m in metric_list if m in scores.columns and not scores[m].isna().all()]
        if not valid_metrics:
            raise ValueError("No valid metrics for consensus.")
        votes: dict[int, int] = {}
        for m in valid_metrics:
            k_best = _best_by_metric(m)
            votes[k_best] = votes.get(k_best, 0) + 1
        max_votes = max(votes.values())
        candidates = [k for k, v in votes.items() if v == max_votes]
        if len(candidates) == 1:
            best = candidates[0]
        else:
            # Tie-break using tie_breaker (e.g., silhouette), restricted to
            # the tied candidates themselves, not the full k_range - the
            # winner must be one of the tied k's, not just whichever k
            # happens to score best on tie_breaker overall.
            tie_df = scores[scores["k"].isin(candidates)]
            best = _best_by_metric(tie_breaker, frame=tie_df)
        return int(best)

    # ------------------------------------------------------------------
    # Multi-metric ranking with diversity weights
    # ------------------------------------------------------------------
    if strategy == "rank":
        # 1. Prepare a copy with k as index
        ranking_df = scores.set_index("k").copy()

        # 2. Orient all metrics so that higher is better
        oriented = ranking_df.copy()
        for col in oriented.columns:
            if col in metric_direction and metric_direction[col] == "min":
                oriented[col] = -oriented[col]
            # If a metric is unknown, we treat as higher-is-better (safe default)

        # 3. Drop metrics with no usable signal (e.g. inertia for a model
        # with no inertia_) rather than ranking on them: an all-NaN column
        # left in would otherwise zero out every metric's diversity weight.
        valid_cols = [c for c in oriented.columns if not oriented[c].isna().all()]
        if not valid_cols:
            raise ValueError("No metric in `scores` has any non-NaN values to rank on.")
        dropped_cols = [c for c in oriented.columns if c not in valid_cols]
        oriented = oriented[valid_cols]

        # 4. Compute ranks per metric (higher rank = better)
        ranks = oriented.rank(axis=0, ascending=True)  # rank across k values

        # 5. Compute diversity weights based on correlation of oriented scores
        weights = _diversity_weights(oriented)

        # 6. Weighted Borda count (sum of ranks x weights)
        ranking_df["weighted_borda"] = (ranks * weights).sum(axis=1)

        # 7. Sort and pick the best k
        ranking_df = ranking_df.sort_values("weighted_borda", ascending=False)
        best_k = ranking_df.index[0]

        # Attach weights (and any dropped metrics) as metadata for transparency
        ranking_df.attrs["metric_weights"] = dict(zip(valid_cols, np.round(weights, 3), strict=True))
        if dropped_cols:
            ranking_df.attrs["dropped_metrics"] = dropped_cols

        if return_ranking:
            return int(best_k), ranking_df
        return int(best_k)

    # ------------------------------------------------------------------
    # Unknown strategy
    # ------------------------------------------------------------------
    raise ValueError(f"Unknown strategy: {strategy}")


def _purity_score(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Fraction of points whose cluster's majority true label matches their own.

    No sklearn one-liner exists for this (unlike ARI/NMI), so it is computed
    directly: for each predicted cluster, count how many of its members share
    the single most common true label in that cluster, then sum across
    clusters and divide by the total number of points.
    """
    cluster_members = cluster_dictionary(np.arange(len(labels_pred)), labels_pred)
    correct = sum(np.unique(labels_true[indices], return_counts=True)[1].max() for indices in cluster_members.values())
    return correct / len(labels_pred)


def external_validity_scores(labels_pred: Sequence, labels_true: Sequence) -> dict[str, float]:
    """Compare predicted cluster labels against known ground-truth labels.

    A different question from `clustering_validity_scores`' internal metrics
    (silhouette, Davies-Bouldin, ...), which need no ground truth at all:
    these three metrics require one, e.g. a real building type (NREL), or an
    injected synthetic archetype used to check recovery.

    Args:
        labels_pred: Predicted cluster labels, one per sample.
        labels_true: Ground-truth labels, one per sample.

    Returns:
        Dict with keys:
            - "ari": Adjusted Rand Index (Hubert & Arabie 1985,
              `hubert1985comparing`); 1.0 is a perfect match, ~0.0 is
              chance-level agreement, corrected for chance.
            - "nmi": Normalized Mutual Information; 1.0 is a perfect match,
              0.0 is independence.
            - "purity": Fraction of points whose cluster's majority true
              label matches their own; biased toward more clusters (a
              cluster of size 1 is always 100% pure), so read alongside ARI/
              NMI rather than alone.
    """
    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)
    if len(labels_pred) != len(labels_true):
        raise ValueError(
            f"labels_pred and labels_true must be the same length: {len(labels_pred)} vs {len(labels_true)}"
        )

    return {
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "purity": float(_purity_score(labels_pred, labels_true)),
    }


def composite_trustworthy_score(
    candidates: pd.DataFrame,
    separation_metrics: Sequence[str] = ("silhouette", "calinski_harabasz", "davies_bouldin"),
    trust_metrics: Sequence[str] = ("balance", "stability"),
    id_column: str = "trial_number",
) -> pd.DataFrame:
    """Rank candidate pipelines by separation quality, discounted by structural trustworthiness.

    Separation-only metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
    cannot distinguish a real archetype from a single extreme outlier
    isolated into its own cluster, since all three only measure how
    separated groups are: an isolated outlier is, by construction,
    perfectly separated from everything else. Folding a trust signal
    (`ark.cluster.stability`'s balance-entropy or resampling-based
    stability) into the *same* additive ensemble as separation does not fix
    this on its own, since a near-perfect separation score can still
    average out to something respectable-looking even when trust is near
    zero. This function instead treats trust as a genuine multiplicative
    gate on separation quality, not another vote alongside it: no amount
    of separation can buy back a low trust factor, matching the property a
    flat weighted average does not have.

    Args:
        candidates: One row per candidate pipeline, with `id_column` plus
            every column named in `separation_metrics` and `trust_metrics`.
        separation_metrics: Column names combined into a single
            diversity-weighted rank ensemble via `recommend_k`'s own
            `strategy="rank"` machinery (reused here, not reimplemented;
            `id_column` is temporarily treated as `k` to do so).
        trust_metrics: Column names combined into a single trust factor via
            their minimum (a Rawlsian "weakest link" reflecting the same
            convention `cluster_stability`'s own `min_cluster_stability`
            already uses), each expected to already be oriented so that
            higher means more trustworthy and scaled to roughly [0, 1]
            (e.g. `balance`, `cheap_stability`, or the full audited
            `min_cluster_stability`).
        id_column: Column identifying each candidate row (e.g. a trial
            number); need not actually be a cluster count.

    Returns:
        DataFrame with `id_column`, `separation_score` (the rank ensemble,
        normalized to [0, 1]), `trust_factor`, and `composite_score`
        (their product), sorted by `composite_score` descending.
    """
    # Built as a fresh frame, not candidates.rename(id_column -> "k"): the
    # caller's own candidates frame may already have a real "k" column
    # (e.g. an Optuna trial's own cluster-count param), and renaming would
    # collide into two same-named columns rather than replacing it.
    renamed = pd.DataFrame({"k": candidates[id_column].to_numpy()})
    for metric in separation_metrics:
        renamed[metric] = candidates[metric].to_numpy()
    _, ranking_df = recommend_k(renamed, strategy="rank", return_ranking=True)
    separation_score = ranking_df["weighted_borda"] / len(renamed)

    trust_factor = candidates.set_index(candidates[id_column])[list(trust_metrics)].min(axis=1)

    result = pd.DataFrame({"separation_score": separation_score, "trust_factor": trust_factor})
    result["composite_score"] = result["separation_score"] * result["trust_factor"]
    result = result.sort_values("composite_score", ascending=False)
    result.index.name = id_column
    return result.reset_index()
