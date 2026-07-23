import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from ark.cluster.cluster_validation import (
    clustering_validity_scores,
    composite_trustworthy_score,
    external_validity_scores,
    recommend_k,
)


def test_clustering_validity_scores_recovers_true_k():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=-8, scale=0.3, size=(20, 3))
    blob_b = rng.normal(loc=0, scale=0.3, size=(20, 3))
    blob_c = rng.normal(loc=8, scale=0.3, size=(20, 3))
    X = np.vstack([blob_a, blob_b, blob_c])

    scores = clustering_validity_scores(X, k_range=range(2, 6), n_jobs=1)

    assert list(scores["k"]) == [2, 3, 4, 5]
    assert "error" not in scores.columns
    best_k = recommend_k(scores, strategy="silhouette")
    assert best_k == 3


def test_clustering_validity_scores_reports_failed_k_without_crashing():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))  # only 5 samples

    # k=10 > n_samples is infeasible for KMeans; every other k should still
    # produce real metrics instead of the whole sweep raising.
    scores = clustering_validity_scores(X, k_range=[2, 10], n_jobs=1)

    row_k2 = scores.loc[scores["k"] == 2].iloc[0]
    row_k10 = scores.loc[scores["k"] == 10].iloc[0]
    assert not np.isnan(row_k2["silhouette"])
    assert np.isnan(row_k10["silhouette"])
    assert isinstance(row_k10["error"], str) and row_k10["error"]


def test_recommend_k_rank_ignores_all_nan_metric_instead_of_zeroing_every_weight():
    # AgglomerativeClustering has no `inertia_`, so a real caller using it
    # with clustering_validity_scores gets an all-NaN inertia column here.
    assert not hasattr(AgglomerativeClustering(), "inertia_")

    scores = pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "inertia": [np.nan, np.nan, np.nan, np.nan],
            "silhouette": [0.30, 0.50, 0.45, 0.20],
            "calinski_harabasz": [100, 200, 180, 90],
            "davies_bouldin": [1.2, 0.8, 0.9, 1.5],
        }
    )

    best_k, ranking = recommend_k(scores, strategy="rank", return_ranking=True)

    # k=3 is the clear winner on all three real metrics; an all-NaN inertia
    # column must not silently collapse every weighted_borda score to 0.
    assert best_k == 3
    assert ranking["weighted_borda"].nunique() > 1
    assert ranking.attrs["dropped_metrics"] == ["inertia"]
    assert "inertia" not in ranking.attrs["metric_weights"]


def test_recommend_k_rank_raises_when_every_metric_is_all_nan():
    scores = pd.DataFrame({"k": [2, 3], "inertia": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="non-NaN"):
        recommend_k(scores, strategy="rank")


def test_recommend_k_consensus_tie_break_stays_within_tied_candidates():
    # davies_bouldin votes k=3, silhouette votes k=5, calinski_harabasz
    # votes k=7: a genuine three-way tie among {3, 5, 7}. tie_breaker is
    # 'inertia', a metric outside the three voting metrics, whose own
    # global-best k (2) has zero votes and must never win.
    scores = pd.DataFrame(
        {
            "k": [2, 3, 4, 5, 6, 7, 8],
            "davies_bouldin": [2.0, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "silhouette": [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
            "calinski_harabasz": [10, 10, 10, 10, 10, 99, 10],
            "inertia": [1.0, 50, 50, 50, 50, 50, 50],
        }
    )

    best_k = recommend_k(scores, strategy="consensus", tie_breaker="inertia")

    assert best_k in (3, 5, 7)


def test_recommend_k_elbow_is_order_independent():
    # A real u-shaped inertia curve, but supplied out of ascending-k order;
    # the elbow heuristic must sort by k itself rather than trust row order.
    ordered = pd.DataFrame({"k": [2, 3, 4, 5, 6], "inertia": [100.0, 60.0, 45.0, 38.0, 34.0]})
    shuffled = ordered.sample(frac=1, random_state=1).reset_index(drop=True)

    assert recommend_k(ordered, strategy="elbow") == recommend_k(shuffled, strategy="elbow")


def test_clustering_validity_scores_supports_gmm_via_n_components():
    # GaussianMixture uses `n_components`, not `n_clusters` - the pre-fix
    # code hardcoded `n_clusters` and would raise a TypeError before the
    # sweep even started for a model like this one.
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=-8, scale=0.3, size=(20, 3))
    blob_b = rng.normal(loc=0, scale=0.3, size=(20, 3))
    blob_c = rng.normal(loc=8, scale=0.3, size=(20, 3))
    X = np.vstack([blob_a, blob_b, blob_c])

    scores = clustering_validity_scores(X, k_range=range(2, 6), model=GaussianMixture(random_state=0), n_jobs=1)

    assert list(scores["k"]) == [2, 3, 4, 5]
    assert "error" not in scores.columns
    assert scores["bic"].notna().all()


def test_clustering_validity_scores_bic_is_nan_for_kmeans():
    # KMeans has no `.bic(X)` method; the column should be present but NaN,
    # not missing, so downstream code can uniformly check for it.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))

    scores = clustering_validity_scores(X, k_range=[2, 3], n_jobs=1)

    assert "bic" in scores.columns
    assert scores["bic"].isna().all()


def test_clustering_validity_scores_rejects_model_without_cluster_count_param():
    class NoClusterParamEstimator(BaseEstimator):
        pass

    with pytest.raises(TypeError, match="n_clusters.*n_components"):
        clustering_validity_scores(np.zeros((5, 2)), k_range=[2], model=NoClusterParamEstimator())


def test_recommend_k_bic_picks_lowest_bic():
    scores = pd.DataFrame({"k": [2, 3, 4, 5], "bic": [500.0, 300.0, 320.0, 400.0]})

    assert recommend_k(scores, strategy="bic") == 3


def test_external_validity_scores_perfect_match_gives_ari_and_nmi_of_one():
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])  # same partition, different label ids

    result = external_validity_scores(labels_pred, labels_true)

    assert result["ari"] == pytest.approx(1.0)
    assert result["nmi"] == pytest.approx(1.0)
    assert result["purity"] == pytest.approx(1.0)


def test_external_validity_scores_purity_with_impure_clusters():
    # cluster 0 = {0,0,1} (majority 0, 2/3 correct), cluster 1 = {1,1} (all correct)
    labels_true = np.array([0, 0, 1, 1, 1])
    labels_pred = np.array([0, 0, 0, 1, 1])

    result = external_validity_scores(labels_pred, labels_true)

    assert result["purity"] == pytest.approx(4 / 5)


def test_external_validity_scores_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        external_validity_scores(np.array([0, 1]), np.array([0, 1, 2]))


def test_composite_trustworthy_score_cannot_be_bought_back_by_separation_alone():
    # The exact real finding from london-cluster.ipynb: trial 1 has
    # near-perfect separation (silhouette 0.999) but near-zero trust
    # (balance/stability ~0.01-0.63, the MAC000037 outlier-isolation
    # signature), while trial 2 has good-but-imperfect separation and high
    # trust. A flat weighted average could let trial 1's huge separation
    # edge outweigh its trust deficit; the multiplicative gate must not.
    candidates = pd.DataFrame(
        {
            "trial_number": [1, 2, 3],
            "silhouette": [0.999, 0.935, 0.5],
            "calinski_harabasz": [3_000_000.0, 3_000.0, 500.0],
            "davies_bouldin": [0.002, 0.3, 0.9],
            "balance": [0.009, 0.85, 0.9],
            "stability": [0.63, 0.95, 0.9],
        }
    )

    result = composite_trustworthy_score(candidates)

    assert result.iloc[0]["trial_number"] == 2
    # trial 1 has the best raw separation_score of the three (it dominates
    # every separation metric) but must not win on composite_score.
    best_separation_trial = result.loc[result["separation_score"].idxmax(), "trial_number"]
    assert best_separation_trial == 1
    assert result.loc[result["trial_number"] == 1, "composite_score"].iloc[0] < result["composite_score"].max()


def test_composite_trustworthy_score_orders_by_composite_not_separation():
    candidates = pd.DataFrame(
        {
            "trial_number": [10, 20],
            "silhouette": [0.9, 0.6],
            "calinski_harabasz": [5000.0, 2000.0],
            "davies_bouldin": [0.2, 0.5],
            "balance": [0.1, 0.95],
            "stability": [0.2, 0.95],
        }
    )

    result = composite_trustworthy_score(candidates)

    assert list(result["trial_number"]) == [20, 10]
    assert result["composite_score"].is_monotonic_decreasing


def test_composite_trustworthy_score_uses_minimum_across_trust_metrics():
    # A candidate that is balanced but not stable (or vice versa) should be
    # penalized by its worst trust axis, not averaged with its best one.
    candidates = pd.DataFrame(
        {
            "trial_number": [1, 2],
            "silhouette": [0.8, 0.8],
            "calinski_harabasz": [1000.0, 1000.0],
            "davies_bouldin": [0.3, 0.3],
            "balance": [0.95, 0.5],
            "stability": [0.5, 0.95],
        }
    )

    result = composite_trustworthy_score(candidates)

    assert result["trust_factor"].tolist() == [0.5, 0.5]
    assert result["composite_score"].iloc[0] == result["composite_score"].iloc[1]


def test_composite_trustworthy_score_handles_a_real_k_column_colliding_with_id_column():
    # The real usage in london-cluster.ipynb: candidates come from Optuna
    # trials that already have their own genuine "k" column (the trial's
    # own suggested cluster count), separate from id_column="trial_number".
    # Renaming trial_number -> "k" naively would collide with that existing
    # column into two same-named columns, breaking recommend_k's own
    # set_index("k") internally.
    candidates = pd.DataFrame(
        {
            "trial_number": [7, 8, 9],
            "k": [3, 5, 4],
            "silhouette": [0.9, 0.6, 0.75],
            "calinski_harabasz": [5000.0, 2000.0, 3000.0],
            "davies_bouldin": [0.2, 0.5, 0.3],
            "balance": [0.9, 0.9, 0.9],
            "stability": [0.9, 0.9, 0.9],
        }
    )

    result = composite_trustworthy_score(candidates)

    assert set(result["trial_number"]) == {7, 8, 9}
    assert result.iloc[0]["trial_number"] == 7
