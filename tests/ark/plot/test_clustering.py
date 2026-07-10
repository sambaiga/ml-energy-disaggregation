import numpy as np
import pandas as pd
import pytest

from ark.plot.clustering import cluster_members, cluster_profiles, cluster_scatter, validity_curve, validity_scores


def test_cluster_members_groups_ids_by_label():
    ids = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]

    members = cluster_members(ids, labels)

    assert members == {"0": ["a", "c"], "1": ["b", "d"]}


def test_cluster_members_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shorter"):
        cluster_members(["a", "b"], [0])


def test_validity_scores_reports_one_row_per_k():
    rng = np.random.default_rng(0)
    # two well-separated blobs: a real 2-cluster structure to check against.
    blob_a = rng.normal(loc=-5, scale=0.3, size=(20, 2))
    blob_b = rng.normal(loc=5, scale=0.3, size=(20, 2))
    X = np.vstack([blob_a, blob_b])

    scores = validity_scores(X, range(2, 5))

    assert list(scores["k"]) == [2, 3, 4]
    assert {"inertia", "silhouette", "davies_bouldin"} <= set(scores.columns)
    # k=2 should score best (closest to the true structure) on both indices.
    best_silhouette_k = scores.loc[scores["silhouette"].idxmax(), "k"]
    best_db_k = scores.loc[scores["davies_bouldin"].idxmin(), "k"]
    assert best_silhouette_k == 2
    assert best_db_k == 2


def test_validity_curve_returns_a_plot_spec():
    scores = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "inertia": [10.0, 8.0, 7.0],
            "silhouette": [0.5, 0.4, 0.3],
            "davies_bouldin": [1.0, 1.2, 1.4],
        }
    )

    chart = validity_curve(scores)

    assert type(chart).__name__ == "PlotSpec"


def test_cluster_scatter_returns_a_plot_spec():
    embedding = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    labels = [0, 0, 1]

    chart = cluster_scatter(embedding, labels)

    assert type(chart).__name__ == "PlotSpec"


def test_cluster_profiles_returns_a_plot_spec():
    profiles = np.array(
        [
            [0.1, 0.5, 0.9, 0.3],
            [0.2, 0.6, 0.8, 0.4],
            [0.9, 0.5, 0.1, 0.7],
        ]
    )
    labels = [0, 0, 1]

    chart = cluster_profiles(profiles, labels)

    assert type(chart).__name__ == "PlotSpec"
