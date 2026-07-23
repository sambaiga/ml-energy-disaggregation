import numpy as np
import pandas as pd
import pytest

from ark.plot.clustering import cluster_members, cluster_profiles, cluster_scatter, validity_curve, validity_grid


def test_cluster_members_groups_ids_by_label():
    ids = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]

    members = cluster_members(ids, labels)

    assert members == {"0": ["a", "c"], "1": ["b", "d"]}


def test_cluster_members_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shorter"):
        cluster_members(["a", "b"], [0])


def _sample_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k": [2, 3, 4],
            "inertia": [10.0, 8.0, 7.0],
            "silhouette": [0.5, 0.4, 0.3],
            "calinski_harabasz": [50.0, 80.0, 70.0],
            "davies_bouldin": [1.0, 1.2, 1.4],
        }
    )


def test_validity_curve_returns_a_plot_spec():
    chart = validity_curve(_sample_scores())

    assert type(chart).__name__ == "PlotSpec"


def test_validity_curve_normalize_returns_a_plot_spec():
    chart = validity_curve(_sample_scores(), normalize=True)

    assert type(chart).__name__ == "PlotSpec"


def test_validity_curve_drops_all_nan_metric_instead_of_charting_it():
    scores = _sample_scores()
    scores["inertia"] = np.nan

    # Should not raise, and should not try to chart a metric with no signal.
    chart = validity_curve(scores)

    assert type(chart).__name__ == "PlotSpec"


def test_validity_curve_raises_when_every_metric_is_all_nan():
    scores = pd.DataFrame({"k": [2, 3], "inertia": [np.nan, np.nan]})

    with pytest.raises(ValueError, match="No usable"):
        validity_curve(scores)


def test_validity_grid_returns_a_plot_spec():
    chart = validity_grid(_sample_scores())

    assert type(chart).__name__ == "PlotSpec"


def test_validity_grid_without_best_k_markers_returns_a_plot_spec():
    chart = validity_grid(_sample_scores(), show_best=False)

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
