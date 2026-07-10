"""Cluster-visualization helpers shared across every clustering case study in this book.

Promoted out of the author's own research notebooks
(`resources/Adaptive-conformal-PV-Forecasting.ipynb`,
`resources/cvar_flexibility/Clustering_And_Forecasting_London.ipynb`,
`resources/profiling 3/src/utils/visualize_cluster.py`), all of which
re-derive the same handful of charts, a "project to 2D, scatter colored by
cluster label" chart, a label-to-member-ids grouping dict, a validity-index
comparison across k, and a per-cluster member-profiles-plus-mean chart, in
matplotlib/seaborn, slightly differently each time. `resources/utils/utils.py`'s
own `clustering_kmeans` even has a real bug, it references an undefined
`df1`. Rebuilt once here through lets-plot and `BRAND_PALETTE` for brand
consistency with every other chart in this book, rather than re-derived per
chapter.

Deliberately doesn't wrap the clustering algorithms themselves (`KMeans`,
`SpectralClustering`, `PCA`, ...): scikit-learn's own API is already clean
and Pythonic, unlike OpenDSS's COM-style interface `ark.dss.Circuit` exists
to hide. Only the repeated visualization and bookkeeping steps are promoted
here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from lets_plot import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    ggsize,
    labs,
    layer_tooltips,
    scale_color_manual,
    theme,
)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from ark.plot.theme import modern_theme
from ark.plot.tokens import BRAND_PALETTE


def cluster_members(ids: Sequence, labels: Sequence) -> dict[str, list]:
    """Group `ids` by their cluster `labels`, e.g. `{"0": [...], "1": [...]}`.

    Args:
        ids: One identifier per item (a customer name, a bus name, ...).
        labels: The matching cluster label for each id, same length and order.

    Returns:
        Cluster label (as a string) mapped to the list of ids in it.
    """
    members: dict[str, list] = defaultdict(list)
    for id_, label in zip(ids, labels, strict=True):
        members[str(label)].append(id_)
    return dict(members)


def cluster_scatter(
    embedding: np.ndarray,
    labels: Sequence,
    *,
    label_names: dict | None = None,
    title: str = "Cluster projection",
    x_label: str = "Component 1",
    y_label: str = "Component 2",
    point_size: float = 4.0,
) -> object:
    """Scatter a 2D embedding, colored by cluster label.

    `embedding` is any already-computed 2D projection (PCA, t-SNE, UMAP, or
    two raw features); this function only plots it, picking and fitting the
    projection is the caller's job, so the same chart works regardless of
    which reduction a given case study needs.

    Args:
        embedding: `(n_samples, 2)` array of 2D coordinates.
        labels: One cluster label per row of `embedding`.
        label_names: Optional label -> display-name mapping (e.g. cluster
            index -> a real phase name), shown in the legend instead of the
            raw label.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        point_size: Marker size.

    Returns:
        A lets-plot figure, ready to display in a notebook cell.
    """
    names = [str(label_names.get(label, label)) if label_names else str(label) for label in labels]
    df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "cluster": names})
    n_clusters = df["cluster"].nunique()

    return (
        ggplot(df, aes(x="x", y="y", color="cluster"))
        + geom_point(size=point_size, alpha=0.85, tooltips=layer_tooltips().disable_splitting())
        + scale_color_manual(values=BRAND_PALETTE[:n_clusters])
        + labs(x=x_label, y=y_label, title=title, color="Cluster")
        + modern_theme()
        + ggsize(600, 450)
    )


def validity_scores(
    X: np.ndarray,
    k_range: Sequence[int],
    *,
    random_state: int = 0,
    n_init: int = 20,
) -> pd.DataFrame:
    """Inertia, silhouette, and Davies-Bouldin index across a range of k.

    A validity comparison should be run, not assumed: no single index
    reliably picks the right k alone, the combination this function reports
    is the same one the reference literature (McLoughlin et al. 2015,
    Rajabi et al. 2020) checks together before choosing k.

    Args:
        X: `(n_samples, n_features)` clustering input.
        k_range: Candidate cluster counts to try.
        random_state: Passed to `KMeans` for reproducibility.
        n_init: Passed to `KMeans`.

    Returns:
        One row per k, columns `k`, `inertia`, `silhouette`, `davies_bouldin`.
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def validity_curve(scores: pd.DataFrame, *, title: str = "Cluster validity across k") -> object:
    """Facet `validity_scores`' three metrics into one small-multiples chart.

    Args:
        scores: Output of `validity_scores`.
        title: Chart title.

    Returns:
        A lets-plot figure, ready to display in a notebook cell.
    """
    long_df = scores.melt(
        id_vars="k", value_vars=["inertia", "silhouette", "davies_bouldin"], var_name="metric", value_name="value"
    )
    return (
        ggplot(long_df, aes(x="k", y="value"))
        + geom_line(color=BRAND_PALETTE[0], size=1)
        + geom_point(color=BRAND_PALETTE[0], size=3)
        + facet_wrap("metric", scales="free_y")
        + labs(x="Number of clusters (k)", y="", title=title)
        + modern_theme()
        + ggsize(750, 320)
    )


def cluster_profiles(
    profiles: np.ndarray,
    labels: Sequence,
    *,
    x: Sequence | None = None,
    label_names: dict | None = None,
    x_label: str = "Hour of day",
    y_label: str = "Normalized demand",
    title: str = "Cluster daily profiles",
) -> object:
    """One small-multiples panel per cluster, member profiles plus the mean.

    Every member's own profile is drawn thin and translucent, with that
    cluster's mean profile drawn bold over it, so a cluster with many
    members doesn't drown out one with few.

    Args:
        profiles: `(n_samples, n_timesteps)`, one customer's own daily (or
            weekly, seasonal, ...) shape per row.
        labels: One cluster label per row of `profiles`.
        x: Optional x-axis values (e.g. hour-of-day), defaults to
            `0..n_timesteps-1`.
        label_names: Optional label -> display-name mapping, shown as each
            facet's title instead of the raw label.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Chart title.

    Returns:
        A lets-plot figure, ready to display in a notebook cell.
    """
    n_samples, n_timesteps = profiles.shape
    x_values = np.asarray(x) if x is not None else np.arange(n_timesteps)
    names = [str(label_names.get(label, label)) if label_names else str(label) for label in labels]

    member_df = pd.concat(
        [pd.DataFrame({"x": x_values, "y": profiles[i], "cluster": names[i], "member": i}) for i in range(n_samples)],
        ignore_index=True,
    )
    mean_df = member_df.groupby(["cluster", "x"], as_index=False)["y"].mean()
    n_clusters = member_df["cluster"].nunique()

    return (
        ggplot()
        + geom_line(aes(x="x", y="y", group="member"), data=member_df, color="#9aa0a6", alpha=0.25, size=0.4)
        + geom_line(aes(x="x", y="y", color="cluster"), data=mean_df, size=1.4)
        + scale_color_manual(values=BRAND_PALETTE[:n_clusters])
        + facet_wrap("cluster")
        + labs(x=x_label, y=y_label, title=title, color="Cluster")
        + modern_theme()
        + theme(legend_position="none")
        + ggsize(750, 500)
    )
