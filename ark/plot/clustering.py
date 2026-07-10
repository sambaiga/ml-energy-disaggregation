"""Cluster-visualization helpers shared across every clustering case study in this book.

Promoted out of two of the author's own research notebooks
(`resources/Adaptive-conformal-PV-Forecasting.ipynb`,
`resources/cvar_flexibility/Clustering_And_Forecasting_London.ipynb`), both
of which re-derive the same "project to 2D, scatter colored by cluster
label" chart and a label-to-member-ids grouping dict, in matplotlib/seaborn,
slightly differently each time. `resources/utils/utils.py`'s own
`clustering_kmeans` even has a real bug, it references an undefined `df1`.
Rebuilt once here through lets-plot and `BRAND_PALETTE` for brand
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

from lets_plot import aes, geom_point, ggplot, ggsize, labs, layer_tooltips, scale_color_manual
import numpy as np
import pandas as pd

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
