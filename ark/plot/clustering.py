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

`validity_curve` below only visualizes a validity-scores table; computing
one (multiple internal indices across a range of k, with the NaN-handling
and diversity-weighted k-recommendation this book's clustering chapters
need) lives in `ark.cluster.cluster_validation.clustering_validity_scores`
instead, not here, since that's model-fitting logic, not a chart.

Deliberately doesn't wrap the clustering algorithms themselves (`KMeans`,
`SpectralClustering`, `PCA`, ...): scikit-learn's own API is already clean
and Pythonic, unlike OpenDSS's COM-style interface `ark.dss.Circuit` exists
to hide. Only the repeated visualization and bookkeeping steps are promoted
here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from lets_plot import (
    aes,
    coord_cartesian,
    coord_fixed,
    facet_wrap,
    geom_density2d,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    ggsize,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
    sampling_random_stratified,
    scale_color_manual,
    scale_shape_manual,
    scale_x_continuous,
    theme,
)
import numpy as np
import pandas as pd

from ark.cluster.feature import normalize_by_daily_max
from ark.plot.theme import modern_theme
from ark.plot.tokens import BRAND_PALETTE, CATEGORICAL_PALETTE, DANGER, SHAPE_PALETTE

# Mirrors `ark.cluster.cluster_validation.recommend_k`'s own direction map:
# kept as a separate, small copy rather than importing across modules, so a
# chart-only import here never has to pull in cluster_validation's sklearn/
# joblib dependency chain. Keep both in sync if a new metric is ever added.
_METRIC_DIRECTION = {
    "inertia": "min",
    "davies_bouldin": "min",
    "silhouette": "max",
    "calinski_harabasz": "max",
}
_METRIC_LABELS = {
    "inertia": "Inertia (lower is better)",
    "silhouette": "Silhouette (higher is better)",
    "calinski_harabasz": "Calinski-Harabasz (higher is better)",
    "davies_bouldin": "Davies-Bouldin (lower is better)",
}
# One fixed color per metric, reused across every chart in this module (see
# the book's own color-consistency rule): the same metric should read as
# the same color everywhere, not a color picked positionally per chart.
VALIDITY_METRIC_COLORS = {
    "inertia": BRAND_PALETTE[0],
    "silhouette": BRAND_PALETTE[1],
    "calinski_harabasz": BRAND_PALETTE[2],
    "davies_bouldin": BRAND_PALETTE[3],
}


def plot_cluster_centroids(
    df: pd.DataFrame,
    labels: Sequence,
    colors: Sequence | None = None,
    weekly: bool = False,
) -> Any:
    """Plot cluster centroids computed from `df`.

    The function normalises the input df table by daily maximum, groups
    per-cluster and returns a lets-plot `ggplot` showing the cluster centroids
    either by hour of day or day of week.

    Parameters
    ----------
    df:
        Wide-format time-indexed DataFrame (index = timestamps, columns = ids).
    labels:
        Sequence of cluster labels aligned with `df.columns`.
    colors:
        Optional sequence of colors for clusters. If None, a default
        palette derived from `CATEGORICAL_PALETTE` is used.
    weekly:
        If True, aggregate by day-of-week. Otherwise aggregate by hour.

    Returns:
    -------
    ggplot object
    """
    if df.empty:  # pragma: no cover - trivial guard
        return ggplot() + labs(title="Cluster centroids") + modern_theme()

    normalized = normalize_by_daily_max(df)
    clust = pd.Series(labels, index=df.T.index, name="Cluster")
    centroid_wide = normalized.T.assign(cluster=clust)
    centroid_wide = centroid_wide.groupby("cluster").mean().T

    centroid_long = (
        centroid_wide.reset_index()
        .rename(columns={"index": "time"})
        .melt(id_vars=["time"], var_name="cluster", value_name="value")
    )
    centroid_long["hour"] = centroid_long["time"].dt.hour
    centroid_long["day"] = centroid_long["time"].dt.dayofweek

    if weekly:
        plot_df = centroid_long.groupby(["cluster", "day"], as_index=False)["value"].mean()
        plot_df = plot_df.rename(columns={"day": "x"})
        x_breaks = list(range(7))
        x_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        x_label = "Day of week"
    else:
        plot_df = centroid_long.groupby(["cluster", "hour"], as_index=False)["value"].mean()
        plot_df = plot_df.rename(columns={"hour": "x"})
        x_breaks = list(range(24))
        x_labels = [str(i) for i in range(24)]
        x_label = "Hour of day"

    plot_df["x"] = plot_df["x"].astype(int)
    n_clusters = int(plot_df["cluster"].nunique())

    if colors is None:
        colors = (CATEGORICAL_PALETTE * (n_clusters // len(CATEGORICAL_PALETTE) + 1))[:n_clusters]

    tooltips = (
        layer_tooltips()
        .line("cluster")
        .line("x = @x")
        .line("value = @value")
        .format("@value", ".3f")
        .disable_splitting()
    )

    plot = (
        ggplot(plot_df, aes(x="x", y="value", color="cluster", group="cluster"))
        + geom_line(size=1.0, alpha=0.95, tooltips=tooltips)
        + scale_color_manual(values=colors)
        + scale_x_continuous(breaks=x_breaks, labels=x_labels)
        + labs(x=x_label, y="Normalized demand", title="Cluster centroids", color="Cluster")
        + modern_theme()
        + ggsize(650, 360)
    )

    return plot


def plot_clustered_profiles(
    df: pd.DataFrame,
    clust: Sequence,
    cluster_dict: dict,
    normalise: bool = True,
) -> Any:
    """Render clustered daily profiles using lets-plot.

    The function draws faint per-member hourly-average traces and bold
    cluster-centroid traces. Input ``df`` must be a time-indexed wide
    DataFrame (columns are meter identifiers). When ``normalise`` is true
    each series is normalised by its daily maximum using
    :func:`ark.cluster.feature.normalize_by_daily_max`.

    Args:
        df: Wide-format time-indexed DataFrame (columns = meters).
        clust: Sequence of cluster labels aligned to ``df.columns``.
        cluster_dict: Mapping from cluster label to iterable of meter ids.
        normalise: If True, normalise each series by its daily maximum.

    Returns:
        Lets-plot ``ggplot`` object with member traces and centroids.
    """
    # Keep behaviour consistent with previous implementation.
    if df.empty:  # pragma: no cover - trivial guard
        return ggplot() + labs(title="Cluster profiles") + modern_theme()

    # Optionally normalise in-place on a copy to avoid mutating caller state.
    data = df.copy()
    if normalise:
        data = normalize_by_daily_max(data)

    cluster_counts = clust.value_counts(normalize=True) * 100
    cluster_labels = {
        str(label): (f"clust: {label}, perc: {cluster_counts.loc[int(label)]:.1f}%") for label in cluster_counts.index
    }

    # Build mapping from meter -> cluster label (string).
    cluster_map: dict[Any, str] = {
        meter: str(cluster_value) for cluster_value, meters in cluster_dict.items() for meter in meters
    }

    plot_df = (
        data.reset_index()
        .rename(columns={data.index.name or "index": "time"})
        .melt(id_vars=["time"], var_name="meter", value_name="value")
    )

    plot_df["cluster"] = plot_df["meter"].map(cluster_map)
    plot_df = plot_df.dropna(subset=["cluster"])  # drop unmatched meters
    plot_df["hour"] = plot_df["time"].dt.hour
    plot_df["cluster_label"] = plot_df["cluster"].map(cluster_labels)

    # Per-member average by hour (used to draw faint member traces).
    member_plot_df = (
        plot_df.groupby(["cluster", "cluster_label", "meter", "hour"], as_index=False)["value"]
        .mean()
        .rename(columns={"hour": "x"})
    )

    # Centroid: mean of member averages.
    mean_plot_df = member_plot_df.groupby(["cluster", "cluster_label", "x"], as_index=False)["value"].mean()

    cluster_colors = {
        label: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        for i, label in enumerate(sorted(cluster_labels.values()))
    }

    x_breaks = list(range(24))
    x_labels = [str(i) for i in range(24)]
    x_label = "Hour of day"

    tooltips = (
        layer_tooltips()
        .line("cluster_label")
        .line("meter")
        .line("x = @x")
        .line("value = @value")
        .format("@value", ".3f")
        .disable_splitting()
    )

    plot = (
        ggplot()
        + geom_line(
            aes(x="x", y="value", group="meter"),
            data=member_plot_df,
            color="#9aa0a6",
            alpha=0.18,
            size=0.3,
            tooltips=tooltips,
        )
        + geom_line(
            aes(x="x", y="value", color="cluster_label", group="cluster_label"),
            data=mean_plot_df,
            size=1.6,
        )
        + scale_color_manual(values=cluster_colors)
        + facet_wrap("cluster_label", ncol=3)
        + scale_x_continuous(breaks=x_breaks, labels=x_labels)
        + coord_cartesian(ylim=(0, 1))
        + labs(x=x_label, y="Normalized demand", title="Cluster daily profiles")
        + modern_theme()
        + theme(legend_position="none")
        + ggsize(780, 420)
    )

    return plot


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
    alpha: float = 0.88,
    stroke: float = 0.6,
    show_legend: bool = True,
    legend_title: str = "Cluster",
    coord_fixed_ratio: float = 1.0,  # New: Force aspect ratio
    show_density: bool = False,  # New: Density contours
    density_color: str = "#6B7280",  # New: Density contour color
    tooltip_info: bool = True,
    max_points: int | None = None,
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
        alpha: Marker transparency.
        stroke: Marker stroke width.
        show_legend: Whether to show the legend.
        legend_title: Title for the legend.
        coord_fixed_ratio: Aspect-ratio setting for the plot.
        show_density: Whether to draw density contours.
        density_color: Color used for the density contours.
        tooltip_info: Whether to include tooltips on the points.
        max_points: If set and the embedding has more rows than this,
            draw a random sample stratified by cluster label instead of
            every point, so one dense majority cluster does not drown out
            a real but small minority one, and dense scatters with
            thousands of points stay readable rather than a solid blob.
            Every cluster still keeps at least a handful of points via
            lets-plot's own `min_subsample`. `None` (the default) draws
            every point, unchanged from this function's prior behaviour.

    Returns:
        A lets-plot figure, ready to display in a notebook cell.
    """
    names = [str(label_names.get(label, label)) if label_names else str(label) for label in labels]
    df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "cluster": names})
    n_clusters = df["cluster"].nunique()

    # Safe palette cycling
    colors = (CATEGORICAL_PALETTE * (n_clusters // len(CATEGORICAL_PALETTE) + 1))[:n_clusters]
    shapes = (SHAPE_PALETTE * (n_clusters // len(SHAPE_PALETTE) + 1))[:n_clusters]

    # Tooltip configuration
    tooltips = None
    if tooltip_info:
        tooltips = (
            layer_tooltips()
            .line("@cluster")
            .line("x = @x")
            .line("y = @y")
            .format("@x", ".3f")
            .format("@y", ".3f")
            .disable_splitting()
        )

    sampling = None
    if max_points is not None and len(df) > max_points:
        min_subsample = max(5, max_points // (n_clusters * 4))
        sampling = sampling_random_stratified(max_points, seed=0, min_subsample=min_subsample)

    plot = ggplot(df, aes(x="x", y="y", color="cluster", shape="cluster")) + geom_point(
        size=point_size,
        alpha=alpha,
        stroke=stroke,
        stroke_color="#1F2937",  # subtle dark edge for definition
        tooltips=tooltips,
        sampling=sampling,
    )
    if show_density:
        plot += geom_density2d(color=density_color, size=0.45, alpha=0.65, tooltips="none")
    plot += (
        scale_color_manual(values=colors)
        + scale_shape_manual(values=shapes)
        + labs(x=x_label, y=y_label, title=title, color="Cluster")
        + modern_theme()
        + ggsize(600, 550)
        + coord_fixed(ratio=coord_fixed_ratio)
    )
    # Legend
    if show_legend:
        plot += guides(
            color=guide_legend(override_aes={"size": 5.5}, title=legend_title),
            shape=guide_legend(override_aes={"size": 5.5}, title=legend_title),
        )
    else:
        plot += theme(legend_position="none")
    return plot


def _resolve_metrics(scores: pd.DataFrame, metrics: Sequence[str] | None) -> list[str]:
    """Pick which validity-metric columns to chart, dropping any with no signal.

    Defaults to whichever of the four known validity metrics are present;
    an all-NaN column (e.g. `inertia` for a model with no `inertia_`) is
    dropped rather than charted as an empty line, and a caller-supplied
    `error` column (from a k that failed to fit) is never treated as a
    metric to chart, known or not.
    """
    candidates = metrics if metrics is not None else _METRIC_DIRECTION.keys()
    resolved = [m for m in candidates if m in scores.columns and not scores[m].isna().all()]
    if not resolved:
        raise ValueError("No usable (non-all-NaN) validity metric found in `scores`.")
    return resolved


def validity_curve(
    scores: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    normalize: bool = False,
    title: str = "Cluster validity across k",
) -> object:
    """Chart `clustering_validity_scores`' metrics across k.

    Two shapes:
        - `normalize=False` (default): small-multiples, one free-y-scale
          facet per metric, each in its own real units.
        - `normalize=True`: every metric min-max scaled to [0, 1] and
          overlaid on one panel, trading real units for a direct visual
          comparison of *where* each metric peaks or bottoms out across k.

    Args:
        scores: Output of `ark.cluster.cluster_validation.clustering_validity_scores`.
        metrics: Which metric columns to chart. Defaults to whichever of
            `inertia`, `silhouette`, `calinski_harabasz`, `davies_bouldin`
            are present in `scores` and not all-NaN; an `error` column, if
            present, is never charted.
        normalize: See above.
        title: Chart title.

    Returns:
        A lets-plot figure, ready to display in a notebook cell.
    """
    resolved = _resolve_metrics(scores, metrics)

    if not normalize:
        long_df = scores.melt(id_vars="k", value_vars=resolved, var_name="metric", value_name="value")
        return (
            ggplot(long_df, aes(x="k", y="value"))
            + geom_line(color=BRAND_PALETTE[0], size=1)
            + geom_point(color=BRAND_PALETTE[0], size=3)
            + facet_wrap("metric", scales="free_y")
            + labs(x="Number of clusters (k)", y="", title=title)
            + modern_theme()
            + ggsize(750, 320)
        )

    normalized = scores[["k", *resolved]].copy()
    for metric in resolved:
        col = normalized[metric]
        span = col.max() - col.min()
        normalized[metric] = 0.5 if span == 0 else (col - col.min()) / span

    long_df = normalized.melt(id_vars="k", value_vars=resolved, var_name="metric", value_name="normalized_score")
    return (
        ggplot(long_df, aes(x="k", y="normalized_score", color="metric"))
        + geom_line(size=1)
        + geom_point(size=3)
        + scale_color_manual(values=VALIDITY_METRIC_COLORS)
        + labs(x="Number of clusters (k)", y="Normalized score (0-1)", color="Metric", title=title)
        + modern_theme()
        + ggsize(650, 420)
    )


def validity_grid(
    scores: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    show_best: bool = True,
    title: str = "Clustering validity metrics",
) -> object:
    """Small-multiples validity chart with each metric's own best k marked.

    The lets-plot, brand-consistent equivalent of a hand-rolled matplotlib
    "one subplot per validity index, direction labeled, best k marked"
    chart: each facet's title already states whether lower or higher is
    better for that metric, so a reader never has to remember the
    direction convention per index.

    Args:
        scores: Output of `ark.cluster.cluster_validation.clustering_validity_scores`.
        metrics: Which metric columns to chart, see `validity_curve`.
        show_best: If True, draw a dashed vertical line at each metric's
            own best k (lowest for inertia/davies_bouldin, highest for
            silhouette/calinski_harabasz).
        title: Overall figure title.

    Returns:
        A lets-plot figure, ready to display in a notebook cell.
    """
    resolved = _resolve_metrics(scores, metrics)
    labels = {m: _METRIC_LABELS.get(m, m) for m in resolved}

    long_df = scores.melt(id_vars="k", value_vars=resolved, var_name="metric", value_name="value")
    long_df["metric_label"] = long_df["metric"].map(labels)

    plot = (
        ggplot(long_df, aes(x="k", y="value"))
        + geom_line(color=BRAND_PALETTE[0], size=1)
        + geom_point(color=BRAND_PALETTE[0], size=3)
        + facet_wrap("metric_label", scales="free_y")
        + labs(x="Number of clusters (k)", y="", title=title)
        + modern_theme()
        + ggsize(750, 550)
    )

    if show_best:

        def _best_idx(metric: str) -> int:
            series = scores[metric]
            return series.idxmin() if _METRIC_DIRECTION.get(metric, "max") == "min" else series.idxmax()

        best_df = pd.DataFrame(
            [{"metric_label": labels[metric], "best_k": scores.loc[_best_idx(metric), "k"]} for metric in resolved]
        )
        plot = plot + geom_vline(aes(xintercept="best_k"), data=best_df, color=DANGER, linetype="dashed", size=0.8)

    return plot


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
