"""Rank forecasting models across several real metrics at once, honestly.

A single metric read alone can crown a misleading winner: Part 5 Chapter 1
found window-average posting the lowest MAE and RMSE of four baselines while
having the second-worst correlation with the real signal, a rolling average
minimizes pointwise error by damping the response, not by tracking it. The
fix there was prose ("check the Corr column before trusting MAE"); this
module turns that same instinct into an actual method, reusing twiga's own
`diversity_weights` (used in `twiga.core.data.selection.select_top_features`
to rank *features* by combining several relevance metrics) applied to
*models* instead: each metric is treated as one ranker, and rankers that
mostly agree with each other (MAE, RMSE, WMAPE, SMAPE all measure pointwise
error and tend to move together) are down-weighted relative to a ranker that
adds genuinely independent information (Corr, which measures whether a
model tracks the real signal's shape at all).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import twiga.core.data.selection as _selection


def _diversity_weights_fixed(score_matrix: pd.DataFrame) -> np.ndarray:
    """Twiga's own `diversity_weights`, with one real bugfix applied.

    `DataFrame.corr().to_numpy()` returns a read-only array under this
    environment's numpy/pandas combination, and the upstream function
    mutates it in place with `np.fill_diagonal`, confirmed directly (see
    Part 5 Chapter 1's own notebook, which found and patched the same
    issue). The one-line fix is an explicit `.copy()`; behaviour is
    otherwise identical to `twiga.core.data.selection.diversity_weights`.
    """
    n = score_matrix.shape[1]
    if n == 1:
        return np.ones(1)
    corr = score_matrix.corr(method="spearman").abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    uniqueness = 1.0 - corr.mean(axis=1)
    weights = np.clip(uniqueness, 0.05, None)
    return weights / weights.sum()


_selection.diversity_weights = _diversity_weights_fixed
from twiga.core.data.selection import diversity_weights  # noqa: E402

LOWER_IS_BETTER = {"mae", "rmse", "wmape", "smape", "nrmse", "mape", "mse"}
HIGHER_IS_BETTER = {"corr"}


def rank_models(
    scores: pd.DataFrame,
    *,
    lower_is_better: set[str] | None = None,
) -> pd.DataFrame:
    """Diversity-weighted multi-metric ranking of models.

    Args:
        scores: One row per model, one column per metric (e.g. MAE, RMSE,
            Corr, WMAPE, SMAPE, NRMSE from `twiga.core.metrics.get_pointwise_metrics`).
            Index should be the model name.
        lower_is_better: Metric names where a smaller value is better
            (case-insensitive). Defaults to the common pointwise-error set
            (MAE, RMSE, WMAPE, SMAPE, NRMSE, MAPE, MSE); everything else is
            treated as higher-is-better (e.g. Corr).

    Returns:
        `scores`, with `arith_rank`, `borda_count`, and `weighted_borda`
        columns appended (higher = better for all three), sorted by
        `weighted_borda` descending. The per-metric weights used are
        attached as `.attrs["metric_weights"]` on the returned frame.
    """
    lower = {m.lower() for m in (lower_is_better or LOWER_IS_BETTER)}
    oriented = scores.copy()
    for col in oriented.columns:
        if col.lower() in lower:
            oriented[col] = -oriented[col]

    ranks = oriented.rank(ascending=True)
    weights = diversity_weights(oriented)

    result = scores.copy()
    result["arith_rank"] = ranks.mean(axis=1)
    result["borda_count"] = ranks.sum(axis=1)
    result["weighted_borda"] = (ranks * weights).sum(axis=1)
    result = result.sort_values("weighted_borda", ascending=False)
    result.attrs["metric_weights"] = dict(zip(oriented.columns, np.round(weights, 3), strict=True))
    return result
