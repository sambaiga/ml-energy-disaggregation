"""Rank real cases by risk, honestly, with a confidence interval on each rank.

Borrowed from information retrieval's learning-to-rank literature, not
invented for this book. `combined_score` deliberately stays a simple,
transparent multi-criteria score rather than a learned ranker: on a labeled
set this small, a learned model has little to earn its complexity back
against, the same lesson Chapter 4 drew from IDEC. `rank_with_interval`
is inspired by recent conformal-ranking work (uncertainty on a rank
position, not just a point rank) but implemented here as a bootstrap
rank-interval, not a from-scratch reimplementation of that paper's exact
distributional method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def combined_score(
    df: pd.DataFrame,
    columns: list[str],
    *,
    method: str = "max",
) -> pd.Series:
    """Combine several risk columns, each on its own scale, into one comparable rank score.

    Each column is min-max normalized to [0, 1] first, so no single risk
    type dominates just because its raw units happen to be larger.

    Args:
        df: A table with one row per case (customer, bus position, feeder, ...).
        columns: Risk column names to combine.
        method: `"max"` (worst-case across risk types, the default, matching
            Chapter 4's own finding that different risk types concentrate in
            different cases) or `"mean"`.

    Returns:
        A `pd.Series` of combined scores, one per row, aligned to `df`'s index.
    """
    normalized = pd.DataFrame(index=df.index)
    for col in columns:
        lo, hi = df[col].min(), df[col].max()
        normalized[col] = (df[col] - lo) / (hi - lo) if hi > lo else 0.0

    if method == "max":
        return normalized.max(axis=1)
    if method == "mean":
        return normalized.mean(axis=1)
    raise ValueError(f"unknown method {method!r}, expected 'max' or 'mean'")


def rank_table(df: pd.DataFrame, score_column: str, *, ascending: bool = False) -> pd.DataFrame:
    """Sort by a score column and attach an explicit rank position.

    Args:
        df: A table with one row per case.
        score_column: Column to rank by.
        ascending: If True, rank 1 is the smallest score instead of the largest.

    Returns:
        `df` sorted by `score_column`, with a new `rank` column (1 = top priority).
    """
    ranked = df.sort_values(score_column, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def rank_with_interval(
    df: pd.DataFrame,
    score_column: str,
    *,
    ascending: bool = False,
    n_bootstrap: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Attach a bootstrap rank-position confidence interval to each case.

    Resamples the labeled set with replacement `n_bootstrap` times,
    re-ranks each time, and reports the 5th-95th percentile of each case's
    own rank position across those resamples, a real, checkable measure of
    how much sampling noise alone could move a case up or down the list,
    not just a bare point rank.

    Args:
        df: A table with one row per case.
        score_column: Column to rank by.
        ascending: If True, rank 1 is the smallest score instead of the largest.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Seed for reproducibility.

    Returns:
        `rank_table`'s output, with `rank_lower`/`rank_upper` columns added.
    """
    rng = np.random.default_rng(random_state)
    ranked = rank_table(df, score_column, ascending=ascending)
    n = len(ranked)

    bootstrap_ranks = {idx: [] for idx in ranked.index}
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(ranked.index, size=n, replace=True)
        sample = ranked.loc[sample_idx, [score_column]].copy()
        sample["_original_idx"] = sample_idx
        sample_sorted = sample.sort_values(score_column, ascending=ascending).reset_index(drop=True)
        for position, original_idx in enumerate(sample_sorted["_original_idx"], start=1):
            bootstrap_ranks[original_idx].append(position)

    ranked["rank_lower"] = [np.percentile(bootstrap_ranks[idx], 5) for idx in ranked.index]
    ranked["rank_upper"] = [np.percentile(bootstrap_ranks[idx], 95) for idx in ranked.index]
    return ranked
