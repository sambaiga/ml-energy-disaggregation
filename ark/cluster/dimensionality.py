"""Choosing how many components to retain from an ordered dimensionality-reduction spectrum.

A different question from `ark.cluster.cluster_validation`: that module
chooses *k*, the number of clusters, in an already-reduced space. This
module chooses the dimensionality of that space in the first place, e.g.
how many principal components to keep before clustering on them.
"""

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

__all__ = ["select_n_components"]

log = logging.getLogger(__name__)


def _profile_log_likelihood(values: NDArray[np.float64]) -> tuple[int, NDArray[np.float64]]:
    """Zhu & Ghodsi (2006) profile-likelihood change point over an ordered spectrum.

    Models the top `q` values and the remaining `p - q` values as two
    independent Gaussian samples sharing one common variance (a different
    variance per group would let the profile log-likelihood diverge at
    q=1 or q=p-1, per the paper's own note), and picks the `q` that
    maximizes the resulting profile log-likelihood, an automatic
    alternative to eyeballing a scree plot for the "gap" or "elbow."

    Args:
        values: Ordered (descending) eigenvalues, singular values, or
            explained-variance ratios. The argmax below is invariant to
            positive rescaling (multiplying every value by a constant
            shifts every candidate q's log-likelihood by the same
            constant), so raw eigenvalues and normalized ratios give the
            same answer.

    Returns:
        `(q_hat, log_likelihoods)`: the selected split point (1-indexed,
        so `q_hat` components are retained) and the profile log-likelihood
        computed at every candidate split `q = 1, ..., p - 1`.
    """
    d = np.asarray(values, dtype=float)
    p = len(d)
    if p < 3:
        raise ValueError("profile_likelihood needs at least 3 values to find a change point")

    log_likelihoods = np.full(p - 1, -np.inf)
    for q in range(1, p):
        s1, s2 = d[:q], d[q:]
        mu1, mu2 = s1.mean(), s2.mean()
        var1 = s1.var(ddof=1) if len(s1) > 1 else 0.0
        var2 = s2.var(ddof=1) if len(s2) > 1 else 0.0
        pooled_var = ((q - 1) * var1 + (p - q - 1) * var2) / (p - 2)
        # A group of identical values (or p small enough that ddof=1
        # variances vanish) can drive the pooled variance to exactly 0,
        # which would make the Gaussian log-density diverge; floor it at
        # machine epsilon instead of dividing by zero.
        pooled_std = np.sqrt(max(pooled_var, np.finfo(float).eps))
        log_likelihoods[q - 1] = (
            norm.logpdf(s1, loc=mu1, scale=pooled_std).sum() + norm.logpdf(s2, loc=mu2, scale=pooled_std).sum()
        )

    q_hat = int(np.argmax(log_likelihoods)) + 1
    return q_hat, log_likelihoods


def select_n_components(
    explained_variance_ratio: NDArray[np.float64],
    method: Literal["variance_threshold", "profile_likelihood"] = "profile_likelihood",
    threshold: float = 0.90,
) -> int:
    """Choose how many components to retain from an ordered variance spectrum.

    Args:
        explained_variance_ratio: Per-component explained-variance ratio (or
            any ordered-descending, non-negative importance measure, e.g.
            raw eigenvalues), such as `PCA(...).explained_variance_ratio_`.
        method: "variance_threshold" retains the smallest number of
            components whose cumulative share reaches `threshold` (this
            book's existing ad hoc rule, e.g. already used for Chronos-2
            embeddings). "profile_likelihood" instead finds an automatic
            change point in the spectrum itself (Zhu & Ghodsi 2006,
            `zhu2006profilelikelihood`), with no threshold to pick.
        threshold: Cumulative-variance target, only used for
            `method="variance_threshold"`.

    Returns:
        Number of components to retain (at least 1).
    """
    ratios = np.asarray(explained_variance_ratio, dtype=float)
    if ratios.ndim != 1 or len(ratios) == 0:
        raise ValueError("explained_variance_ratio must be a non-empty 1D array")

    if method == "variance_threshold":
        cumulative = np.cumsum(ratios)
        return int(np.searchsorted(cumulative, threshold) + 1)

    if method == "profile_likelihood":
        q_hat, _ = _profile_log_likelihood(ratios)
        return q_hat

    raise ValueError(f"Unknown method: {method}")
