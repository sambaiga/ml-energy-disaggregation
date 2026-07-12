"""Feature engineering for anomaly detection: shape, not raw readings.

Mirrors Part 2's own feature-engineering lesson for a different signal (raw
current alone underperforms a real feature representation; see Chapter 2,
Feature Engineering) applied here to voltage/power timeseries: a bare
instantaneous reading throws away the shape information that actually
separates a real anomaly from ordinary diurnal variation. Every function
here turns a raw profile into a feature vector; detection never runs on a
single instantaneous value alone.
"""

from __future__ import annotations

import numpy as np


def rolling_statistics(profile: np.ndarray) -> np.ndarray:
    """Real, cheap summary statistics of one profile's own moment-to-moment behaviour.

    Args:
        profile: A real timeseries, shape `(n_steps,)`.

    Returns:
        A 4-element feature vector: mean, standard deviation, mean absolute
        rate of change (first difference), and peak-to-trough range.

    Examples:
        >>> rolling_statistics(np.array([1.0, 2.0, 1.0, 2.0])).round(2)
        array([1.5, 0.5, 1. , 1. ])
    """
    diffs = np.diff(profile)
    return np.array(
        [
            profile.mean(),
            profile.std(),
            np.abs(diffs).mean() if len(diffs) else 0.0,
            profile.max() - profile.min(),
        ]
    )


def fft_magnitude_features(profile: np.ndarray, *, n_features: int = 2) -> np.ndarray:
    """The strongest non-DC frequency components of one profile, real periodicity a bare reading cannot carry.

    Args:
        profile: A real timeseries, shape `(n_steps,)`.
        n_features: How many of the largest non-DC magnitude components to keep.

    Returns:
        An `n_features`-element vector of FFT magnitudes, largest first.

    Examples:
        >>> t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        >>> profile = np.sin(t) + 0.1 * np.sin(4 * t)
        >>> fft_magnitude_features(profile, n_features=2).round(1)
        array([24. ,  2.4])
    """
    spectrum = np.abs(np.fft.rfft(profile))[1:]  # drop the DC component
    top = np.sort(spectrum)[::-1][:n_features]
    if len(top) < n_features:
        top = np.pad(top, (0, n_features - len(top)))
    return top


def distance_matrix_features(profile: np.ndarray, *, n_bins: int = 12) -> np.ndarray:
    """A WRG-style pairwise distance-matrix embedding of one profile's own self-similarity structure.

    Chapter 2's own WRG/AWRG pipeline (`resources/nilm-code/AWRGNILM`'s
    `get_distance_measure()`) turns a high-frequency current window into a
    pairwise distance matrix, an "image" carrying far more identifying
    signal than any single reading. Same idea, reimplemented directly for
    this chapter's own low-frequency (30-minute AMI) profile rather than
    depending on the reference repo's own `torch`-based implementation:
    block-average the profile down to `n_bins` points (a simple PAA,
    matching the reference repo's own downsampling step), then compute the
    pairwise absolute-difference matrix between those points. The
    reference repo feeds this full symmetric matrix to a CNN as an image,
    where the redundancy is harmless; a classical multivariate detector
    needs a well-posed covariance matrix, so only the strictly upper
    triangle is kept here (checked directly: fitting `EllipticEnvelope` on
    the full flattened matrix warns that the covariance is rank-deficient,
    since a symmetric matrix with a zero diagonal has no more real degrees
    of freedom than its own upper triangle).

    Args:
        profile: A real timeseries, shape `(n_steps,)`.
        n_bins: How many points to reduce the profile to before computing
            the distance matrix. The returned vector has
            `n_bins * (n_bins - 1) / 2` entries.

    Returns:
        The distance matrix's own upper-triangular entries, flattened.

    Examples:
        >>> distance_matrix_features(np.array([1.0, 2.0, 3.0, 4.0]), n_bins=2)
        array([2.])
    """
    n_steps = len(profile)
    bin_edges = np.linspace(0, n_steps, n_bins + 1).astype(int)
    binned = np.array(
        [
            profile[bin_edges[i] : bin_edges[i + 1]].mean() if bin_edges[i + 1] > bin_edges[i] else 0.0
            for i in range(n_bins)
        ]
    )
    dist = np.abs(binned[:, None] - binned[None, :])
    rows, cols = np.triu_indices(n_bins, k=1)
    return dist[rows, cols]


def event_features(mask: np.ndarray, severity: np.ndarray, *, step_hours: float = 0.5) -> np.ndarray:
    """Occurrence, severity, and timing of a real event series, the general shape behind any LV network event.

    A voltage violation, a thermal overload, a reverse-power-flow episode,
    a PV export window, a reactive-power excursion, every one of these is,
    structurally, the same real question asked of a different signal: not
    "what is the instantaneous value," but how often does this event
    happen, how bad is it when it does, how long does it last, and when.
    This function is that question, generalized: given a boolean event
    mask and a real severity series (already computed for the caller's own
    signal, e.g. `abs(vmag_pu - 1.0)` beyond a compliant band, or export
    depth below zero), it returns how many real steps are inside an event,
    how many distinct real episodes those steps form, the event's own
    mean, max, and total severity, and the real timing (mean hour, and its
    spread) of when those steps occur.

    Args:
        mask: Boolean event indicator, one entry per real timestep.
        severity: A real, non-negative severity value per timestep (only
            entries where `mask` is True are used); e.g. how far a voltage
            reading sits outside a compliant band, or how deep a customer's
            own net load swings negative.
        step_hours: Real hours per timestep, for converting a step index
            into an hour-of-day.

    Returns:
        A 7-element feature vector: event-step count, episode count, mean
        severity, max severity, total severity, mean event hour, and hour
        spread (std). All zero when no event occurs.

    Examples:
        >>> mask = np.array([False, True, True, False, True])
        >>> severity = np.array([0.0, 0.02, 0.05, 0.0, 0.01])
        >>> event_features(mask, severity, step_hours=1.0).round(3)
        array([3.   , 2.   , 0.027, 0.05 , 0.08 , 2.333, 1.247])
    """
    if not mask.any():
        return np.zeros(7)
    n_steps = float(mask.sum())
    episode_starts = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int)])) == 1)
    n_episodes = float(len(episode_starts))
    event_severity = severity[mask]
    event_hours = np.flatnonzero(mask) * step_hours
    return np.array(
        [
            n_steps,
            n_episodes,
            event_severity.mean(),
            event_severity.max(),
            event_severity.sum(),
            event_hours.mean(),
            event_hours.std(),
        ]
    )


def extract_features(profile: np.ndarray, *, n_fft_features: int = 2) -> np.ndarray:
    """Turn one raw profile into the real feature vector the anomaly detectors run on.

    Concatenates the profile's own peak-normalized shape (the same
    representation Chapters 4-5 already cluster and retrieve on) with
    rolling statistics and FFT-magnitude features, not the raw instantaneous
    reading alone.

    Args:
        profile: A real timeseries, shape `(n_steps,)`.
        n_fft_features: How many FFT-magnitude features to append.

    Returns:
        A `(n_steps + 4 + n_fft_features,)` feature vector.

    Examples:
        >>> extract_features(np.array([1.0, 2.0, 1.0, 2.0]), n_fft_features=1).shape
        (9,)
    """
    peak = profile.max()
    shape = profile / peak if peak > 0 else profile
    return np.concatenate(
        [shape, rolling_statistics(profile), fft_magnitude_features(profile, n_features=n_fft_features)]
    )
