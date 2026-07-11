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
