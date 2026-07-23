import numpy as np
import pytest

from ark.cluster.dimensionality import select_n_components


def test_profile_likelihood_matches_the_papers_own_worked_example():
    # Zhu & Ghodsi (2006), Fig. 1: eigenvalues 10, 9, 3, 2, 1 have a clear
    # gap after the second value; the paper's own MLE of q is 2.
    eigenvalues = np.array([10.0, 9.0, 3.0, 2.0, 1.0])

    n = select_n_components(eigenvalues, method="profile_likelihood")

    assert n == 2


def test_profile_likelihood_is_invariant_to_positive_rescaling():
    # Explained-variance ratios are eigenvalues divided by their sum, a
    # positive rescaling; the method must pick the same q either way, since
    # a downstream caller might pass raw eigenvalues or normalized ratios.
    eigenvalues = np.array([10.0, 9.0, 3.0, 2.0, 1.0])
    ratios = eigenvalues / eigenvalues.sum()

    assert select_n_components(eigenvalues, method="profile_likelihood") == select_n_components(
        ratios, method="profile_likelihood"
    )


def test_profile_likelihood_recovers_a_known_true_elbow_in_a_longer_spectrum():
    # A real signal/noise spectrum: 5 signal eigenvalues forming a tight
    # cluster near 9-10, then a long, flat noise tail clustered around 0.1 -
    # a synthetic case constructed so the true break is known in advance
    # (a large gap between two internally-tight groups), not read off a plot.
    signal = np.array([10.0, 9.7, 9.4, 9.1, 8.8])
    rng = np.random.default_rng(0)
    noise = 0.1 + rng.normal(scale=0.01, size=15)
    spectrum = np.concatenate([signal, np.sort(noise)[::-1]])

    n = select_n_components(spectrum, method="profile_likelihood")

    assert n == 5


def test_variance_threshold_and_profile_likelihood_can_disagree():
    # Demonstrates the actual value of having both methods available: a
    # spectrum with a real, obvious 3-component elbow, but a long enough
    # noise tail that reaching 90% cumulative variance requires many more
    # components than the true break point.
    signal = np.array([40.0, 30.0, 25.0])
    rng = np.random.default_rng(1)
    noise = np.abs(rng.normal(scale=1.0, size=20)) + 0.5
    spectrum = np.concatenate([signal, np.sort(noise)[::-1]])
    ratios = spectrum / spectrum.sum()

    n_profile = select_n_components(ratios, method="profile_likelihood")
    n_variance = select_n_components(ratios, method="variance_threshold", threshold=0.90)

    assert n_profile == 3
    assert n_variance > n_profile


def test_variance_threshold_matches_manual_cumulative_sum():
    ratios = np.array([0.5, 0.3, 0.1, 0.1])

    n = select_n_components(ratios, method="variance_threshold", threshold=0.80)

    # cumulative: 0.5, 0.8, 0.9, 1.0 -> first index reaching 0.80 is index 1 (2 components)
    assert n == 2


def test_select_n_components_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        select_n_components(np.array([]))


def test_profile_likelihood_rejects_fewer_than_three_values():
    with pytest.raises(ValueError, match="at least 3"):
        select_n_components(np.array([0.6, 0.4]), method="profile_likelihood")


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        select_n_components(np.array([0.5, 0.3, 0.2]), method="bogus")
