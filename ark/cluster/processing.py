from collections import defaultdict
from collections.abc import Hashable
import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def cluster_dictionary(
    labels_indices: list[int] | np.ndarray | pd.Series,
    labels: list[Any] | np.ndarray | pd.Series,
) -> dict[Hashable, list[int]]:
    """Creates a dictionary mapping cluster labels to lists of indices.

    Groups the provided indices according to their corresponding cluster labels.

    Args:
        labels_indices: Sequence of indices (usually integers) corresponding
            to each label.
        labels: Sequence of cluster labels. Can be integers, strings, or any
            hashable type.

    Returns:
        A dictionary where:
        - Keys are the unique cluster labels (preserving original type)
        - Values are lists of indices belonging to that cluster

    Raises:
        ValueError: If `labels_indices` and `labels` have different lengths.
    """
    # Convert numpy arrays and pandas Series to lists for consistent handling
    if isinstance(labels_indices, np.ndarray | pd.Series):
        labels_indices = labels_indices.tolist()
    if isinstance(labels, np.ndarray | pd.Series):
        labels = labels.tolist()

    if len(labels_indices) != len(labels):
        raise ValueError(f"Length mismatch: {len(labels_indices)} indices vs {len(labels)} labels")

    cluster: defaultdict[Hashable, list[int]] = defaultdict(list)

    for idx, label in zip(labels_indices, labels, strict=True):
        cluster[label].append(idx)

    # Convert defaultdict to regular dict for cleaner return type
    return dict(cluster)
