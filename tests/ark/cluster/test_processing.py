import pytest

from ark.cluster.processing import cluster_dictionary


def test_cluster_dictionary_groups_indices_by_label():
    result = cluster_dictionary([0, 1, 2, 3], [0, 1, 0, 1])

    assert result == {0: [0, 2], 1: [1, 3]}


def test_cluster_dictionary_length_mismatch_raises_valueerror():
    with pytest.raises(ValueError, match="Length mismatch"):
        cluster_dictionary([0, 1, 2], [0, 1])


def test_cluster_dictionary_preserves_label_type():
    # string labels stay strings, int labels stay ints, per the docstring's
    # own stated guarantee about preserving the original label type.
    result = cluster_dictionary(["a", "b", "c"], ["cold", "hot", "cold"])

    assert set(result.keys()) == {"cold", "hot"}
    assert all(isinstance(k, str) for k in result)
