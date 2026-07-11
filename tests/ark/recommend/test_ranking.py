import pandas as pd

from ark.recommend.ranking import combined_score, rank_table, rank_with_interval


def test_combined_score_normalizes_each_column_before_combining():
    df = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [100.0, 100.0, 200.0]})

    score = combined_score(df, ["a", "b"], method="max")

    assert score.iloc[0] == 0.0
    assert score.iloc[2] == 1.0


def test_combined_score_mean_averages_the_normalized_columns():
    df = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0]})

    score = combined_score(df, ["a", "b"], method="mean")

    assert list(score.round(3)) == [0.0, 1.0]


def test_combined_score_rejects_unknown_method():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    try:
        combined_score(df, ["a"], method="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown method")


def test_rank_table_orders_by_score_descending_by_default():
    df = pd.DataFrame({"name": ["a", "b", "c"], "score": [0.5, 0.9, 0.1]})

    ranked = rank_table(df, "score")

    assert list(ranked["name"]) == ["b", "a", "c"]
    assert list(ranked["rank"]) == [1, 2, 3]


def test_rank_with_interval_brackets_the_point_rank():
    df = pd.DataFrame({"name": [f"c{i}" for i in range(20)], "score": range(20)})

    ranked = rank_with_interval(df, "score", n_bootstrap=50, random_state=0)

    assert (ranked["rank_lower"] <= ranked["rank"]).all()
    assert (ranked["rank"] <= ranked["rank_upper"]).all()
