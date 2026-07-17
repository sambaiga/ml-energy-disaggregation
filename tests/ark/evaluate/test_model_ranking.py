import pandas as pd

from ark.evaluate.ranking import rank_models


def test_rank_models_prefers_correlation_over_a_narrow_mae_win():
    # Part 5 Chapter 1's own real baseline numbers: window_average has the
    # lowest MAE/RMSE/WMAPE of the four, but the second-worst Corr, a rolling
    # average that minimizes pointwise error by damping the response rather
    # than tracking the real signal. seasonal_naive's MAE is narrowly worse
    # but its Corr is more than 3x higher, the best of the four.
    scores = pd.DataFrame(
        {
            "mae": [0.3595, 0.2947, 0.3701, 0.2889],
            "rmse": [0.5905, 0.5237, 0.6084, 0.4360],
            "corr": [0.0769, 0.2578, 0.0755, 0.0727],
            "wmape": [79.0624, 65.6525, 81.4465, 63.3271],
            "smape": [63.5501, 51.9019, 66.8983, 60.4267],
        },
        index=["naive", "seasonal_naive", "drift", "window_average"],
    )

    ranked = rank_models(scores)

    assert ranked.index[0] == "seasonal_naive"
    assert ranked.attrs["metric_weights"]["corr"] > ranked.attrs["metric_weights"]["mae"]


def test_rank_models_weights_sum_to_one():
    scores = pd.DataFrame({"mae": [1.0, 2.0, 3.0], "corr": [0.9, 0.5, 0.1]}, index=["a", "b", "c"])

    ranked = rank_models(scores)

    assert sum(ranked.attrs["metric_weights"].values()) == 1.0


def test_rank_models_respects_custom_lower_is_better():
    # Flip the convention: treat "corr" as lower-is-better instead of the default.
    scores = pd.DataFrame({"corr": [0.9, 0.1]}, index=["a", "b"])

    ranked = rank_models(scores, lower_is_better={"corr"})

    assert ranked.index[0] == "b"


def test_rank_models_higher_borda_count_ranks_first():
    scores = pd.DataFrame(
        {"mae": [1.0, 2.0, 3.0], "corr": [0.9, 0.6, 0.3]},
        index=["best", "middle", "worst"],
    )

    ranked = rank_models(scores)

    assert list(ranked.index) == ["best", "middle", "worst"]
    assert ranked.loc["best", "borda_count"] > ranked.loc["worst", "borda_count"]
