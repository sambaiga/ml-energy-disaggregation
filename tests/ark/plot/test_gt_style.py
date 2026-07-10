from great_tables import GT
import pandas as pd

from ark.plot.gt_style import themed_gt


def test_themed_gt_accepts_a_gt_object():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    table = themed_gt(GT(df), n_rows=len(df))

    assert isinstance(table, GT)


def test_themed_gt_accepts_a_dataframe_directly():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    table = themed_gt(df)

    assert isinstance(table, GT)


def test_themed_gt_infers_n_rows_from_the_dataframe():
    df = pd.DataFrame({"a": range(5)})

    striped = themed_gt(df)
    unstriped = themed_gt(df, striped=False)

    assert isinstance(striped, GT)
    assert isinstance(unstriped, GT)
