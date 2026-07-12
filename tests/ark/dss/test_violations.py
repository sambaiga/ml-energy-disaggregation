import pandas as pd
import pytest

from ark.dss.violations import flag_violations


def test_flag_violations_keeps_only_out_of_band_rows():
    df = pd.DataFrame({"vmag_pu": [0.90, 0.95, 1.0, 1.05, 1.15]})

    violations = flag_violations(df, low=0.94, high=1.10)

    assert sorted(violations["vmag_pu"].tolist()) == [0.90, 1.15]


def test_flag_violations_severity_measures_distance_past_the_nearer_limit():
    df = pd.DataFrame({"vmag_pu": [0.90, 1.15]})

    violations = flag_violations(df, low=0.94, high=1.10).set_index("vmag_pu")

    assert violations.loc[0.90, "severity"] == pytest.approx(0.04)
    assert violations.loc[1.15, "severity"] == pytest.approx(0.05)


def test_flag_violations_respects_custom_thresholds_and_column_name():
    df = pd.DataFrame({"reading": [0.80, 1.0, 1.30]})

    violations = flag_violations(df, low=0.85, high=1.25, value_col="reading")

    assert sorted(violations["reading"].tolist()) == [0.80, 1.30]


def test_flag_violations_returns_empty_frame_when_nothing_violates():
    df = pd.DataFrame({"vmag_pu": [0.96, 1.0, 1.05]})

    violations = flag_violations(df)

    assert violations.empty
