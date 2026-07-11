"""Flag ANSI-style voltage violations in a tidy voltage timeseries.

Generalizes the inline `LOW, HIGH` threshold-and-severity pattern Chapter 5's
own real-time alarm-triage section built once, so any chapter with a real
voltage timeseries (one row per bus/customer/step, a `vmag_pu` column) can
flag violations the same way without repeating the comparison and severity
math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flag_violations(
    voltage_df: pd.DataFrame,
    *,
    low: float = 0.94,
    high: float = 1.10,
    value_col: str = "vmag_pu",
) -> pd.DataFrame:
    """Flag rows outside a real voltage compliance band and score how far outside.

    Args:
        voltage_df: A tidy voltage timeseries, one row per (bus/customer,
            step), with a per-unit voltage column.
        low: Lower compliance limit, per unit (default 0.94, the same
            convention Chapter 5's own alarm-triage section used).
        high: Upper compliance limit, per unit (default 1.10, same
            convention).
        value_col: Column holding the per-unit voltage magnitude.

    Returns:
        The subset of `voltage_df` outside `[low, high]`, with a new
        `severity` column: how far past the nearer limit, always positive.

    Examples:
        >>> df = pd.DataFrame({"vmag_pu": [0.95, 1.12, 1.0]})
        >>> flag_violations(df)["severity"].round(3).tolist()
        [0.02]
    """
    violations = voltage_df[(voltage_df[value_col] < low) | (voltage_df[value_col] > high)].copy()
    violations["severity"] = np.where(
        violations[value_col] > high, violations[value_col] - high, low - violations[value_col]
    )
    return violations
