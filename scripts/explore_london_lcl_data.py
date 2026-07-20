#!/usr/bin/env python3
"""Exploratory look at the Low Carbon London dataset: a handful of real household load profiles.

Not part of any book chapter. Same purpose as `explore_goiener_data.py`:
see what the real data actually looks like before deciding whether it is
worth adopting as a third generalization-check population.

Reads directly out of the downloaded zip's one partitioned CSV
(`Small LCL Data/LCL-June2015v2_0.csv`, ~1M rows, 30 real households) via
`zipfile`, no full extraction to disk needed since a single partition
already fits comfortably in memory.

Real, checked file layout (not assumed from the dataset description
alone): columns `LCLid, stdorToU, DateTime, KWH/hh (per half hour)`,
half-hourly resolution, no separate per-household metadata file bundled in
this particular archive.

Usage:
    uv run python scripts/explore_london_lcl_data.py
    uv run python scripts/explore_london_lcl_data.py --n-households 8 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "resources" / "london-lcl" / "data"
ARCHIVE = DATA_DIR / "Partitioned LCL Data.zip"
PARTITION = "Small LCL Data/LCL-June2015v2_0.csv"
OUT_PNG = REPO_ROOT / "resources" / "london-lcl" / "sample-profiles.png"

MIN_DAYS = 300  # at least ~10 months of real half-hourly history


def load_partition() -> pd.DataFrame:
    with zipfile.ZipFile(ARCHIVE) as z, z.open(PARTITION) as f:
        df = pd.read_csv(f, parse_dates=["DateTime"])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"KWH/hh (per half hour)": "kwh_hh"})
    df["kwh_hh"] = pd.to_numeric(df["kwh_hh"], errors="coerce")
    return df


def pick_households(df: pd.DataFrame, n_households: int, seed: int) -> list[str]:
    """Sample real household ids with at least MIN_DAYS of real half-hourly history."""
    span_days = df.groupby("LCLid")["DateTime"].agg(lambda s: (s.max() - s.min()).days)
    candidates = span_days[span_days >= MIN_DAYS].index
    return pd.Series(candidates).sample(n=n_households, random_state=seed).tolist()


def plot_sample_week(df: pd.DataFrame, household_ids: list[str], out_path: Path) -> None:
    """One real week per household, the most recent full Monday-Sunday span each series has."""
    fig, axes = plt.subplots(len(household_ids), 1, figsize=(9, 2.2 * len(household_ids)))
    if len(household_ids) == 1:
        axes = [axes]
    for ax, lclid in zip(axes, household_ids, strict=True):
        series = df[df["LCLid"] == lclid].set_index("DateTime").sort_index()
        mondays = series.index[series.index.weekday == 0]
        last_monday = mondays[-1] if len(mondays) else series.index[0]
        week = series.loc[last_monday : last_monday + pd.Timedelta(days=7)]
        ax.plot(week.index, week["kwh_hh"], color="#0369A1", linewidth=1.2)
        ax.set_title(f"household {lclid}  (n={len(series):,} half-hourly readings)", fontsize=9, loc="left")
        ax.set_ylabel("kWh/hh")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path.relative_to(REPO_ROOT)}")


def main(n_households: int, seed: int) -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"{ARCHIVE} not found; run scripts/fetch_london_lcl_data.py first")
    df = load_partition()
    print(f"loaded partition: {len(df):,} rows, {df['LCLid'].nunique()} households")
    household_ids = pick_households(df, n_households, seed)
    print(f"sampled {len(household_ids)} households: {household_ids}")
    plot_sample_week(df, household_ids, OUT_PNG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-households", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_households=args.n_households, seed=args.seed)
