#!/usr/bin/env python3
"""Exploratory look at the UK PV dataset: a handful of real household PV generation profiles.

Not part of any book chapter. Same purpose as the GoiEner/London/Norway
scripts: see what the real data actually looks like before deciding
whether it is worth adopting as a real, per-household PV complement.

Real, checked schema (from the dataset's own README, confirmed directly):
- metadata.csv: ss_id, start_datetime_GMT, end_datetime_GMT,
  latitude_rounded, longitude_rounded, orientation, tilt, kWp.
- {year}-{month}.parquet (this project's own local filename for the
  upstream `30_minutely/year=YYYY/month=MM/data.parquet` partition):
  ss_id, datetime_GMT, generation_Wh, the *cumulative* energy generated in
  that half hour. Average power in kW = generation_Wh * 2 / 1000.

Usage:
    uv run python scripts/fetch_uk_pv_data.py   # first, needs HF_TOKEN
    uv run python scripts/explore_uk_pv_data.py
    uv run python scripts/explore_uk_pv_data.py --n-systems 8 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "resources" / "uk-pv" / "data"
METADATA = DATA_DIR / "metadata.csv"
OUT_PNG = REPO_ROOT / "resources" / "uk-pv" / "sample-profiles.png"

MIN_KWP = 1.0  # exclude implausibly tiny/likely-mislabelled systems


def find_month_parquet() -> Path:
    parquets = sorted(DATA_DIR.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no *.parquet found under {DATA_DIR}; run scripts/fetch_uk_pv_data.py first")
    return parquets[0]


def pick_systems(meta: pd.DataFrame, df: pd.DataFrame, n_systems: int, seed: int) -> list[int]:
    """Sample real systems with a plausible nominal capacity and real data in this month's own partition."""
    present = set(df["ss_id"].unique())
    candidates = meta[(meta["kWp"] >= MIN_KWP) & (meta["ss_id"].isin(present))]
    return candidates["ss_id"].sample(n=n_systems, random_state=seed).tolist()


def plot_sample_week(df: pd.DataFrame, meta: pd.DataFrame, ss_ids: list[int], out_path: Path) -> None:
    """One real week per system, the first full Monday-Sunday span each system's own month has."""
    fig, axes = plt.subplots(len(ss_ids), 1, figsize=(9, 2.2 * len(ss_ids)))
    if len(ss_ids) == 1:
        axes = [axes]
    for ax, ss_id in zip(axes, ss_ids, strict=True):
        series = df[df["ss_id"] == ss_id].set_index("datetime_GMT").sort_index()
        mondays = series.index[series.index.weekday == 0]
        first_monday = mondays[0] if len(mondays) else series.index[0]
        week = series.loc[first_monday : first_monday + pd.Timedelta(days=7)]
        kw = week["generation_Wh"] * 2 / 1000
        kwp = meta.loc[meta["ss_id"] == ss_id, "kWp"].iloc[0]
        ax.plot(week.index, kw, color="#EA580C", linewidth=1.2)
        ax.set_title(f"system {ss_id}  (kWp={kwp:.2f}, n={len(series):,} half-hourly readings)", fontsize=9, loc="left")
        ax.set_ylabel("kW")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path.relative_to(REPO_ROOT)}")


def main(n_systems: int, seed: int) -> None:
    if not METADATA.exists():
        raise SystemExit(f"{METADATA} not found; run scripts/fetch_uk_pv_data.py first")
    meta = pd.read_csv(METADATA)
    month_file = find_month_parquet()
    df = pd.read_parquet(month_file)
    print(f"loaded {month_file.name}: {len(df):,} rows, {df['ss_id'].nunique()} systems")
    ss_ids = pick_systems(meta, df, n_systems, seed)
    print(f"sampled {len(ss_ids)} systems: {ss_ids}")
    plot_sample_week(df, meta, ss_ids, OUT_PNG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-systems", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_systems=args.n_systems, seed=args.seed)
