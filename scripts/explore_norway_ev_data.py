#!/usr/bin/env python3
"""Exploratory look at the Norway EV charging dataset: a handful of real household charging histories.

Not part of any book chapter. Same purpose as the GoiEner/London scripts:
see what the real data actually looks like before deciding whether it is
worth adopting as a real, per-household EV-charging complement.

Unlike continuous half-hourly load/PV series, EV charging data is
naturally discrete: one row per real charging session, not a fixed-step
time series, so the "profile" plotted here is each session's own real
energy charged (kWh) over real calendar time, not a continuous curve.

Real, checked file layout (confirmed directly, not assumed from the
paper alone): semicolon-delimited, comma-decimal CSV, columns `location,
user_id, session_id, plugin_time, plugout_time, connection_time,
energy_session` (kWh actually delivered in that session).

Usage:
    uv run python scripts/explore_norway_ev_data.py
    uv run python scripts/explore_norway_ev_data.py --n-users 8 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "resources" / "norway-ev" / "data" / "Dataset1_charging_reports.csv"
OUT_PNG = REPO_ROOT / "resources" / "norway-ev" / "sample-profiles.png"

MIN_SESSIONS = 30  # enough real sessions to see a real charging rhythm, not one or two data points


def load_sessions() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, sep=";", decimal=",", parse_dates=["plugin_time", "plugout_time"])


def pick_users(df: pd.DataFrame, n_users: int, seed: int) -> list[str]:
    """Sample real users with at least MIN_SESSIONS real charging sessions, not the first N rows blindly."""
    counts = df.groupby("user_id").size()
    candidates = counts[counts >= MIN_SESSIONS].index
    return pd.Series(candidates).sample(n=n_users, random_state=seed).tolist()


def plot_sample_users(df: pd.DataFrame, user_ids: list[str], out_path: Path) -> None:
    """Each real charging session's own energy delivered (kWh), plotted against its real plug-in time."""
    fig, axes = plt.subplots(len(user_ids), 1, figsize=(9, 2.2 * len(user_ids)))
    if len(user_ids) == 1:
        axes = [axes]
    for ax, user_id in zip(axes, user_ids, strict=True):
        sessions = df[df["user_id"] == user_id].sort_values("plugin_time")
        ax.stem(sessions["plugin_time"], sessions["energy_session"], basefmt=" ", linefmt="#0369A1", markerfmt=" ")
        ax.set_title(f"user {user_id}  (n={len(sessions)} real charging sessions)", fontsize=9, loc="left")
        ax.set_ylabel("kWh/session")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path.relative_to(REPO_ROOT)}")


def main(n_users: int, seed: int) -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"{DATA_FILE} not found; run scripts/fetch_norway_ev_data.py first")
    df = load_sessions()
    print(f"loaded {len(df):,} real charging sessions, {df['user_id'].nunique()} users")
    user_ids = pick_users(df, n_users, seed)
    print(f"sampled {len(user_ids)} users: {user_ids}")
    plot_sample_users(df, user_ids, OUT_PNG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_users=args.n_users, seed=args.seed)
