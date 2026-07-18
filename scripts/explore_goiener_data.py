#!/usr/bin/env python3
"""Exploratory look at the GoiEner dataset: a handful of real household load profiles.

Not part of any book chapter. Answers one question directly: what does
GoiEner's own real data actually look like, before deciding whether it is
worth adopting as a third generalization-check population (alongside
AusNet and NREL ResStock).

Streams a small sample of customer CSVs directly out of the compressed
`imp-post.tzst` archive (a `tarfile` member iterator over a `zstandard`
decompression stream) rather than extracting the full ~4GB archive to disk,
since only a handful of profiles are needed for a first look.

Real, checked archive layout (not assumed from the paper alone): each
member is `goi4_pst/imp_csv/{64-hex customer id}.csv`, columns
`timestamp, kWh, imputed`, hourly resolution, header row present.

Requires `zstandard` (not a project dependency, GoiEner is still only
under evaluation): `uv run --with zstandard python scripts/explore_goiener_data.py`

Usage:
    uv run --with zstandard python scripts/explore_goiener_data.py
    uv run --with zstandard python scripts/explore_goiener_data.py --n-customers 8 --seed 7
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import tarfile

import matplotlib.pyplot as plt
import pandas as pd
import zstandard as zstd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "resources" / "goiener" / "data"
ARCHIVE = DATA_DIR / "imp-post.tzst"
METADATA = DATA_DIR / "metadata.csv"
OUT_PNG = REPO_ROOT / "resources" / "goiener" / "sample-profiles.png"


def pick_customer_ids(n_customers: int, seed: int) -> list[str]:
    """Sample real customer ids with low missingness and a full year of data, not the first N rows blindly.

    Also restricted to `end_date >= 2021-05-30`, the real cutoff for the
    `imp-post` archive this script fetches; metadata.csv itself spans all
    three periods (pre/in/post), so an unfiltered sample can pick a
    customer whose own series only exists in a different archive.
    """
    meta = pd.read_csv(METADATA, dtype={"user": str}, parse_dates=["end_date"])
    candidates = meta[
        (meta["missing_samples_pct"] < 1.0) & (meta["length_years"] >= 1.0) & (meta["end_date"] >= "2021-05-30")
    ]
    sample = candidates.sample(n=n_customers, random_state=seed)
    return sample["user"].tolist()


def stream_customer_series(customer_ids: set[str]) -> dict[str, pd.DataFrame]:
    """Iterate the compressed archive once, pulling out only the requested customers' own CSVs."""
    found: dict[str, pd.DataFrame] = {}
    dctx = zstd.ZstdDecompressor()
    with ARCHIVE.open("rb") as fh, dctx.stream_reader(fh) as reader, tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            if stem not in customer_ids:
                continue
            raw = tar.extractfile(member)
            if raw is None:
                continue
            df = pd.read_csv(io.BytesIO(raw.read()), parse_dates=["timestamp"])
            found[stem] = df
            if len(found) == len(customer_ids):
                break
    return found


def plot_sample_week(series: dict[str, pd.DataFrame], out_path: Path) -> None:
    """One real week per customer, the most recent full Monday-Sunday span each series has."""
    fig, axes = plt.subplots(len(series), 1, figsize=(9, 2.2 * len(series)), sharex=False)
    if len(series) == 1:
        axes = [axes]
    for ax, (customer_id, df) in zip(axes, series.items(), strict=True):
        df = df.set_index("timestamp").sort_index()
        last_monday = df.index[df.index.weekday == 0][-1] if (df.index.weekday == 0).any() else df.index[0]
        week = df.loc[last_monday : last_monday + pd.Timedelta(days=7)]
        ax.plot(week.index, week["kWh"], color="#0369A1", linewidth=1.2)
        ax.set_title(f"customer {customer_id[:10]}...  (n={len(df):,} hourly readings)", fontsize=9, loc="left")
        ax.set_ylabel("kWh")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path.relative_to(REPO_ROOT)}")


def main(n_customers: int, seed: int) -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"{ARCHIVE} not found; run scripts/fetch_goiener_data.py first")
    customer_ids = pick_customer_ids(n_customers, seed)
    print(f"sampled {len(customer_ids)} customers: {[c[:10] + '...' for c in customer_ids]}")
    series = stream_customer_series(set(customer_ids))
    print(f"streamed {len(series)}/{len(customer_ids)} matching series out of the compressed archive")
    plot_sample_week(series, OUT_PNG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_customers=args.n_customers, seed=args.seed)
