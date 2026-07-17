#!/usr/bin/env python3
"""Sample real ResStock + ComStock buildings from NREL's public OEDI Data Lake.

A generalization check for Part 5: does clustering and forecasting behavior
found on AusNet's 342 real Australian households hold up on a very different,
much more diverse real building stock? Complements AusNet, does not replace
it: ResStock/ComStock are physics-based building energy *simulations*
calibrated to real end-use survey data, not metered readings the way AusNet
is, and a US building on US climate/tariff assumptions besides. Honest about
that distinction wherever this data gets used.

Stratified by real building subtype, not flat random: subtype prevalence is
heavily skewed (ResStock's Single-Family Detached is 338K of 550K rows,
ComStock's Hospital is 2,125 of 209K), so a flat random sample of a few
hundred can easily miss rare subtypes entirely. ~40 per subtype gives 200
ResStock (5 subtypes) + 560 ComStock (14 subtypes) = 760 buildings, tied to
how many real subtypes exist rather than a round number picked in advance.

Verified directly against the actual 2024 release on S3 (not assumed from
NREL's own general docs, which describe a different partition scheme for an
older release):
- ResStock: one national metadata file, real per-building rows.
- ComStock: `bldg_id` is a *prototype* model ID that recurs across many
  (state, county) locations with a sampling weight, not a unique building;
  the national metadata file is a per-(bldg_id, state) table, so a sampled
  row is a (bldg_id, state) pair, not bldg_id alone.
- Both datasets use the same timeseries path structure in this release:
  timeseries_individual_buildings/by_state/upgrade=0/state={state}/{bldg_id}-0.parquet
- The real total-electricity column is `out.electricity.total.energy_consumption`,
  confirmed from an actual downloaded file, 15-minute resolution, 35,040 rows/year.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/nrel-buildstock/data/{metadata.csv,load_timeseries_30min.parquet}

Usage:
    uv run python scripts/fetch_part5_nrel_buildstock_data.py
    uv run python scripts/fetch_part5_nrel_buildstock_data.py --n-per-subtype 20 --seed 7
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "nrel-buildstock" / "data"

BUCKET = "oedi-data-lake"
RELEASE_ROOT = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024"
RESSTOCK_DIR = f"{RELEASE_ROOT}/resstock_amy2018_release_2"
COMSTOCK_DIR = f"{RELEASE_ROOT}/comstock_amy2018_release_2"
TARGET_FREQ = "30min"  # matches AusNet's own resolution, not this dataset's native 15-min

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _read_parquet_s3(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buf)
    buf.seek(0)
    return pd.read_parquet(buf, columns=columns)


def _stratified_sample(df: pd.DataFrame, *, n_per_group: int, seed: int) -> pd.DataFrame:
    """Up to `n_per_group` rows per real building_type, not a flat random sample."""
    parts = [group.sample(min(len(group), n_per_group), random_state=seed) for _, group in df.groupby("building_type")]
    return pd.concat(parts, ignore_index=True)


def sample_resstock(n_per_subtype: int, seed: int) -> pd.DataFrame:
    """One row per real residential building, already unique."""
    meta = _read_parquet_s3(
        f"{RESSTOCK_DIR}/metadata/baseline.parquet",
        columns=["bldg_id", "in.state", "in.geometry_building_type_recs"],
    )
    sample = meta.reset_index().rename(columns={"in.geometry_building_type_recs": "building_type", "in.state": "state"})
    return _stratified_sample(sample, n_per_group=n_per_subtype, seed=seed).assign(dataset="resstock")


def sample_comstock(n_per_subtype: int, seed: int) -> pd.DataFrame:
    """One row per (bldg_id, state).

    ComStock's bldg_id is a prototype model simulated in multiple
    representative locations, not a unique building.
    """
    meta = _read_parquet_s3(
        f"{COMSTOCK_DIR}/metadata_and_annual_results_aggregates/national/full/parquet/baseline_agg.parquet",
        columns=["bldg_id", "in.state", "in.comstock_building_type"],
    )
    meta = meta.rename(columns={"in.comstock_building_type": "building_type", "in.state": "state"}).drop_duplicates(
        subset=["bldg_id", "state"]
    )
    return _stratified_sample(meta, n_per_group=n_per_subtype, seed=seed).assign(dataset="comstock")


def timeseries_key(dataset: str, bldg_id: int, state: str) -> str:
    base = RESSTOCK_DIR if dataset == "resstock" else COMSTOCK_DIR
    return f"{base}/timeseries_individual_buildings/by_state/upgrade=0/state={state}/{bldg_id}-0.parquet"


def fetch_hourly_load(dataset: str, bldg_id: int, state: str) -> pd.Series:
    key = timeseries_key(dataset, bldg_id, state)
    df = _read_parquet_s3(key, columns=["timestamp", "out.electricity.total.energy_consumption"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    load = df.set_index("timestamp")["out.electricity.total.energy_consumption"]
    # native resolution is 15-min energy (kWh) per interval; resample to match
    # AusNet's own 30-min resolution, summing since this is energy, not power
    return load.resample(TARGET_FREQ).sum()


def fetch(n_per_subtype: int, seed: int, max_workers: int, force: bool) -> None:
    metadata_path = DATA_ROOT / "metadata.csv"
    timeseries_path = DATA_ROOT / "load_timeseries_30min.parquet"
    if metadata_path.exists() and timeseries_path.exists() and not force:
        print(f"skip: {metadata_path.relative_to(REPO_ROOT)} already present, use --force to re-fetch")
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    res_sample = sample_resstock(n_per_subtype, seed)
    com_sample = sample_comstock(n_per_subtype, seed)
    sampled = pd.concat([res_sample, com_sample], ignore_index=True)
    print(f"sampled {len(sampled)} buildings: {sampled['dataset'].value_counts().to_dict()}")

    loads: dict[tuple[str, int, str], pd.Series] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_hourly_load, row["dataset"], row["bldg_id"], row["state"]): (
                row["dataset"],
                row["bldg_id"],
                row["state"],
            )
            for _, row in sampled.iterrows()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                loads[key] = future.result()
            except Exception as e:  # noqa: BLE001
                failures.append((*key, str(e)))

    print(f"fetched {len(loads)} real load series, {len(failures)} failures")
    if failures:
        print(failures[:5])

    fetched_keys = set(loads.keys())
    sampled = sampled[
        sampled.apply(lambda r: (r["dataset"], r["bldg_id"], r["state"]) in fetched_keys, axis=1)
    ].reset_index(drop=True)
    sampled.to_csv(metadata_path, index=False)

    long_rows = []
    for (dataset, bldg_id, state), series in loads.items():
        frame = series.rename("load_kwh").reset_index().rename(columns={"index": "timestamp"})
        frame["dataset"] = dataset
        frame["bldg_id"] = bldg_id
        frame["state"] = state
        long_rows.append(frame)
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_parquet(timeseries_path, index=False)

    print(
        f"\nDone. Saved {metadata_path.relative_to(REPO_ROOT)} ({len(sampled)} buildings) and "
        f"{timeseries_path.relative_to(REPO_ROOT)} ({len(long_df):,} rows), gitignored, not committed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-subtype", type=int, default=40, help="Real buildings sampled per building subtype.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--max-workers", type=int, default=12, help="Concurrent S3 downloads.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if local files already exist.")
    args = parser.parse_args()
    fetch(args.n_per_subtype, args.seed, args.max_workers, args.force)
