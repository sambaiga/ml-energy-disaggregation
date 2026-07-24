"""Real household load-shape loaders for AusNet, London, and GoiEner.

Each function reads one utility's own already-vendored real data
(`resources/*`, gitignored, fetched locally by its own `scripts/fetch_*`
script). Two levels are exposed per utility:

- `load_*_pivot`: the raw meter time series, real households as columns,
  a real (or, for AusNet, a synthetic but consistently-spaced) datetime
  index as rows. This is what `ark.cluster.feature.extract_features`
  itself needs, the same shape `london-cluster.ipynb`'s own `pivot`
  variable already is.
- `load_*_shape`: a peak-normalised, whole-year-mean daily demand matrix,
  real households as rows, timesteps as columns, built by averaging the
  pivot above. This is what Chapter 3's own AusNet analysis clusters on
  directly, with no dimension-reduction step.

Promoted here once so a chapter comparing feature or clustering choices
*across* utilities loads each the same way, not three subtly different
ways, each with its own copy of the archive-reading code.

Deliberately returns a `pd.DataFrame`, not a bare `np.ndarray`: the
household ID is real, checkable information (which real customer
clustered where), not something to discard at the loading boundary.
"""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd

from ark.cluster.preprocessing import normalize_shape

__all__ = [
    "load_ausnet_pivot",
    "load_ausnet_shape",
    "load_london_pivot",
    "load_london_shape",
    "load_goiener_pivot",
    "load_goiener_shape",
]

_RESOURCES = Path(__file__).resolve().parents[2] / "resources"


def load_ausnet_pivot(window_days: int = 360, synthetic_start: str = "2020-01-01") -> pd.DataFrame:
    """Load AusNet's 342 real households as a raw, half-hourly meter pivot table.

    The vendored `.npy` stores real half-hourly readings but no real
    calendar dates, only a `(household, day, half-hour)` array; a synthetic,
    consistently-spaced `DatetimeIndex` is assigned so `extract_features`'s
    calendar-dependent feature sets (`calendar_peak`, `profile_seasonal`)
    can run, exactly as Chapter 3's own AusNet cells already implicitly
    treat day-index 0 as a real season boundary without ever storing an
    actual date. The synthetic year itself carries no meaning beyond
    ordering; only relative calendar position within it does.

    Args:
        window_days: Number of real calendar days from the start of the
            year included.
        synthetic_start: Arbitrary start date for the assigned index.

    Returns:
        DataFrame indexed by a synthetic half-hourly `DatetimeIndex`,
        columns `0..341` (real household IDs from the vendored array).

    Raises:
        SystemExit: if the real data has not been fetched locally yet
            (`scripts/fetch_part4_ausnet_data.py`).
    """
    path = _RESOURCES / "cvar_flexibility" / "data" / "timeseries-lv" / "Residential load data 30-min resolution.npy"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/fetch_part4_ausnet_data.py first")

    load_data = np.load(path)[:, 0:window_days, :]
    n_households = load_data.shape[0]
    flat = load_data.reshape(n_households, -1).T.astype(np.float64)  # (timesteps, households)
    index = pd.date_range(synthetic_start, periods=flat.shape[0], freq="30min")
    return pd.DataFrame(flat, index=index, columns=pd.RangeIndex(n_households, name="household"))


def load_ausnet_shape(window_days: int = 360) -> pd.DataFrame:
    """Load AusNet's 342 real households as a peak-normalised whole-year-mean shape matrix.

    Averages across the full real window (360 days, all four real calendar
    quarters), not a single arbitrary 90-day slice: Chapter 3's own
    cross-quarter stability check found archetypes drift meaningfully
    across a single year (ARI as low as 0.06), so any one quarter is a
    noisier, less representative read on a household's own typical
    behaviour than the whole real year averaged together. This also
    sidesteps a real bug the single-season convention had: AusNet is
    Southern Hemisphere, so its first 90 real calendar days are summer,
    while the same slice on London/GoiEner's own Northern Hemisphere data
    is winter, silently comparing different real seasons across utilities
    under the same "season" label.

    Args:
        window_days: Number of real calendar days from the start of the
            year averaged into one whole-year-mean daily shape.

    Returns:
        `(342, 48)` DataFrame, households indexed `0..341`, columns the
        half-hour of day.

    Raises:
        SystemExit: if the real data has not been fetched locally yet
            (`scripts/fetch_part4_ausnet_data.py`).
    """
    path = _RESOURCES / "cvar_flexibility" / "data" / "timeseries-lv" / "Residential load data 30-min resolution.npy"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/fetch_part4_ausnet_data.py first")

    load_data = np.load(path)
    window = load_data[:, 0:window_days, :]
    shape = normalize_shape(window.mean(axis=1))
    return pd.DataFrame(shape, index=pd.RangeIndex(shape.shape[0], name="household"))


def load_london_pivot(
    n_partitions: int = 50,
    window_start: str = "2013-01-01",
    window_days: int = 360,
    min_coverage: float = 0.999,
) -> pd.DataFrame:
    """Load real Low Carbon London households as a raw, half-hourly meter pivot table.

    Defaults match `london-cluster.ipynb`'s own settled-recipe population,
    1,284 real households after coverage filtering.

    Args:
        n_partitions: Number of the archive's 168 real CSV partitions to
            load.
        window_start: Start of the real, continuous window kept.
        window_days: Length of that window (360 days, matching the
            AusNet/GoiEner convention).
        min_coverage: Minimum fraction of real half-hourly readings a
            household must have across the whole window to be kept.

    Returns:
        DataFrame indexed by a real half-hourly `DatetimeIndex`, columns
        real `LCLid` values.

    Raises:
        SystemExit: if the real data has not been fetched locally yet
            (`scripts/fetch_london_lcl_data.py`).
    """
    archive = _RESOURCES / "london-lcl" / "data" / "Partitioned LCL Data.zip"
    if not archive.exists():
        raise SystemExit(f"{archive} not found; run scripts/fetch_london_lcl_data.py first")

    with zipfile.ZipFile(archive) as z:
        partitions = sorted(z.namelist())[:n_partitions]
        frames = []
        for name in partitions:
            with z.open(name) as f:
                df = pd.read_csv(f, parse_dates=["DateTime"])
            df.columns = [c.strip() for c in df.columns]
            df = df[["LCLid", "DateTime", "KWH/hh (per half hour)"]].rename(
                columns={"KWH/hh (per half hour)": "kwh_hh"}
            )
            df["kwh_hh"] = pd.to_numeric(df["kwh_hh"], errors="coerce")
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    start = pd.Timestamp(window_start)
    end = start + pd.Timedelta(days=window_days)
    year = raw[(raw["DateTime"] >= start) & (raw["DateTime"] < end)]
    pivot = year.pivot_table(index="DateTime", columns="LCLid", values="kwh_hh", aggfunc="first")
    full_index = pd.date_range(start, end, freq="30min", inclusive="left")
    pivot = pivot.reindex(full_index)

    coverage = pivot.notna().mean(axis=0)
    complete_ids = coverage[coverage >= min_coverage].index
    pivot = pivot[complete_ids].interpolate(method="linear", limit_direction="both").astype(np.float64)
    pivot.columns.name = "household"
    return pivot


def load_london_shape(
    n_partitions: int = 50,
    window_start: str = "2013-01-01",
    window_days: int = 360,
    min_coverage: float = 0.999,
) -> pd.DataFrame:
    """Load real Low Carbon London households as a peak-normalised whole-year-mean shape matrix.

    Defaults (`n_partitions=50`) match `london-cluster.ipynb`'s own
    settled-recipe population, 1,284 real households after coverage
    filtering, not the smaller ~295-household sample the exploratory
    `04-customer-feeder-clustering-london-code.ipynb` notebook uses for a
    lighter check; anything auditing that search's own cached trials needs
    this same population, not a different-sized one.

    Averages across the full real window (360 days, all four real calendar
    quarters), not a single arbitrary 90-day slice, for the same reason
    `load_ausnet_shape` does: one quarter is a noisier read on a real
    household's own typical behaviour than the whole year, and a single
    hemisphere-blind slice would not mean the same real season on
    London's own Northern Hemisphere data as it does on AusNet's Southern
    Hemisphere data.

    Args:
        n_partitions: Number of the archive's 168 real CSV partitions to
            load.
        window_start: Start of the real, continuous window kept.
        window_days: Length of that window (360 days, 4 real 90-day
            quarters, matching the AusNet/GoiEner convention).
        min_coverage: Minimum fraction of real half-hourly readings a
            household must have across the whole window to be kept.

    Returns:
        DataFrame indexed by real `LCLid`, columns the half-hour of day.

    Raises:
        SystemExit: if the real data has not been fetched locally yet
            (`scripts/fetch_london_lcl_data.py`).
    """
    pivot = load_london_pivot(n_partitions, window_start, window_days, min_coverage)
    load_data = pivot.T.to_numpy().reshape(pivot.shape[1], window_days, 48)
    shape = normalize_shape(load_data.mean(axis=1))
    return pd.DataFrame(shape, index=pd.Index(pivot.columns, name="household"))


def _goiener_target_ids(n_households: int, window_start: str, window_days: int, random_state: int) -> set[str]:
    """Pick a real, stratified GoiEner household sample from `metadata.csv`."""
    metadata = _RESOURCES / "goiener" / "data" / "metadata.csv"
    meta = pd.read_csv(metadata, dtype={"user": str}, parse_dates=["start_date", "end_date"])
    start_utc = pd.Timestamp(window_start, tz="UTC")
    end_utc = start_utc + pd.Timedelta(days=window_days)
    candidates = meta[
        (meta["missing_samples_pct"] < 1.0)
        & (meta["length_years"] >= 1.0)
        & (meta["start_date"] <= start_utc)
        & (meta["end_date"] >= end_utc)
    ].copy()
    candidates["is_residential"] = candidates["cnae"] == 9820.0
    frac = min(1.0, n_households / len(candidates))
    return set(
        candidates.groupby("is_residential", group_keys=False)[["user"]].apply(
            lambda g: g.sample(frac=frac, random_state=random_state)
        )["user"]
    )


def load_goiener_pivot(
    n_households: int = 800,
    window_start: str = "2021-06-06",
    window_days: int = 360,
    min_coverage: float = 0.99,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load real GoiEner households as a raw, hourly meter pivot table.

    GoiEner is real hourly data, not half-hourly like AusNet/London.

    Args:
        n_households: Target real household sample size, stratified by
            residential/commercial status (CNAE 9820 marks a household).
        window_start: Start of the real, continuous window kept.
        window_days: Length of that window (360 days, matching the
            AusNet/London convention).
        min_coverage: Minimum fraction of real hourly readings a household
            must have across the whole window to be kept.
        random_state: Seed for the stratified household sample.

    Returns:
        DataFrame indexed by a real hourly `DatetimeIndex`, columns real
        household IDs.

    Raises:
        SystemExit: if the real data has not been fetched locally yet
            (`scripts/fetch_goiener_data.py`).
    """
    import zstandard as zstd

    archive = _RESOURCES / "goiener" / "data" / "imp-post.tzst"
    if not archive.exists():
        raise SystemExit(f"{archive} not found; run scripts/fetch_goiener_data.py first")

    start = pd.Timestamp(window_start)
    end = start + pd.Timedelta(days=window_days)
    target_ids = _goiener_target_ids(n_households, window_start, window_days, random_state)

    found: dict[str, bytes] = {}
    dctx = zstd.ZstdDecompressor()
    with archive.open("rb") as fh, dctx.stream_reader(fh) as reader, tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            if stem not in target_ids:
                continue
            raw = tar.extractfile(member)
            if raw is None:
                continue
            found[stem] = raw.read()
            if len(found) == len(target_ids):
                break

    full_index = pd.date_range(start, end, freq="1h", inclusive="left")
    series: dict[str, np.ndarray] = {}
    for uid, raw_bytes in found.items():
        df = pd.read_csv(io.BytesIO(raw_bytes), parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        win = df.reindex(full_index)
        if win["kWh"].notna().mean() >= min_coverage:
            series[uid] = win["kWh"].interpolate(method="linear", limit_direction="both").to_numpy()

    pivot = pd.DataFrame(series, index=full_index).astype(np.float64)
    pivot.columns.name = "household"
    return pivot


def load_goiener_shape(
    n_households: int = 800,
    window_start: str = "2021-06-06",
    window_days: int = 360,
    min_coverage: float = 0.99,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load real GoiEner households as a peak-normalised whole-year-mean shape matrix.

    GoiEner is real hourly data, not half-hourly like AusNet/London, so the
    returned shape has 24 columns per day, not 48; comparable to the other
    two utilities only as a daily pattern, not at matching time resolution.
    `n_households=800` matches `london-cluster.ipynb`'s own settled-recipe
    cross-dataset check population, not the larger 2,000-household sample
    the exploratory GoiEner notebook uses for its own separate purposes.

    Averages across the full real window (360 days, all four real calendar
    quarters), not a single arbitrary 90-day slice, for the same reason
    `load_ausnet_shape` does.

    Args:
        n_households: Target real household sample size, stratified by
            residential/commercial status (CNAE 9820 marks a household).
        window_start: Start of the real, continuous window kept.
        window_days: Length of that window (360 days, matching the
            AusNet/London convention).
        min_coverage: Minimum fraction of real hourly readings a household
            must have across the whole window to be kept.
        random_state: Seed for the stratified household sample.

    Returns:
        DataFrame indexed by real household ID, columns the hour of day.

    Raises:
        SystemExit: if the real data has not been fetched locally yet
            (`scripts/fetch_goiener_data.py`).
    """
    pivot = load_goiener_pivot(n_households, window_start, window_days, min_coverage, random_state)
    steps_per_day = 24
    load_data = pivot.T.to_numpy().reshape(pivot.shape[1], window_days, steps_per_day)
    shape = normalize_shape(load_data.mean(axis=1))
    return pd.DataFrame(shape, index=pd.Index(pivot.columns, name="household"))
