#!/usr/bin/env python3
"""Vendor real SMART-DS feeder data Part 4 Chapter 4's feeder-level clustering needs.

Downloads OpenDSS network models plus a real, kW-stratified sample of
per-building 15-minute load profiles for a sample of AUS/P1U substations
(Austin, TX region) from NREL's SMART-DS dataset (CC-BY-4.0,
oedi-data-lake.s3.amazonaws.com), a validated synthetic distribution
testbed, real building footprints and real per-building demand curves,
not toy data.

Each feeder's own `Loads.dss` references 200-350 unique profile ids, real
data but drawn from one shared, region-wide pool (`.../AUS/P1U/profiles/`),
not stored per-feeder; fetching every one across ~30 feeders would mean
thousands of individual downloads (~690KB each) for a section of the
chapter that only needs each feeder's real aggregate *shape*, not its
full customer population. Sampling a real subset per feeder, spread across
the kW range so small and large customers are both represented, keeps
this tractable while still building a genuinely real feeder-level curve,
not a synthetic one.

Only used for this chapter's secondary feeder-level clustering section
(Chapter 4's own customer-level core runs entirely on AusNet's already-
vendored real smart-meter pool). AusNet's single 31-customer feeder can't
support feeder-level clustering on its own, and SMART-DS's AUS/P1U region
has 27 substations, each with a handful of LV feeders, real structure to
cluster.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/smart-ds/AUS_P1U/<substation>/<feeder>/ for network
models, resources/smart-ds/profiles/ for the shared, deduplicated sample
of real profile CSVs.

Usage:
    uv run python scripts/fetch_part4_smartds_data.py
    uv run python scripts/fetch_part4_smartds_data.py --force   # re-download existing files
    uv run python scripts/fetch_part4_smartds_data.py --n-substations 4
    uv run python scripts/fetch_part4_smartds_data.py --n-profiles-per-feeder 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "smart-ds" / "AUS_P1U"
PROFILES_ROOT = REPO_ROOT / "resources" / "smart-ds" / "profiles"

S3_BASE = "https://oedi-data-lake.s3.amazonaws.com"
S3_PREFIX = "SMART-DS/v1.0/2018/AUS/P1U/scenarios/base_timeseries/opendss"
S3_PROFILES_PREFIX = "SMART-DS/v1.0/2018/AUS/P1U/profiles"

# A fixed, real substation sample, checked directly against the S3 listing
# (not the full 27, a manageable slice: ~30 feeders total across 8
# substations). Substation IDs are stable, real SMART-DS naming.
SUBSTATIONS = [
    "p1uhs0_1247",
    "p1uhs10_1247",
    "p1uhs11_1247",
    "p1uhs12_1247",
    "p1uhs13_1247",
    "p1uhs14_1247",
    "p1uhs15_1247",
    "p1uhs16_1247",
]

# Only the files a feeder-level clustering exercise actually needs: the
# network model and its real load profiles. Skips CYME/geojson exports,
# Buscoords (only needed for plotting), and Intermediates.txt (a large
# NREL-internal scratch file, not needed to solve the circuit).
FEEDER_FILES = ["LineCodes.dss", "Lines.dss", "LoadShapes.dss", "Loads.dss", "Master.dss", "Transformers.dss"]


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _list_feeders(substation: str) -> list[str]:
    """List real feeder subfolder names for a substation via the S3 REST API."""
    url = f"{S3_BASE}/?list-type=2&prefix={S3_PREFIX}/{substation}/&delimiter=/"
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", body)
    feeders = []
    for prefix in prefixes:
        parts = prefix.rstrip("/").split("/")
        name = parts[-1]
        if name.startswith(f"{substation}--"):
            feeders.append(name)
    return sorted(feeders)


def _is_valid(dest: Path) -> bool:
    """A previously-downloaded file is only worth skipping if it's non-empty.

    A process killed mid-download (this happened once, during development:
    an earlier run hung and was force-killed) can leave a 0-byte file
    behind; existence alone isn't enough to trust it.
    """
    return dest.exists() and dest.stat().st_size > 0


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response, dest.open("wb") as f:  # noqa: S310
        total_read = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            total_read += len(chunk)
    if total_read == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: downloaded 0 bytes, SMART-DS's S3 layout may have changed.")
    return total_read


def _feeder_load_profiles(loads_dss: Path) -> list[tuple[str, float, str]]:
    """Parse a feeder's `Loads.dss` into `(load_name, kW, profile_id)` triples."""
    text = loads_dss.read_text()
    triples = []
    for line in text.splitlines():
        name_match = re.search(r"New Load\.(\S+)", line)
        kw_match = re.search(r"kW=([\d.]+)", line)
        profile_match = re.search(r"yearly=(\S+)", line)
        if name_match and kw_match and profile_match:
            triples.append((name_match.group(1), float(kw_match.group(1)), profile_match.group(1)))
    return triples


def _sample_profiles(triples: list[tuple[str, float, str]], n_samples: int, seed: int) -> list[str]:
    """Pick a real, kW-stratified sample of profile ids, not just the first N.

    Sorts by each load's own kW rating and takes evenly spaced picks across
    that range, so a small sample still spans small and large customers
    instead of whatever happened to be listed first in `Loads.dss`.
    """
    unique_by_id: dict[str, float] = {}
    for _name, kw, profile_id in triples:
        unique_by_id.setdefault(profile_id, kw)
    ranked = sorted(unique_by_id.items(), key=lambda item: item[1])
    if len(ranked) <= n_samples:
        return [profile_id for profile_id, _kw in ranked]
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security-sensitive
    indices = sorted(rng.sample(range(len(ranked)), n_samples))
    return [ranked[i][0] for i in indices]


def fetch(force: bool, n_substations: int, n_profiles_per_feeder: int) -> None:
    substations = SUBSTATIONS[:n_substations]
    total_bytes = 0
    total_files = 0
    profile_ids_needed: set[str] = set()

    for substation in substations:
        print(f"\n{substation}:")
        try:
            feeders = _list_feeders(substation)
        except OSError as exc:
            print(f"  error listing feeders: {exc}", file=sys.stderr)
            continue
        print(f"  {len(feeders)} feeders found")
        for feeder in feeders:
            for filename in FEEDER_FILES:
                dest = DATA_ROOT / substation / feeder / filename
                if _is_valid(dest) and not force:
                    print(f"  skip  {feeder}/{filename} (already present)")
                    continue
                url = f"{S3_BASE}/{S3_PREFIX}/{substation}/{feeder}/{filename}"
                try:
                    size = _download(url, dest)
                except (OSError, RuntimeError) as exc:
                    # OSError covers urllib.error.URLError and transient
                    # connection failures (BrokenPipeError, ConnectionReset,
                    # ...): one flaky download should skip that file and
                    # keep going, not crash the whole fetch run.
                    print(f"  error {feeder}/{filename}: {exc}", file=sys.stderr)
                    dest.unlink(missing_ok=True)
                    continue
                total_bytes += size
                total_files += 1
                print(f"  fetch {feeder}/{filename} ({_human_size(size)})")

            loads_dss = DATA_ROOT / substation / feeder / "Loads.dss"
            if loads_dss.exists():
                triples = _feeder_load_profiles(loads_dss)
                sampled = _sample_profiles(triples, n_profiles_per_feeder, seed=hash(feeder) & 0xFFFF)
                profile_ids_needed.update(sampled)

    # Real per-building profile CSVs live in one shared, region-wide pool
    # (each feeder's Loads.dss just references a subset by id), so fetch
    # each id once here rather than per-feeder, even if several feeders
    # reference the same profile.
    print(f"\nprofiles: {len(profile_ids_needed)} unique ids needed across all feeders (kW-stratified sample)")
    for profile_id in sorted(profile_ids_needed):
        # e.g. "res_kw_17316_pu" -> also fetch its "res_kvar_17316_pu" pair.
        kvar_id = profile_id.replace("_kw_", "_kvar_", 1)
        for filename in (f"{profile_id}.csv", f"{kvar_id}.csv"):
            dest = PROFILES_ROOT / filename
            if _is_valid(dest) and not force:
                continue
            url = f"{S3_BASE}/{S3_PROFILES_PREFIX}/{filename}"
            try:
                size = _download(url, dest)
            except (OSError, RuntimeError) as exc:
                print(f"  error {filename}: {exc}", file=sys.stderr)
                dest.unlink(missing_ok=True)
                continue
            total_bytes += size
            total_files += 1
            print(f"  fetch profiles/{filename} ({_human_size(size)})")

    print(f"\nDone. {total_files} files, {_human_size(total_bytes)} total.")
    print(f"Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist locally.")
    parser.add_argument(
        "--n-substations", type=int, default=8, help="How many substations to fetch (default 8, max 8)."
    )
    parser.add_argument(
        "--n-profiles-per-feeder",
        type=int,
        default=12,
        help="Real per-customer profile CSVs to sample per feeder, kW-stratified (default 12). Each feeder's "
        "Loads.dss references 200-350 unique profiles; a real, licensed, but shared regional pool, sampling "
        "instead of fetching every one keeps this from turning into thousands of small downloads.",
    )
    args = parser.parse_args()
    try:
        fetch(
            force=args.force,
            n_substations=min(args.n_substations, len(SUBSTATIONS)),
            n_profiles_per_feeder=args.n_profiles_per_feeder,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
