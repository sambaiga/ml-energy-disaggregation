#!/usr/bin/env python3
"""Vendor real SMART-DS feeder data Part 4 Chapter 4's feeder-level clustering needs.

Downloads real per-building 15-minute load profiles and OpenDSS network
models for a sample of AUS/P1U substations (Austin, TX region) from NREL's
SMART-DS dataset (CC-BY-4.0, oedi-data-lake.s3.amazonaws.com), a validated
synthetic distribution testbed, real building footprints and real per-
building demand curves, not toy data.

Only used for this chapter's secondary feeder-level clustering section
(Chapter 4's own customer-level core runs entirely on AusNet's already-
vendored real smart-meter pool). AusNet's single 31-customer feeder can't
support feeder-level clustering on its own, and SMART-DS's AUS/P1U region
has 27 substations, each with a handful of LV feeders, real structure to
cluster.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/smart-ds/AUS_P1U/<substation>/<feeder>/

Usage:
    uv run python scripts/fetch_part4_smartds_data.py
    uv run python scripts/fetch_part4_smartds_data.py --force   # re-download existing files
    uv run python scripts/fetch_part4_smartds_data.py --n-substations 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "smart-ds" / "AUS_P1U"

S3_BASE = "https://oedi-data-lake.s3.amazonaws.com"
S3_PREFIX = "SMART-DS/v1.0/2018/AUS/P1U/scenarios/base_timeseries/opendss"

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
    with urllib.request.urlopen(request) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", body)
    feeders = []
    for prefix in prefixes:
        parts = prefix.rstrip("/").split("/")
        name = parts[-1]
        if name.startswith(f"{substation}--"):
            feeders.append(name)
    return sorted(feeders)


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request) as response, dest.open("wb") as f:  # noqa: S310
        total_read = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            total_read += len(chunk)
    if total_read == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: downloaded 0 bytes, SMART-DS's S3 layout may have changed.")
    return total_read


def fetch(force: bool, n_substations: int) -> None:
    substations = SUBSTATIONS[:n_substations]
    total_bytes = 0
    total_files = 0
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
                if dest.exists() and not force:
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

    print(f"\nDone. {total_files} files, {_human_size(total_bytes)} total.")
    print(f"Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist locally.")
    parser.add_argument(
        "--n-substations", type=int, default=8, help="How many substations to fetch (default 8, max 8)."
    )
    args = parser.parse_args()
    try:
        fetch(force=args.force, n_substations=min(args.n_substations, len(SUBSTATIONS)))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
