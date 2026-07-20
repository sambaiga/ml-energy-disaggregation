#!/usr/bin/env python3
"""Fetch the Low Carbon London smart-meter dataset for evaluation, not yet wired into any chapter.

Exploratory only: checking whether Low Carbon London is a viable third
generalization-check population (alongside AusNet and NREL ResStock), per
the author's own request, not yet a committed data source for this book.

Real, published dataset: UK Power Networks' Low Carbon London project,
energy consumption for 5,567 London households, November 2011-February
2014, half-hourly. Hosted directly by the Greater London Authority on the
London Datastore, no registration required, CC-BY-4.0. Confirmed directly
via the Datastore's own API (data.london.gov.uk/api/v2/dataset/
smartmeter-energy-use-data-in-london-households), not assumed from
third-party mirrors (a Kaggle re-upload of this same data also exists but
needs a Kaggle account; this script goes to the primary source instead).

Fetches the pre-partitioned version (168 separate CSVs, ~1M rows each,
same data as the single 10GB file, just split for practical loading), not
the single monolithic archive, since only a handful of household profiles
are needed here.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/london-lcl/data/Partitioned LCL Data.zip

Usage:
    uv run python scripts/fetch_london_lcl_data.py
    uv run python scripts/fetch_london_lcl_data.py --force   # re-download existing file
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "london-lcl" / "data"

# Confirmed directly from data.london.gov.uk/api/v2/dataset/
# smartmeter-energy-use-data-in-london-households, not a guessed URL.
URL = "https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip"
FILENAME = "Partitioned LCL Data.zip"
EXPECTED_SIZE = 795_722_689


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download(url: str, dest: Path, expected_size: int) -> None:
    if not url.startswith("https://data.london.gov.uk/"):
        raise ValueError(f"refusing to fetch non-London-Datastore URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request) as response, dest.open("wb") as f:  # noqa: S310
        total_read = 0
        total_size = int(response.headers.get("Content-Length", expected_size))
        while chunk := response.read(1 << 20):
            f.write(chunk)
            total_read += len(chunk)
            print(f"\r  {_human_size(total_read)} / {_human_size(total_size)}", end="", flush=True)
    print()
    if abs(total_read - expected_size) > max(64, expected_size // 100):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: downloaded {_human_size(total_read)}, expected ~{_human_size(expected_size)}. "
            "The London Datastore record may have changed; check by hand."
        )


def fetch(force: bool) -> None:
    dest = DATA_ROOT / FILENAME
    print(f"Low Carbon London data -> {DATA_ROOT.relative_to(REPO_ROOT)}/")
    if dest.exists() and not force:
        print(f"  skip  {FILENAME} (already present, use --force to re-download)")
    else:
        print(f"  fetch {FILENAME} ({_human_size(EXPECTED_SIZE)})")
        _download(URL, dest, EXPECTED_SIZE)
    print(f"\nDone. Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download the file if it already exists locally.")
    args = parser.parse_args()
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
