#!/usr/bin/env python3
"""Fetch the Norway residential EV charging dataset for evaluation, not yet wired into any chapter.

Exploratory only: checking whether this is a viable real, per-household EV
charging source (real energy charged in kWh per session, from actual
residential locations, not public charging infrastructure), per the
author's own request.

Real, published dataset: 35,000 charging sessions from 267 users across 12
residential locations in Norway, published as a data article in *Data in
Brief* (2024), hosted on Zenodo (record 12730566, confirmed directly via
the record's own API, not a guessed URL). No access gate: openly
downloadable, no registration.

Real, checked file layout (from a direct read of the first rows, not
assumed from the paper's own description alone): `Dataset1_charging_
reports.csv`, semicolon-delimited, columns `location, user_id, session_id,
plugin_time, plugout_time, connection_time, energy_session` (the real
energy charged, in kWh, comma-decimal formatted). Three further files
(`Dataset2_user_predictions.csv`, `Dataset3_session_predictions.csv`,
`Dataset4_hourly_predictions.csv`) hold the paper's own derived/predicted
data, not fetched here since only the real charging reports are needed for
a first look.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/norway-ev/data/Dataset1_charging_reports.csv

Usage:
    uv run python scripts/fetch_norway_ev_data.py
    uv run python scripts/fetch_norway_ev_data.py --force   # re-download existing file
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "norway-ev" / "data"

# Confirmed directly via zenodo.org/api/records/12730566, not a guessed URL.
ZENODO_BASE = "https://zenodo.org/api/records/12730566/files"
FILENAME = "Dataset1_charging_reports.csv"
EXPECTED_SIZE = 3_105_240


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download(url: str, dest: Path, expected_size: int) -> None:
    if not url.startswith("https://zenodo.org/"):
        raise ValueError(f"refusing to fetch non-Zenodo URL: {url}")
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
            "Zenodo record 12730566 may have changed; check by hand."
        )


def fetch(force: bool) -> None:
    dest = DATA_ROOT / FILENAME
    print(f"Norway EV charging data (Zenodo 12730566) -> {DATA_ROOT.relative_to(REPO_ROOT)}/")
    if dest.exists() and not force:
        print(f"  skip  {FILENAME} (already present, use --force to re-download)")
    else:
        url = f"{ZENODO_BASE}/{FILENAME}/content"
        print(f"  fetch {FILENAME} ({_human_size(EXPECTED_SIZE)})")
        _download(url, dest, EXPECTED_SIZE)
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
