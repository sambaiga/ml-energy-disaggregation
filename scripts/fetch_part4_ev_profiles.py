#!/usr/bin/env python3
"""Vendor real EV charging profiles for Part 4 Chapter 2's EV demand section.

Downloads diversified electric-vehicle home-charging profiles from
Team-Nando's `EV-Demand-Profiles` repo (github.com/Team-Nando), 1-minute,
24-hour profiles derived by the University of Melbourne from the UK's real
"Electric Nation" trial, for Level 1 (3.68kW) and Level 2 (7.36kW) chargers,
weekday and weekend, diversified across 100/1,200 EVs. Only the "with a
Daily Plug-in Factor (70%)" set is fetched, the README's own recommended
default (not every EV charges every day).

Unlike the other Team-Nando repos this project vendors from, this one ships
no LICENSE file (confirmed via the GitHub API: `license: None`), so this
data is used as a reference for building this book's own, adapted EV
charging scenario, not redistributed or reproduced verbatim in any
committed output.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/cvar_flexibility/data/ev-profiles/

Usage:
    uv run python scripts/fetch_part4_ev_profiles.py
    uv run python scripts/fetch_part4_ev_profiles.py --force   # re-download existing files
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "cvar_flexibility" / "data" / "ev-profiles"

RAW_BASE = "https://raw.githubusercontent.com/Team-Nando/EV-Demand-Profiles/main"
SOURCE_FOLDER = "Diversified EV Profiles/with a Daily Plug-in Factor (70%)/npy files (python)"

# All twelve npy files are ~11.4KB (1,440 one-minute points, float64), so a
# single approximate size threshold catches a truncated download without
# hardcoding twelve near-identical numbers.
EXPECTED_SIZE = 11_648
FILES = [
    "Weekday - Level 1 - Diversified (Median) of 100 EVs.npy",
    "Weekday - Level 1 - Diversified (75th percentile) of 100 EVs.npy",
    "Weekday - Level 1 - Diversified (Average) of 1,200 EVs.npy",
    "Weekday - Level 2 - Diversified (Median) of 100 EVs.npy",
    "Weekday - Level 2 - Diversified (75th percentile) of 100 EVs.npy",
    "Weekday - Level 2 - Diversified (Average) of 1,200 EVs.npy",
    "Weekend - Level 1 - Diversified (Median) of 100 EVs.npy",
    "Weekend - Level 1 - Diversified (75th percentile) of 100 EVs.npy",
    "Weekend - Level 1 - Diversified (Average) of 1,200 EVs.npy",
    "Weekend - Level 2 - Diversified (Median) of 100 EVs.npy",
    "Weekend - Level 2 - Diversified (75th percentile) of 100 EVs.npy",
    "Weekend - Level 2 - Diversified (Average) of 1,200 EVs.npy",
]


def _human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download(url: str, dest: Path) -> None:
    if not url.startswith("https://raw.githubusercontent.com/"):
        raise ValueError(f"refusing to fetch non-GitHub-raw URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request) as response, dest.open("wb") as f:  # noqa: S310
        total_read = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            total_read += len(chunk)
    if abs(total_read - EXPECTED_SIZE) > max(64, EXPECTED_SIZE // 5):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: downloaded {_human_size(total_read)}, expected ~{_human_size(EXPECTED_SIZE)}. "
            "Team-Nando's repo may have changed; check the URL by hand."
        )


def fetch(force: bool) -> None:
    print(f"Team-Nando/EV-Demand-Profiles -> {DATA_ROOT.relative_to(REPO_ROOT)}/")
    folder_path = "/".join(urllib.parse.quote(segment) for segment in SOURCE_FOLDER.split("/"))
    for filename in FILES:
        dest = DATA_ROOT / filename
        if dest.exists() and not force:
            print(f"  skip  {filename} (already present, use --force to re-download)")
            continue
        url = f"{RAW_BASE}/{folder_path}/{urllib.parse.quote(filename)}"
        print(f"  fetch {filename}")
        _download(url, dest)

    print(f"\nDone. Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")
    print("No LICENSE ships with this source repo: use as reference for a custom scenario, not verbatim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist locally.")
    args = parser.parse_args()
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
