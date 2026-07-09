#!/usr/bin/env python3
"""Vendor the real AusNet LV network + smart-meter data Part 4 Chapter 2 needs.

Downloads the network model and 30-minute smart-meter data from Team-Nando's
BSD-3-Clause DER hosting-capacity tutorials (github.com/Team-Nando), the same
342-customer, 31-node AusNet feeder the author's own local notebooks
(resources/cvar_flexibility/notebook/Oendss_timeseries.ipynb,
Opendss_der-model.ipynb) already build on via a hardcoded Windows path.
Vendoring it here makes those notebooks runnable on any machine.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/cvar_flexibility/data/{timeseries-lv,voltwatt-lv}/

Usage:
    uv run python scripts/fetch_part4_ausnet_data.py
    uv run python scripts/fetch_part4_ausnet_data.py --force   # re-download existing files
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "cvar_flexibility" / "data"

RAW_BASE = "https://raw.githubusercontent.com/Team-Nando"

# Source: Tutorial-DERHostingCapacity-2-TimeSeries_LV (Part 2 of Team-Nando's
# series). Canonical source for the network model and smart-meter data: the
# same files are duplicated under Part 3's TestLVCircuit/ subfolder, so only
# one copy is fetched. Sizes noted from the GitHub contents API so a partial
# download is easy to spot.
TIMESERIES_LV_FILES: dict[str, int] = {
    "LVcircuit-linecodes.txt": 246,
    "LVcircuit-lines.txt": 3457,
    "LVcircuit-loads.txt": 4153,
    "LVcircuit-master.txt": 264,
    "LVcircuit-servicelines.txt": 3660,
    "LVcircuit-transformers.txt": 152,
    "LVcircuit-topology.png": 259_920,
    "Residential PV data 30-min resolution.npy": 140_240,
    "Residential load data 30-min resolution.npy": 47_934_800,
    "Residential_PV_profiles_Autumn.npy": 1280,
    "Residential_PV_profiles_Spring.npy": 1280,
    "Residential_PV_profiles_Summer.npy": 1280,
    "Residential_PV_profiles_Winter.npy": 1280,
    "Tutorial-DERHC-2.ipynb": 37_339,
    "LICENSE": 1540,
}

# Source: Tutorial-DERHostingCapacity-3-VoltWatt_LV. Only the reference
# notebook and figures, for cross-checking Chapter 2's Volt-Watt/Volt-VAr
# section against Team-Nando's own worked example; the network/meter data
# it uses is identical to TIMESERIES_LV_FILES above and not re-fetched.
VOLTWATT_LV_FILES: dict[str, int] = {
    "Tutorial-DERHC-3.ipynb": 90_923,
    "Figure_1.png": 17_438,
    "Figure_2.jpg": 37_562,
    "Figure_3.png": 259_920,
    "LICENSE": 1540,
}

SOURCES = {
    "timeseries-lv": ("Tutorial-DERHostingCapacity-2-TimeSeries_LV", TIMESERIES_LV_FILES),
    "voltwatt-lv": ("Tutorial-DERHostingCapacity-3-VoltWatt_LV", VOLTWATT_LV_FILES),
}


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download(url: str, dest: Path, expected_size: int) -> None:
    if not url.startswith("https://raw.githubusercontent.com/"):
        raise ValueError(f"refusing to fetch non-GitHub-raw URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request) as response, dest.open("wb") as f:  # noqa: S310
        total_read = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            total_read += len(chunk)
    if abs(total_read - expected_size) > max(64, expected_size // 100):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: downloaded {_human_size(total_read)}, expected ~{_human_size(expected_size)}. "
            "Team-Nando's repo may have changed; check the URL by hand."
        )


def fetch(force: bool) -> None:
    for subdir, (repo, files) in SOURCES.items():
        target_dir = DATA_ROOT / subdir
        print(f"\n{repo} -> {target_dir.relative_to(REPO_ROOT)}/")
        for filename, expected_size in files.items():
            dest = target_dir / filename
            if dest.exists() and not force:
                print(f"  skip  {filename} (already present, use --force to re-download)")
                continue
            url = f"{RAW_BASE}/{repo}/main/{urllib.parse.quote(filename)}"
            print(f"  fetch {filename} ({_human_size(expected_size)})")
            _download(url, dest, expected_size)

    print(f"\nDone. Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist locally.")
    args = parser.parse_args()
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
