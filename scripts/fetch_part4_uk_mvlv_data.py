#!/usr/bin/env python3
"""Vendor real UK MV-LV network data Part 4 Chapter 5's ranking/recommendation work needs.

Downloads the `HV_UG_full` hybrid MV-LV network model from Deakin et al.'s
`uk-mvlv-models` (github.com/deakinmt/uk-mvlv-models), built by allocating
real LVNS (Electricity North West's low-carbon-networks-fund trial) LV
circuits onto a real UKGDS medium-voltage backbone [@deakin2021hybridmvlv].
Verified directly before adding this script: the network solves cleanly in
OpenDSS (112,887 buses, 19,072 loads, voltage range 0.966-1.082 pu, no
structural defect), real `.dss` format, no MATLAB or other extra toolchain
needed. The repository has no license file (checked directly: none of the
common filenames resolve, and the GitHub API reports `license: null`); its
README only asks that use be cited, the citation above.

Every load in this model is a static placeholder (`kW=1` or `kW=3`), not a
real time-varying reading, the same "needs a real load shape assigned"
state AusNet's own network was in before Chapter 1 assigned it real
smart-meter readings. This chapter borrows AusNet's own real customer
shapes onto this network's real bus positions, the same reuse pattern.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/uk-mvlv/HV_UG_full/

Usage:
    uv run python scripts/fetch_part4_uk_mvlv_data.py
    uv run python scripts/fetch_part4_uk_mvlv_data.py --force   # re-download and re-extract
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "uk-mvlv"

RAW_URL = "https://raw.githubusercontent.com/deakinmt/uk-mvlv-models/master/HV_UG_full.zip"
EXPECTED_SIZE = 9_257_503


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response, dest.open("wb") as f:  # noqa: S310
        total_read = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            total_read += len(chunk)
    if abs(total_read - EXPECTED_SIZE) > max(64, EXPECTED_SIZE // 100):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name}: downloaded {_human_size(total_read)}, expected ~{_human_size(EXPECTED_SIZE)}. "
            "The uk-mvlv-models repo may have changed; check the URL by hand."
        )
    return total_read


def fetch(force: bool) -> None:
    zip_dest = DATA_ROOT / "HV_UG_full.zip"
    extract_dir = DATA_ROOT / "HV_UG_full"

    if extract_dir.exists() and not force:
        print(f"skip  HV_UG_full/ (already present at {extract_dir.relative_to(REPO_ROOT)}, use --force to re-fetch)")
        return

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"fetch HV_UG_full.zip ({_human_size(EXPECTED_SIZE)}) from deakinmt/uk-mvlv-models")
    _download(RAW_URL, zip_dest)

    print(f"extract -> {extract_dir.relative_to(REPO_ROOT)}/")
    with zipfile.ZipFile(zip_dest) as archive:
        archive.extractall(DATA_ROOT)
    zip_dest.unlink()

    master = extract_dir / "master_mvlv.dss"
    if not master.exists():
        raise RuntimeError(
            f"extraction did not produce {master.relative_to(REPO_ROOT)}; archive layout may have changed."
        )

    print(f"\nDone. Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")
    print("No license file ships with this repository; its README asks only that use be cited (Deakin et al. 2021).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract even if already present.")
    args = parser.parse_args()
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
