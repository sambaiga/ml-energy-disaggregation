#!/usr/bin/env python3
"""Vendor the Maree et al. Mendeley smart-meter dataset Part 3 needs.

"Energy distribution models with AMI smart meter sensor dataset" (Maree et
al., Mendeley Data, DOI 10.17632/jv3rz8k35r.1, CC BY 4.0): a real Norwegian
DSO network (Lede AS, Porsgrunn) with ~6,809 customers, hourly AMI active/
reactive power from 2022-01-01 to 2024-09-30, plus regional weather and
electricity spot prices and network topology in pandapower/CGMES format.

The dataset has no public, paginated file-listing API (Mendeley Data's own
frontend only exposes the first 100 of 7,372 individual files through its
REST endpoint), so this script uses the same whole-dataset zip Mendeley's
own "Download All" button requests: a redirect through
/public-api/zip/{id}/download/{version} to a presigned, time-limited S3 URL.

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/mendeley-lede-porsgrunn-ami/

Usage:
    uv run python scripts/fetch_part3_mendeley_data.py
    uv run python scripts/fetch_part3_mendeley_data.py --skip-extract   # keep the zip only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "resources" / "mendeley-lede-porsgrunn-ami"

DATASET_ID = "jv3rz8k35r"
DATASET_VERSION = "1"
ZIP_ENDPOINT = f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{DATASET_VERSION}"


def _human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download_zip(dest: Path) -> None:
    request = urllib.request.Request(ZIP_ENDPOINT, headers={"User-Agent": "ml-energy-disaggregation-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(request) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        print(f"downloading {_human_size(total)} -> {dest.relative_to(REPO_ROOT)}")
        written = 0
        last_pct = -1
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                written += len(chunk)
                pct = int(written * 100 / total) if total else 0
                if pct != last_pct and pct % 5 == 0:
                    print(f"  {pct}% ({_human_size(written)}/{_human_size(total)})")
                    last_pct = pct
    print(f"done: {_human_size(written)}")


def fetch(skip_extract: bool) -> None:
    zip_path = DATA_DIR / f"{DATASET_ID}-{DATASET_VERSION}.zip"
    if zip_path.exists():
        print(f"skip download, {zip_path.relative_to(REPO_ROOT)} already present")
    else:
        _download_zip(zip_path)

    if skip_extract:
        return

    extract_dir = DATA_DIR / "extracted"
    if extract_dir.exists():
        print(f"skip extract, {extract_dir.relative_to(REPO_ROOT)} already present")
        return

    print(f"extracting -> {extract_dir.relative_to(REPO_ROOT)}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print("done extracting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-extract", action="store_true", help="Download the zip but don't extract it.")
    args = parser.parse_args()
    try:
        fetch(skip_extract=args.skip_extract)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
