#!/usr/bin/env python3
"""Fetch the GoiEner Spanish smart-meter dataset for evaluation, not yet wired into any chapter.

Exploratory only: this is checking whether GoiEner is a viable third
generalization-check population (alongside AusNet and NREL ResStock), per
the author's own request, not yet a committed data source for this book.

Real, published dataset: Quesada, Astigarraga, Merveille, Borges (2024),
"An electricity smart meter dataset of Spanish households: insights into
consumption patterns," Scientific Data 11:59, doi.org/10.1038/s41597-023-02846-0.
Hosted on Zenodo, record 7362094 (confirmed via the record's own API,
zenodo.org/api/records/7362094), CC-BY-4.0.

Real, checked characteristics (from the paper, not assumed):
- 25,559 anonymized customers total; this script fetches `imp-post.tzst`,
  the most recent, already-imputed snapshot (17,519 series, period after
  2021-05-30), rather than the raw or earlier-period archives, since it is
  both the largest already-cleaned slice and the most representative of
  current metering conditions.
- Hourly resolution (not this book's usual 30-minute AusNet/NREL
  convention, a real mismatch worth resolving before any chapter use, not
  glossed over).
- `.tzst` = a Zstandard-compressed tar archive: one CSV per customer,
  columns `timestamp, kWh, imputed` (a 0/1 flag), no header row.
- `metadata.csv` carries `self_consumption_type` (flags customers with a
  real PV/self-generation component) and `province` (heavily concentrated
  in Basque Country/Navarre, ~86% of customers, confirmed in the paper,
  not broad Spain-wide climate diversity despite nominal province coverage).

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/goiener/data/{imp-post.tzst,metadata.csv}

Usage:
    uv run python scripts/fetch_goiener_data.py
    uv run python scripts/fetch_goiener_data.py --force   # re-download existing files
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "goiener" / "data"

ZENODO_BASE = "https://zenodo.org/api/records/7362094/files"

# Sizes confirmed directly from the Zenodo record's own API
# (zenodo.org/api/records/7362094), not assumed from the paper's own
# rounded figures.
FILES: dict[str, int] = {
    "imp-post.tzst": 508_168_212,
    "metadata.csv": 5_579_767,
}


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
            "Zenodo record 7362094 may have changed; check by hand."
        )


def fetch(force: bool) -> None:
    print(f"GoiEner smart meters data (Zenodo 7362094) -> {DATA_ROOT.relative_to(REPO_ROOT)}/")
    for filename, expected_size in FILES.items():
        dest = DATA_ROOT / filename
        if dest.exists() and not force:
            print(f"  skip  {filename} (already present, use --force to re-download)")
            continue
        url = f"{ZENODO_BASE}/{filename}/content"
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
