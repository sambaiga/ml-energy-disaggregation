#!/usr/bin/env python3
"""Fetch the UK PV dataset for evaluation, not yet wired into any chapter.

Exploratory only: checking whether this is a viable real, per-household PV
generation source (kW/kWh, not just population-shared shapes the way
AusNet's own PV component is), per the author's own request.

Real dataset: "UK PV" (Sheffield Solar / Open Climate Fix), over 30,000
real domestic solar PV systems across Great Britain, 2010-2025, 30-minute
cumulative energy generation. Hosted on Hugging Face
(huggingface.co/datasets/openclimatefix/uk_pv), CC-BY-4.0. Confirmed
directly from the dataset's own README and a live API check, not assumed:
no accompanying academic paper exists for this dataset (a real gap against
"supported by publication," disclosed rather than glossed over), and the
repository is *gated*, real access requires a Hugging Face account and a
personal access token, not just a public URL (confirmed via a direct
unauthenticated request returning HTTP 401 GatedRepo).

Real, checked schema (from the dataset's own README):
- metadata.csv: ss_id, start_datetime_GMT, end_datetime_GMT,
  latitude_rounded, longitude_rounded, orientation, tilt, kWp (nominal
  capacity).
- 30_minutely/year=YYYY/month=MM/data.parquet: ss_id, datetime_GMT,
  generation_Wh (multiply by 2 for average power in Watts over that
  half-hour).

Fetches one representative month (June 2023, a real summer month with
strong UK solar output) rather than the full 15-year, 30,000-system
archive, since only a handful of sample profiles are needed here.

Requires a Hugging Face access token with this dataset's own terms
accepted:
    1. Visit https://huggingface.co/datasets/openclimatefix/uk_pv and
       accept the dataset's access terms (auto-approved).
    2. Create a token at https://huggingface.co/settings/tokens.
    3. export HF_TOKEN=<your token>

Target directory (resources/ is entirely gitignored, so nothing here is
committed): resources/uk-pv/data/{metadata.csv,2023-06.parquet}

Usage:
    uv run python scripts/fetch_uk_pv_data.py
    uv run python scripts/fetch_uk_pv_data.py --year 2022 --month 6 --force
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "resources" / "uk-pv" / "data"
REPO_ID = "openclimatefix/uk_pv"


def fetch(year: int, month: int, force: bool) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. openclimatefix/uk_pv is a gated dataset: visit "
            "https://huggingface.co/datasets/openclimatefix/uk_pv to accept its access terms, "
            "create a token at https://huggingface.co/settings/tokens, then `export HF_TOKEN=<token>`."
        )

    from huggingface_hub import hf_hub_download

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    meta_dest = DATA_ROOT / "metadata.csv"
    if meta_dest.exists() and not force:
        print("  skip  metadata.csv (already present, use --force to re-download)")
    else:
        print("  fetch metadata.csv")
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="metadata.csv", token=token)
        shutil.copy(path, meta_dest)

    month_filename = f"30_minutely/year={year}/month={month:02d}/data.parquet"
    month_dest = DATA_ROOT / f"{year}-{month:02d}.parquet"
    if month_dest.exists() and not force:
        print(f"  skip  {month_dest.name} (already present, use --force to re-download)")
    else:
        print(f"  fetch {month_filename}")
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=month_filename, token=token)
        shutil.copy(path, month_dest)

    print(f"\nDone. Vendored data lives under {DATA_ROOT.relative_to(REPO_ROOT)}/ (gitignored, not committed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist locally.")
    args = parser.parse_args()
    try:
        fetch(year=args.year, month=args.month, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
