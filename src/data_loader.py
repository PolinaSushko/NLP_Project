#!/usr/bin/env python3
"""
Utility to fetch a CSV dataset and copy/save it to  data/raw/<file_name>.csv

Examples
--------
# Local file → data/raw/imdb_reviews.csv
python data_load.py --source ./downloads/imdb_reviews.csv --file_name imdb_reviews

# Remote URL → data/raw/reviews.csv
python data_load.py \
       --source https://example.com/datasets/reviews.csv \
       --file_name reviews
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw")


def _download(url: str, dest: Path) -> None:
    """Stream-download a remote file to *dest*."""
    CHUNK = 1 << 16  # 64 KB
    try:
        with requests.get(url, stream=True, timeout=15) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:  # filter out keep-alive chunks
                        fh.write(chunk)
    except requests.exceptions.RequestException as exc:
        sys.exit(f"[ERROR] Failed to download {url} → {exc}")


def _copy_local(src: Path, dest: Path) -> None:
    """Copy a local file to *dest* (overwrite if exists)."""
    try:
        shutil.copyfile(src, dest)
    except OSError as exc:
        sys.exit(f"[ERROR] Copy failed: {exc}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Fetch/Copy a CSV dataset to data/raw/")
    p.add_argument(
        "--source",
        required=True,
        help="Local path or HTTP(S) URL of the CSV dataset",
    )
    p.add_argument(
        "--file_name",
        required=True,
        help="Target file name (without .csv) inside data/raw/",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DIR / f"{args.file_name}.csv"

    # Decide local copy vs. download
    parsed = urlparse(args.source)
    if parsed.scheme in ("http", "https"):
        print(f"⬇️  Downloading {args.source} → {dest_path}")
        _download(args.source, dest_path)
    else:
        src_path = Path(args.source).expanduser().resolve()
        if not src_path.is_file():
            sys.exit(f"[ERROR] File not found: {src_path}")
        print(f"📄 Copying {src_path} → {dest_path}")
        _copy_local(src_path, dest_path)

    print(f"✅ Dataset saved: {dest_path.absolute()}")


if __name__ == "__main__":
    main()
