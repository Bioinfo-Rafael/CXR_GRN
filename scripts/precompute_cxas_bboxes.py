#!/usr/bin/env python3
"""Precompute CXAS masks -> bounding rectangles for NIH images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from adapters.cxas_bbox_cache import ANATOMICAL_REGIONS, precompute_cxas_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA index")
    args = parser.parse_args()

    selected = None
    if args.split_file:
        selected = {
            line.strip() for line in Path(args.split_file).read_text().splitlines() if line.strip()
        }
    with Path(args.csv).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("NIH CSV is empty")
    name_key = "Image Index" if "Image Index" in rows[0] else "filename"
    names = [row[name_key] for row in rows if selected is None or row[name_key] in selected]
    if args.limit is not None:
        names = names[: args.limit]

    image_root = Path(args.image_dir)
    direct = {name: image_root / name for name in names}
    missing = [name for name, path in direct.items() if not path.is_file()]
    if missing:
        index = {path.name: path for path in image_root.rglob("*") if path.is_file()}
        direct.update({name: index[name] for name in missing if name in index})
    unresolved = [name for name in names if not direct[name].is_file()]
    if unresolved:
        raise FileNotFoundError(f"Could not find NIH images: {unresolved[:5]}")

    output = precompute_cxas_cache([direct[name] for name in names], args.output, args.device)
    counts = {}
    for mapping in ANATOMICAL_REGIONS:
        counts[mapping.status] = counts.get(mapping.status, 0) + 1
    print("cache", output)
    print("images", len(names))
    print("mapping_status", counts)


if __name__ == "__main__":
    main()
