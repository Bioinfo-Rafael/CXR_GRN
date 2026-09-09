#!/usr/bin/env python3
"""Reuse CXR_LLM image bytes via symlinks; never copy or create labels."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.mimic_manifest import load_mimic_manifest, image_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, default=ROOT / 'data/mimic_smoke/images')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/mimic_smoke/manifest.jsonl')
    args = parser.parse_args()
    rows = load_mimic_manifest(args.source_manifest)
    plans = []
    for row in rows:
        destination = args.image_dir.absolute() / (row['dicom_id'] + '.jpg')
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or image_sha256(destination) != row['sha256']:
                raise FileExistsError(f'Refusing to overwrite: {destination}')
        plans.append((row, destination))
    args.image_dir.mkdir(parents=True, exist_ok=True)
    for row, destination in plans:
        source = Path(row['local_image_path']).resolve()
        if not destination.exists():
            destination.symlink_to(source)
        row.update(local_image_path=str(destination), source_image_path=str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({'images': len(rows), 'symlinks': sum(p.is_symlink() for _, p in plans),
                      'copied_image_bytes': 0, 'manifest': str(args.output)}))


if __name__ == '__main__':
    main()
