#!/usr/bin/env python3
"""Reuse existing CXAS 26-node mapping with label-free MIMIC manifest inputs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.mimic_manifest import load_mimic_manifest
from adapters.cxas_bbox_cache import precompute_cxas_cache, CXASBBoxCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    rows = load_mimic_manifest(args.manifest)
    cache_path = precompute_cxas_cache([r['local_image_path'] for r in rows], args.output, args.device)
    cache = CXASBBoxCache(cache_path)
    Path(str(cache_path) + '.images.json').write_text(
        json.dumps({r['dicom_id']: r['sha256'] for r in rows}, indent=2) + '\n')
    for row in rows:
        boxes, valid = cache.get(row['dicom_id'] + '.jpg')
        print(json.dumps({'dicom_id': row['dicom_id'], 'valid_nodes': int(valid.sum()),
                          'bbox_shape': list(boxes.shape), 'coordinate_format': 'normalized_xyxy'}))
    print('MIMIC CXAS CACHE OK', cache_path)


if __name__ == '__main__':
    main()
