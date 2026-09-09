#!/usr/bin/env python3
"""Assert label-free real-MIMIC inference through the existing MRGL classifier."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from adapters.mimic_manifest import MimicManifestDataset
from adapters.semantic_graph import TableVGroupSemanticAdapter
from adapters.nih_dataset import NIH_LABELS
from models.mrgl_classifier import MRGLClassifier


def assert_finite_tree(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), 'Non-finite tensor in MRGL pipeline'
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bbox-cache', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--result-path', type=Path, default=ROOT / 'outputs/mimic_mrgl_inference.json')
    args = parser.parse_args()
    torch.manual_seed(0)
    dataset = MimicManifestDataset(args.manifest, args.bbox_cache)
    model = MRGLClassifier(num_classes=14, pretrained_backbone=True).to(args.device).eval()
    semantic = TableVGroupSemanticAdapter()
    results = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            assert 'target' not in batch, 'Inference must not fabricate labels'
            image, boxes, valid = (batch[k].to(args.device) for k in ('image', 'bbox', 'bbox_valid'))
            adjacency = semantic(len(image), valid, image.device)
            output, details = model(image, boxes, adjacency, bbox_valid=valid, return_details=True)
            assert_finite_tree({'image': image, 'bbox': boxes, 'output': output, **details})
            assert not output.requires_grad
            assert valid.any(dim=1).all(), 'No usable anatomical nodes for an image'
            assert (details['node_features'][~valid] == 0).all(), 'Invalid nodes must remain masked'
            assert output.shape == (len(image), 14)
            assert details['node_features'].shape == (len(image), 26, 2048)
            assert details['feature_map'].shape == (len(image), 2048, 8, 8)
            assert (output >= 0).all() and (output <= 1).all()
            for key in ('spatial_adj', 'semantic_adj', 'implicit_adj'):
                assert details[key].shape == (len(image), 26, 26)
            for i, dicom in enumerate(batch['dicom_id']):
                results.append({'dicom_id': dicom, 'study_id': batch['study_id'][i],
                                'subject_id': batch['subject_id'][i], 'local_image_path': batch['local_image_path'][i],
                                'valid_nodes': int(valid[i].sum()), 'placeholder_image': False,
                                'output_shape': list(output[i:i+1].shape), 'output': output[i].cpu().tolist(),
                                'all_tensors_finite': True, 'no_grad': True})
    summary = {'images': len(results), 'all_checks_passed': True, 'results': results,
               'model': 'ImageNet ResNet-50 + randomly initialized MRGL graph/head',
               'head_dimension': len(NIH_LABELS), 'head_taxonomy': list(NIH_LABELS),
               'semantic_graph': 'TableV group fallback, NOT paper exact',
               'training_performed': False, 'labels_available': False,
               'manifest': str(args.manifest), 'bbox_cache': str(args.bbox_cache)}
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(summary, indent=2) + '\n')
    print('MIMIC REAL IMAGE MRGL INFERENCE OK', len(results), args.result_path)


if __name__ == '__main__':
    main()
