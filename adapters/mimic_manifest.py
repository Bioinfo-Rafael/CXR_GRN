"""Label-free MIMIC interface to CXR_LLM's manual-image manifest."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from .cxas_bbox_cache import CXASBBoxCache


def image_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_mimic_manifest(path):
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        relative = source.get('relative_path') or source['physionet_relative_path']
        parts = PurePosixPath(relative).parts
        dicom = source['dicom_id']
        if (not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{8}){4}', dicom)
                or len(parts) != 5 or parts[0] != 'files'
                or not re.fullmatch(r'p\d{8}', parts[2])
                or not re.fullmatch(r's\d{8}', parts[3])
                or parts[1] != parts[2][:3] or parts[4] != dicom + '.jpg'
                or source['study_id'] != parts[3]
                or source.get('subject_id', parts[2]) != parts[2]):
            raise ValueError('Manifest dicom/study/subject/relative_path mismatch')
        if dicom in seen:
            raise ValueError('Duplicate dicom_id in image manifest')
        seen.add(dicom)
        local = Path(source.get('local_image_path') or source['flat_local_path']).absolute()
        if not local.is_file():
            raise FileNotFoundError(local)
        with Image.open(local) as image:
            if image.format != 'JPEG':
                raise ValueError('MIMIC manifest input must be real JPEG')
            image.verify()
        digest = image_sha256(local)
        if source.get('sha256') and source['sha256'] != digest:
            raise ValueError(f'Image checksum mismatch: {dicom}')
        rows.append({'dicom_id': dicom, 'study_id': parts[3], 'subject_id': parts[2],
                     'relative_path': relative, 'local_image_path': str(local), 'sha256': digest})
    if not rows:
        raise ValueError('Image manifest is empty')
    return rows


class MimicManifestDataset(Dataset):
    """Inference only: deliberately provides NO target or fabricated labels."""
    def __init__(self, manifest, bbox_cache, image_size=256):
        self.records = load_mimic_manifest(manifest)
        self.image_size = image_size
        self.cache = CXASBBoxCache(bbox_cache)
        # Wrapper sidecar binds cached anatomy to the exact image bytes.
        sidecar = Path(str(bbox_cache) + '.images.json')
        fingerprints = json.loads(sidecar.read_text())
        for row in self.records:
            if fingerprints.get(row['dicom_id']) != row['sha256']:
                raise ValueError('Stale/missing bbox provenance: rerun MIMIC CXAS precompute')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row['local_image_path']) as source:
            source.load()
            image = source.convert('RGB')
        image = TF.resize(image, [self.image_size, self.image_size],
                          interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        boxes_norm, valid = self.cache.get(row['dicom_id'] + '.jpg')
        if (boxes_norm.shape != (26, 4) or valid.shape != (26,)
                or not np.isfinite(boxes_norm).all()
                or np.any(boxes_norm < 0) or np.any(boxes_norm > 1)
                or np.any(boxes_norm[valid, 2:] <= boxes_norm[valid, :2])):
            raise ValueError('Invalid normalized CXAS bbox cache')
        boxes = torch.from_numpy(boxes_norm) * float(self.image_size)
        valid_tensor = torch.from_numpy(valid)
        # Computational ROI sentinel only. Validity remains false, features masked.
        boxes[~valid_tensor] = torch.tensor([0., 0., 1., 1.])
        boxes[:, 2:] = torch.maximum(boxes[:, 2:], boxes[:, :2] + 1.)
        boxes.clamp_(0, float(self.image_size))
        return {**row, 'image': tensor, 'bbox': boxes, 'bbox_valid': valid_tensor}
