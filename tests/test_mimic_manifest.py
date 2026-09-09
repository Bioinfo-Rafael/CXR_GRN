"""Synthetic file fixtures test interfaces only; never used as MIMIC training data."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from adapters.mimic_manifest import load_mimic_manifest, image_sha256, MimicManifestDataset

DICOM = '02aa804e-bde0afdd-112c0b34-7bc16630-4e384014'


class MimicManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mrgl-unit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / (DICOM + '.jpg')
        Image.new('RGB', (32, 32), 'white').save(image)
        self.row = {'dicom_id': DICOM, 'study_id': 's50414267', 'subject_id': 'p10000032',
                    'relative_path': f'files/p10/p10000032/s50414267/{DICOM}.jpg',
                    'flat_local_path': str(image), 'sha256': image_sha256(image)}
        self.manifest = self.root / 'manifest.jsonl'
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.row) + '\n')

    def test_interface_without_labels(self):
        row = load_mimic_manifest(self.manifest)[0]
        self.assertEqual(row['dicom_id'], DICOM)
        self.assertNotIn('target', row)

    def test_reject_study_mismatch(self):
        self.row['study_id'] = 's00000000'
        self.write_manifest()
        with self.assertRaises(ValueError):
            load_mimic_manifest(self.manifest)

    def test_reject_checksum_mismatch(self):
        self.row['sha256'] = 'wrong'
        self.write_manifest()
        with self.assertRaises(ValueError):
            load_mimic_manifest(self.manifest)

    def test_reject_missing_image(self):
        self.row['flat_local_path'] = str(self.root / 'missing.jpg')
        self.write_manifest()
        with self.assertRaises(FileNotFoundError):
            load_mimic_manifest(self.manifest)

    def test_reject_duplicate_id(self):
        self.manifest.write_text((json.dumps(self.row) + '\n') * 2)
        with self.assertRaises(ValueError):
            load_mimic_manifest(self.manifest)

    def test_dataset_no_target_invalid_nodes_remain_invalid(self):
        cache = self.root / 'boxes.npz'
        boxes = np.zeros((1, 26, 4), dtype=np.float32)
        valid = np.zeros((1, 26), dtype=bool)
        boxes[0, 0] = [0.1, 0.1, 0.8, 0.8]
        valid[0, 0] = True
        np.savez(cache, image_names=[DICOM + '.jpg'], bboxes_norm=boxes, valid=valid,
                 metadata_json=np.asarray('{}'))
        Path(str(cache) + '.images.json').write_text(json.dumps({DICOM: self.row['sha256']}))
        sample = MimicManifestDataset(self.manifest, cache)[0]
        self.assertEqual(tuple(sample['image'].shape), (3, 256, 256))
        self.assertNotIn('target', sample)
        self.assertEqual(int(sample['bbox_valid'].sum()), 1)
        Path(str(cache) + '.images.json').write_text('{}')
        with self.assertRaises(ValueError):
            MimicManifestDataset(self.manifest, cache)


if __name__ == '__main__':
    unittest.main()
