"""CXAS segmentation-to-bounding-box cache for the paper's 26 nodes.

The mapping below was checked against Table V of the MRGL paper and CXAS
0.0.18's PAX-Ray++ class list.  A missing CXAS class stays missing; it is not
silently replaced by an anatomically different mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AnatomyMapping:
    paper_name: str
    cxas_classes: Tuple[str, ...]
    status: str
    table_v_group: str
    note: str = ""


# Order is identical to anatomical_feature_extract.py and graph_mimic_new.py.
# CXAS names are the exact strings in cxas/data/paxray_labels.json.
ANATOMICAL_REGIONS: Tuple[AnatomyMapping, ...] = (
    AnatomyMapping("Right lung", ("right lung",), "exact", "right_lung"),
    AnatomyMapping("Right upper lung", ("right upper zone lung",), "exact", "right_lung"),
    AnatomyMapping("Right mid lung", ("right mid zone lung",), "exact", "right_lung"),
    AnatomyMapping(
        "Right lower lung", ("right lung base",), "approximation", "right_lung",
        "CXAS has a lung-base zone, not the paper's lower-lung zone.",
    ),
    AnatomyMapping("Hilar of right lung", (), "unavailable", "right_lung"),
    AnatomyMapping("Apical of right lung", ("right apical zone lung",), "exact", "right_lung"),
    AnatomyMapping("Right costophrenic sulcus", (), "unavailable", "right_lung"),
    AnatomyMapping("Right hemidiaphragm", ("right hemidiaphragm",), "exact", "right_lung"),
    AnatomyMapping("Left lung", ("left lung",), "exact", "left_lung"),
    AnatomyMapping("Left upper lung", ("left upper zone lung",), "exact", "left_lung"),
    AnatomyMapping("Left mid lung", ("left mid zone lung",), "exact", "left_lung"),
    AnatomyMapping(
        "Left lower lung", ("left lung base",), "approximation", "left_lung",
        "CXAS has a lung-base zone, not the paper's lower-lung zone.",
    ),
    AnatomyMapping("Hilar of left lung", (), "unavailable", "left_lung"),
    AnatomyMapping("Apical of left lung", ("left apical zone lung",), "exact", "left_lung"),
    AnatomyMapping("Left costophrenic sulcus", (), "unavailable", "left_lung"),
    AnatomyMapping("Left hemidiaphragm", ("left hemidiaphragm",), "exact", "left_lung"),
    AnatomyMapping("Main bronchus", (), "unavailable", "others"),
    AnatomyMapping("Right clavicle", ("clavicle right",), "exact", "others"),
    AnatomyMapping("Left clavicle", ("clavicle left",), "exact", "others"),
    AnatomyMapping("Aortic arch structure", ("aortic arch",), "exact", "others"),
    AnatomyMapping(
        "Mediastinum",
        (
            "cardiomediastinum", "upper mediastinum", "lower mediastinum",
            "anterior mediastinum", "middle mediastinum", "posterior mediastinum",
        ),
        "composed",
        "others",
        "Union of the six CXAS mediastinal masks.",
    ),
    AnatomyMapping("Superior vena cava structure", (), "unavailable", "others"),
    AnatomyMapping("Cardiac", ("heart",), "exact", "cardiac"),
    AnatomyMapping("Cavoatrial", (), "unavailable", "cardiac"),
    AnatomyMapping("Descending aorta", ("descending aorta",), "exact", "cardiac"),
    AnatomyMapping(
        "Structure of carina", ("tracheal bifurcation",), "terminology", "cardiac",
        "The CXAS tracheal bifurcation mask is used for the carina.",
    ),
)

NUM_ANATOMICAL_NODES = len(ANATOMICAL_REGIONS)
assert NUM_ANATOMICAL_NODES == 26


def masks_to_normalized_bboxes(
    masks: np.ndarray,
    class_to_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert CXAS masks [C,H,W] to normalized XYXY boxes and validity."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"Expected CXAS masks [C,H,W], got {masks.shape}")
    _, height, width = masks.shape
    boxes = np.zeros((NUM_ANATOMICAL_NODES, 4), dtype=np.float32)
    valid = np.zeros(NUM_ANATOMICAL_NODES, dtype=bool)

    for node_index, mapping in enumerate(ANATOMICAL_REGIONS):
        if not mapping.cxas_classes:
            continue
        missing = [name for name in mapping.cxas_classes if name not in class_to_index]
        if missing:
            raise KeyError(f"CXAS class list changed; missing {missing}")
        selected = masks[[class_to_index[name] for name in mapping.cxas_classes]]
        union = np.any(selected, axis=0)
        ys, xs = np.nonzero(union)
        if xs.size == 0:
            continue
        # XYXY uses an exclusive maximum, as expected by torchvision ROIAlign.
        boxes[node_index] = (
            xs.min() / width,
            ys.min() / height,
            (xs.max() + 1) / width,
            (ys.max() + 1) / height,
        )
        valid[node_index] = True
    return boxes, valid


class CXASBBoxCache:
    """Read a compressed cache created by :func:`precompute_cxas_cache`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"BBox cache not found: {self.path}. Run scripts/precompute_cxas_bboxes.py first."
            )
        with np.load(self.path, allow_pickle=False) as data:
            names = data["image_names"].astype(str).tolist()
            boxes = data["bboxes_norm"].astype(np.float32)
            valid = data["valid"].astype(bool)
            self.metadata = json.loads(str(data["metadata_json"].item()))
        if boxes.shape != (len(names), NUM_ANATOMICAL_NODES, 4):
            raise ValueError(f"Invalid bbox cache shape: {boxes.shape}")
        self._records = {
            name: (boxes[index], valid[index]) for index, name in enumerate(names)
        }

    def __contains__(self, image_name: str) -> bool:
        return Path(image_name).name in self._records

    def get(self, image_name: str) -> Tuple[np.ndarray, np.ndarray]:
        key = Path(image_name).name
        if key not in self._records:
            raise KeyError(f"{key} is missing from bbox cache {self.path}")
        boxes, valid = self._records[key]
        return boxes.copy(), valid.copy()


def precompute_cxas_cache(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    device: str = "cpu",
) -> Path:
    """Run CXAS once per image and store normalized bounding rectangles."""
    try:
        from cxas import CXAS
        from cxas.label_mapper import category_ids
    except ImportError as exc:
        raise RuntimeError("Install cxas before precomputing anatomy boxes") from exc

    if device != "cpu" and not device.isdigit():
        raise ValueError("CXAS accepts 'cpu' or a CUDA index such as '0'")
    segmentor = CXAS(model_name="UNet_ResNet50_default", gpus=device)
    class_to_index = {str(name): int(index) for name, index in category_ids.items()}

    names = []
    all_boxes = []
    all_valid = []
    for image_path_value in image_paths:
        image_path = Path(image_path_value)
        prediction = segmentor.process_file(str(image_path))
        masks = prediction["segmentation_preds"][0].detach().cpu().numpy()
        boxes, valid = masks_to_normalized_bboxes(masks, class_to_index)
        names.append(image_path.name)
        all_boxes.append(boxes)
        all_valid.append(valid)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "source": "CXAS 0.0.18 UNet_ResNet50_default segmentation_preds",
        "coordinate_format": "normalized_xyxy",
        "mapping": [asdict(mapping) for mapping in ANATOMICAL_REGIONS],
    }
    np.savez_compressed(
        output,
        image_names=np.asarray(names),
        bboxes_norm=np.asarray(all_boxes, dtype=np.float32),
        valid=np.asarray(all_valid, dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return output
