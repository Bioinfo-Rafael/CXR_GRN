"""Minimal NIH ChestX-ray14 multi-label dataset adapter."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from .cxas_bbox_cache import CXASBBoxCache


NIH_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

_LABEL_ALIASES = {
    "Pleural Thickening": "Pleural_Thickening",
    "Pleural thickening": "Pleural_Thickening",
}


class NIHChestXray14Dataset(Dataset):
    """Return image, 14-way target, cached 26 boxes, and a box-valid mask."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        bbox_cache: str | Path,
        split_file: Optional[str | Path] = None,
        image_size: int = 256,
        limit: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.cache = CXASBBoxCache(bbox_cache)
        selected = None
        if split_file is not None:
            selected = {
                line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()
            }

        with self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise ValueError(f"No rows found in {self.csv_path}")

        image_key = self._choose_key(rows[0], ("Image Index", "image", "filename"))
        label_key = self._choose_key(rows[0], ("Finding Labels", "labels"))
        self.records = []
        for row in rows:
            name = row[image_key]
            if selected is not None and name not in selected:
                continue
            if name not in self.cache:
                continue
            self.records.append((name, row[label_key]))
            if limit is not None and len(self.records) >= limit:
                break
        if not self.records:
            raise ValueError("No CSV rows matched both the split and bbox cache")

        self._image_index: Optional[Dict[str, Path]] = None

    @staticmethod
    def _choose_key(row: Dict[str, str], choices: Sequence[str]) -> str:
        for key in choices:
            if key in row:
                return key
        raise KeyError(f"Expected one of {choices}, found {tuple(row)}")

    def _resolve_image(self, name: str) -> Path:
        direct = self.image_dir / name
        if direct.is_file():
            return direct
        if self._image_index is None:
            self._image_index = {
                path.name: path for path in self.image_dir.rglob("*") if path.is_file()
            }
        try:
            return self._image_index[Path(name).name]
        except KeyError as exc:
            raise FileNotFoundError(f"Image {name} not found below {self.image_dir}") from exc

    @staticmethod
    def _target(label_text: str) -> torch.Tensor:
        active = {
            _LABEL_ALIASES.get(label.strip(), label.strip())
            for label in label_text.split("|")
            if label.strip() and label.strip() != "No Finding"
        }
        return torch.tensor([float(label in active) for label in NIH_LABELS], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        image_name, labels = self.records[index]
        image = Image.open(self._resolve_image(image_name)).convert("RGB")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        boxes_norm, valid = self.cache.get(image_name)
        boxes = torch.from_numpy(boxes_norm) * float(self.image_size)
        valid_tensor = torch.from_numpy(valid)
        # ROIAlign requires positive-area boxes even though invalid node features
        # are subsequently zeroed using bbox_valid.
        boxes[~valid_tensor] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes[:, 2:] = torch.maximum(boxes[:, 2:], boxes[:, :2] + 1.0)
        boxes.clamp_(0.0, float(self.image_size))

        return {
            "image": image_tensor,
            "target": self._target(labels),
            "bbox": boxes,
            "bbox_valid": valid_tensor,
            "image_name": image_name,
        }


def make_nih_loader(
    csv_path: str | Path,
    image_dir: str | Path,
    bbox_cache: str | Path,
    split_file: Optional[str | Path] = None,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0,
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    dataset = NIHChestXray14Dataset(
        csv_path=csv_path,
        image_dir=image_dir,
        bbox_cache=bbox_cache,
        split_file=split_file,
        image_size=image_size,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
