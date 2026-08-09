"""Shared setup for the two one-batch MRGL entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from adapters.nih_dataset import make_nih_loader
from adapters.semantic_graph import PrecomputedSemanticGraphAdapter, TableVGroupSemanticAdapter
from models.mrgl_classifier import MRGLClassifier


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", required=True, help="NIH Data_Entry_2017.csv")
    parser.add_argument("--image-dir", required=True, help="NIH image root")
    parser.add_argument("--bbox-cache", required=True, help="CXAS .npz cache")
    parser.add_argument("--split-file", help="Optional NIH filename list")
    parser.add_argument("--semantic-adj", help="Optional paper-compatible/precomputed adjacency .npy/.npz")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Avoid downloading ImageNet ResNet-50 weights (architecture is unchanged)",
    )


def prepare(args) -> Tuple[torch.utils.data.DataLoader, MRGLClassifier, object, torch.device]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    loader = make_nih_loader(
        csv_path=args.csv,
        image_dir=args.image_dir,
        bbox_cache=args.bbox_cache,
        split_file=args.split_file,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.batch_size,
        shuffle=False,
    )
    model = MRGLClassifier(
        num_classes=14,
        hidden_dim=256,
        graph_layers=3,
        spatial_weight=0.3,
        semantic_weight=0.4,
        iou_threshold=args.iou_threshold,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)
    semantic_adapter = (
        PrecomputedSemanticGraphAdapter(args.semantic_adj)
        if args.semantic_adj
        else TableVGroupSemanticAdapter()
    )
    return loader, model, semantic_adapter, device


def move_batch(batch, device: torch.device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_one_batch(model, semantic_adapter, batch):
    semantic_adj = semantic_adapter(
        batch_size=batch["image"].shape[0],
        valid=batch["bbox_valid"],
        device=batch["image"].device,
    )
    return model(
        batch["image"],
        batch["bbox"],
        semantic_adj=semantic_adj,
        bbox_valid=batch["bbox_valid"],
        return_details=True,
    )


def print_shapes(batch, output, details, loss=None) -> None:
    print("image.shape", tuple(batch["image"].shape))
    print("target.shape", tuple(batch["target"].shape))
    print("bbox.shape", tuple(batch["bbox"].shape))
    print("bbox_valid_per_image", batch["bbox_valid"].sum(dim=1).tolist())
    print("node_features.shape", tuple(details["node_features"].shape))
    print("spatial_adj.shape", tuple(details["spatial_adj"].shape))
    print("semantic_adj.shape", tuple(details["semantic_adj"].shape))
    print("implicit_adj.shape", tuple(details["implicit_adj"].shape))
    print("output.shape", tuple(output.shape))
    if loss is not None:
        print("loss", loss.item())
