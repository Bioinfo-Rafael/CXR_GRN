"""Single-CXR multi-relationship graph learning classifier.

This module follows Figure 3 and Equations 3, 6, and 7 of the MRGL paper.  It
intentionally does not depend on the two-image/question-conditioned
``ChangeDetector`` in ``models/modules.py``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import roi_align


def spatial_iou_adjacency(
    boxes: torch.Tensor,
    threshold: float = 0.3,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paper Eq. 3: connect anatomical boxes whose IoU reaches ``threshold``."""
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes [B,N,4], got {tuple(boxes.shape)}")
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    area = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    inter_x0 = torch.maximum(x0[:, :, None], x0[:, None, :])
    inter_y0 = torch.maximum(y0[:, :, None], y0[:, None, :])
    inter_x1 = torch.minimum(x1[:, :, None], x1[:, None, :])
    inter_y1 = torch.minimum(y1[:, :, None], y1[:, None, :])
    intersection = (inter_x1 - inter_x0).clamp_min(0) * (inter_y1 - inter_y0).clamp_min(0)
    union = area[:, :, None] + area[:, None, :] - intersection
    iou = intersection / union.clamp_min(torch.finfo(boxes.dtype).eps)
    adjacency = (iou >= threshold).to(boxes.dtype)
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    if valid is not None:
        pair_valid = valid.bool()
        adjacency *= (pair_valid[:, :, None] & pair_valid[:, None, :]).to(adjacency.dtype)
    return adjacency


def implicit_fully_connected_adjacency(
    batch_size: int,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Paper implicit graph: all anatomical nodes are mutually connected."""
    adjacency = torch.ones(batch_size, num_nodes, num_nodes, device=device, dtype=dtype)
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    return adjacency


def center_edge_features(boxes: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    """Build normalized [x_i,y_i,x_j,y_j] features described before Eq. 6."""
    height, width = image_hw
    centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    scale = boxes.new_tensor([width, height])
    centers = centers / scale
    target = centers[:, :, None, :].expand(-1, -1, boxes.shape[1], -1)
    source = centers[:, None, :, :].expand(-1, boxes.shape[1], -1, -1)
    return torch.cat((target, source), dim=-1)


class MRGLGraphConv(nn.Module):
    """Small vectorized graph convolution corresponding to paper Eq. 6."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)  # W1
        self.neighbor_projection = nn.Linear(input_dim, output_dim)  # W2
        self.message_projection = nn.Linear(output_dim * 2, output_dim)  # W3
        self.edge_projection = nn.Linear(4, output_dim)  # W4 on center edge features

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid is not None:
            # CXAS-unavailable or empty masks are zero nodes. Mask both ends so
            # Linear biases cannot turn those missing nodes into messages.
            pair_valid = valid[:, :, None] & valid[:, None, :]
            adjacency = adjacency * pair_valid.to(adjacency.dtype)
        self_term = self.self_projection(node_features)
        projected_neighbors = self.neighbor_projection(node_features)
        # adjacency[target i, source j] * W2 F_j
        neighbor_messages = adjacency[..., None] * projected_neighbors[:, None, :, :]
        embedded_edges = adjacency[..., None] * self.edge_projection(edge_features)
        messages = self.message_projection(torch.cat((neighbor_messages, embedded_edges), dim=-1))
        messages = messages * (adjacency > 0)[..., None]
        output = F.relu(self_term + messages.sum(dim=2))
        if valid is not None:
            output = output * valid[..., None].to(output.dtype)
        return output


class MRGLBranch(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            MRGLGraphConv(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        features = node_features
        for layer in self.layers:
            features = layer(features, adjacency, edge_features, valid)
        return features


class MRGLClassifier(nn.Module):
    """ResNet ROI nodes -> spatial/semantic/implicit branches -> Eq. 7 output."""

    def __init__(
        self,
        num_classes: int = 14,
        hidden_dim: int = 256,
        graph_layers: int = 3,
        spatial_weight: float = 0.3,
        semantic_weight: float = 0.4,
        iou_threshold: float = 0.3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if spatial_weight < 0 or semantic_weight < 0 or spatial_weight + semantic_weight > 1:
            raise ValueError("Graph ensemble weights must be non-negative and sum to at most one")
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone_output_dim = 2048
        self.hidden_dim = hidden_dim
        self.iou_threshold = iou_threshold
        self.spatial_weight = spatial_weight
        self.semantic_weight = semantic_weight
        self.implicit_weight = 1.0 - spatial_weight - semantic_weight

        self.branches = nn.ModuleDict({
            name: MRGLBranch(self.backbone_output_dim, hidden_dim, graph_layers)
            for name in ("spatial", "semantic", "implicit")
        })
        self.predictors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes),
            )
            for name in ("spatial", "semantic", "implicit")
        })

    def extract_node_features(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.backbone(images)
        image_width = images.shape[-1]
        spatial_scale = feature_map.shape[-1] / float(image_width)
        box_list = [batch_boxes for batch_boxes in boxes]
        pooled = roi_align(
            feature_map,
            box_list,
            output_size=(1, 1),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        node_features = pooled.flatten(1).reshape(images.shape[0], boxes.shape[1], -1)
        if valid is not None:
            node_features = node_features * valid[..., None].to(node_features.dtype)
        return node_features, feature_map

    @staticmethod
    def _node_average(features: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        if valid is None:
            return features.mean(dim=1)
        weights = valid[..., None].to(features.dtype)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        semantic_adj: torch.Tensor,
        bbox_valid: Optional[torch.Tensor] = None,
        spatial_adj: Optional[torch.Tensor] = None,
        implicit_adj: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        node_features, feature_map = self.extract_node_features(images, boxes, bbox_valid)
        if spatial_adj is None:
            spatial_adj = spatial_iou_adjacency(boxes, self.iou_threshold, bbox_valid)
        if implicit_adj is None:
            implicit_adj = implicit_fully_connected_adjacency(
                images.shape[0], boxes.shape[1], images.device, images.dtype
            )
        edge_features = center_edge_features(boxes, images.shape[-2:])
        adjacencies = {
            "spatial": spatial_adj,
            "semantic": semantic_adj,
            "implicit": implicit_adj,
        }

        graph_features: Dict[str, torch.Tensor] = {}
        branch_predictions: Dict[str, torch.Tensor] = {}
        for name in ("spatial", "semantic", "implicit"):
            graph_features[name] = self.branches[name](
                node_features, adjacencies[name], edge_features, bbox_valid
            )
            pooled = self._node_average(graph_features[name], bbox_valid)
            branch_predictions[name] = torch.sigmoid(self.predictors[name](pooled))

        # Eq. 7. The paper reports the best alpha/beta near 0.3/0.4.
        output = (
            self.spatial_weight * branch_predictions["spatial"]
            + self.semantic_weight * branch_predictions["semantic"]
            + self.implicit_weight * branch_predictions["implicit"]
        )
        if not return_details:
            return output
        return output, {
            "feature_map": feature_map,
            "node_features": node_features,
            "spatial_adj": spatial_adj,
            "semantic_adj": semantic_adj,
            "implicit_adj": implicit_adj,
            "graph_features": graph_features,
            "branch_predictions": branch_predictions,
        }
