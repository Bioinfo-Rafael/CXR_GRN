"""Semantic adjacency adapters.

The paper's semantic graph depends on graph Grad-CAM top-1/top-2 labels and the
unreleased machine-readable Figure 7 knowledge graph.  The public repository's
combine_dicts.py instead joins disease-location detector nodes; it is not used
as if it were the paper graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .cxas_bbox_cache import ANATOMICAL_REGIONS, NUM_ANATOMICAL_NODES


class TableVGroupSemanticAdapter:
    """Explicit fallback: connect 26 anatomy nodes in the same Table V group.

    This is deliberately a separate adapter and is not claimed to reproduce the
    Grad-CAM/disease-co-occurrence semantic graph in the paper.
    """

    is_paper_exact = False

    def __call__(
        self,
        batch_size: int,
        valid: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        groups = [mapping.table_v_group for mapping in ANATOMICAL_REGIONS]
        adjacency = torch.zeros(
            batch_size,
            NUM_ANATOMICAL_NODES,
            NUM_ANATOMICAL_NODES,
            dtype=torch.float32,
            device=device,
        )
        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                if i != j and group_i == group_j:
                    adjacency[:, i, j] = 1.0
        if valid is not None:
            pair_valid = valid.to(device=adjacency.device, dtype=torch.bool)
            adjacency *= (pair_valid[:, :, None] & pair_valid[:, None, :]).float()
        return adjacency


class PrecomputedSemanticGraphAdapter:
    """Load a user-supplied [26,26] or [B,26,26] semantic adjacency."""

    is_paper_exact = None

    def __init__(self, path: str | Path):
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                array = loaded["semantic_adj"]
        else:
            array = loaded
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.shape[-2:] != (NUM_ANATOMICAL_NODES, NUM_ANATOMICAL_NODES):
            raise ValueError(f"Expected semantic adjacency [...,26,26], got {tensor.shape}")
        self.adjacency = tensor

    def __call__(
        self,
        batch_size: int,
        valid: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        adjacency = self.adjacency
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        elif adjacency.shape[0] != batch_size:
            raise ValueError(f"Semantic batch {adjacency.shape[0]} != image batch {batch_size}")
        adjacency = adjacency.to(device=device)
        adjacency = adjacency.clone()
        adjacency.diagonal(dim1=-2, dim2=-1).zero_()
        if valid is not None:
            pair_valid = valid.to(device=adjacency.device, dtype=torch.bool)
            adjacency *= (pair_valid[:, :, None] & pair_valid[:, None, :]).float()
        return adjacency
