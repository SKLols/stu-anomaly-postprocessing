"""
Query Selection Filter
======================
Post-processing query selection for Mask4Former-3D based anomaly detection.

This module implements the spatial area-based query filtering strategy introduced
in the master's thesis:

    "Query-Optimized 3D Anomaly Segmentation: Comprehensive Uncertainty
     Evaluation for Autonomous Driving"
    Sourabh Lolge, Technische Hochschule Ingolstadt, 2025

Key idea: Filter object queries from Mask4Former-3D based on the number of
active points in their predicted masks. Queries with fewer points than the
area threshold are discarded before anomaly scoring, reducing noise from
fragmented or empty masks.

Achieves ~62% query reduction (100 → ~38 queries) while improving AUROC.
"""

import numpy as np
import torch
from typing import Tuple


def select_queries(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
    min_points: int = 30,
    activation_threshold: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select object queries based on spatial mask area in raw point space.

    Filters out queries whose predicted masks cover fewer than `min_points`
    raw points. This removes noise from fragmented or background queries
    before anomaly scoring, improving detection performance especially at
    mid-range distances (10-20m).

    Args:
        mask:                 Predicted mask logits, shape [N_voxels, N_queries].
        logit:                Predicted class logits, shape [N_queries, N_classes].
        inv_map:              Inverse map from voxels to raw points, shape [N_raw_points].
        min_points:           Minimum number of raw points a query must cover to
                              be retained. Default is 30 (optimal from ablation study).
        activation_threshold: Sigmoid threshold above which a point is considered
                              active for a query. Default is 0.1.

    Returns:
        selected_mask:  Filtered mask tensor, shape [N_voxels, N_selected_queries].
        selected_logit: Filtered logit tensor, shape [N_selected_queries, N_classes].

    Notes:
        - Operates after full Mask4Former-3D inference, requires no retraining.
        - Falls back to all queries if no queries pass the area threshold.
        - The optimal threshold of 30 points was determined via ablation study
          across thresholds [5, 10, 20, 30, 40] on the STU validation set.

    Example:
        >>> selected_mask, selected_logit = select_queries(
        ...     mask=pred_masks[b_idx],
        ...     logit=pred_logits[b_idx],
        ...     inv_map=inverse_maps[b_idx],
        ...     min_points=30,
        ... )
    """
    mask_sigmoid = mask.float().sigmoid()  # [N_voxels, N_queries]

    # Map voxel-space masks to raw point space via inverse map
    mask_np = mask_sigmoid.detach().cpu().numpy()
    inv_map_np = inv_map.cpu().numpy()

    n_raw_points = len(inv_map_np)
    n_queries = mask_np.shape[1]

    mask_raw = np.zeros((n_raw_points, n_queries), dtype=np.float32)
    for raw_idx, voxel_idx in enumerate(inv_map_np):
        mask_raw[raw_idx, :] = mask_np[voxel_idx, :]

    # Select queries whose masks cover at least min_points raw points
    selected_indices = [
        q for q in range(n_queries)
        if np.sum(mask_raw[:, q] > activation_threshold) >= min_points
    ]

    if len(selected_indices) == 0:
        # Fallback: retain all queries to avoid empty scoring
        print(
            f"[QuerySelection] Warning: No queries passed area threshold "
            f"(min_points={min_points}). Falling back to all {n_queries} queries."
        )
        return mask_sigmoid, logit

    n_selected = len(selected_indices)
    n_removed = n_queries - n_selected
    reduction_pct = (n_removed / n_queries) * 100
    print(
        f"[QuerySelection] Selected {n_selected}/{n_queries} queries "
        f"({reduction_pct:.1f}% reduction, threshold={min_points} pts)"
    )

    selected_mask = mask_sigmoid[:, selected_indices]
    selected_logit = logit[selected_indices]

    return selected_mask, selected_logit