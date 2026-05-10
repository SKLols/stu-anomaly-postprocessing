"""
Uncertainty Scoring Methods
============================
Point-level anomaly scoring functions for 3D LiDAR anomaly detection.

This module implements and extends the uncertainty methods evaluated in the
master's thesis:

    "Query-Optimized 3D Anomaly Segmentation: Comprehensive Uncertainty
     Evaluation for Autonomous Driving"
    Sourabh Lolge, Technische Hochschule Ingolstadt, 2025

Methods implemented:
    - MaxLogit  [Hendrycks et al., 2019]  — STU baseline
    - MSP       [Hendrycks & Gimpel, 2016] — extended evaluation
    - RbA       [Nayal et al., 2023]       — STU baseline
    - Entropy   [Chan et al., 2021]        — extended evaluation
    - Energy    [Tian et al., 2022]        — extended evaluation
    - SML       [Jung et al., 2021]        — extended evaluation (best FPR@95)

All functions:
    - Accept pre-sigmoid mask tensors and raw logits from Mask4Former-3D.
    - Return anomaly scores where HIGHER values = MORE anomalous.
    - Are designed to be called after query selection (filter.py).

References:
    [1] Nekrasov et al., "Spotting the Unexpected (STU)", CVPR 2025.
    [2] Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and
        Out-of-Distribution Examples", ICLR 2017.
    [3] Jung et al., "Standardized Max Logits", ICCV 2021.
    [4] Tian et al., "Pixel-wise Energy-biased Abstention Learning", ECCV 2022.
    [5] Chan et al., "Entropy Maximization and Meta Classification", ICCV 2021.
    [6] Nayal et al., "RbA: Segmenting Unknown Regions Rejected by All", ICCV 2023.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor to [0, 1] range with numerical stability.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor normalized to [0, 1] range.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _compute_confid(
    mask_sigmoid: torch.Tensor,
    logit: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-point confidence scores via mask-weighted logit aggregation.

    Each point's logit is computed as a weighted sum of query logits,
    where weights are the sigmoid mask values for that point.

    Args:
        mask_sigmoid: Sigmoid-activated mask tensor, shape [N_voxels, N_queries].
                      Can be raw masks (will be sigmoid-activated) or
                      already-sigmoid masks.
        logit:        Class logit tensor, shape [N_queries, N_classes].

    Returns:
        Per-voxel confidence scores, shape [N_voxels, N_classes].
    """
    return mask_sigmoid.matmul(logit)


# ---------------------------------------------------------------------------
# Uncertainty scoring methods
# ---------------------------------------------------------------------------

def score_maxlogit(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MaxLogit anomaly scores [Hendrycks et al., 2019].

    Uses the maximum pre-softmax logit value per point as the inlier
    confidence. Lower max logit = more anomalous. Score is inverted so
    that higher output = more anomalous, consistent with all other methods.

    Args:
        mask:    Sigmoid mask tensor, shape [N_voxels, N_queries].
        logit:   Class logit tensor, shape [N_queries, N_classes].
        inv_map: Inverse map from voxels to raw points, shape [N_raw_points].

    Returns:
        Anomaly scores per raw point, shape [N_raw_points].
        Range: unbounded, higher = more anomalous.
    """
    confid = _compute_confid(mask, logit)
    max_logit = torch.max(confid, dim=1).values[inv_map]
    return (max_logit * -1) + 1


def score_msp(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Maximum Softmax Probability (MSP) anomaly scores [Hendrycks & Gimpel, 2016].

    Uses 1 - max(softmax(logits)) as the anomaly score. In-distribution
    points should have high softmax confidence; anomalies will have lower
    confidence, yielding higher anomaly scores.

    This method was found to match or exceed Deep Ensemble performance
    (92.94% vs 90.93% AUROC) with 3x lower computational cost.

    Args:
        mask:    Sigmoid mask tensor, shape [N_voxels, N_queries].
        logit:   Class logit tensor, shape [N_queries, N_classes].
        inv_map: Inverse map from voxels to raw points, shape [N_raw_points].

    Returns:
        Anomaly scores per raw point, shape [N_raw_points].
        Range: [0, 1], higher = more anomalous.
    """
    confid = _compute_confid(mask, logit)
    probs = F.softmax(confid, dim=1)
    msp = torch.max(probs, dim=1).values[inv_map]
    return 1 - msp


def score_rba(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Region-based Anomaly (RbA) scores [Nayal et al., 2023].

    Exploits Mask4Former-3D's query deactivation mechanism. Points with
    no active query predictions receive the highest anomaly scores.
    Uses tanh-based aggregation across all queries.

    Args:
        mask:    Sigmoid mask tensor, shape [N_voxels, N_queries].
        logit:   Class logit tensor, shape [N_queries, N_classes].
        inv_map: Inverse map from voxels to raw points, shape [N_raw_points].

    Returns:
        Anomaly scores per raw point, shape [N_raw_points].
        Range: [0, 2], higher = more anomalous.
    """
    confid = _compute_confid(mask, logit)
    rba = -confid.tanh().sum(dim=1)[inv_map]
    rba = torch.clamp(rba, min=-1)
    return rba + 1


def score_entropy(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
) -> torch.Tensor:
    """
    Compute entropy-based anomaly scores [Chan et al., 2021].

    Uses the Shannon entropy of the softmax distribution as the anomaly
    score. Higher entropy = higher uncertainty = more likely anomalous.

    Args:
        mask:    Sigmoid mask tensor, shape [N_voxels, N_queries].
        logit:   Class logit tensor, shape [N_queries, N_classes].
        inv_map: Inverse map from voxels to raw points, shape [N_raw_points].

    Returns:
        Anomaly scores per raw point, shape [N_raw_points].
        Range: [0, log(N_classes)], higher = more anomalous.
    """
    confid = _compute_confid(mask, logit)
    probs = F.softmax(confid, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)[inv_map]
    return entropy


def score_energy(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
) -> torch.Tensor:
    """
    Compute energy-based anomaly scores [Tian et al., 2022].

    Uses the Helmholtz free energy formulation: -log(sum(exp(logits))).
    Considers the entire logit distribution rather than just the maximum,
    mitigating the overconfidence problem in OOD detection.

    Args:
        mask:    Sigmoid mask tensor, shape [N_voxels, N_queries].
        logit:   Class logit tensor, shape [N_queries, N_classes].
        inv_map: Inverse map from voxels to raw points, shape [N_raw_points].

    Returns:
        Anomaly scores per raw point, shape [N_raw_points].
        Range: [0, 1] after normalization, higher = more anomalous.
    """
    confid = _compute_confid(mask, logit)
    energy = -torch.logsumexp(confid, dim=1)[inv_map]
    return _normalize(energy)


def score_sml(
    mask: torch.Tensor,
    logit: torch.Tensor,
    inv_map: torch.Tensor,
    class_mean: torch.Tensor,
    class_var: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Standardized Max Logits (SML) anomaly scores [Jung et al., 2021].

    Standardizes the max logit per point by per-class statistics computed
    on the training set. This handles the issue that logit distributions
    differ across classes, enabling fair cross-class comparison.

    SML achieved the best FPR@95 of all evaluated methods:
    17.40% with query selection vs 37.34% for Deep Ensemble.

    Args:
        mask:        Sigmoid mask tensor, shape [N_voxels, N_queries].
        logit:       Class logit tensor, shape [N_queries, N_classes].
        inv_map:     Inverse map from voxels to raw points, shape [N_raw_points].
        class_mean:  Per-class mean logit values from training set,
                     shape [N_classes].
        class_var:   Per-class logit variance from training set,
                     shape [N_classes].

    Returns:
        Anomaly scores per raw point, shape [N_raw_points].
        Higher = more anomalous.

    Notes:
        The formula used is: sml = z + y, where
            z = (max_logit - class_mean) / sqrt(class_var)  [standardized z-score]
            y = (max_logit - class_mean)^2 / N_classes      [quadratic penalty]
        This is then inverted so higher = more anomalous.
    """
    eps = 1e-6
    n_classes = len(class_mean)

    confid = _compute_confid(mask, logit)
    max_logit, prediction = torch.max(confid, dim=1)
    max_logit = max_logit[inv_map]
    prediction = prediction[inv_map]

    sml = torch.zeros_like(max_logit)
    for c in range(n_classes):
        mask_c = prediction == c
        if mask_c.any():
            z = (max_logit[mask_c] - class_mean[c]) / torch.sqrt(class_var[c] + eps)
            y = (max_logit[mask_c] - class_mean[c]) ** 2 / n_classes
            sml[mask_c] = z + y

    return (sml * -1) + 1