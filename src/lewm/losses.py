"""Loss functions for LeWM-style training.

SIGReg (Sketched Isotropic Gaussian Regularizer) prevents representation
collapse by projecting embeddings onto random 1D directions and testing
each projection for Gaussianity via the Epps-Pulley statistic. By the
Cramer-Wold theorem, if all 1D marginals are Gaussian then so is the
joint distribution, so SIGReg -> 0 implies the embeddings fill space
isotropically.

This is the unsupervised replacement for MCR2/EMA-based collapse prevention
used in the LeWorldModel paper (Maes, Le Lidec, LeCun et al. 2026).

Phase 3 adds `contrastive_centroid_loss`: an InfoNCE-style auxiliary that
prevents predictions from collapsing to the global mean by requiring each
predicted cell's embedding to be closer to its own perturbation's actual
centroid than to other perturbations' centroids in the same batch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def epps_pulley_statistic(x: torch.Tensor) -> torch.Tensor:
    """Epps-Pulley test statistic for Gaussianity of a 1D sample.

    Lower values = more Gaussian. Returns a scalar.

    x: (n,) 1D tensor, assumed already zero-mean.
    """
    n = x.shape[0]
    var = x.var() + 1e-8

    # Term 1: pairwise interactions  (2/n^2) sum_{i,j} exp(-||x_i - x_j||^2 / (4*sigma^2))
    diffs_sq = (x.unsqueeze(0) - x.unsqueeze(1)) ** 2
    T1 = torch.exp(-diffs_sq / (4.0 * var)).sum() / (n * n)

    # Term 2: -2(1+2*sigma^2)^{-1/2} * (1/n) sum_i exp(-x_i^2 / (2+4*sigma^2))
    scale2 = 1.0 + 2.0 * var
    T2 = -2.0 * (scale2 ** -0.5) * torch.exp(-x ** 2 / (2.0 * scale2)).mean()

    # Term 3: constant
    T3 = (1.0 + 4.0 * var) ** -0.5

    return T1 + T2 + T3


# Module-level cache for random projection directions, keyed by (dim, device).
# Cached so SIGReg uses the same projections across an entire training run
# (more stable signal than re-sampling every batch).
_DIRECTIONS_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def sigreg_loss(
    Z: torch.Tensor,
    n_projections: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """SIGReg: aggregate Epps-Pulley over random 1D projections.

    Z: (batch_size, embed_dim) embeddings (NOT pre-normalized).
    Returns scalar; lower = more Gaussian = less collapse.
    """
    d = Z.shape[1]
    key = (d, Z.device.type)
    if key not in _DIRECTIONS_CACHE or _DIRECTIONS_CACHE[key].shape[0] != n_projections:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        dirs = torch.randn(n_projections, d, generator=gen)
        dirs = F.normalize(dirs, dim=1).to(Z.device)
        _DIRECTIONS_CACHE[key] = dirs
    directions = _DIRECTIONS_CACHE[key]

    projections = Z @ directions.T                          # (B, K)
    projections = projections - projections.mean(dim=0, keepdim=True)

    total = projections.new_zeros(())
    for k in range(n_projections):
        total = total + epps_pulley_statistic(projections[:, k])
    return total / n_projections


def contrastive_centroid_loss(
    z_pred: torch.Tensor,
    z_actual: torch.Tensor,
    pert_id: torch.Tensor,
    is_control: torch.Tensor | None = None,
    temperature: float = 0.1,
    min_perts: int = 2,
) -> tuple[torch.Tensor, dict]:
    """InfoNCE-style loss: each predicted cell should sit closer to its own
    perturbation's centroid (in z_actual space) than to other perturbations'
    centroids.

    Steps:
      1. For each unique non-control perturbation in the batch, build its
         actual-centroid c_g = mean(z_actual[pert_id == g]).
      2. For each non-control predicted cell z_pred_i with pert g_i, compute
         softmax cross-entropy where the target class is the index of
         centroid g_i and the logits are -|| z_pred_i - c_g ||^2 / τ
         (negative squared distance, equivalently the squared cosine if z's
         are unit-normalized — but our embeddings live in R^d so we use
         negative L2 distance which has the same monotonicity).
      3. Average over predicted cells.

    Controls (pert_id == 0 or is_control == True) are skipped — they don't
    have a target perturbation centroid.

    Returns
    -------
    loss : scalar tensor
    diag : dict of diagnostics (n_perts, n_pred_cells, mean_logit_gap)
    """
    if is_control is None:
        is_control = pert_id == 0

    # Indices of non-control cells
    nc_mask = ~is_control
    if nc_mask.sum() == 0:
        return z_pred.new_zeros(()), {"n_perts": 0, "n_cells": 0}

    nc_pert_ids = pert_id[nc_mask]                     # (M,)
    z_actual_nc = z_actual[nc_mask]                    # (M, D)
    z_pred_nc = z_pred[nc_mask]                        # (M, D)

    # Unique perturbations present in the batch (non-control only)
    unique_perts, inv = torch.unique(nc_pert_ids, return_inverse=True)
    n_perts = unique_perts.shape[0]
    if n_perts < min_perts:
        return z_pred.new_zeros(()), {"n_perts": int(n_perts), "n_cells": 0}

    # Build per-pert actual centroids by scatter-mean
    D = z_actual_nc.shape[1]
    centroids = z_actual_nc.new_zeros(n_perts, D)
    counts = z_actual_nc.new_zeros(n_perts)
    centroids.index_add_(0, inv, z_actual_nc)
    counts.index_add_(0, inv, torch.ones_like(inv, dtype=z_actual_nc.dtype))
    centroids = centroids / counts.unsqueeze(-1).clamp(min=1.0)

    # Negative squared L2 distance from each predicted cell to each centroid
    # logits[i, j] = -|| z_pred_i - c_j ||^2 / τ
    # = (-||z_pred||^2 + 2 z_pred · c - ||c||^2) / τ
    # We can compute it via cdist for clarity:
    dists_sq = torch.cdist(z_pred_nc, centroids, p=2).pow(2)   # (M, n_perts)
    logits = -dists_sq / temperature

    # Cross-entropy with target = index of own centroid
    loss = torch.nn.functional.cross_entropy(logits, inv)

    # Diagnostic: average gap between own-centroid logit and best-other logit
    with torch.no_grad():
        own_logit = logits.gather(1, inv.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, inv.unsqueeze(1), float("-inf"))
        best_other = masked.max(dim=1).values
        gap = (own_logit - best_other).mean().item()

    diag = {
        "n_perts": int(n_perts),
        "n_cells": int(nc_mask.sum().item()),
        "mean_logit_gap": gap,
    }
    return loss, diag
