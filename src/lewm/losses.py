"""Loss functions for LeWM-style training.

SIGReg (Sketched Isotropic Gaussian Regularizer) prevents representation
collapse by projecting embeddings onto random 1D directions and testing
each projection for Gaussianity via the Epps-Pulley statistic. By the
Cramer-Wold theorem, if all 1D marginals are Gaussian then so is the
joint distribution, so SIGReg -> 0 implies the embeddings fill space
isotropically.

This is the unsupervised replacement for MCR2/EMA-based collapse prevention
used in the LeWorldModel paper (Maes, Le Lidec, LeCun et al. 2026).
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
