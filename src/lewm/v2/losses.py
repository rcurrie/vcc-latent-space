"""Loss functions for v2.

  - sigreg_loss              : re-export from v1 (unchanged Gaussianity prior).
  - contrastive_centroid_loss: re-export from v1 (Phase B baseline subspace prior).
  - augmentation_invariance  : MSE between encoder outputs of paired views.
                               New in v2; the Van-Assel-motivated invariance term.
  - mcr2_loss                : MCR² implementation, gated behind a flag (dormant
                               by default, becomes ablation A1).

All losses operate on unnormalized latents (R^d), consistent with SIGReg's
isotropic-Gaussian target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# v1 primitives we reuse unchanged.
from lewm.losses import (  # noqa: F401
    contrastive_centroid_loss,
    epps_pulley_statistic,
    sigreg_loss,
)


def sigreg_loss_fresh(
    Z: torch.Tensor,
    n_projections: int = 64,
) -> torch.Tensor:
    """Same SIGReg as v1, but re-samples directions on every call.

    v1's sigreg_loss caches K=64 directions for the whole training run.
    That cache was identified as a collapse failure mode in Phase A
    sanity: the encoder learns to satisfy Gaussianity on the specific
    cached directions while leaving the joint distribution rank-deficient.

    Re-sampling each call means the encoder cannot overfit to a specific
    direction set — every batch tests Gaussianity along a fresh basis,
    pushing toward a closer approximation of the Cramér-Wold ideal.

    Cost: ~K·d extra FLOPs to sample new directions; negligible vs
    K·n² for the Epps-Pulley terms themselves.
    """
    d = Z.shape[1]
    dirs = torch.randn(n_projections, d, device=Z.device, dtype=Z.dtype)
    dirs = F.normalize(dirs, dim=1)

    projections = Z @ dirs.T                          # (B, K)
    projections = projections - projections.mean(dim=0, keepdim=True)

    total = projections.new_zeros(())
    for k in range(n_projections):
        total = total + epps_pulley_statistic(projections[:, k])
    return total / n_projections


def variance_floor_loss(
    Z: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg-style per-dimension variance floor.

    For each dim d: penalty = max(0, target_std - std(Z[:, d])).
    Returned as the mean over dims.

    Anchors per-dimension std at `target_std`, preventing the encoder
    from collapsing to a low-variance manifold while satisfying SIGReg
    or augmentation-invariance trivially. The scale-invariant nature of
    Epps-Pulley means SIGReg alone does not prevent variance shrinkage;
    this term supplies the missing anchor.

    Use additively in Phase A with a small weight (start λ=1.0).
    """
    std = (Z.var(dim=0, unbiased=False) + eps).sqrt()
    return torch.clamp(target_std - std, min=0.0).mean()


def covariance_decorrelation_loss(
    Z: torch.Tensor,
) -> torch.Tensor:
    """VICReg-style off-diagonal covariance penalty.

    L_cov = (1/d) · Σ_{i≠j} cov(Z)[i,j]² where cov is over the batch dim.

    Drives the embedding covariance matrix toward diagonal — i.e. forces
    embedding dimensions to be linearly decorrelated. Stacks with a
    per-dim variance floor (which prevents magnitude collapse) to give
    the standard SSL anti-collapse recipe (invariance + variance + cov).

    SIGReg with finite K-direction sampling is theoretically supposed to
    cover decorrelation via Cramér-Wold, but in practice K=64 cached or
    even resampled directions are too weak to fight invariance pressure
    — confirmed empirically in Phase A sanity runs. This is the missing
    explicit decorrelation term.
    """
    B, D = Z.shape
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / (B - 1)
    off_diag_sq = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag_sq / D


def augmentation_invariance_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """MSE between two paired encoder outputs.

    z1, z2 : (B, D) tensors; z_i = encoder(view_i(x)) where view_1 and view_2
             are independent binomial-subsamples of the same raw counts.

    Motivation: Van Assel et al. NeurIPS 2025 prove that joint-embedding
    methods need their augmentations to span the irrelevant-feature subspace
    for OOD generalization. Sequencing-depth noise is *the* dominant
    irrelevant feature in scRNA-seq, so binomial subsampling is the
    biologically faithful augmentation to enforce invariance against.

    Returns scalar; lower = more invariant.
    """
    return (z1 - z2).pow(2).mean()


def mcr2_marginal_loss(
    Z: torch.Tensor,
    eps_sq: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Marginal coding rate: -R(Z) where R(Z) = (1/2) logdet(I + α · ZᵀZ / B).

    This is the *marginal* half of MCR² (Yu/Ma 2020, eq. 4). Without a class
    partition (e.g. Phase A controls-only), the conditional term R(Z | Π) is
    unavailable; but maximizing R(Z) alone is itself a principled anti-
    collapse pressure — it pushes the eigenvalues of the embedding covariance
    upward, naturally enforcing both unit-ish variance and decorrelation in a
    single log-det. Mathematically equivalent to a maximum-entropy prior under
    the Gaussian rate-distortion source model.

    We return `loss = -R(Z)` so it minimizes like a normal training loss.

    For Phase B, switch to `mcr2_loss(Z, labels)` which adds the conditional
    term R(Z | Π) and computes the full ΔR objective.

    eps_sq : ε² distortion parameter. Default 0.5 per Yu/Ma reference settings
             for normalized features; sensitivity will need a quick sweep.
    """
    B, D = Z.shape
    alpha = D / (B * eps_sq)
    I = torch.eye(D, device=Z.device, dtype=Z.dtype)
    cov = Z.T @ Z
    # slogdet on MPS doesn't have a kernel — compute on CPU for safety.
    M = (I + alpha * cov).cpu()
    R = 0.5 * torch.linalg.slogdet(M).logabsdet
    R = R.to(Z.device)
    loss = -R
    diag = {"R": float(R.detach()), "alpha": float(alpha)}
    return loss, diag


def mcr2_loss(
    Z: torch.Tensor,
    labels: torch.Tensor,
    eps_sq: float = 0.5,
    min_class_size: int = 2,
) -> tuple[torch.Tensor, dict]:
    """Maximal Coding Rate Reduction (MCR²).

    DORMANT in v2 baseline; promoted to active loss in ablation A1.

    Implements ΔR(Z, Π) = R(Z) - R(Z | Π) per Yu/Ma 2020 (NeurIPS) eqs (4-6).
    Maximizing ΔR pushes the conditional distributions Z|class into
    orthogonal subspaces while keeping the marginal coverage high.

    Returns -(ΔR) so it can be minimized like a normal loss.

    Z          : (B, D) embeddings.
    labels     : (B,) integer class labels. Classes with fewer than
                 `min_class_size` samples in the batch are dropped from
                 the conditional term (still counted in the marginal).
    eps_sq     : ε² distortion parameter. The single most sensitive knob;
                 default 0.5 follows the Yu/Ma reference setting for
                 normalized features. Will need tuning for our scale.

    Returns
    -------
    loss : scalar tensor (negative ΔR; minimize)
    diag : dict with R_total, R_conditional, n_classes_used, n_dropped.

    Note: implementation kept conservative — uses a single-precision Cholesky-
    free formulation via logdet of (I + α · Z Zᵀ / B) which is the standard
    rate-distortion form for Gaussian sources. This is good enough for
    diagnostic use; production runs may want a numerically stabler variant.
    """
    B, D = Z.shape
    alpha = D / (B * eps_sq)
    I = torch.eye(D, dtype=Z.dtype)  # CPU; slogdet has no MPS kernel

    # Marginal coding rate R(Z) = (1/2) logdet(I + α Zᵀ Z). Compute on CPU
    # (autograd preserves the graph across the device hop).
    Zc = Z.cpu()
    cov = Zc.T @ Zc
    R_total_cpu = 0.5 * torch.linalg.slogdet(I + alpha * cov).logabsdet
    R_total = R_total_cpu.to(Z.device)

    # Conditional rate Σ_j (n_j / B) · (1/2) logdet(I + α_j Zⱼᵀ Zⱼ).
    labels_cpu = labels.cpu()
    unique, counts = torch.unique(labels_cpu, return_counts=True)
    R_cond_cpu = Zc.new_zeros(())
    n_used = 0
    n_dropped = 0
    for j, n_j in zip(unique.tolist(), counts.tolist()):
        if n_j < min_class_size:
            n_dropped += 1
            continue
        Zj = Zc[labels_cpu == j]
        alpha_j = D / (n_j * eps_sq)
        cov_j = Zj.T @ Zj
        Rj = 0.5 * torch.linalg.slogdet(I + alpha_j * cov_j).logabsdet
        R_cond_cpu = R_cond_cpu + (n_j / B) * Rj
        n_used += 1
    R_cond = R_cond_cpu.to(Z.device)
    R_total = R_total_cpu.to(Z.device)

    delta_R = R_total - R_cond
    loss = -delta_R
    diag = {
        "R_total": float(R_total.detach()),
        "R_cond": float(R_cond.detach()),
        "delta_R": float(delta_R.detach()),
        "n_classes_used": int(n_used),
        "n_classes_dropped": int(n_dropped),
    }
    return loss, diag
