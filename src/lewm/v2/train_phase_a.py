"""Phase A: controls-only pretraining for v2.

Trains the encoder (and a JEPA helper predictor) on the 38k non-targeting
control cells. Loss is the sum of three terms:

  L_jepa  : MSE(JEPA(z_context), z_target.detach()) — gene-set masking
            (kept from v1 for spatial-context invariance over the gene axis)
  L_inv   : MSE(encoder(view_1), encoder(view_2))   — augmentation-invariance
            over binomial-subsample views (new in v2; Van-Assel-motivated)
  L_sig   : SIGReg over encoder(view_1)             — isotropic-Gaussian prior
            (kept from v1 for collapse prevention)

Each view is an independent binomial subsample of the same raw count
vector, then log1p(CP10k)-normalized.

Outputs:
  results/v2/phase_a/checkpoint.pt — encoder state dict (only what Phase B needs)
  results/v2/phase_a/metrics.jsonl — per-epoch metrics
  results/v2/phase_a/config.json   — TrainConfigA dump
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import AugmentConfig, V2Dataset, collate_v2, load_split
from .losses import (
    augmentation_invariance_loss,
    covariance_decorrelation_loss,
    mcr2_marginal_loss,
    sigreg_loss_fresh,
    variance_floor_loss,
)
from .models import JEPAPredictor, MLPEncoder, gene_set_mask


@dataclass
class PhaseAConfig:
    # data
    batch_size: int = 512
    tau: float = 0.5                 # binomial-subsample retention probability
    # model
    embed_dim: int = 256
    hidden_dim: int = 512
    jepa_hidden: int = 256
    context_ratio: float = 0.75
    # losses
    # Defaults match the v2-final recipe (no JEPA, MCR² weight 0.01).
    # The 30-epoch Phase A used --jepa-weight 0 --mcr2-weight 0.01 explicitly;
    # defaults updated here so re-runs don't need the flags to reproduce.
    jepa_weight: float = 0.0       # dropped: competed with MCR² for budget
    invariance_weight: float = 1.0
    # Path C (default for v2): MCR² marginal-rate anti-collapse loss.
    # Single log-det term, rate-distortion-grounded, naturally enforces both
    # variance and decorrelation. Phase B will extend to full ΔR with class
    # partition by perturbation. weight=0.01 balances against inv/JEPA scale.
    mcr2_weight: float = 0.01
    mcr2_eps_sq: float = 0.5
    # Path A/B (legacy, available via CLI for ablations). Disabled by default.
    sigreg_weight: float = 0.0
    sigreg_projections: int = 64
    variance_weight: float = 0.0
    variance_target_std: float = 1.0
    covariance_weight: float = 0.0
    # optimization
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # logging
    out_dir: str = "results/v2/phase_a"
    seed: int = 0
    # diagnostics
    rank_eps_ratio: float = 0.01     # for effective-rank threshold
    # DataLoader workers. 4 gives ~5x speedup over 0 on M4; 8 gives ~7x.
    num_workers: int = 4


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _svdvals_cpu(Z: torch.Tensor) -> torch.Tensor:
    """SVD on CPU (MPS lacks an SVD kernel as of this write)."""
    with torch.no_grad():
        Zc = Z - Z.mean(dim=0, keepdim=True)
        return torch.linalg.svdvals(Zc.detach().cpu())


def effective_rank(Z: torch.Tensor, eps_ratio: float = 0.01) -> int:
    """Count singular values of Z that exceed eps_ratio · sigma_max.

    Coarse diagnostic: how many directions of the embedding space are
    being used? Drops near 1 if the encoder collapses; should grow during
    healthy Phase A.
    """
    s = _svdvals_cpu(Z)
    thresh = eps_ratio * s.max().clamp(min=1e-12)
    return int((s > thresh).sum().item())


def participation_ratio(Z: torch.Tensor) -> float:
    """Continuous rank proxy: (Σ σ_i²)² / Σ σ_i⁴, a smooth effective-rank.

    Equal to D for white noise (all singular values equal). Falls to 1
    when one direction dominates.
    """
    s = _svdvals_cpu(Z)
    s2 = s ** 2
    return float((s2.sum() ** 2 / (s2 ** 2).sum()).item())


def train_phase_a(cfg: PhaseAConfig) -> dict:
    """Run Phase A and return the final metrics dict (last-epoch averages)."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").unlink(missing_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = select_device()
    print(f"device: {device}")

    # Data: controls only.
    split = load_split("train")
    control_idx = np.where(split.pert_ids == 0)[0]
    print(f"controls: {len(control_idx)} of {split.n_cells} cells")

    aug = AugmentConfig(tau=cfg.tau, paired_views=True)
    dataset = V2Dataset(split, indices=control_idx, aug=aug, seed=cfg.seed)
    # Force *fork* context: macOS spawn would re-pickle the CSR matrix.
    fork_ctx = mp.get_context("fork") if cfg.num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=fork_ctx,
        collate_fn=collate_v2,
        drop_last=True,
    )

    encoder = MLPEncoder(split.n_genes, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim).to(device)
    jepa = JEPAPredictor(embed_dim=cfg.embed_dim, hidden_dim=cfg.jepa_hidden).to(device)

    params = list(encoder.parameters()) + list(jepa.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    print(f"encoder + jepa params: {sum(p.numel() for p in params)/1e6:.2f}M")
    print(f"batches/epoch: {len(loader)} @ batch_size={cfg.batch_size}")

    metrics_final = {}
    total_time = 0.0
    for epoch in range(1, cfg.epochs + 1):
        encoder.train(); jepa.train()
        t0 = time.perf_counter()
        sums = {"jepa": 0.0, "inv": 0.0, "mcr2": 0.0,
                "sig": 0.0, "var": 0.0, "cov": 0.0,
                "total": 0.0, "pr": 0.0, "eff_rank": 0.0}
        n_batches = 0

        for x1, x2, _pert, _batch, _ctrl in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            # Encoder forward on both views (shared encoder).
            z1 = encoder(x1)
            z2 = encoder(x2)

            # Augmentation invariance.
            L_inv = augmentation_invariance_loss(z1, z2)

            # JEPA masked-latent prediction on view 1.
            x_ctx, x_tgt = gene_set_mask(x1, cfg.context_ratio)
            z_ctx = encoder(x_ctx)
            with torch.no_grad():
                z_tgt = encoder(x_tgt)
            z_pred_tgt = jepa(z_ctx)
            L_jepa = F.mse_loss(z_pred_tgt, z_tgt)

            # Anti-collapse losses. Default: MCR² marginal only. Other paths
            # (SIGReg / VICReg var+cov) are available behind weight flags for
            # ablation but compute is skipped when weight is 0.
            if cfg.mcr2_weight > 0:
                L_mcr2, _ = mcr2_marginal_loss(z1, eps_sq=cfg.mcr2_eps_sq)
            else:
                L_mcr2 = z1.new_zeros(())

            L_sig = (sigreg_loss_fresh(z1, n_projections=cfg.sigreg_projections)
                     if cfg.sigreg_weight > 0 else z1.new_zeros(()))
            L_var = (variance_floor_loss(z1, target_std=cfg.variance_target_std)
                     if cfg.variance_weight > 0 else z1.new_zeros(()))
            L_cov = (covariance_decorrelation_loss(z1)
                     if cfg.covariance_weight > 0 else z1.new_zeros(()))

            L = (cfg.jepa_weight * L_jepa
                 + cfg.invariance_weight * L_inv
                 + cfg.mcr2_weight * L_mcr2
                 + cfg.sigreg_weight * L_sig
                 + cfg.variance_weight * L_var
                 + cfg.covariance_weight * L_cov)

            optim.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            sums["jepa"] += float(L_jepa.detach())
            sums["inv"] += float(L_inv.detach())
            sums["mcr2"] += float(L_mcr2.detach())
            sums["sig"] += float(L_sig.detach())
            sums["var"] += float(L_var.detach())
            sums["cov"] += float(L_cov.detach())
            sums["total"] += float(L.detach())
            sums["pr"] += participation_ratio(z1)
            sums["eff_rank"] += effective_rank(z1, cfg.rank_eps_ratio)
            n_batches += 1

        sched.step()
        epoch_time = time.perf_counter() - t0
        total_time += epoch_time
        avg = {k: v / n_batches for k, v in sums.items()}
        avg["epoch"] = epoch
        avg["lr"] = optim.param_groups[0]["lr"]
        avg["epoch_time_s"] = epoch_time

        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(avg, default=float) + "\n")

        print(
            f"[A {epoch:3d}/{cfg.epochs}] "
            f"L={avg['total']:.4f} jepa={avg['jepa']:.4f} "
            f"inv={avg['inv']:.4f} mcr2={avg['mcr2']:.4f} "
            f"sig={avg['sig']:.4f} var={avg['var']:.4f} cov={avg['cov']:.4f} "
            f"PR={avg['pr']:.1f} eff_rank={avg['eff_rank']:.1f} "
            f"({epoch_time:.1f}s)"
        )
        metrics_final = avg

    # Save checkpoint: encoder only (jepa is a Phase A artifact, not reused).
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "config": asdict(cfg),
            "final_metrics": metrics_final,
            "n_train_cells": len(control_idx),
            "total_time_s": total_time,
        },
        str(ckpt_path),
    )
    print(f"saved checkpoint to {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    print(f"total train time: {total_time:.1f}s")

    return metrics_final
