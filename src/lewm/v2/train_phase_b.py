"""Phase B: world-model training for v2.

Builds on the frozen Phase A encoder checkpoint and jointly trains:

  - encoder                  (resumed from Phase A; lower LR)
  - ProteinActionEmbedV2     (PCA-1280 panel, single-linear projection)
  - PerturbationPredictor    (AdaLN-zero, identity-start)

No decoder in the loop. All losses live in latent space:

  L_pred    : MSE(z_post, stopgrad(z_target))   — core world-model objective
  L_inv     : MSE(z_view1, z_view2)             — augmentation-invariance
                                                   (binomial-subsample views,
                                                   extended to perturbed cells)
  L_mcr2    : -ΔR(z, pert_id) full conditional  — orthogonal pert subspaces

Sampler: 25% controls + 8 perts × 48 cells per batch (stratified). Each
perturbed cell is paired with a random control from the SAME batch as its
z_source (the "what cell state did this perturbation start from" prior).
Controls source themselves with gene_idx=-1, which routes through
ProteinActionEmbedV2's zero/fallback path → AdaLN identity → z_post ≈
encoder(control) (autoencoder-like signal).

Held-out perts (from data/vcc/v2_internal_val_split.json) are excluded from
training and used for periodic Latent-PDS monitoring.
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

from .data import (
    AugmentConfig,
    StratifiedPerturbationSampler,
    V2Dataset,
    collate_v2,
    load_split,
)
from .eval_latent import latent_pds
from .losses import (
    augmentation_invariance_loss,
    contrastive_centroid_loss,
    mcr2_loss,
)
from .models import (
    MLPEncoder,
    PerturbationPredictor,
    ProteinActionEmbedV2,
)
from .splits import load_internal_val_split, partition_indices_for_internal_val
from .train_phase_a import (
    effective_rank,
    participation_ratio,
    select_device,
)


DEFAULT_PCA_PANEL_PATH = "data/vcc/v2_gene_esm2_panel_pca1280.pt"
DEFAULT_PHASE_A_CKPT = "results/v2/phase_a/checkpoint.pt"


@dataclass
class PhaseBConfig:
    # data
    batch_size: int = 512
    n_perts_per_batch: int = 8
    control_fraction: float = 0.25
    tau: float = 0.5
    # model
    embed_dim: int = 256
    hidden_dim: int = 512
    action_dim: int = 64
    adaln_layers: int = 4
    adaln_heads: int = 4
    # paths
    phase_a_checkpoint: str = DEFAULT_PHASE_A_CKPT
    protein_panel_path: str = DEFAULT_PCA_PANEL_PATH
    out_dir: str = "results/v2/phase_b"
    # losses
    pred_weight: float = 1.0
    invariance_weight: float = 1.0
    mcr2_weight: float = 0.01
    mcr2_eps_sq: float = 0.5
    # v1-style contrastive aux, kept available as ablation
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 1.0
    # optimization
    epochs: int = 40
    encoder_lr: float = 1e-4
    predictor_lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # eval cadence
    eval_every: int = 2
    control_sample_size: int = 256
    # diagnostic cadence: PR/eff_rank do SVD on CPU; sampling avoids paying
    # the cost every batch. 0 disables; default samples once per epoch.
    diag_every_n_batches: int = 0      # 0 = once at end of epoch only
    # DataLoader parallelism. Profiled at 4× speedup with 4 workers; 8 workers
    # gives ~7×. Default 4 to balance memory (each worker forks the in-memory
    # CSR matrix on macOS).
    num_workers: int = 4
    seed: int = 0


def _build_pert_to_gene_idx(split) -> torch.Tensor:
    """Map pert_id → panel column index; controls / unknowns → -1."""
    var_to_col = {g: i for i, g in enumerate(split.var_names)}
    out = np.full(len(split.pert_vocab), -1, dtype=np.int64)
    for pid, gname in enumerate(split.pert_vocab):
        if pid == 0:               # control label
            continue
        out[pid] = var_to_col.get(gname, -1)
    return torch.from_numpy(out)


def train_phase_b(cfg: PhaseBConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").unlink(missing_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = select_device()
    print(f"device: {device}")

    split = load_split("train")
    print(f"split: {split.n_cells} cells × {split.n_genes} genes, "
          f"{split.n_perts - 1} non-control perts")

    iv = load_internal_val_split()
    holdout_pert_names = iv["holdout_pert_names"]
    print(f"holdout perts ({len(holdout_pert_names)}): "
          f"{holdout_pert_names[:5]}... (total {len(holdout_pert_names)})")
    train_idx, val_idx = partition_indices_for_internal_val(split, holdout_pert_names)
    print(f"  train indices: {len(train_idx)}, internal-val indices: {len(val_idx)}")

    # ---------------------------------------------------------------- data
    aug = AugmentConfig(tau=cfg.tau, paired_views=True)
    train_ds = V2Dataset(split, indices=train_idx, aug=aug, seed=cfg.seed)
    sampler = StratifiedPerturbationSampler(
        train_ds,
        batch_size=cfg.batch_size,
        n_perts_per_batch=cfg.n_perts_per_batch,
        control_fraction=cfg.control_fraction,
        seed=cfg.seed,
    )
    # macOS DataLoader defaults to *spawn* (Python 3.8+), which re-pickles the
    # ~10GB in-memory CSR matrix into each worker → many minutes of startup.
    # Forcing *fork* makes workers share the CSR via copy-on-write, dropping
    # startup from minutes to ~1s. Requires MPS not be initialized when fork
    # happens; we accomplish this by constructing the DataLoader before the
    # first .to(device) call below, but the DataLoader actually only fires
    # workers on the first iter() — which IS after MPS init. So we must use
    # an explicit fork context. PyTorch+MPS handles fork-after-MPS-init in
    # this configuration as long as workers don't touch the MPS context.
    fork_ctx = mp.get_context("fork") if cfg.num_workers > 0 else None
    loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_v2,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=fork_ctx,
    )
    print(f"batches/epoch: {len(sampler)} | batch_size_actual={sampler.batch_size_actual}")

    # ---------------------------------------------------------------- models
    encoder = MLPEncoder(
        gene_dim=split.n_genes, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
    ).to(device)
    ckpt = torch.load(cfg.phase_a_checkpoint, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    print(f"loaded Phase A encoder from {cfg.phase_a_checkpoint}")
    print(f"  Phase A final inv={ckpt['final_metrics'].get('inv', 'n/a')}, "
          f"PR={ckpt['final_metrics'].get('pr', 'n/a')}, "
          f"eff_rank={ckpt['final_metrics'].get('eff_rank', 'n/a')}")

    panel = torch.load(cfg.protein_panel_path, weights_only=False, map_location="cpu")
    if list(panel["var_names"]) != list(split.var_names):
        raise RuntimeError(
            "Protein panel var_names != split var_names; rebuild PCA panel."
        )
    print(f"protein panel: dim={panel['embed_dim']}, "
          f"covered={panel['n_covered']} / {len(panel['var_names'])}")
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=panel["embeddings"],
        coverage=panel["coverage"],
        action_dim=cfg.action_dim,
    ).to(device)

    predictor = PerturbationPredictor(
        embed_dim=cfg.embed_dim,
        action_embed=action_embed,
        n_layers=cfg.adaln_layers,
        n_heads=cfg.adaln_heads,
    ).to(device)

    enc_params = list(encoder.parameters())
    pred_params = list(predictor.parameters())     # includes action_embed
    n_enc = sum(p.numel() for p in enc_params)
    n_pred = sum(p.numel() for p in pred_params)
    print(f"params — encoder: {n_enc/1e6:.2f}M | predictor+action: {n_pred/1e6:.2f}M")

    optim = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.encoder_lr},
            {"params": pred_params, "lr": cfg.predictor_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    pert_to_gene_idx = _build_pert_to_gene_idx(split).to(device)

    # ---------------------------------------------------------------- train
    total_time = 0.0
    final_metrics = {}
    use_contrastive = cfg.contrastive_weight > 0.0

    # Track best PDS observed during training as a sanity sidecar — we report
    # the final-epoch checkpoint as the headline number, but having the best
    # tells us whether "train to convergence" cost us PDS (peak-then-leak) or
    # not (monotone or noisy-but-stable).
    best_pds = -float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        encoder.train(); predictor.train()
        t0 = time.perf_counter()
        sums = {"pred": 0.0, "inv": 0.0, "mcr2": 0.0, "con": 0.0,
                "total": 0.0, "pr": 0.0, "eff_rank": 0.0, "logit_gap": 0.0}
        n_batches = 0

        for x1, x2, pert_id, _batch, is_control in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            pert_id = pert_id.to(device)
            is_control = is_control.to(device)
            gene_idx = pert_to_gene_idx[pert_id]                  # (B,)

            B = x1.shape[0]
            ctrl_pos = torch.where(is_control)[0]
            if len(ctrl_pos) == 0:
                continue
            source_idx = torch.where(
                is_control,
                torch.arange(B, device=device),
                ctrl_pos[torch.randint(len(ctrl_pos), (B,), device=device)],
            )

            z1 = encoder(x1)                                       # paired view 1
            z2 = encoder(x2)                                       # paired view 2 (invariance only)
            z_source = z1[source_idx]                              # source for predictor
            z_target = z1                                          # target = encoder(x1)
            z_post = predictor(z_source, gene_idx)

            L_pred = F.mse_loss(z_post, z_target.detach())
            L_inv = augmentation_invariance_loss(z1, z2)
            L_mcr2, mcr2_diag = mcr2_loss(z1, pert_id, eps_sq=cfg.mcr2_eps_sq)

            L = (cfg.pred_weight * L_pred
                 + cfg.invariance_weight * L_inv
                 + cfg.mcr2_weight * L_mcr2)

            if use_contrastive:
                L_con, con_diag = contrastive_centroid_loss(
                    z_post, z_target, pert_id, is_control,
                    temperature=cfg.contrastive_temperature,
                )
                L = L + cfg.contrastive_weight * L_con
                sums["con"] += float(L_con.detach())
                sums["logit_gap"] += con_diag.get("mean_logit_gap", 0.0)

            optim.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + pred_params, cfg.grad_clip)
            optim.step()

            sums["pred"] += float(L_pred.detach())
            sums["inv"] += float(L_inv.detach())
            sums["mcr2"] += float(L_mcr2.detach())
            sums["total"] += float(L.detach())
            # SVD-based diagnostics are CPU-only and slow; sample sparingly.
            if (cfg.diag_every_n_batches and
                    n_batches % cfg.diag_every_n_batches == 0):
                sums["pr"] += participation_ratio(z1)
                sums["eff_rank"] += effective_rank(z1)
                sums["_diag_n"] = sums.get("_diag_n", 0) + 1
            n_batches += 1

        # One final diag sample at end of epoch if we never sampled mid-epoch.
        if "_diag_n" not in sums:
            sums["pr"] = participation_ratio(z1)
            sums["eff_rank"] = float(effective_rank(z1))
            sums["_diag_n"] = 1

        sched.step()
        epoch_time = time.perf_counter() - t0
        total_time += epoch_time
        diag_n = sums.pop("_diag_n", 1)
        avg = {}
        for k, v in sums.items():
            if k in ("pr", "eff_rank"):
                avg[k] = v / max(diag_n, 1)
            else:
                avg[k] = v / n_batches
        avg["epoch"] = epoch
        avg["lr_enc"] = optim.param_groups[0]["lr"]
        avg["lr_pred"] = optim.param_groups[1]["lr"]
        avg["epoch_time_s"] = epoch_time

        # Periodic latent-PDS on held-out
        lp = None
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            lp = latent_pds(
                encoder=encoder, predictor=predictor,
                split=split, holdout_pert_names=holdout_pert_names,
                control_sample_size=cfg.control_sample_size,
                device=device, seed=cfg.seed,
            )
            avg["latent_pds"] = lp.pds
            avg["latent_top1"] = lp.top1_acc
            if lp.pds > best_pds:
                best_pds = lp.pds
                best_epoch = epoch
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "action_embed": {k: v.detach().cpu().clone() for k, v in action_embed.state_dict().items()},
                    "predictor": {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
                    "epoch": epoch,
                    "pds": lp.pds,
                    "top1": lp.top1_acc,
                }

        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(avg, default=float) + "\n")

        line = (
            f"[B {epoch:3d}/{cfg.epochs}] "
            f"L={avg['total']:.4f} pred={avg['pred']:.4f} "
            f"inv={avg['inv']:.4f} mcr2={avg['mcr2']:.2f} "
            f"PR={avg['pr']:.1f} eff_rank={avg['eff_rank']:.1f} "
        )
        if use_contrastive:
            line += f"con={avg['con']:.3f} gap={avg['logit_gap']:+.2f} "
        if lp is not None:
            line += f"| PDS={lp.pds:.3f} top1={lp.top1_acc:.3f} "
        line += f"({epoch_time:.1f}s)"
        print(line)
        final_metrics = avg

    # Final checkpoint (headline number for the run).
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "action_embed": action_embed.state_dict(),
            "predictor": predictor.state_dict(),
            "config": asdict(cfg),
            "final_metrics": final_metrics,
            "n_train_cells": int(len(train_idx)),
            "total_time_s": total_time,
        },
        str(ckpt_path),
    )
    print(f"saved checkpoint to {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # Best-PDS sidecar checkpoint (sanity comparison, not the headline result).
    if best_state is not None:
        best_path = out_dir / "checkpoint_best_pds.pt"
        torch.save(
            {
                **best_state,
                "config": asdict(cfg),
                "note": "Best-PDS checkpoint over the run. Sidecar for cherry-pick check; the final-epoch checkpoint.pt is the recipe's headline result.",
            },
            str(best_path),
        )
        print(f"best-PDS sidecar: ep{best_epoch}, PDS={best_pds:.3f} → {best_path}")

    print(f"total train time: {total_time:.1f}s")
    return final_metrics
