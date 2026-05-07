"""Internal validation metrics for the VCC LeWM model.

These are cheap signals computed during training on a held-out subset of
training perturbations. They don't replicate the official VCC metrics but
correlate with what we expect them to measure.

  pred_emb_mse  : MSE between predicted z_post and actual perturbed-cell
                  embedding, averaged over held-out perts.
  ctrl_emb_mse  : Same as above but using the control's embedding as a
                  no-op baseline. We want pred_emb_mse < ctrl_emb_mse.
  pred_gene_mse : Decoder MSE in gene space against actual perturbed cells.
  ctrl_gene_mse : Same but using the source control cell directly. We want
                  pred_gene_mse < ctrl_gene_mse for the model to be useful.
  pert_dr       : Average per-perturbation discrimination ratio: ratio of
                  distance-to-self-pert-mean over distance-to-other-perts-
                  mean. >1 means predictions land closer to their own
                  perturbation's actual cells than to other perturbations'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .data import VCCDataset, normalize, CONTROL_LABEL

if TYPE_CHECKING:
    from .data import VCCSplit
    from .models import MLPEncoder, PerturbationPredictor, Decoder
    from .train import TrainConfig


@torch.no_grad()
def internal_val_metrics(
    encoder,
    pert_predictor,
    decoder,
    val_dataset: VCCDataset,
    split: "VCCSplit",
    device: torch.device,
    cfg: "TrainConfig",
    max_cells_per_pert: int = 256,
    n_source_controls: int = 256,
) -> dict[str, float]:
    """Quick internal validation. Returns scalar metrics suitable for logging."""
    encoder.eval(); pert_predictor.eval(); decoder.eval()

    # Map dataset perturbation IDs to gene column indices for ActionEmbed.
    var_to_col = {g: i for i, g in enumerate(split.var_names)}

    # Sample a pool of control cells, normalized, on device.
    ctrl_global = np.where(split.control_mask)[0]
    rng = np.random.default_rng(0)
    ctrl_sample = rng.choice(ctrl_global, size=min(n_source_controls, len(ctrl_global)), replace=False)
    x_ctrl_dense = split.X[ctrl_sample].toarray()
    x_ctrl = torch.from_numpy(normalize(x_ctrl_dense)).to(device)
    z_ctrl = encoder(x_ctrl)                          # (n_ctrl, embed_dim)

    # Group val cells by perturbation
    val_pert_ids = split.pert_ids[val_dataset.indices]
    val_perts_present = sorted(set(int(p) for p in val_pert_ids if p != 0))
    if not val_perts_present:
        encoder.train(); pert_predictor.train(); decoder.train()
        return {"n_val_perts": 0}

    # Compute per-perturbation: predicted z, actual z, decoder predictions
    pred_z_per_pert: dict[int, torch.Tensor] = {}    # mean predicted z
    actual_z_per_pert: dict[int, torch.Tensor] = {}
    pred_emb_mse_sum = 0.0
    ctrl_emb_mse_sum = 0.0
    pred_gene_mse_sum = 0.0
    ctrl_gene_mse_sum = 0.0

    for pid in val_perts_present:
        pert_pos = np.where(val_pert_ids == pid)[0][:max_cells_per_pert]
        cell_idx = val_dataset.indices[pert_pos]
        x_pert_dense = split.X[cell_idx].toarray()
        x_pert = torch.from_numpy(normalize(x_pert_dense)).to(device)
        z_actual = encoder(x_pert)                   # (n_p, embed_dim)
        actual_z_per_pert[pid] = z_actual

        # Predict from a random sample of controls (no need to use all)
        n_pred = z_actual.shape[0]
        ctrl_pick = rng.choice(z_ctrl.shape[0], size=n_pred, replace=True)
        z_source = z_ctrl[torch.from_numpy(ctrl_pick).to(device)]
        x_source = x_ctrl[torch.from_numpy(ctrl_pick).to(device)]
        gene_name = split.pert_vocab[pid]
        col = var_to_col.get(gene_name, -1)
        gene_idx = torch.full((n_pred,), col, dtype=torch.long, device=device)

        z_pred = pert_predictor(z_source, gene_idx)
        x_pred = decoder(z_pred)

        pred_z_per_pert[pid] = z_pred.mean(dim=0)

        pred_emb_mse_sum += F.mse_loss(z_pred, z_actual).item()
        ctrl_emb_mse_sum += F.mse_loss(z_source, z_actual).item()
        pred_gene_mse_sum += F.mse_loss(x_pred, x_pert).item()
        ctrl_gene_mse_sum += F.mse_loss(x_source, x_pert).item()

    n_perts = len(val_perts_present)
    pred_emb_mse = pred_emb_mse_sum / n_perts
    ctrl_emb_mse = ctrl_emb_mse_sum / n_perts
    pred_gene_mse = pred_gene_mse_sum / n_perts
    ctrl_gene_mse = ctrl_gene_mse_sum / n_perts

    # Discrimination ratio: for each pert, distance from predicted-mean to
    # the actual-pert centroid vs. distance to the closest other pert centroid.
    actual_centroids = torch.stack(
        [actual_z_per_pert[p].mean(dim=0) for p in val_perts_present], dim=0,
    )
    pred_centroids = torch.stack(
        [pred_z_per_pert[p] for p in val_perts_present], dim=0,
    )
    drs = []
    for i in range(n_perts):
        d_self = (pred_centroids[i] - actual_centroids[i]).norm()
        # Closest *other* pert centroid
        d_others = (
            (pred_centroids[i].unsqueeze(0) - actual_centroids).norm(dim=-1)
        )
        d_others[i] = float("inf")
        d_other = d_others.min()
        drs.append((d_other / (d_self + 1e-8)).item())
    pert_dr = float(np.mean(drs))

    encoder.train(); pert_predictor.train(); decoder.train()
    return {
        "n_val_perts": n_perts,
        "pred_emb_mse": pred_emb_mse,
        "ctrl_emb_mse": ctrl_emb_mse,
        "pred_gene_mse": pred_gene_mse,
        "ctrl_gene_mse": ctrl_gene_mse,
        "pert_dr": pert_dr,
    }


@torch.no_grad()
def score_against_split(
    encoder,
    pert_predictor,
    decoder,
    train_split: "VCCSplit",
    eval_split: "VCCSplit",
    device: torch.device,
    n_pred_per_pert: int = 256,
    deg_top_k: int = 100,
) -> dict:
    """End-to-end scoring against a held-out split (VCC validation/test file).

    Approximates VCC's three official metrics without depending on cell-eval:

      pds  : Perturbation Discrimination Score. For each predicted-pert
             centroid, fraction of *other* perturbations whose actual centroid
             is *farther* (in L1 over gene space) than the matching one.
             1.0 = perfect, 0.5 = chance.
      des  : Differential Expression Jaccard. For each pert, take top-K
             upregulated genes and top-K downregulated genes vs control.
             Jaccard(predicted_DEGs, actual_DEGs), averaged.
      mae  : Mean absolute error in log1p(CP10k) gene space, per pert,
             averaged.

    Returns a dict with the three aggregate metrics plus per-pert tables.
    """
    encoder.eval(); pert_predictor.eval(); decoder.eval()

    # ---- Source controls drawn from training split (model never saw eval split)
    ctrl_idx = np.where(train_split.control_mask)[0]
    rng = np.random.default_rng(0)
    n_ctrl = min(len(ctrl_idx), 4 * n_pred_per_pert)
    ctrl_pick = rng.choice(ctrl_idx, size=n_ctrl, replace=False)
    x_ctrl_dense = train_split.X[ctrl_pick].toarray()
    x_ctrl = torch.from_numpy(normalize(x_ctrl_dense)).to(device)
    x_ctrl_mean = x_ctrl.mean(dim=0)                    # (G,)

    var_to_col = {g: i for i, g in enumerate(train_split.var_names)}

    # ---- Group eval cells by perturbation
    eval_perts_present = sorted(set(int(p) for p in eval_split.pert_ids if p != 0))
    eval_var_match = (eval_split.var_names == train_split.var_names)
    if not eval_var_match:
        # Should not happen for the VCC files but check anyway
        raise RuntimeError("eval split var_names do not match train split var_names")

    actual_centroids: dict[int, torch.Tensor] = {}
    pred_centroids: dict[int, torch.Tensor] = {}
    actual_means: dict[int, torch.Tensor] = {}
    pred_means: dict[int, torch.Tensor] = {}
    pert_names: list[str] = []
    mae_per_pert: list[float] = []
    pred_emb_mse_per_pert: list[float] = []

    for pid in eval_perts_present:
        gene_name = eval_split.pert_vocab[pid]
        col = var_to_col.get(gene_name, -1)
        if col == -1:
            continue
        pert_names.append(gene_name)

        cell_pos = np.where(eval_split.pert_ids == pid)[0]
        if len(cell_pos) > n_pred_per_pert:
            cell_pos = rng.choice(cell_pos, size=n_pred_per_pert, replace=False)
        x_actual = torch.from_numpy(
            normalize(eval_split.X[cell_pos].toarray())
        ).to(device)
        z_actual = encoder(x_actual)

        # Predict from random source controls
        n_pred = x_actual.shape[0]
        src_pick = torch.from_numpy(
            rng.choice(n_ctrl, size=n_pred, replace=True)
        ).to(device)
        z_source = encoder(x_ctrl[src_pick])
        gene_idx = torch.full((n_pred,), col, dtype=torch.long, device=device)
        z_pred = pert_predictor(z_source, gene_idx)
        x_pred = decoder(z_pred)

        actual_centroids[pid] = z_actual.mean(dim=0)
        pred_centroids[pid] = z_pred.mean(dim=0)
        actual_means[pid] = x_actual.mean(dim=0)
        pred_means[pid] = x_pred.mean(dim=0)

        mae_per_pert.append((x_pred.mean(dim=0) - x_actual.mean(dim=0)).abs().mean().item())
        pred_emb_mse_per_pert.append(F.mse_loss(z_pred.mean(dim=0), z_actual.mean(dim=0)).item())

    # ---- PDS approximation: gene-space L1 ranking
    perts = list(actual_means.keys())
    n = len(perts)
    if n < 2:
        encoder.train(); pert_predictor.train(); decoder.train()
        return {"n_perts": n, "error": "fewer than 2 perts available, can't score"}

    pred_stack = torch.stack([pred_means[p] for p in perts], dim=0)         # (n, G)
    actual_stack = torch.stack([actual_means[p] for p in perts], dim=0)     # (n, G)
    # L1 distance from each predicted pert to each actual pert
    dist = (pred_stack.unsqueeze(1) - actual_stack.unsqueeze(0)).abs().sum(dim=-1)  # (n, n)
    diag = dist.diag()
    rank_correct = (dist > diag.unsqueeze(1)).sum(dim=1)        # how many actuals farther than the matching one
    pds = (rank_correct.float() / max(n - 1, 1)).mean().item()

    # ---- DES approximation: top-k up + down DEG Jaccard vs control
    # Compute per-pert delta = pert_mean - control_mean. Top-K (up), Bottom-K (down).
    ctrl_mean_cpu = x_ctrl_mean.cpu().numpy()
    des_per_pert = []
    for p in perts:
        actual_delta = actual_means[p].cpu().numpy() - ctrl_mean_cpu
        pred_delta = pred_means[p].cpu().numpy() - ctrl_mean_cpu
        actual_up = set(np.argsort(-actual_delta)[:deg_top_k])
        actual_dn = set(np.argsort(actual_delta)[:deg_top_k])
        pred_up = set(np.argsort(-pred_delta)[:deg_top_k])
        pred_dn = set(np.argsort(pred_delta)[:deg_top_k])
        j_up = len(actual_up & pred_up) / max(len(actual_up | pred_up), 1)
        j_dn = len(actual_dn & pred_dn) / max(len(actual_dn | pred_dn), 1)
        des_per_pert.append(0.5 * (j_up + j_dn))
    des = float(np.mean(des_per_pert))

    mae_overall = float(np.mean(mae_per_pert))
    pred_emb_mse_overall = float(np.mean(pred_emb_mse_per_pert))

    encoder.train(); pert_predictor.train(); decoder.train()
    return {
        "n_perts": n,
        "pds": pds,
        "des": des,
        "mae_logcp10k": mae_overall,
        "pred_emb_mse": pred_emb_mse_overall,
        "per_pert": {
            "names": pert_names,
            "mae": mae_per_pert,
            "des": des_per_pert,
        },
    }
