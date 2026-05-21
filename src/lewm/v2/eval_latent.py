"""Latent-space evaluation metrics for v2.

The core monitor is **Latent-PDS** on held-out perturbations: for each
held-out gene g, does the predicted-z centroid sit closer to its own actual-z
centroid than to other held-out perturbations' actual-z centroids?

Decoder-free, gene-space-free, faithful to the v2 sketch's primary metric.
Chance baseline is 0.5; perfect is 1.0.

Two summary numbers per call:

  top1_acc : fraction of held-out perts whose predicted centroid is nearest
             to their *own* actual centroid (argmin over the holdout set).
             Stricter, higher variance.

  pds      : LeCun-faithful rank version. For each held-out pert g, count
             how many *other* held-out actual centroids are farther from
             c_pred[g] than c_actual[g] is. Normalize by (n_holdout-1).
             Average across g. Chance = 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    AugmentConfig,
    V2Dataset,
    VCCSplit,
    collate_v2,
)


@dataclass
class LatentPDSResult:
    """Summary of a single Latent-PDS evaluation call."""
    n_holdout_perts: int
    n_actual_cells: int        # held-out perturbed cells encoded
    n_control_cells: int       # controls used as source for predictions
    top1_acc: float            # fraction of perts where own centroid is nearest
    pds: float                 # mean rank-based PDS over perts (chance 0.5)


def _gene_idx_for_pert(
    pert_name: str,
    var_names: list[str],
    var_index: dict[str, int] | None = None,
) -> int:
    """Look up the panel column index for a perturbation gene name."""
    if var_index is None:
        var_index = {g: i for i, g in enumerate(var_names)}
    return var_index.get(pert_name, -1)


@torch.no_grad()
def latent_pds(
    *,
    encoder,
    predictor,
    split: VCCSplit,
    holdout_pert_names: list[str],
    control_sample_size: int = 256,
    device: torch.device,
    seed: int = 0,
    encode_batch_size: int = 512,
) -> LatentPDSResult:
    """Compute Latent-PDS for a set of held-out perturbations.

    Steps:
      1. Encode all cells whose target_gene is in holdout_pert_names ⇒ z_actual.
      2. Per held-out gene g: c_actual[g] = mean(z_actual over cells with pert g).
      3. Encode `control_sample_size` randomly-sampled control cells ⇒ z_ctrl.
      4. For each g: z_pred[g] = predictor(z_ctrl, gene_idx(g)); centroid c_pred[g].
      5. Score top-1 accuracy and rank-based PDS.

    The encoder/predictor are toggled into eval() and restored on exit.
    """
    enc_was_training = encoder.training
    pred_was_training = predictor.training
    encoder.eval(); predictor.eval()

    var_index = {g: i for i, g in enumerate(split.var_names)}

    # ---- 1. Held-out actual cells (per-pert) -----------------------------
    name_to_pert_id = {g: i for i, g in enumerate(split.pert_vocab)}
    holdout_pert_ids = [name_to_pert_id[g] for g in holdout_pert_names if g in name_to_pert_id]
    if len(holdout_pert_ids) < len(holdout_pert_names):
        missing = set(holdout_pert_names) - set(split.pert_vocab)
        raise ValueError(f"held-out perts missing from vocab: {missing}")

    holdout_mask = np.isin(split.pert_ids, holdout_pert_ids)
    holdout_indices = np.where(holdout_mask)[0].astype(np.int64)
    if len(holdout_indices) == 0:
        raise ValueError("no held-out cells found in this split")

    # Plain (no-augmentation) views for evaluation.
    aug_eval = AugmentConfig(tau=1.0, paired_views=False)
    held_ds = V2Dataset(split, indices=holdout_indices, aug=aug_eval, seed=seed)
    loader = DataLoader(
        held_ds, batch_size=encode_batch_size, shuffle=False,
        collate_fn=collate_v2, num_workers=0,
    )
    z_all = []
    pert_all = []
    for x1, _x2, pert_id, _batch, _ctrl in loader:
        z_all.append(encoder(x1.to(device)).detach().cpu())
        pert_all.append(pert_id)
    z_actual = torch.cat(z_all, dim=0)              # (N_actual, D)
    pert_actual = torch.cat(pert_all, dim=0)        # (N_actual,)

    # Per-pert actual centroids in holdout order.
    c_actual = []
    valid_perts = []
    for pid in holdout_pert_ids:
        mask = pert_actual == pid
        if mask.sum() == 0:
            continue
        c_actual.append(z_actual[mask].mean(dim=0))
        valid_perts.append(pid)
    c_actual = torch.stack(c_actual, dim=0)         # (P, D)

    # ---- 2. Predicted centroids per held-out pert ------------------------
    rng = np.random.default_rng(seed)
    ctrl_pool = np.where(split.pert_ids == 0)[0]
    if len(ctrl_pool) == 0:
        raise ValueError("split has no controls — cannot build z_pred sources")
    ctrl_idx = rng.choice(ctrl_pool, size=min(control_sample_size, len(ctrl_pool)),
                          replace=False)
    ctrl_ds = V2Dataset(split, indices=ctrl_idx, aug=aug_eval, seed=seed)
    loader = DataLoader(
        ctrl_ds, batch_size=encode_batch_size, shuffle=False,
        collate_fn=collate_v2, num_workers=0,
    )
    z_ctrl = []
    for x1, _x2, _pert_id, _batch, _ctrl in loader:
        z_ctrl.append(encoder(x1.to(device)).detach())
    z_ctrl = torch.cat(z_ctrl, dim=0)               # (N_ctrl, D)

    c_pred = []
    for pid in valid_perts:
        gname = split.pert_vocab[pid]
        gidx = _gene_idx_for_pert(gname, split.var_names, var_index)
        gene_idx_batch = torch.full(
            (z_ctrl.shape[0],), gidx, dtype=torch.long, device=device,
        )
        z_post = predictor(z_ctrl, gene_idx_batch).detach().cpu()
        c_pred.append(z_post.mean(dim=0))
    c_pred = torch.stack(c_pred, dim=0)             # (P, D)

    # ---- 3. Score top-1 + PDS -------------------------------------------
    # dists[i, j] = || c_pred[i] - c_actual[j] ||
    dists = torch.cdist(c_pred.unsqueeze(0), c_actual.unsqueeze(0)).squeeze(0)  # (P, P)
    P = dists.shape[0]
    own = dists.diag()                                  # (P,)
    nearest = dists.argmin(dim=1)
    top1 = float((nearest == torch.arange(P)).float().mean())

    # PDS: fraction of other actuals farther than own, averaged over perts.
    others = dists.clone()
    others[range(P), range(P)] = float("inf")
    farther_than_own = (others > own.unsqueeze(1)).float().sum(dim=1)
    pds = float((farther_than_own / max(P - 1, 1)).mean()) if P > 1 else float("nan")

    if enc_was_training:
        encoder.train()
    if pred_was_training:
        predictor.train()

    return LatentPDSResult(
        n_holdout_perts=P,
        n_actual_cells=int(z_actual.shape[0]),
        n_control_cells=int(z_ctrl.shape[0]),
        top1_acc=top1,
        pds=pds,
    )
