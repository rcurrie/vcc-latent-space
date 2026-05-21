"""Gene-space scoring against an external VCC split (Validation or Test file).

Adapts v1's `score_against_split` for v2's action-conditioned decoder.

The three approximated VCC metrics:

  pds  : Perturbation Discrimination Score. For each predicted-pert centroid,
         fraction of *other* actual centroids that are FARTHER (in L1 over
         log1p(CP10k) gene space) than the matching one. 1.0 = perfect,
         0.5 = chance. Same definition as VCC's official PDS but with L1
         distance computed on log1p(CP10k) means rather than the full
         cell-eval gene-set version.

  des  : Differential Expression Score (Jaccard approximation). For each
         pert, compute delta = pert_mean - control_mean. Take top-K
         upregulated and top-K downregulated genes. Compare predicted DEGs
         vs actual DEGs via Jaccard, average up+down per pert, then mean
         over perts. Default K=100.

  mae  : Mean absolute error in log1p(CP10k) space between predicted-mean
         and actual-mean expression, averaged over perts.

These are the same approximations v1 reported and the scoring numbers are
directly comparable (v1 Phase 3.2: PDS=0.544, DES=0.076, MAE=0.015).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lewm.data import normalize  # reuse v1's normalize
from .data import VCCSplit, load_split
from .models import ActionConditionedDecoder


@dataclass
class GeneSpaceScores:
    n_perts: int
    pds: float
    des: float
    mae_logcp10k: float
    pred_emb_mse: float       # MSE between predicted-z and actual-z centroids
    per_pert_mae: dict[str, float]
    per_pert_des: dict[str, float]


@torch.no_grad()
def score_v2_against_split(
    *,
    encoder,
    predictor,
    action_embed,
    decoder: ActionConditionedDecoder,
    train_split: VCCSplit,
    eval_split: VCCSplit,
    device: torch.device,
    n_pred_per_pert: int = 256,
    deg_top_k: int = 100,
    seed: int = 0,
) -> GeneSpaceScores:
    encoder.eval(); predictor.eval(); action_embed.eval(); decoder.eval()

    if list(eval_split.var_names) != list(train_split.var_names):
        raise RuntimeError("eval split var_names != train split var_names")

    rng = np.random.default_rng(seed)
    var_to_col = {g: i for i, g in enumerate(train_split.var_names)}

    # Sample a pool of source controls from the training split.
    ctrl_idx = np.where(train_split.control_mask)[0]
    n_ctrl = min(len(ctrl_idx), 4 * n_pred_per_pert)
    ctrl_pick = rng.choice(ctrl_idx, size=n_ctrl, replace=False)
    x_ctrl_dense = train_split.X[ctrl_pick].toarray()
    x_ctrl = torch.from_numpy(normalize(x_ctrl_dense)).to(device)
    z_ctrl_pool = encoder(x_ctrl)
    x_ctrl_mean = x_ctrl.mean(dim=0)

    # Group eval cells by perturbation; score one pert at a time.
    eval_perts_present = sorted(set(int(p) for p in eval_split.pert_ids if p != 0))

    actual_means: dict[int, torch.Tensor] = {}
    pred_means: dict[int, torch.Tensor] = {}
    actual_centroids_z: dict[int, torch.Tensor] = {}
    pred_centroids_z: dict[int, torch.Tensor] = {}
    pert_names: dict[int, str] = {}

    for pid in eval_perts_present:
        gene_name = eval_split.pert_vocab[pid]
        col = var_to_col.get(gene_name, -1)
        if col == -1:
            continue
        pert_names[pid] = gene_name

        cell_pos = np.where(eval_split.pert_ids == pid)[0]
        if len(cell_pos) > n_pred_per_pert:
            cell_pos = rng.choice(cell_pos, size=n_pred_per_pert, replace=False)
        x_actual = torch.from_numpy(
            normalize(eval_split.X[cell_pos].toarray())
        ).to(device)
        z_actual = encoder(x_actual)

        n_pred = x_actual.shape[0]
        src_pick = torch.from_numpy(
            rng.choice(n_ctrl, size=n_pred, replace=True)
        ).to(device)
        z_source = z_ctrl_pool[src_pick]

        gene_idx_batch = torch.full((n_pred,), col, dtype=torch.long, device=device)
        z_pred = predictor(z_source, gene_idx_batch)
        action_vec = action_embed(gene_idx_batch)
        x_pred = decoder(z_pred, action_vec)

        actual_means[pid] = x_actual.mean(dim=0)
        pred_means[pid] = x_pred.mean(dim=0)
        actual_centroids_z[pid] = z_actual.mean(dim=0)
        pred_centroids_z[pid] = z_pred.mean(dim=0)

    perts = list(actual_means.keys())
    n = len(perts)
    if n < 2:
        encoder.train(); predictor.train(); action_embed.train(); decoder.train()
        return GeneSpaceScores(
            n_perts=n, pds=float("nan"), des=float("nan"),
            mae_logcp10k=float("nan"), pred_emb_mse=float("nan"),
            per_pert_mae={}, per_pert_des={},
        )

    pred_stack = torch.stack([pred_means[p] for p in perts], dim=0)        # (n, G)
    actual_stack = torch.stack([actual_means[p] for p in perts], dim=0)    # (n, G)

    # PDS: L1 in gene space.
    dist = (pred_stack.unsqueeze(1) - actual_stack.unsqueeze(0)).abs().sum(dim=-1)  # (n, n)
    diag = dist.diag()
    rank_correct = (dist > diag.unsqueeze(1)).sum(dim=1)
    pds = float((rank_correct.float() / max(n - 1, 1)).mean().item())

    # DES: top-K up/down DEG Jaccard vs control mean.
    ctrl_mean_cpu = x_ctrl_mean.cpu().numpy()
    per_pert_des: dict[str, float] = {}
    per_pert_mae: dict[str, float] = {}
    des_vals = []
    mae_vals = []
    pred_z_mse_vals = []
    for p in perts:
        actual_delta = actual_means[p].cpu().numpy() - ctrl_mean_cpu
        pred_delta = pred_means[p].cpu().numpy() - ctrl_mean_cpu
        actual_up = set(np.argsort(-actual_delta)[:deg_top_k])
        actual_dn = set(np.argsort(actual_delta)[:deg_top_k])
        pred_up = set(np.argsort(-pred_delta)[:deg_top_k])
        pred_dn = set(np.argsort(pred_delta)[:deg_top_k])
        j_up = len(actual_up & pred_up) / max(len(actual_up | pred_up), 1)
        j_dn = len(actual_dn & pred_dn) / max(len(actual_dn | pred_dn), 1)
        per_pert_des[pert_names[p]] = 0.5 * (j_up + j_dn)
        des_vals.append(per_pert_des[pert_names[p]])

        m = (pred_means[p] - actual_means[p]).abs().mean().item()
        per_pert_mae[pert_names[p]] = m
        mae_vals.append(m)

        pred_z_mse_vals.append(
            F.mse_loss(pred_centroids_z[p], actual_centroids_z[p]).item()
        )

    encoder.train(); predictor.train(); action_embed.train(); decoder.train()
    return GeneSpaceScores(
        n_perts=n,
        pds=pds,
        des=float(np.mean(des_vals)),
        mae_logcp10k=float(np.mean(mae_vals)),
        pred_emb_mse=float(np.mean(pred_z_mse_vals)),
        per_pert_mae=per_pert_mae,
        per_pert_des=per_pert_des,
    )
