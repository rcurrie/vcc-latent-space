"""S4: train the simple head on per-neighborhood delta SETS (the core experiment).

Fork of run_hybrid.py for the neighborhood experiment (docs/des_neighborhood_plan.md).
Two differences from the baseline simple head:

  TARGET  — per perturbation, the target is the mean of the matched
            per-neighborhood deltas {Δ_nb} (count-corrected pseudo-bulk,
            phase-matched control; see lewm.v2.nbhd_targets). For a deterministic
            action->delta head, MSE against the SET equals MSE against its mean
            (same gradient), so we regress the mean. No rank loss yet (that's S6).

  INFERENCE — handled by score_celleval.py --base neighborhoods: the predicted
            delta is applied to control NEIGHBORHOOD pseudo-bulks to emit a
            population whose spread is the neighborhood-level (not single-cell)
            spread. That is the "spread" the official cell-eval DES can reward.

    uv run python scripts/v2/run_hybrid_nbhd.py --k 50
    uv run python scripts/v2/score_celleval.py --checkpoint results/nbhd/hybrid_nbhd_k50/checkpoint.pt --split val --base neighborhoods --k 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.v2.data import load_split
from lewm.v2.models import MLPEncoder, ProteinActionEmbedV2
from lewm.v2.nbhd_targets import build_matched_deltas
from lewm.v2.splits import load_internal_val_split
from lewm.v2.train_phase_a import select_device

PHASE_A_CKPT = "results/v2/phase_a/checkpoint.pt"
PANEL_PATH = "data/vcc/v2_gene_esm2_panel_pca1280.pt"


class SimpleDeltaHead(nn.Module):
    """Linear(action_dim -> gene_dim), zero-init. Matches the baseline simple head."""

    def __init__(self, action_dim: int, gene_dim: int):
        super().__init__()
        self.lin = nn.Linear(action_dim, gene_dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.lin(action)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--action-dim", type=int, default=64)
    ap.add_argument("--n-ctrl", type=int, default=5000)
    ap.add_argument("--n-pert-max", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"results/nbhd/hybrid_nbhd_k{args.k}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"device: {device}  k={args.k}")

    split = load_split("train")
    var_to_col = {g: i for i, g in enumerate(split.var_names)}

    encoder = MLPEncoder(split.n_genes, embed_dim=256, hidden_dim=512).to(device)
    ck = torch.load(PHASE_A_CKPT, weights_only=False, map_location=device)
    encoder.load_state_dict(ck["encoder"]); encoder.eval()

    # Training perts = non-control, excluding the frozen internal-val holdout
    # (mirrors the baseline hybrid so the comparison is fair).
    holdout = set(load_internal_val_split()["holdout_pert_names"])
    pert_ids = [pid for pid in range(1, split.n_perts)
                if split.pert_vocab[pid] not in holdout]
    print(f"training perts: {len(pert_ids)} (excluded {len(holdout)} holdout)")

    print("building matched per-neighborhood delta targets...")
    t0 = time.perf_counter()
    targets, _ctrl_nb_pb = build_matched_deltas(
        encoder=encoder, device=device, split=split, pert_ids=pert_ids,
        k=args.k, n_ctrl=args.n_ctrl, n_pert_max=args.n_pert_max, seed=args.seed,
    )
    print(f"  built targets for {len(targets)} perts in {time.perf_counter()-t0:.1f}s")

    gene_cols, target_rows = [], []
    for t in targets:
        col = var_to_col.get(t.pert, -1)
        if col == -1:
            continue
        gene_cols.append(col)
        target_rows.append(t.deltas.mean(axis=0))      # mean over neighborhoods
    gene_cols = torch.tensor(gene_cols, dtype=torch.long, device=device)
    Y = torch.from_numpy(np.stack(target_rows)).float().to(device)   # (P, G)
    print(f"target matrix: {tuple(Y.shape)}")

    panel = torch.load(PANEL_PATH, weights_only=False, map_location="cpu")
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=panel["embeddings"], coverage=panel["coverage"],
        action_dim=args.action_dim,
    ).to(device)
    delta_head = SimpleDeltaHead(args.action_dim, split.n_genes).to(device)

    params = list(delta_head.parameters()) + list(action_embed.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Full-batch regression over perts (P ~ 120, cheap).
    for epoch in range(1, args.epochs + 1):
        delta_head.train(); action_embed.train()
        pred = delta_head(action_embed(gene_cols))     # (P, G)
        loss = F.mse_loss(pred, Y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step(); sched.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"[S4 {epoch:4d}/{args.epochs}] mse={float(loss):.6f} lr={optim.param_groups[0]['lr']:.2e}")

    cfg = {"head_type": "simple", "use_simple_head": True, "action_dim": args.action_dim,
           "embed_dim": 256, "hidden_dim": 512, "k": args.k,
           "protein_panel_path": PANEL_PATH, "phase_a_checkpoint": PHASE_A_CKPT,
           "nbhd_trained": True, "epochs": args.epochs, "lr": args.lr,
           "n_train_perts": int(Y.shape[0])}
    torch.save({
        "encoder": encoder.state_dict(),
        "action_embed": action_embed.state_dict(),
        "delta_head": delta_head.state_dict(),
        "config": cfg,
        "final_mse": float(loss),
    }, str(out_dir / "checkpoint.pt"))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {out_dir}/checkpoint.pt")


if __name__ == "__main__":
    main()
