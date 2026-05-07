"""Score the trained model on the VCC validation split.

The competition is closed, so we treat this as a standard held-out test.
Loads phase2_checkpoint.pt, predicts post-perturbation expression for the
50 unseen perturbations in the validation file, and reports our own
approximations of VCC's three metrics (PDS, DES, MAE).

Usage:
    python scripts/score_validation.py [--checkpoint results/vcc/phase2_checkpoint.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from lewm.data import load_split
from lewm.eval import score_against_split
from lewm.models import (
    ActionEmbed,
    Decoder,
    MLPEncoder,
    PerturbationPredictor,
    compute_gene_features,
)
from lewm.train import TrainConfig, select_device


def load_models_from_checkpoint(ckpt_path: Path, train_split, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**cfg_dict)

    encoder = MLPEncoder(train_split.n_genes, cfg.embed_dim, cfg.hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    print("computing per-gene features from training controls ...")
    ctrl_idx = np.where(train_split.control_mask)[0]
    gene_feats = compute_gene_features(train_split.X, ctrl_idx).to(device)
    action_embed = ActionEmbed(
        gene_feats, action_dim=cfg.action_dim, hidden_dim=cfg.action_dim,
    ).to(device)
    pert_predictor = PerturbationPredictor(
        cfg.embed_dim, action_embed, n_layers=cfg.adaln_layers, n_heads=cfg.adaln_heads,
    ).to(device)
    pert_predictor.load_state_dict(ckpt["pert_predictor"])

    decoder = Decoder(cfg.embed_dim, train_split.n_genes, cfg.decoder_hidden).to(device)
    decoder.load_state_dict(ckpt["decoder"])

    return encoder, pert_predictor, decoder, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="results/vcc/phase2_checkpoint.pt",
        type=Path,
    )
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")

    print("\nloading training split (for source controls) ...")
    train_split = load_split("train")
    print(f"  {train_split.n_cells} cells, {train_split.n_perts} perts")

    print(f"\nloading checkpoint {args.checkpoint} ...")
    encoder, pert_predictor, decoder, cfg = load_models_from_checkpoint(
        args.checkpoint, train_split, device,
    )

    print("\nloading validation split (for scoring) ...")
    # Use the same vocab as training so unseen-pert ids are consistent (-1).
    # But validation has its own perturbations, which we want to evaluate.
    # Use validation's own vocab here.
    val_split = load_split("val")
    print(f"  {val_split.n_cells} cells, {val_split.n_perts} perts")

    # Confirm gene panels match (they do, but assert)
    assert val_split.var_names == train_split.var_names, (
        "var_names mismatch between train and val"
    )

    print("\nscoring ...")
    results = score_against_split(
        encoder, pert_predictor, decoder, train_split, val_split, device,
    )

    print("\n=== validation scores ===")
    print(f"  n_perts evaluated  : {results.get('n_perts', 0)}")
    print(f"  PDS (1.0 = perfect): {results.get('pds', float('nan')):.3f}")
    print(f"  DES (DEG Jaccard)  : {results.get('des', float('nan')):.3f}")
    print(f"  MAE (log1p CP10k)  : {results.get('mae_logcp10k', float('nan')):.4f}")
    print(f"  pred-vs-actual emb : {results.get('pred_emb_mse', float('nan')):.4f}")

    out_dir = Path(cfg.out_dir if hasattr(cfg, 'out_dir') else "results/vcc")
    out_path = out_dir / "validation_score.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    main()
