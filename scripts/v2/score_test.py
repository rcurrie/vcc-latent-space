"""Score a v2 checkpoint against the held-out VCC Test file.

The Test file (`data/vcc/adata_Test.h5ad`) has 100 perturbations disjoint
from both the training set and the Validation set. We never iterated
against it — model recipe choices (A1, A3, big-decoder ablations) were
made on Validation. So this score is the closest thing we have to a
leaderboard-comparable held-out number.

Run after Phase B + Phase C have produced checkpoints:

    uv run python scripts/v2/score_test.py
        --phase-b-checkpoint results/v2/A1_phase_b/checkpoint.pt
        --phase-c-checkpoint results/v2/A1_phase_c/checkpoint.pt
        --out results/v2/A1_phase_c/test_scores.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch

from lewm.v2.data import load_split
from lewm.v2.eval_gene import score_v2_against_split
from lewm.v2.models import (
    ActionConditionedDecoder,
    MLPEncoder,
    PerturbationPredictor,
    ProteinActionEmbedV2,
)
from lewm.v2.train_phase_a import select_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-b-checkpoint", default="results/v2/A1_phase_b/checkpoint.pt")
    ap.add_argument("--phase-c-checkpoint", default="results/v2/A1_phase_c/checkpoint.pt")
    ap.add_argument("--protein-panel-path", default="data/vcc/v2_gene_esm2_panel_pca1280.pt")
    ap.add_argument("--out", default="results/v2/A1_phase_c/test_scores.json")
    ap.add_argument("--n-pred-per-pert", type=int, default=256)
    ap.add_argument("--deg-top-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")

    train_split = load_split("train")
    test_split = load_split("test")
    print(f"train cells: {train_split.n_cells} | test cells: {test_split.n_cells}")
    print(f"test perts (non-control unique): {test_split.n_perts - 1}")

    ckpt_b = torch.load(args.phase_b_checkpoint, weights_only=False, map_location=device)
    cfg_b = ckpt_b["config"]
    print(f"loaded Phase B: {args.phase_b_checkpoint}")
    print(f"  final latent PDS: {ckpt_b.get('final_metrics', {}).get('latent_pds', ckpt_b.get('pds', 'n/a'))}")

    ckpt_c = torch.load(args.phase_c_checkpoint, weights_only=False, map_location=device)
    cfg_c = ckpt_c["config"]
    print(f"loaded Phase C: {args.phase_c_checkpoint}")

    panel = torch.load(args.protein_panel_path, weights_only=False, map_location="cpu")
    encoder = MLPEncoder(
        gene_dim=train_split.n_genes,
        embed_dim=cfg_b["embed_dim"],
        hidden_dim=cfg_b["hidden_dim"],
    ).to(device)
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=panel["embeddings"],
        coverage=panel["coverage"],
        action_dim=cfg_b["action_dim"],
    ).to(device)
    predictor = PerturbationPredictor(
        embed_dim=cfg_b["embed_dim"],
        action_embed=action_embed,
        n_layers=cfg_b["adaln_layers"],
        n_heads=cfg_b["adaln_heads"],
    ).to(device)
    encoder.load_state_dict(ckpt_b["encoder"])
    action_embed.load_state_dict(ckpt_b["action_embed"])
    predictor.load_state_dict(ckpt_b["predictor"])

    decoder = ActionConditionedDecoder(
        embed_dim=cfg_b["embed_dim"],
        action_dim=cfg_b["action_dim"],
        gene_dim=train_split.n_genes,
        hidden_dim=cfg_c["decoder_hidden"],
        n_blocks=cfg_c["decoder_blocks"],
    ).to(device)
    decoder.load_state_dict(ckpt_c["decoder"])

    print(f"\nScoring against TEST file (100 disjoint perts)...")
    scores = score_v2_against_split(
        encoder=encoder, predictor=predictor, action_embed=action_embed,
        decoder=decoder,
        train_split=train_split, eval_split=test_split,
        device=device,
        n_pred_per_pert=args.n_pred_per_pert,
        deg_top_k=args.deg_top_k,
        seed=args.seed,
    )

    print(f"\nVCC Test scores:")
    print(f"  n_perts:  {scores.n_perts}")
    print(f"  PDS:      {scores.pds:.4f}   (val: ~0.571, chance: 0.500)")
    print(f"  DES:      {scores.des:.4f}   (val: ~0.089)")
    print(f"  MAE:      {scores.mae_logcp10k:.4f}   (val: ~0.017)")
    print(f"  pred_emb_mse: {scores.pred_emb_mse:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {
            "split": "test",
            "n_perts": scores.n_perts,
            "pds": scores.pds,
            "des": scores.des,
            "mae_logcp10k": scores.mae_logcp10k,
            "pred_emb_mse": scores.pred_emb_mse,
            "per_pert_mae": scores.per_pert_mae,
            "per_pert_des": scores.per_pert_des,
            "phase_b_checkpoint": args.phase_b_checkpoint,
            "phase_c_checkpoint": args.phase_c_checkpoint,
        },
        indent=2, default=float,
    ))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
