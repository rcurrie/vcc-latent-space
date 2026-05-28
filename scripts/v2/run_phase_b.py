"""Launch v2 Phase B (world-model training).

    uv run python scripts/v2/run_phase_b.py                       # default 40 epochs
    uv run python scripts/v2/run_phase_b.py --epochs 3            # sanity
    uv run python scripts/v2/run_phase_b.py --mcr2-weight 0.005   # less MCR² pressure
    uv run python scripts/v2/run_phase_b.py --contrastive-weight 1.0  # ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lewm.v2.train_phase_b import PhaseBConfig, train_phase_b


def main():
    cfg = PhaseBConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--tau", type=float, default=cfg.tau)
    ap.add_argument("--encoder-lr", type=float, default=cfg.encoder_lr)
    ap.add_argument("--predictor-lr", type=float, default=cfg.predictor_lr)
    ap.add_argument("--pred-weight", type=float, default=cfg.pred_weight)
    ap.add_argument("--invariance-weight", type=float, default=cfg.invariance_weight)
    ap.add_argument("--mcr2-weight", type=float, default=cfg.mcr2_weight)
    ap.add_argument("--mcr2-eps-sq", type=float, default=cfg.mcr2_eps_sq)
    ap.add_argument("--contrastive-weight", type=float, default=cfg.contrastive_weight)
    ap.add_argument("--pseudobulk-weight", type=float, default=cfg.pseudobulk_weight)
    ap.add_argument("--eval-every", type=int, default=cfg.eval_every)
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--phase-a-checkpoint", type=str, default=cfg.phase_a_checkpoint)
    ap.add_argument("--protein-panel-path", type=str, default=cfg.protein_panel_path)
    ap.add_argument("--out-dir", type=str, default=cfg.out_dir)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    args = ap.parse_args()

    cfg = PhaseBConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        tau=args.tau,
        encoder_lr=args.encoder_lr,
        predictor_lr=args.predictor_lr,
        pred_weight=args.pred_weight,
        invariance_weight=args.invariance_weight,
        mcr2_weight=args.mcr2_weight,
        mcr2_eps_sq=args.mcr2_eps_sq,
        contrastive_weight=args.contrastive_weight,
        pseudobulk_weight=args.pseudobulk_weight,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        phase_a_checkpoint=args.phase_a_checkpoint,
        protein_panel_path=args.protein_panel_path,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    train_phase_b(cfg)


if __name__ == "__main__":
    main()
