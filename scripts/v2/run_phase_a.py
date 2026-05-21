"""Launch v2 Phase A pretraining.

    uv run python scripts/v2/run_phase_a.py                 # default 30 epochs
    uv run python scripts/v2/run_phase_a.py --epochs 3      # sanity run
    uv run python scripts/v2/run_phase_a.py --tau 0.7       # gentler augmentation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lewm.v2.train_phase_a import PhaseAConfig, train_phase_a


def main():
    cfg = PhaseAConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--tau", type=float, default=cfg.tau)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--jepa-weight", type=float, default=cfg.jepa_weight)
    ap.add_argument("--invariance-weight", type=float, default=cfg.invariance_weight)
    ap.add_argument("--mcr2-weight", type=float, default=cfg.mcr2_weight)
    ap.add_argument("--mcr2-eps-sq", type=float, default=cfg.mcr2_eps_sq)
    ap.add_argument("--sigreg-weight", type=float, default=cfg.sigreg_weight)
    ap.add_argument("--sigreg-projections", type=int, default=cfg.sigreg_projections)
    ap.add_argument("--variance-weight", type=float, default=cfg.variance_weight)
    ap.add_argument("--covariance-weight", type=float, default=cfg.covariance_weight)
    ap.add_argument("--out-dir", type=str, default=cfg.out_dir)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    args = ap.parse_args()

    cfg = PhaseAConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        tau=args.tau,
        lr=args.lr,
        jepa_weight=args.jepa_weight,
        invariance_weight=args.invariance_weight,
        mcr2_weight=args.mcr2_weight,
        mcr2_eps_sq=args.mcr2_eps_sq,
        sigreg_weight=args.sigreg_weight,
        sigreg_projections=args.sigreg_projections,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    train_phase_a(cfg)


if __name__ == "__main__":
    main()
