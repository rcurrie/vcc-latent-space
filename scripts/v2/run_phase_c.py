"""Launch v2 Phase C (post-hoc decoder + VCC scoring).

    uv run python scripts/v2/run_phase_c.py
    uv run python scripts/v2/run_phase_c.py --epochs 3      # sanity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lewm.v2.train_phase_c import PhaseCConfig, train_phase_c


def main():
    cfg = PhaseCConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--decoder-hidden", type=int, default=cfg.decoder_hidden)
    ap.add_argument("--decoder-blocks", type=int, default=cfg.decoder_blocks)
    ap.add_argument("--z-actual-weight", type=float, default=cfg.z_actual_weight)
    ap.add_argument("--z-post-weight", type=float, default=cfg.z_post_weight)
    ap.add_argument("--phase-b-checkpoint", type=str, default=cfg.phase_b_checkpoint)
    ap.add_argument("--protein-panel-path", type=str, default=cfg.protein_panel_path)
    ap.add_argument("--out-dir", type=str, default=cfg.out_dir)
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    args = ap.parse_args()

    cfg = PhaseCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        decoder_hidden=args.decoder_hidden,
        decoder_blocks=args.decoder_blocks,
        z_actual_weight=args.z_actual_weight,
        z_post_weight=args.z_post_weight,
        phase_b_checkpoint=args.phase_b_checkpoint,
        protein_panel_path=args.protein_panel_path,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_phase_c(cfg)


if __name__ == "__main__":
    main()
