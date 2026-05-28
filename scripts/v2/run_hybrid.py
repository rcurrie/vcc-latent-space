"""Launch the v2 Option B Hybrid: frozen encoder + gene-space delta head.

    uv run python scripts/v2/run_hybrid.py
    uv run python scripts/v2/run_hybrid.py --epochs 3       # sanity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lewm.v2.train_hybrid import HybridConfig, train_hybrid


def main():
    cfg = HybridConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--delta-hidden", type=int, default=cfg.delta_hidden)
    ap.add_argument("--delta-blocks", type=int, default=cfg.delta_blocks)
    ap.add_argument("--use-simple-head", action="store_true", default=cfg.use_simple_head)
    ap.add_argument("--no-simple-head", dest="use_simple_head", action="store_false")
    ap.add_argument("--head-type", choices=["simple", "concat", "adaln"], default=cfg.head_type)
    ap.add_argument("--phase-a-checkpoint", type=str, default=cfg.phase_a_checkpoint)
    ap.add_argument("--protein-panel-path", type=str, default=cfg.protein_panel_path)
    ap.add_argument("--out-dir", type=str, default=cfg.out_dir)
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--eval-every", type=int, default=cfg.eval_every)
    args = ap.parse_args()
    cfg = HybridConfig(
        epochs=args.epochs, lr=args.lr,
        delta_hidden=args.delta_hidden, delta_blocks=args.delta_blocks,
        use_simple_head=args.use_simple_head,
        head_type=args.head_type,
        phase_a_checkpoint=args.phase_a_checkpoint,
        protein_panel_path=args.protein_panel_path,
        out_dir=args.out_dir, num_workers=args.num_workers,
        eval_every=args.eval_every,
    )
    train_hybrid(cfg)


if __name__ == "__main__":
    main()
