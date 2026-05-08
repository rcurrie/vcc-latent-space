"""Full Phase 2 training with the contrastive auxiliary loss enabled.

Same architecture and hyperparameters as the baseline (`python -m lewm.train`)
but with `contrastive_weight=1.0` and `contrastive_temperature=1.0`. Outputs
to `results/vcc-contrastive/` so the baseline is preserved at `results/vcc/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lewm.train import TrainConfig, main


if __name__ == "__main__":
    cfg = TrainConfig(
        contrastive_weight=1.0,
        contrastive_temperature=1.0,
        out_dir="results/vcc-contrastive",
    )
    main(cfg)
