"""Phase 2 smoke test with tiny config.

Runs 1 epoch of Phase 2.1 + 1 epoch of Phase 2.2 on the actual VCC data
to verify wiring end-to-end before running the full training. Should
complete in a couple of minutes.

Run:
    python scripts/phase2_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lewm.train import TrainConfig, main


if __name__ == "__main__":
    cfg = TrainConfig(
        batch_size=256,
        n_perts_per_batch=4,
        embed_dim=128,
        hidden_dim=256,
        jepa_hidden=128,
        action_dim=32,
        adaln_layers=2,
        adaln_heads=4,
        decoder_hidden=256,
        sigreg_projections=16,
        phase1_epochs=1,
        phase2_epochs=2,
        eval_every=1,
        out_dir="results/vcc-smoke",
        # Phase 3 #1: contrastive auxiliary loss
        contrastive_weight=1.0,
        contrastive_temperature=1.0,
    )
    main(cfg)
