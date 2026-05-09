"""Phase 3.2: full training with ESM2 protein action embeddings + contrastive aux loss.

Same architecture and hyperparameters as the baseline / contrastive runs,
but with ProteinActionEmbed (5120-d ESM2 -> 64-d MLP) replacing the
3-feature ActionEmbed. Contrastive loss is on with weight=1.0, τ=1.0.

Outputs to results/vcc-protein-contrastive/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lewm.train import TrainConfig, main


if __name__ == "__main__":
    cfg = TrainConfig(
        # ESM2 protein action conditioning
        use_protein_action_embed=True,
        protein_proj_hidden=256,
        # contrastive aux loss (Phase 3.1)
        contrastive_weight=1.0,
        contrastive_temperature=1.0,
        # output dir
        out_dir="results/vcc-protein-contrastive",
    )
    main(cfg)
