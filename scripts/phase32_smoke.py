"""Phase 3.2 smoke: tiny config with ESM2 protein embeddings as the action.

Verifies the wiring of ProteinActionEmbed end-to-end on the real VCC data.
1 epoch each of phase 2.1 and phase 2.2. Should take ~2 min.
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
        out_dir="results/vcc-smoke-esm2",
        # Phase 3.2: ESM2 protein action embedding
        use_protein_action_embed=True,
        protein_proj_hidden=256,
        # also enable contrastive to see them stack
        contrastive_weight=1.0,
        contrastive_temperature=1.0,
    )
    main(cfg)
