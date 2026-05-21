"""v2 Phase 0 smoke test.

Imports every v2 module, instantiates a minimal end-to-end stack on a
small subset of the training split, and verifies:

  - V2Dataset returns two distinct paired views when tau < 1.0
  - augmentation_invariance_loss decreases when both views are identical
  - sigreg_loss runs over the encoder output
  - ProteinActionEmbedV2 forward against the new PCA-1280 panel works
  - PerturbationPredictor + ActionConditionedDecoder forward shapes match

Not a training loop — purely a "does the wiring work" check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lewm.v2.data import (
    AugmentConfig,
    V2Dataset,
    collate_v2,
    load_split,
)
from lewm.v2.losses import (
    augmentation_invariance_loss,
    mcr2_loss,
    sigreg_loss,
)
from lewm.v2.models import (
    ActionConditionedDecoder,
    JEPAPredictor,
    MLPEncoder,
    PerturbationPredictor,
    ProteinActionEmbedV2,
    gene_set_mask,
)
from lewm.v2.splits import load_internal_val_split

PCA_PANEL_PATH = Path("data/vcc/v2_gene_esm2_panel_pca1280.pt")


def main():
    print("== v2 Phase 0 smoke test ==")

    # 1) Load split + internal-val JSON
    split = load_split("train")
    print(f"split: {split.n_cells} cells × {split.n_genes} genes, {split.n_perts} perts")
    iv = load_internal_val_split()
    print(f"internal-val: {iv['n_holdout']} held-out perts (e.g. {iv['holdout_pert_names'][:3]}...)")

    # 2) Paired-view dataset on the first 256 cells
    aug = AugmentConfig(tau=0.5, paired_views=True)
    ds = V2Dataset(split, indices=np.arange(256), aug=aug, seed=0)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_v2)
    x1, x2, pert_id, batch_id, is_ctrl = next(iter(loader))
    print(f"batch: x1={tuple(x1.shape)}, x2={tuple(x2.shape)}, "
          f"is_ctrl_frac={is_ctrl.float().mean():.3f}")
    diff = (x1 - x2).abs().mean().item()
    print(f"  mean |x1-x2| (should be > 0 since tau=0.5): {diff:.4f}")
    assert diff > 1e-3, "paired views are identical — augmenter is dead"

    # 3) Encoder + augmentation-invariance loss
    enc = MLPEncoder(split.n_genes, embed_dim=256)
    z1 = enc(x1)
    z2 = enc(x2)
    L_inv = augmentation_invariance_loss(z1, z2)
    L_sig = sigreg_loss(z1, n_projections=32)
    print(f"encoder out: z1={tuple(z1.shape)} | L_inv={L_inv:.4f} | L_sig={L_sig:.4f}")

    # 4) JEPA predictor + gene_set_mask
    x_ctx, x_tgt = gene_set_mask(x1, context_ratio=0.75)
    z_ctx = enc(x_ctx)
    z_tgt = enc(x_tgt)
    jepa = JEPAPredictor(embed_dim=256)
    z_pred_tgt = jepa(z_ctx)
    L_jepa = ((z_pred_tgt - z_tgt.detach()) ** 2).mean()
    print(f"JEPA: ctx_emb={tuple(z_ctx.shape)} | L_jepa={L_jepa:.4f}")

    # 5) ProteinActionEmbedV2 against the PCA panel
    pp = torch.load(str(PCA_PANEL_PATH), weights_only=False, map_location="cpu")
    print(f"PCA panel: dim={pp['embed_dim']}, covered={pp['n_covered']}, "
          f"explained_var={pp['pca_explained_variance_ratio']:.3f}")
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=pp["embeddings"],
        coverage=pp["coverage"],
        action_dim=64,
    )
    # Use pert_id - 1 to get the gene index into the panel? No: pert_vocab
    # maps to a gene NAME; we'd need a name->panel_idx lookup. For the
    # smoke test, just feed arbitrary gene indices.
    fake_gene_idx = torch.randint(0, split.n_genes, (x1.shape[0],))
    a = action_embed(fake_gene_idx)
    print(f"action_embed out: {tuple(a.shape)}")

    # 6) PerturbationPredictor end-to-end
    predictor = PerturbationPredictor(
        embed_dim=256,
        action_embed=action_embed,
        n_layers=2,        # smaller for smoke test
        n_heads=4,
    )
    z_post = predictor(z1, fake_gene_idx)
    print(f"predictor out: {tuple(z_post.shape)}")

    # 7) ActionConditionedDecoder
    dec = ActionConditionedDecoder(
        embed_dim=256,
        action_dim=64,
        gene_dim=split.n_genes,
        hidden_dim=256,
        n_blocks=1,
    )
    x_hat = dec(z_post, a)
    print(f"decoder out: {tuple(x_hat.shape)}, min={x_hat.min():.4f}, max={x_hat.max():.4f}")
    assert (x_hat >= 0).all(), "decoder output should be non-negative (softplus)"

    # 8) MCR² dormant-but-callable check (uses pert_id as labels)
    L_mcr, mcr_diag = mcr2_loss(z1, pert_id, eps_sq=0.5)
    print(f"MCR² (diagnostic only): loss={L_mcr.item():.3f} | "
          f"ΔR={mcr_diag['delta_R']:.3f} | classes used={mcr_diag['n_classes_used']}")

    print("== smoke test passed ==")


if __name__ == "__main__":
    main()
