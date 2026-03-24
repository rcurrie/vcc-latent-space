# Geo-JEPA Roadmap

## Current status — proxy dataset validation (geo_jepa_simple.py)

- MLP encoder (2000D → 512 → 512 → 128, L2-normalized) + MCR² (d×d eigvalsh) + JEPA predictor
- 7-room proxy dataset from MNIST [0,1,3,4,7,8,9] with MNAR dropout simulating scRNA-seq
- AdamW, batch 512, 40 epochs (~30s on MPS)
- DR 1.78–2.29x, off-diagonal cosine ~0.02, clean UMAP separation
- Hard labels (known digit identity) used for Π in MCR² and for diagnostic coloring

## Completed — trajectory prediction (Phase B)

- TrajectoryPredictor with residual displacement: z_dest = normalize(z_source + delta)
- Perturbation embedding (32D) per type, scaled by dose
- Cosine similarity loss, 40 epochs, frozen encoder
- **Result**: Predicted DR ~1.25-1.37x vs actual DR ~1.9-2.3x
- Predictions land in the correct room neighborhood but lack precision
- Tried: MSE loss, cosine loss, absolute prediction, learned room embeddings, residual displacement — all converge to similar predicted DR
- **Bottleneck hypothesis**: frozen encoder compresses within-room variance, so z_source carries no cell-specific signal; predictor can only learn room-level mappings

## Next steps — proxy dataset

1. **Improve trajectory prediction** — unfreeze encoder with low LR during Phase B, or add richer perturbation representations (dose as separate input, larger pert embedding)
2. **OOD detection** — digit 7 reserved as unseen fate target; measure whether predicted trajectories toward 7 have detectably higher uncertainty or coding-rate anomaly

## Future — toward unsupervised partitioning

3. **Soft / learned Π** — replace hard labels with a clustering head that outputs soft assignments; MCR² natively supports soft partition matrices; the loss itself drives room discovery
4. **R-only baseline** — maximize total coding rate R(Z) without explicit Π; rely on JEPA structure to implicitly organize rooms; compare with soft-Π version
5. **Validate on known biology** — run on a well-characterized scRNA-seq dataset (e.g., Tabula Muris or PBMC 10x) where cell-type labels exist for ground-truth comparison

## Future — architecture upgrades

6. **Gene-set attention encoder** — replace MLP with GeneInteractionPrior (attention over pathway-grouped genes) from geo_jepa_mnist.py once MLP baseline is solid
7. **ReduNet layers** — revisit iterative compression layers; need stable d×d formulation (not Cholesky) to run on MPS
8. **Phase A/B curriculum** — homeostatic scaffold (Phase A) → perturbation learning (Phase B) with EMA anchor; originally implemented in geo_jepa_mnist.py but collapsed; retry on simpler backbone
