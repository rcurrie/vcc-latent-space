# Plan: Pivot to VCC 2025

## Reset rationale

The MNIST proxy validated three things:
1. SIGReg prevents collapse unsupervised
2. AdaLN + joint training beats frozen-encoder MLP (3/4 perts pass DR>2 vs 0/4)
3. The world model framing works

It can't tell us anything more. Proxy data is too clean, too low-dim, and lacks real biological noise. Time to reset and aim at the real benchmark.

## Reset what

Move all MNIST/proxy code to a `legacy/` directory (preserve it, don't delete — git history is one ref but a working baseline is useful for sanity checks). Keep the `lewm_scrna.py` insights (SIGReg, AdaLN, joint training) but reimplement against real data structure.

| Move to legacy/ | Reason |
|---|---|
| `build_proxy_dataset.py` | MNIST proxy generator |
| `geo_jepa_simple.py` | MCR² Phase A/B prototype |
| `geo_jepa_mnist.py` | Earlier prototype |
| `lewm_scrna.py` | Will be reborn as the `src/lewm/` package with the same architecture |
| `u-ctrl-mnist-*.py` | u-CTRL prototypes, unrelated |
| `GEO_JEPA_PLAN.md` | Stale roadmap; replaced by this plan |
| `GEO_JEPA_SESSION_BRIEFING.md` | Pre-pivot context |

Keep at root: `README.md`, `pyproject.toml`, `.python-version`, `.gitignore`, `.venv/`, `data/`, `results/`.

## Environment refresh

Current: Python 3.12, torch 2.10. Latest stable as of May 2026 (per research):

| Package | Current | Update to | Reason |
|---|---|---|---|
| Python | 3.12 | **stay 3.12** | scvi-tools / scanpy fully tested; 3.13 offers no scRNA-seq advantage |
| torch | 2.10 | **2.11.0** | minor MPS perf improvements; low risk |
| scanpy | — | **1.12.1** | needed for h5ad ops |
| anndata | — | **0.12.9** | backed mode + zarr support |
| zarr | — | **3.x** | for h5ad → zarr conversion |
| h5py | — | latest | direct h5ad reads |

Drop unused: `torchvision`, `umap-learn` (move to dev/optional — only needed for diagnostics).

## VCC 2025 evaluation (target metrics)

Submission: AnnData h5ad of predicted post-perturbation gene expression. Evaluator: `cell-eval` from ArcInstitute. Three metrics:

1. **Perturbation Discrimination Score (PDS)** — L1 ranking: does predicted profile sit closest to its true target perturbation vs all others?
2. **Differential Expression Score (DES)** — does the predicted DEG set match the real DEG set?
3. **MAE** — gene-level error on log-normalized expression

Held-out test set: 100 perturbations whose identities we don't see during training. Architecture must generalize to unseen perturbations.

## Phased plan

### Phase 0 — Reset and environment (target: clean working tree)

**Milestone 0.1**: legacy/ directory created, all MNIST code moved, git committed
**Milestone 0.2**: pyproject.toml updated, `uv sync` succeeds, torch 2.11 + scanpy + anndata install cleanly
**Milestone 0.3**: Smoke test — load training h5ad in backed mode, print shape, list 5 perturbations, embed 100 cells through a randomly-initialized MLP. Validates the data path end-to-end.

### Phase 1 — Data pipeline (target: feed cells into PyTorch reliably) — DONE

**Strategy decision**: Skip zarr conversion. The VCC training file's CSR is ~15.5GB in memory (1.93B nnz), which fits comfortably in 32GB. Loading takes 6.5s. Random batch access from in-memory scipy CSR is ~13ms/batch. `num_workers=0` is fine at this scale — model compute dominates, not data loading. Simpler, no upfront 30min wait, no 30GB extra disk.

**Milestone 1.1** (done): Data loading. `lewm.data.load_split(name)` reads h5ad fully into memory as scipy CSR + per-cell metadata (target_gene, batch, perturbation IDs against a vocab).

**Milestone 1.2** (done): `VCCDataset` — wraps a `VCCSplit` plus an `indices` array, so the same split can be subset (controls only for Phase 1, train minus held-out perts, etc). Per-cell normalization (log1p of CP10k) on the fly.

**Milestone 1.3** (done): `StratifiedPerturbationSampler` — each batch contains 25% control cells + 8 perturbations × 48 cells each = 512 total. Guarantees SIGReg sees a non-degenerate distribution.

**Milestone 1.4** (done): `make_internal_val_split` — holds out 10 randomly-chosen non-control perturbations from training as our internal eval set. ~9.5k cells held out from 221k.

**Smoke test results** (on M4, MPS): 6.5s load, 120ms/batch (data + 9.4M-param MLP forward). Full epoch projected at ~52s. 9 unique perts per batch as designed; batch normalization stats look reasonable (x.mean≈0.30, x.std≈0.45).

### Phase 2 — Baseline LeWM on VCC — DONE

**Architecture (`src/lewm/`)**:
- Encoder: 18080 → 512 → 512 → 256, BatchNorm projector, no L2 norm
- JEPAPredictor: 256 → 256 MLP (gene-set masking, Phase 2.1 only)
- ActionEmbed: gene_idx → 64-dim action via MLP over 3 frozen per-gene features (mean expression, dispersion, fraction expressing) computed from training controls. Generalizes to unseen perts because the gene panel is shared across train/val/test.
- PerturbationPredictor: 4 AdaLN-conditioned transformer blocks (zero-init), `(z_cell, gene_idx) → z_post`
- Decoder: 256 → 1024 → 18080 with softplus output

**Training**:
- Phase 2.1: 30 epochs encoder + JEPA on 38k controls. Loss = MSE_pred + SIGReg.
- Phase 2.2: 40 epochs joint encoder + predictor + decoder on 211k cells (10 perts held out as internal val). Each batch contains 25% controls + 8 perts × 48 cells; perturbed cells pair with random controls from the batch as their source. Loss = MSE_pred + decoder_MSE + SIGReg.
- Total: 57 min on M4/MPS, no NaN, SIGReg stays converged at -0.31.

**Internal val** (10 held-out training perts):
- pred_emb_mse: 1.11 → 0.72 (encoder predictions are 56% closer to actual perturbed embeddings than the source control)
- pert_dr: started ~1.0, dropped to 0.5, recovered to 0.64. Below the 1.0 "useful discrimination" line.

**Validation scores** (50 unseen perturbations from VCC validation file):

| Metric | Value | Read |
|---|---|---|
| PDS | **0.500** | Chance. Predictions don't distinguish their target perturbation from others. |
| DES | **0.075** | Faint signal — few of the actual top DEGs appear in our predicted top DEGs. |
| MAE (log1p CP10k) | 0.014 | Low, but most genes don't move under any one knockdown — largely measuring "predict no change". |
| pred_emb MSE | 0.012 | Embeddings are well-aligned in absolute terms. |

**Diagnosis: mean collapse.** The model learned a single "average post-perturbation displacement vector" instead of N=150 different vectors. Classic failure of MSE-trained conditional generators when the conditioning signal is weak relative to within-condition variance: the MSE-optimal point estimate is the conditional mean `E[z_actual | gene_idx, z_ctrl]`, and with ~1500 cells per pert + a small action MLP, that mean is approximately the global mean.

This is exactly the failure mode the flow-matching paper read predicted: MSE point prediction collapses to the conditional mean; distributional matching would preserve modes.

### Phase 3 — Fix mean collapse

1. **Contrastive auxiliary loss on perturbation centroids — DONE, NULL ON OOD.** `lewm.losses.contrastive_centroid_loss`: per batch, compute the actual-centroid `c_g = mean(z_actual over cells with pert g)` for each perturbation present. Each predicted cell `(z_pred_i, gene_g_i)` is a query whose target class is its own centroid (negative-L2² / τ logits, softmax cross-entropy). Default τ=1.0, weight 1.0; gradient flows into both predictor (via `z_pred`) and encoder (via centroid `z_target`).

   **Training-time signal: clearly works.** The training loss converged from 5.3 → 0.16, and the logit gap (own-centroid vs best-other) climbed from -4.5 to +9.1 — predictions sit firmly closer to their own centroid than to others. Internal val DR (10 held-out training perts) held at ~0.77 throughout vs baseline degrading to 0.64.

   **Validation-time signal (50 unseen perts): essentially unchanged.**

   | Metric | Baseline | Contrastive | Δ |
   |---|---|---|---|
   | PDS | 0.500 | 0.506 | +0.006 (noise) |
   | DES | 0.075 | 0.071 | -0.004 |
   | MAE (log1p CP10k) | 0.014 | 0.015 | +0.001 |
   | pred_emb_mse (val) | 0.012 | **0.18** | +0.17 ⚠️ |

   **Diagnosis: in-distribution gain, no out-of-distribution transfer.** The internal val perts share the encoder's learned structure (the encoder has seen sister cells from the same perturbations during training, since the held-out perts were drawn from the *training file* not the *validation file*). The 50 validation perturbations are truly unseen and the model's `ActionEmbed(gene_idx) → action_emb` mapping has no incentive to generalize: the contrastive loss can be satisfied by memorizing the 140 training centroids in the action MLP, which is exactly what it appears to do. The val-time pred_emb_mse jump from 0.012 → 0.18 means contrastive training spread predictions further from actuals on truly-unseen perts (because predictions were pushed apart in the latent space without anchoring around actual val-pert structure).

   **Implication for next steps**: contrastive alone won't fix the architecture's weak inductive bias for perturbation generalization. The 3-feature gene-stat embedding (`MLP(mean, dispersion, frac_expr)`) is too thin a signal — the action MLP can fit training perts arbitrarily without learning a transferable rule. Move to fix #2 (richer action conditioning) and re-test contrastive on top of that.

2. **ESM2 protein-embedding action conditioning (UCE-style) — DONE, FIRST REAL OOD SIGNAL.** Replace `ActionEmbed = MLP(3 features)` with `ProteinActionEmbed = MLP(5120-d ESM2 row)`. ESM2 embeddings are precomputed by `scripts/build_esm2_panel.py` from the human ESM2-15B table; covers 98.3% of the 18,080-gene panel and 100% of all train/val/test perturbations. Fallback to a small learned per-gene embedding for the 1.7% of panel genes (mostly antisense / lincRNA / pseudogenes) without an ESM2 row.

   **Cost**: +2M trainable params (the projection MLP), ~10% slower per epoch due to the larger forward in the action path. ESM2 buffer is 370MB on disk, lives on GPU as a frozen tensor.

   **Validation scores** (50 unseen perturbations from VCC validation file):

   | Metric | Baseline | Contrastive only | ESM2 + Contrastive |
   |---|---|---|---|
   | **PDS** | 0.500 | 0.506 | **0.544** (+0.044) |
   | DES | 0.075 | 0.071 | 0.076 (~equal) |
   | MAE (log1p CP10k) | 0.014 | 0.015 | 0.015 (~equal) |
   | pred_emb_mse (val) | 0.012 | 0.181 | 0.057 (much better than contrastive-only) |

   **First real OOD signal.** PDS is the metric most directly testing perturbation discrimination on unseen genes — the model has a measurable improvement on perturbations it has truly never seen. DES (top-K DEG Jaccard) didn't move, suggesting we're capturing aggregate-direction effects but not yet the specific DEG signatures.

   **The trade-off**: training-time discrimination is much stronger (gap +13 vs +9 for contrastive-only by ep15), and *internal* val DR is *worse* (0.66 vs 0.77) — clear training-pert memorization. But the model trades that in-distribution memorization for some generalization to OOD perts via protein-sequence neighborhoods. This is a healthy sign: the model is learning something protein-similarity-aware, not just memorizing.

   **Why DES didn't move**: PDS rewards L1-distance ranking on the full profile, which is an aggregate signal. DES rewards specific DEG identification, which requires the decoder to map from latent z to fine-grained per-gene effects. The decoder MSE was steady at 0.028 throughout, suggesting the decoder isn't yet specializing per-perturbation. A natural next step is decoder conditioning on the perturbation embedding too (the action signal currently only feeds into the predictor).
2. **Stronger action conditioning.** Replace the 3-feature gene MLP with the target gene's full expression vector across controls (18080-dim) → MLP. Gives the predictor much more signal about what "knocking down gene X" actually means.
3. **Pseudobulk-only training.** Predict per-perturbation mean expression directly. Easier task, won't capture cell-level heterogeneity but should fix PDS immediately. Useful as a sanity check baseline.
4. **Flow matching predictor** — replace MSE point prediction with conditional flow matching. Predicts a velocity field; samples land in the actual post-pert distribution. Larger structural change, addresses root cause.
5. **MCR² overlay** — add MCR² loss with target_gene as 150-class partition labels. Stack on SIGReg.
6. **Gene-set attention encoder** — replace MLP with attention over pathway-grouped genes. Future, important for biology.
7. **Multi-step rollouts** — MPC-style for sequence optimization. Future research.

**Tried and not pursued — population-risk gate (Litman & Guo 2026).** Implemented as `PopRiskAdamW` in `src/lewm/optimizers.py`, kept dormant behind `cfg.use_population_gate`. A/B on Phase 2.1 (4 epochs, fixed seeds, identical loaders):

| α | final pred MSE | SIGReg | params killed |
|---|---|---|---|
| gate off | **0.044** | -0.310 | — |
| α=1.0 (paper default) | 0.338 | -0.276 | 100% |
| α=0.1 | 0.061 | -0.305 | 94% |
| α=0.01 | 0.051 | -0.308 | 77% |

The gate kills too many parameters at any α value where it does anything; at the formal `α = b/(n-b) ≈ 0.0024` it would be essentially a no-op. Diagnosis: scRNA-seq has very high per-parameter gradient variance from biological noise (dropout, batch effects, intra-perturbation cell variation) — the "noise" the gate was designed to suppress is mostly real-but-small signal in our regime. The paper's wins (grokking, PINN, DPO) all involve more structured signal-vs-noise contrasts than ours. Code retained but disabled by default.

## Hardware sanity checks (M4 + 32GB)

Estimated memory: 18080 genes × float32 × 512 batch = 36MB per batch (dense). Encoder + predictor ~5M params = ~20MB weights, ~100MB with optimizer state. Activation memory dominates: rough estimate 500MB-2GB depending on batch size. Should fit comfortably in 32GB.

Bottleneck: data loading from zarr via Apple SSD. Should saturate at ~1GB/s; 15GB train at 5GB raw decompressed → ~3s/epoch raw I/O budget. Realistic: 1-3min/epoch with worker overhead.

If anything blows up, fallbacks: (a) reduce batch size, (b) reduce gene_dim (HVG selection — keep top 4000 highly variable genes), (c) move to k8s GPU cluster.

## File structure post-reset

```
latent-space/
  README.md                    # rewritten for VCC focus
  pyproject.toml               # updated deps
  .python-version              # 3.12
  data/
    vcc/                       # h5ad files (gitignored)
  src/lewm/                    # structured package
    __init__.py
    data.py                    # CSR dataset, stratified sampler
    models.py                  # encoder, AdaLN predictor, decoder
    losses.py                  # SIGReg
    train.py                   # main training loop
    eval.py                    # internal val + score against held-out split
  scripts/
    smoke_test.py              # Phase 1 validation
    phase2_smoke.py            # Phase 2 wiring sanity (1+1 epochs)
    score_validation.py        # load checkpoint, score on validation file
    plot_training.py           # post-training summary figures
  results/vcc/                 # outputs (gitignored)
  legacy/                      # ALL OLD CODE
    geo_jepa_simple.py
    lewm_scrna.py              # save the LeWM v1 for reference
    build_proxy_dataset.py
    GEO_JEPA_PLAN.md
    ...
```

Move from single-file scripts to a small package — VCC code will be too large to keep as one file, and we'll need to import models from training and submission scripts.

## Note on cell-eval / VCC submission

The 2025 competition is closed, so we don't generate cell-eval submission files. Instead `scripts/score_validation.py` runs our own approximations of VCC's three metrics (PDS via L1 ranking, DES via top-K DEG Jaccard, MAE in log1p CP10k) against the validation file. This lets us iterate locally without a network round-trip.

## Open questions for Phase 3

1. **Mean collapse**: which fix is most cost-effective? Try the contrastive auxiliary loss first (smallest change, largest expected uplift on PDS).
2. **Batch effects**: the data has a `batch` column we ignore. Are predictions of unseen perturbations confounded by which batch they came from? Worth a check before/after Phase 3.
3. **Per-cell vs pseudobulk training**: since cell-eval scores can be computed from per-pert pseudobulks, do we lose anything by training on pseudobulks directly? Faster + may sidestep mean collapse entirely.
