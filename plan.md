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
| `lewm_scrna.py` | Will be reborn as `lewm_vcc.py` with same architecture |
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

### Phase 1 — Data pipeline (target: feed cells into PyTorch reliably)

**Milestone 1.1**: Convert h5ad → zarr v3 once (one-time, ~30min). Result: 3 zarr stores in `data/vcc/zarr/{train,val,test}/`. Reason: HDF5 isn't process-safe, breaks `num_workers>0`. Zarr supports concurrent readers.

**Milestone 1.2**: PyTorch `Dataset` that yields `(cell_expression, perturbation_id, batch_id, is_control)` tuples from zarr. Sparse → dense conversion happens per-batch on the worker. Test: iterate one full epoch, time it, target <5min/epoch on M4.

**Milestone 1.3**: Stratified perturbation sampler — within each batch, ensure mix of (control cells, multiple perturbations). Avoids degenerate batches where SIGReg can't compute meaningful Gaussianity.

**Milestone 1.4**: Train/val split tooling — VCC gives us the splits, but we also want a held-out perturbation set within training for our own development metrics. Reserve 10 perturbations from training for "internal val".

### Phase 2 — Baseline LeWM on VCC (target: first end-to-end submission)

**Milestone 2.1**: Phase 1 — homeostatic pretraining on non-targeting controls (~38k cells). Architecture matches `lewm_scrna.py`:
- Encoder: 18080 → 512 → 512 → embed_dim (256), BN projector, no L2 norm
- JEPA gene-set masking (75% context / 25% target)
- Loss: MSE(z_pred, z_target.detach()) + λ·SIGReg(z)
- Eval: UMAP of control cells, check coherent structure (no collapse, batch-effect awareness)

**Milestone 2.2**: Phase 2 — perturbation prediction with AdaLN. Architecture additions:
- Perturbation embedding: `nn.Embedding(num_perturbations, 64)` (~150 train + 50 val + 100 test = up to 300, but we only see ~150 during train; need a strategy for unseen)
- AdaLN-conditioned predictor (4 layers, 4 heads, like `lewm_vcc.py` Phase 2)
- Loss: MSE(z_pred, z_target.detach()) + λ·SIGReg(z_source)
- Train: predict (control cell, perturbation) → perturbed cell embedding

**Decision point — unseen perturbations**: VCC test set has perturbations we never see at training. Options:
  - (a) Train a perturbation embedding per gene; for unseen, use a learned default ("zero perturbation embedding" or mean of seen embeddings)
  - (b) Condition on a **gene-level feature vector** (e.g., target gene's homeostatic expression mean across controls) instead of a learned embedding. This generalizes to unseen genes by construction.
  - (c) Use the gene's expression vector itself as the conditioning signal (knockdown ≈ "remove this gene's contribution")

For first submission: option (b) — use the target gene's row in the var matrix (or an embedding from a frozen pretrained encoder of that gene's expression pattern across controls). Defer (c) as a future refinement.

**Milestone 2.3**: Decoder back to gene space. The world model lives in embedding space, but VCC requires predictions in gene-expression space. Need a decoder: `embed_dim → gene_dim`. Train as part of Phase 2 with reconstruction loss against actual perturbed cells.

**Milestone 2.4**: First end-to-end forward pass — given a (control cell, perturbation gene) pair, produce a predicted gene expression vector. Validate output is in valid range (non-negative log-normalized values).

**Milestone 2.5**: Run on internal val (the 10 training perts we held out). Compute proxy metrics: cosine similarity to actual perturbed cells; per-gene MAE.

**Milestone 2.6**: Generate submission. For each test perturbation, sample N control cells, run prediction, write h5ad in cell-eval format. Run `cell-eval prep` locally to validate format.

**Milestone 2.7**: First VCC submission. Score it. Establish baseline numbers.

### Phase 3 — Iterate (after baseline submitted)

In priority order, gated on baseline performance:

1. **Population-risk gate** (Litman & Guo 2026) — add the per-parameter SNR mask to AdamW. ~10 lines, free experiment, may suppress noise-fitting on real scRNA-seq data.
2. **MCR² overlay** — add MCR² loss with target_gene as the partition labels (we have ~150 classes). Stack on top of SIGReg.
3. **Flow matching predictor** — replace MSE point prediction with conditional flow matching. Treat perturbation outcome as a distribution, not a point. Needed if VCC scoring favors distributional fidelity (it mostly doesn't, but if PDS rewards it, this matters).
4. **Gene-set attention encoder** — replace MLP with attention over pathway-grouped genes. Important if MLP underfits real biology.
5. **Multi-step rollouts** — MPC-style for perturbation sequence optimization. Future research, not for VCC.

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
    vcc/                       # already exists
    vcc-zarr/                  # to be created in Phase 1
  src/                         # NEW: structured code instead of single scripts
    lewm_vcc/
      __init__.py
      data.py                  # zarr dataset, samplers
      models.py                # encoder, predictor, AdaLN
      losses.py                # SIGReg, MSE wrappers
      train.py                 # main training loop
      eval.py                  # internal val, submission generation
      submit.py                # cell-eval format writer
  scripts/
    convert_h5ad_to_zarr.py    # one-time conversion
    smoke_test.py              # phase 0 validation
  results/
    vcc/                       # outputs go here
  legacy/                      # ALL OLD CODE
    geo_jepa_simple.py
    lewm_scrna.py              # save the LeWM v1 for reference
    build_proxy_dataset.py
    GEO_JEPA_PLAN.md
    ...
```

Move from single-file scripts to a small package — VCC code will be too large to keep as one file, and we'll need to import models from training and submission scripts.

## What we're explicitly NOT doing in Phase 2

- Hyperparameter search beyond reasonable defaults (defer until baseline scored)
- Multi-stage curriculum (Phase 1 then Phase 2 separately, no joint pre-training)
- Architecture exploration (one MLP encoder, one AdaLN predictor — match `lewm_scrna.py`)
- Pretrained gene embeddings or external biological priors (defer)
- Ensembling, test-time augmentation
- The population-risk gate (Phase 3 item)
- MCR² (Phase 3 item)

Goal of Phase 2 is "submit something honest and find out where we stand," not "win the leaderboard."

## Verification — concrete pass/fail criteria

| Phase | Pass criterion |
|---|---|
| 0 | `python scripts/smoke_test.py` runs in <30s, prints data shapes |
| 1.1 | Zarr conversion completes without errors, files exist on disk |
| 1.2 | Full epoch iterates in <5min |
| 2.1 | Phase 1 training completes 20 epochs, SIGReg converges, no NaN, UMAP shows non-degenerate structure |
| 2.2 | Phase 2 training completes, prediction MSE on internal val < trivial baseline (predict the control cell unchanged) |
| 2.5 | Internal val: predicted cells closer to actual perturbed cells than to control cells (cosine similarity, mean over perturbations) |
| 2.6 | `cell-eval prep` accepts our h5ad |
| 2.7 | VCC leaderboard returns a score (any score — establishing baseline) |

## Open questions to resolve as we go

1. How does VCC handle multiple control cells per perturbation? Do we predict per-cell or per-perturbation pseudobulks?
2. What's the exact gene order/identity in the test set? Need to align with training var.
3. Are batch effects something we should explicitly model (the data has a `batch` column)?
4. cell-eval — do we run the evaluator locally before submission to validate format?

These get answered during Phase 1 by reading the cell-eval source and any starter docs.
