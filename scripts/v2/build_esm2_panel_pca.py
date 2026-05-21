"""PCA-reduce the existing 5120-dim ESM2-15B panel down to a smaller dim.

Output: data/vcc/v2_gene_esm2_panel_pca{D}.pt with the same dict layout as
the v1 panel, plus a `pca_components` field for reproducibility / inverse
mapping.

Rationale: v2 wants a smaller protein action representation without a
deep projection MLP that can memorize training perts (the v1 failure
mode on Phase 3.1). We have ESM2-15B (5120-dim) locally but no native
ESM2-650M lookup, so we PCA-reduce 5120 -> target_dim using ONLY the
covered panel rows (~98.3% coverage). Uncovered rows are kept as zero
in the new panel; the downstream ProteinActionEmbedV2 routes them
through its per-gene learned fallback.

Default target dim: 1280 (4× compression, retains the dominant linear
variance of ESM2-15B). Override via --dim.

Run once:
    uv run python scripts/v2/build_esm2_panel_pca.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

IN_PATH = Path("data/vcc/gene_esm2_panel.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1280, help="PCA target dim")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: data/vcc/v2_gene_esm2_panel_pca{DIM}.pt)",
    )
    args = ap.parse_args()

    out_path = args.out or Path(f"data/vcc/v2_gene_esm2_panel_pca{args.dim}.pt")
    if out_path.exists():
        print(f"output {out_path} exists — overwriting")

    print(f"loading source panel: {IN_PATH}")
    src = torch.load(str(IN_PATH), weights_only=False, map_location="cpu")
    emb = src["embeddings"]                     # (n_genes, 5120) float32
    cov = src["coverage"]                       # (n_genes,) bool
    var_names = src["var_names"]
    P_in = emb.shape[1]
    print(f"  shape={tuple(emb.shape)}, covered={int(cov.sum())} / {len(cov)}")
    if args.dim >= P_in:
        raise ValueError(f"--dim {args.dim} must be < source dim {P_in}")

    # Fit PCA on covered rows only. Center first, then SVD.
    X = emb[cov].numpy()
    print(f"  PCA on {X.shape[0]} covered rows × {P_in} dims -> {args.dim} dims")
    t0 = time.perf_counter()
    mean = X.mean(axis=0, keepdims=True)             # (1, P_in)
    Xc = X - mean
    # Economy SVD: U (n, r) Σ (r,) Vᵀ (r, P_in). We want top-k Vᵀ rows.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[: args.dim]                       # (D, P_in)
    explained = (S[: args.dim] ** 2).sum() / (S ** 2).sum()
    print(f"  SVD: {time.perf_counter() - t0:.1f}s; explained variance = {explained:.3f}")

    # Project the full panel. Uncovered rows are zero in `emb`, project to zero
    # offset by -mean·componentsᵀ — but we want uncovered rows to stay zero in
    # the output panel (the fallback path handles them downstream), so we
    # mask after projecting.
    proj_full = (emb.numpy() - mean) @ components.T   # (n_genes, D)
    proj_full[~cov.numpy()] = 0.0
    proj_full = proj_full.astype(np.float32, copy=False)

    payload = {
        "embeddings": torch.from_numpy(proj_full),
        "coverage": cov,                              # unchanged
        "var_names": var_names,
        "n_covered": int(cov.sum()),
        "embed_dim": int(args.dim),
        "source": "PCA of ESM2-15B (5120-dim) panel",
        "pca_components": torch.from_numpy(components.astype(np.float32)),
        "pca_mean": torch.from_numpy(mean.astype(np.float32)),
        "pca_explained_variance_ratio": float(explained),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    sz_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
